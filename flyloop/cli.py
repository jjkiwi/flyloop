"""Command line entry point: ``flyloop <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load(args) -> object:
    from .connectome.schema import Connectome
    from .connectome.synthetic import synthetic_connectome

    if getattr(args, "data_root", None):
        from .connectome.data_prep import load_dataset

        # The rate model rejects raw synapse counts, so the commands that use it
        # ask for input proportions. Without this every one of them failed on
        # real data with "expects input-proportion weights". An explicit
        # --matrix still wins: a subparser default would silently override the
        # flag the user typed, which is worse than the error they asked for.
        default = "inprop" if getattr(args, "rate_model", False) else "syncount"
        return load_dataset(
            args.data_root,
            args.dataset,
            matrix=getattr(args, "matrix", None) or default,
            min_synapses=args.min_synapses,
            sign_source=args.sign_source,
        )
    if args.connectome:
        return Connectome.load(args.connectome)
    return synthetic_connectome(seed=args.seed)


def cmd_info(args) -> int:
    c = _load(args)
    print(c)
    print(c.report())
    if c.meta.get("synthetic"):
        print(
            "\nNOTE: this is the synthetic fixture, not a nervous system.\n"
            "      Point --connectome at real data before drawing conclusions.\n"
            "      See docs/DATA.md."
        )
    return 0


def cmd_theory(args) -> int:
    from .brain.lif import LIFParams, peak_psp_factor, synapses_to_threshold

    p = LIFParams()
    print("LIF model constants (Shiu et al. 2024 reference values)")
    for f in ("v_rest", "v_threshold", "tau_m", "tau_syn", "t_refractory", "t_delay", "w_syn"):
        print(f"  {f:14s} {getattr(p, f)}")
    print(f"\n  peak PSP per unit g ....... {peak_psp_factor(p):.4f}")
    print(
        f"  synapses to threshold ..... {synapses_to_threshold(p):.0f} (single spike, at rest)"
    )
    print(
        "\nA connectome edge of a handful of synapses cannot fire anything on its\n"
        "own. Behaviour in this model comes from populations firing together."
    )
    return 0


def cmd_looming(args) -> int:
    from .experiments import looming_experiment

    c = _load(args)
    res = looming_experiment(c, n_trials=args.trials, progress=args.verbose)
    print(res.report())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.trials.to_csv(args.out, index=False)
        print(f"  wrote {args.out}")
    return 0 if res.passed else 1


def cmd_baseline(args) -> int:
    from .body import Arena, Pillar, looming_arena
    from .experiments import run_baseline

    for label, arena in (
        ("looming", looming_arena()),
        ("static", Arena([Pillar(x=1.5, y=0.0, radius=0.3, height=0.8)])),
    ):
        df = run_baseline(arena)
        esc = df.index[df["escape"] > 0]
        when = f"{df['t'].iloc[esc[0]]:.2f}s" if len(esc) else "never"
        print(f"  reactive baseline, {label:8s} escape: {when}")
    print(
        "\nCompare against `flyloop looming`. If the baseline is faster and no\n"
        "less discriminating, the connectome is not yet earning its keep."
    )
    return 0


def cmd_activation(args) -> int:
    from .experiments import activation_with_controls

    c = _load(args)
    print(c)
    cmp = activation_with_controls(
        c,
        drive=args.drive,
        side=args.side,
        rate_hz=args.rate,
        duration=args.duration,
        control_seed=args.seed,
        progress=args.verbose,
    )
    print()
    print(cmp.report(top=args.top))
    for note in cmp.results["original"].notes:
        print(f"  note: {note}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cmp.table.to_csv(args.out)
        print(f"  wrote {args.out}")
    return 0


def cmd_sweep(args) -> int:
    from .connectome.controls import rewire_degree_preserving
    from .experiments import recruitment_sweep

    c = _load(args)
    print(c)
    rates = tuple(float(r) for r in args.rates.split(",")) if args.rates else None
    seeds = tuple(range(args.seeds))
    graphs = [("original", c)]
    if args.control:
        graphs.append(("rewired", rewire_degree_preserving(c, seed=args.seed)))

    kw = dict(drive=args.drive, side=args.side, duration=args.duration, seeds=seeds)
    if rates:
        kw["rates"] = rates
    for name, graph in graphs:
        sweep = recruitment_sweep(graph, graph_name=name, progress=args.verbose, **kw)
        print()
        print(sweep.report(criterion=args.criterion))
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            path = out.with_name(f"{out.stem}_{name}{out.suffix or '.csv'}")
            sweep.rates.to_csv(path)
            print(f"  wrote {path}")
    return 0


def cmd_approach(args) -> int:
    from .loop_rate import RateLoop, Target, approach_score

    c = _load(args)
    print(c)
    target = Target(x=args.target_x, y=args.target_y, radius=args.target_radius)
    loop = RateLoop(
        c,
        target=target,
        hops=args.hops,
        dopamine=not args.no_dopamine,
        speed=args.speed,
        learning_rate=args.learning_rate,
    )
    print(
        f"  {loop.plastic.n_synapses:,} KC->MBON synapses under plasticity, "
        f"{len(loop.pam)} PAM neurons carry the reward"
    )
    log = loop.run(args.steps, progress=args.verbose)
    score = approach_score(log, radius=target.radius)
    print(
        f"\n  distance {score['start']:.3f} -> {score['final']:.3f} "
        f"({score['closed_fraction']:.1%} closed) in {score['steps']} steps"
    )
    print(f"  mean |bearing| {score['mean_abs_bearing']:.1f} deg, reached={score['reached']}")
    print(f"  KC->MBON depression {log['dopamine_depression'].iloc[-1]:.5f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(args.out, index=False)
        print(f"  wrote {args.out}")
    return 0


def cmd_embodied(args) -> int:
    from .connectome.controls import rewire_degree_preserving
    from .experiments.embodied import embodied_experiment

    c = _load(args)
    print(c)
    print(
        "\nEach row is a mirror pair from one body seed: the same fly walks once at a\n"
        "target on its right and once at its mirror image on the left. NeuroMechFly\n"
        "drifts tens of degrees on its own, so only the difference means anything.\n"
    )
    graphs = {"real": c}
    if args.control:
        graphs["rewired"] = rewire_degree_preserving(c, seed=args.seed)
    for label, g in graphs.items():
        table = embodied_experiment(
            g, seeds=tuple(range(args.seeds)), bearing_deg=args.bearing, steps=args.steps
        )
        print(
            f"  {label:8s} fixation {table.fixation_deg.mean():+7.2f} deg "
            f"(sd {table.fixation_deg.std():.2f}), "
            f"commanded turn {table.commanded_turn.mean():+.4f}"
        )
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.with_name(f"{path.stem}_{label}{path.suffix or '.csv'}")
            table.to_csv(path, index=False)
            print(f"    wrote {path}")
    return 0


#: Stages a visual decision passes through, for `flyloop atlas`.
_ATLAS_STAGES = (
    ("optic lobe", ("ol_intrinsic", "ol_sensory")),
    ("visual projection", ("visual_projection", "visual_centrifugal")),
    ("central brain", ("cb_intrinsic", "cb_sensory")),
    ("descending", ("descending_neuron",)),
    ("nerve cord", ("vnc_intrinsic", "vnc_motor")),
)


def cmd_atlas(args) -> int:
    import numpy as np
    import pandas as pd

    from .viz import Atlas, lateralised_activation

    c = _load(args)
    print(c)
    atlas = Atlas.from_connectome(c)
    print(" ", atlas.report())

    df = lateralised_activation(c, bearing_deg=args.bearing)
    lat, resp = df["lateralised"].to_numpy(), df["both"].to_numpy() > 0
    print(
        f"\nObject at {args.bearing:+.0f} deg against its mirror image at "
        f"{-args.bearing:+.0f} deg.\n{resp.sum():,} neurons respond at all; the "
        "difference between the two is what encodes which way to turn.\n"
    )
    sc = c.neurons["super_class"].to_numpy()
    for label, classes in _ATLAS_STAGES:
        m = np.isin(sc, classes) & resp
        if m.any():
            print(f"  {label:20s} {np.abs(lat[m]).mean():.3f}   ({int(m.sum()):,} cells)")
    print("\n  most lateralised cell types:")
    types = c.neurons["type"].astype(str).to_numpy()
    by_type = pd.Series(np.abs(lat)).groupby(types).mean().sort_values(ascending=False)
    for name, v in by_type.head(8).items():
        print(f"    {str(name):12s} {v:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n  wrote {args.out}")
    return 0


def cmd_record(args) -> int:
    from .app import record_episode, write_atlas, write_index
    from .experiments.embodied import target_at
    from .hybrid import HybridLoop

    c = _load(args)
    print(c)
    body = None
    if args.physics:
        from .body.nmf_body import NeuroMechFlyBody

        body = NeuroMechFlyBody(control_dt=args.control_dt, seed=args.seed)
    loop = HybridLoop(
        c,
        target=target_at(args.bearing),
        dt=args.control_dt,
        gain=args.gain,
        coupling_hops=args.coupling_hops,
        body=body,
    )
    print(" ", loop.learned.report())
    print(
        f"  body {type(loop.body).__name__}, joints "
        f"{'recorded' if loop.body_has_joints else 'absent (kinematic stub)'}"
    )
    ep = record_episode(
        loop,
        train_trials=args.train,
        behave_steps=args.steps,
        behave_odour=None if args.no_odour else "trained",
        progress=args.verbose,
    )
    out = Path(args.out)
    # Episodes live in their own folders; the rig and atlas are shared, so a
    # gain comparison does not reload 8 MB of geometry to change one number.
    name = args.name or f"gain{args.gain:g}"
    ep.save(out / "episodes" / name)
    write_atlas(c, out)
    write_index(out)
    print(
        f"\n  {ep.manifest['live_types']:,} of {ep.manifest['all_types']:,} cell "
        f"types responded; wrote {out}/episodes/{name}"
    )
    return 0


def cmd_controls(args) -> int:
    from .experiments import looming_with_controls

    c = _load(args)
    cmp = looming_with_controls(c, n_trials=args.trials, seed=args.seed, progress=args.verbose)
    print(cmp.report())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cmp.table.to_csv(args.out)
        print(f"  wrote {args.out}")
    return 0 if cmp.wiring_dependent else 1


def cmd_columns(args) -> int:
    from .vision.columns import load_columnar_table

    t = load_columnar_table(args.table)
    types = [c for c in t.columns if c not in ("hex1", "hex2", "x", "y")]
    print(f"{args.table}: {len(t)} hexagonal columns, {len(types)} cell types")
    print(f"  cell types: {', '.join(types)}")
    print(
        "\nThese are the published columnar tables that give a real retinotopy.\n"
        "The right optic lobe only -- there is no left-eye table."
    )
    return 0


def cmd_loop(args) -> int:
    from .body import KinematicBody, cluttered_arena, looming_arena
    from .loop import ClosedLoop

    c = _load(args)
    arena = cluttered_arena(seed=args.seed) if args.arena == "clutter" else looming_arena()
    body = KinematicBody(arena, dt=args.control_dt)
    loop = ClosedLoop(c, body, seed=args.seed)
    res = loop.run(args.duration, progress=args.verbose)
    print(res.summary())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.log.to_csv(args.out, index=False)
        print(f"  wrote {args.out}")
    return 0


def cmd_selftest(args) -> int:
    """Everything the project can check about itself without real data."""
    from .brain.lif import LIFParams, synapses_to_threshold
    from .connectome.synthetic import synthetic_connectome
    from .experiments import looming_experiment

    ok = True
    c = synthetic_connectome()
    print(c.report())
    print(f"\nsynapses to threshold: {synapses_to_threshold(LIFParams()):.0f}")
    for t in ("LC4", "DNp01", "DNp02", "DNp04", "DNp09", "DNa01", "DNa02", "MDN"):
        n = len(c.population(t))
        print(f"  {t:6s} {n:3d} neurons {'ok' if n else 'MISSING'}")
        ok &= n > 0
    res = looming_experiment(c, n_trials=2)
    print()
    print(res.report())
    ok &= res.passed
    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flyloop", description=__doc__)
    ap.add_argument("--connectome", help="directory written by Connectome.save()")
    ap.add_argument(
        "--data-root",
        help="clone of YijieYin/connectome_data_prep, to load a real connectome",
    )
    ap.add_argument("--dataset", default="malecns", help="dataset name under --data-root")
    ap.add_argument("--min-synapses", type=int, default=5)
    ap.add_argument(
        "--matrix",
        choices=("syncount", "inprop", "outprop"),
        default=None,
        help="weighting to load; rate-model commands default to inprop",
    )
    ap.add_argument(
        "--sign-source",
        default="flyloop",
        choices=("flyloop", "dataset"),
        help="whose neurotransmitter signs to use; they differ on ~5%% of neurons",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="describe a connectome and sanity-check it").set_defaults(
        func=cmd_info
    )
    sub.add_parser("theory", help="model constants and what they imply").set_defaults(
        func=cmd_theory
    )
    sub.add_parser("baseline", help="run the non-connectome control").set_defaults(
        func=cmd_baseline
    )
    sub.add_parser("selftest", help="end-to-end check on the synthetic fixture").set_defaults(
        func=cmd_selftest
    )

    p = sub.add_parser("looming", help="escape acceptance experiment with controls")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--out", help="write per-trial CSV here")
    p.set_defaults(func=cmd_looming)

    p = sub.add_parser("activation", help="drive a population and read the descending neurons")
    p.add_argument("--drive", default="LC4", help="cell type to stimulate")
    p.add_argument("--side", default="L", choices=("L", "R", "both"))
    p.add_argument("--rate", type=float, default=50.0, help="Poisson drive rate, Hz")
    p.add_argument("--duration", type=float, default=0.2)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", help="write the comparison table CSV here")
    p.set_defaults(func=cmd_activation)

    p = sub.add_parser(
        "sweep", help="how hard must a population be driven before each target joins"
    )
    p.add_argument("--drive", default="LC4")
    p.add_argument("--side", default="L", choices=("L", "R", "both"))
    p.add_argument("--rates", help="comma-separated drive rates in Hz")
    p.add_argument("--seeds", type=int, default=3, help="repeats per drive rate")
    p.add_argument("--duration", type=float, default=0.2)
    p.add_argument("--criterion", type=float, default=5.0, help="recruited above this Hz")
    p.add_argument("--control", action="store_true", help="also sweep a rewired graph")
    p.add_argument("--out", help="write per-graph CSVs based on this path")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("approach", help="fly at a target, with dopamine that grows nearer")
    p.add_argument("--steps", type=int, default=45)
    p.add_argument("--hops", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.2)
    p.add_argument("--target-x", type=float, default=1.4)
    p.add_argument("--target-y", type=float, default=0.6)
    p.add_argument("--target-radius", type=float, default=0.15)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--no-dopamine", action="store_true")
    p.add_argument("--out", help="write the per-step log CSV here")
    p.set_defaults(func=cmd_approach, rate_model=True)

    p = sub.add_parser("embodied", help="drive NeuroMechFly with the same descending readout")
    p.add_argument("--steps", type=int, default=30, help="control steps per episode")
    p.add_argument("--seeds", type=int, default=3, help="body seeds, two episodes each")
    p.add_argument("--bearing", type=float, default=35.0, help="target bearing, degrees")
    p.add_argument("--control", action="store_true", help="also run a rewired graph")
    p.add_argument("--out", help="write per-graph CSVs based on this path")
    p.set_defaults(func=cmd_embodied, rate_model=True)

    p = sub.add_parser("atlas", help="where the deciding neurons are in the animal")
    p.add_argument("--bearing", type=float, default=35.0, help="object bearing, degrees")
    p.add_argument("--out", help="write the per-neuron activation CSV here")
    p.set_defaults(func=cmd_atlas, rate_model=True)

    p = sub.add_parser("record", help="record an episode for the web viewer")
    p.add_argument("--out", default="app/data", help="directory to write into")
    p.add_argument("--steps", type=int, default=40, help="behaviour control steps")
    p.add_argument("--train", type=int, default=8, help="conditioning trials")
    p.add_argument("--bearing", type=float, default=35.0)
    p.add_argument("--gain", type=float, default=1.0, help="1.0 is the measured anatomy")
    p.add_argument("--control-dt", type=float, default=0.05)
    p.add_argument("--physics", action="store_true", help="NeuroMechFly instead of the stub")
    p.add_argument("--no-odour", action="store_true", help="behave with no odour present")
    p.add_argument("--name", help="episode folder name; defaults to gain<N>")
    p.add_argument(
        "--coupling-hops",
        type=int,
        default=1,
        help="how far back to trace MBON input to DNa02; 4 includes the indirect routes",
    )
    p.set_defaults(func=cmd_record, rate_model=True)

    p = sub.add_parser(
        "controls", help="looming experiment vs shuffled/relabelled control graphs"
    )
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--out", help="write the comparison table CSV here")
    p.set_defaults(func=cmd_controls)

    p = sub.add_parser("columns", help="describe a published columnar retinotopy table")
    p.add_argument("--table", default="mcns_right", choices=("mcns_right", "fafb_right"))
    p.set_defaults(func=cmd_columns)

    p = sub.add_parser("loop", help="run the closed sensorimotor loop")
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--arena", choices=("loom", "clutter"), default="loom")
    p.add_argument("--out", help="write the per-step log CSV here")
    p.set_defaults(func=cmd_loop)

    args = ap.parse_args(argv)
    if getattr(args, "side", None) == "both":
        args.side = None
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
