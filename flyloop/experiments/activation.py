"""Driving a named population and seeing what answers.

This is the experiment the Shiu et al. model was built for: activate a cell
type, as an optogenetics experiment would, and read the rest of the network. It
needs no eye, no body and no stimulus design, so it is the first thing to run
against a real connectome -- long before a closed loop, and long before
believing anything a closed loop reports.

It is also the cleanest test of a specific published claim. The ommatid project
measured, on real MaleCNS wiring in a robot, that looming recruits DNp04 and
DNp02 while DNp01, DNp09 and MDN stay silent. Driving LC4 directly here asks
the connectome the same question without the camera, the optic lobe model or
the gain calibration in the way, so a disagreement localises to one of those
rather than to the wiring.

As always the number that matters is the comparison with the control graphs,
not the raw firing rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..brain.lif import LIFBrain, LIFParams
from ..connectome.schema import Connectome

#: Populations worth reading out of an adult fly on any visual manipulation.
DEFAULT_READOUT = (
    "DNp01",
    "DNp02",
    "DNp04",
    "DNp09",
    "DNa01",
    "DNa02",
    "MDN",
)


def _sided(c: Connectome, type_: str) -> dict[str, np.ndarray]:
    types = c.neurons["type"].astype(str).to_numpy()
    out: dict[str, np.ndarray] = {}
    if "side" in c.neurons.columns:
        sides = c.neurons["side"].astype(str).to_numpy()
        for side in ("L", "R"):
            out[f"{type_}_{side}"] = np.flatnonzero((types == type_) & (sides == side))
    else:  # pragma: no cover - datasets without a side column
        out[type_] = np.flatnonzero(types == type_)
    return {k: v for k, v in out.items() if len(v)}


@dataclass
class ActivationResult:
    """Firing rates of the readout populations while one population is driven."""

    driven: str
    rate_hz: float
    duration: float
    rates: pd.Series
    total_spikes: int
    active_neurons: int
    notes: list[str] = field(default_factory=list)

    def report(self, threshold: float = 0.5) -> str:
        lines = [
            f"activation: {self.driven} at {self.rate_hz:.0f} Hz for "
            f"{self.duration * 1e3:.0f} ms",
            f"  {self.total_spikes:,} spikes, {self.active_neurons:,} neurons active",
        ]
        live = self.rates[self.rates > threshold].sort_values(ascending=False)
        if len(live):
            lines += [f"    {k:10s} {v:8.1f} Hz" for k, v in live.items()]
        else:
            lines.append("    nothing above threshold -- the network stayed silent")
        silent = sorted(self.rates[self.rates <= threshold].index)
        if silent:
            lines.append(f"  silent: {', '.join(silent)}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def activation_experiment(
    connectome: Connectome,
    *,
    drive: str = "LC4",
    side: str | None = "L",
    rate_hz: float = 50.0,
    duration: float = 0.2,
    readout: tuple[str, ...] = DEFAULT_READOUT,
    params: LIFParams | None = None,
    seed: int = 0,
) -> ActivationResult:
    """Drive one population with Poisson input and measure the readout populations.

    ``side`` restricts the drive to one hemisphere, which is what makes a
    lateralised response interpretable; pass ``None`` to drive both.
    """
    pops = _sided(connectome, drive)
    if side is not None:
        pops = {k: v for k, v in pops.items() if k.endswith(f"_{side}")}
    driven = np.concatenate(list(pops.values())) if pops else np.empty(0, dtype=np.int64)
    if len(driven) == 0:
        raise KeyError(
            f"connectome {connectome.name!r} has no {drive!r} neurons"
            + (f" on side {side!r}" if side else "")
        )

    record: dict[str, np.ndarray] = {}
    for t in readout:
        record.update(_sided(connectome, t))
    record.update({f"{drive}_driven": driven})

    brain = LIFBrain(connectome, params, seed=seed)
    brain.drive.set(driven, rate_hz)
    res = brain.run(duration, record=record)

    rates = pd.Series(
        {k: res.rate(v) for k, v in record.items()}, dtype=float, name="Hz"
    )
    notes = []
    ceiling = 1.0 / (params or LIFParams()).t_refractory
    if (rates > 0.8 * ceiling).any():
        notes.append(
            f"some populations are near the refractory ceiling ({ceiling:.0f} Hz); "
            "the drive is saturating them, so treat the rates as ordinal"
        )
    if res.spike_counts.sum() == 0:
        notes.append("the network never spiked -- raise LIFParams.w_syn or the drive")

    return ActivationResult(
        driven=f"{drive}" + (f"-{side}" if side else ""),
        rate_hz=rate_hz,
        duration=res.duration,
        rates=rates,
        total_spikes=int(res.spike_counts.sum()),
        active_neurons=int((res.spike_counts > 0).sum()),
        notes=notes,
    )


@dataclass
class ActivationComparison:
    """One activation experiment repeated on the real graph and its controls."""

    results: dict[str, ActivationResult]
    table: pd.DataFrame

    def report(self, top: int = 8) -> str:
        real = self.results["original"].rates
        interesting = real.sort_values(ascending=False).head(top).index
        cols = list(self.results)
        lines = [
            f"activation vs control graphs: {self.results['original'].driven} "
            f"at {self.results['original'].rate_hz:.0f} Hz",
            "",
            "  " + f"{'population':14s}" + "".join(f"{c:>17s}" for c in cols),
        ]
        for pop in interesting:
            row = "".join(f"{self.results[c].rates.get(pop, 0.0):17.1f}" for c in cols)
            lines.append(f"  {pop:14s}{row}")
        lines += [
            "",
            "  total spikes  "
            + "".join(f"{self.results[c].total_spikes:17,d}" for c in cols),
        ]
        return "\n".join(lines)


def activation_with_controls(
    connectome: Connectome,
    *,
    controls: tuple[str, ...] = ("rewired", "relabelled", "signs_scrambled"),
    control_seed: int = 0,
    progress: bool = False,
    **kwargs,
) -> ActivationComparison:
    """Run :func:`activation_experiment` on the real graph and on each control.

    A response on the real wiring is only informative next to what the same
    drive does to a graph that has the same degrees but different targets.
    """
    from ..connectome.controls import CONTROLS

    graphs = {"original": connectome}
    for key in controls:
        if key not in CONTROLS:
            raise KeyError(f"unknown control {key!r}; available: {sorted(CONTROLS)}")
        graphs[key] = CONTROLS[key](connectome, seed=control_seed)

    results: dict[str, ActivationResult] = {}
    for name, graph in graphs.items():
        if progress:  # pragma: no cover
            print(f"  {name} ...", flush=True)
        results[name] = activation_experiment(graph, **kwargs)

    table = pd.DataFrame({k: v.rates for k, v in results.items()})
    return ActivationComparison(results=results, table=table)


#: Seed-to-seed spread in total spikes above which a drive rate is not usable
#: for a threshold. Measured on MaleCNS: 8.2x at 5 Hz, 2.7x at 10 Hz, 1.2x at
#: 20 Hz, with three to six seeds.
UNRELIABLE_SPREAD = 2.0

#: Drive rates for a recruitment sweep, spaced so the low end is resolved.
DEFAULT_SWEEP = (5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)


@dataclass
class RecruitmentSweep:
    """Response of each readout population across a range of drive strengths.

    The question this answers is not "does the circuit respond" but "how hard do
    you have to push before each population joins in". A pathway that only
    appears once the drive is far above anything a real stimulus delivers is a
    pathway the connectome contains but the animal may never use, and that
    distinction is invisible at a single drive rate.
    """

    rates: pd.DataFrame  # index: drive rate, columns: population
    spikes: pd.Series  # median network activity per drive rate
    graph: str
    driven: str
    n_seeds: int
    #: Spread of total spikes across seeds at each drive rate, as max / min.
    #: Poisson drive at a low rate is so sparse that the whole-network response
    #: is dominated by whether a few input spikes happen to coincide, and a
    #: handful of seeds cannot average that out.
    variability: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    #: Rates whose spread exceeds :data:`UNRELIABLE_SPREAD`.
    unreliable: tuple[float, ...] = ()

    def thresholds(
        self, criterion: float = 5.0, *, skip_unreliable: bool = True
    ) -> pd.Series:
        """Lowest drive rate at which each population exceeds ``criterion`` Hz.

        NaN means the population never crossed it anywhere in the sweep, which
        is a result: it says the pathway is absent, not merely weak.

        By default the rates flagged in :attr:`unreliable` are skipped. A
        threshold read off a rate whose seed-to-seed spread is eightfold is not
        a threshold, it is the luckiest of a few trials, and leaving those rows
        in makes every weakly-driven population look recruited at the bottom of
        the sweep.
        """
        table = self.rates
        if skip_unreliable and self.unreliable:
            table = table.loc[~table.index.isin(self.unreliable)]
        out = {}
        for pop in table.columns:
            above = table.index[table[pop] > criterion]
            out[pop] = float(above[0]) if len(above) else float("nan")
        return pd.Series(out, name="recruitment_threshold_Hz").sort_values()

    def report(self, criterion: float = 5.0, top: int = 10) -> str:
        th = self.thresholds(criterion)
        recruited = th.dropna()
        never = sorted(th[th.isna()].index)
        lines = [
            f"recruitment sweep on {self.graph}: {self.driven}, "
            f"{self.n_seeds} seed(s) per rate",
            f"  drive rates: {', '.join(f'{r:g}' for r in self.rates.index)} Hz",
            f"  criterion: first rate where the population exceeds {criterion:g} Hz",
            "",
        ]
        if len(recruited):
            lines.append(f"  {'population':12s} {'recruited at':>13s} {'peak Hz':>9s}")
            for pop, rate in recruited.head(top).items():
                lines.append(
                    f"  {pop:12s} {rate:10.0f} Hz {self.rates[pop].max():9.1f}"
                )
        else:
            lines.append("  nothing was recruited anywhere in the sweep")
        if never:
            lines.append("")
            lines.append(f"  never recruited: {', '.join(never)}")
        if self.unreliable:
            lines += [
                "",
                "  excluded as too variable across seeds (max/min spike count > "
                f"{UNRELIABLE_SPREAD:g}x): "
                + ", ".join(f"{r:g} Hz" for r in self.unreliable),
                f"  run more than {self.n_seeds} seeds to use the bottom of the sweep",
            ]
        return "\n".join(lines)


def recruitment_sweep(
    connectome: Connectome,
    *,
    rates: tuple[float, ...] = DEFAULT_SWEEP,
    seeds: tuple[int, ...] = (0,),
    graph_name: str = "original",
    progress: bool = False,
    **kwargs,
) -> RecruitmentSweep:
    """Run :func:`activation_experiment` across drive strengths.

    Repeats at several ``seeds`` and takes the median, because the drive is
    Poisson and a single trial at a low rate is mostly noise.
    """
    rows: dict[float, pd.Series] = {}
    spikes: dict[float, float] = {}
    spread: dict[float, float] = {}
    driven = ""
    for rate in rates:
        per_seed = []
        counts = []
        for seed in seeds:
            res = activation_experiment(connectome, rate_hz=rate, seed=seed, **kwargs)
            per_seed.append(res.rates)
            counts.append(res.total_spikes)
            driven = res.driven
        rows[rate] = pd.concat(per_seed, axis=1).median(axis=1)
        spikes[rate] = float(np.median(counts))
        spread[rate] = max(counts) / max(min(counts), 1)
        if progress:  # pragma: no cover
            print(
                f"    {rate:6.0f} Hz -> {spikes[rate]:,.0f} spikes "
                f"(spread {spread[rate]:.1f}x)",
                flush=True,
            )

    table = pd.DataFrame(rows).T
    table.index.name = "drive_hz"
    variability = pd.Series(spread, name="seed_spread")
    return RecruitmentSweep(
        rates=table,
        spikes=pd.Series(spikes, name="total_spikes"),
        graph=graph_name,
        driven=driven,
        n_seeds=len(seeds),
        variability=variability,
        unreliable=tuple(variability.index[variability > UNRELIABLE_SPREAD]),
    )
