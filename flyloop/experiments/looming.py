"""The acceptance experiment: does a looming object trigger escape?

This is the first thing to run against any connectome, real or synthetic.  It
is deliberately built as a *comparison*, not a demo:

  * ``looming``      an object on a collision course -- escape expected
  * ``static``       the same object, not moving -- no escape expected
  * ``receding``     the object moving away -- no escape expected

A run that escapes in every condition has learned nothing; it has a broken
adaptation stage.  A run that escapes in none has a silent network -- sweep
``LIFParams.w_syn`` before concluding anything about the connectome.

Reporting only the looming condition, which is what most of the community
demos do, makes any wiring diagram look like it works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..body.kinematic import Arena, KinematicBody, Pillar, looming_arena
from ..brain.lif import LIFParams
from ..connectome.schema import Connectome
from ..loop import ClosedLoop

CONDITIONS = ("looming", "static", "receding")


def _arena(condition: str, speed: float, distance: float) -> Arena:
    if condition == "looming":
        return looming_arena(approach_speed=speed, start_distance=distance)
    if condition == "static":
        return Arena([Pillar(x=distance, y=0.0, radius=0.3, height=0.8)])
    if condition == "receding":
        return Arena([Pillar(x=0.45, y=0.0, radius=0.3, height=0.8, vx=+speed)])
    raise ValueError(f"unknown condition {condition!r}")


@dataclass
class LoomingResult:
    """Per-trial outcomes and the verdict."""

    trials: pd.DataFrame
    escape_rate: dict[str, float]
    median_latency: float | None
    passed: bool
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = ["looming acceptance experiment"]
        for cond in CONDITIONS:
            if cond in self.escape_rate:
                lines.append(f"  {cond:9s} escape rate {self.escape_rate[cond]:5.0%}")
        if self.median_latency is not None:
            lines.append(f"  median escape latency {self.median_latency:.2f} s")
        lines.append(f"  VERDICT: {'PASS' if self.passed else 'FAIL'}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def looming_experiment(
    connectome: Connectome,
    *,
    n_trials: int = 5,
    duration: float = 2.6,
    speed: float = 0.5,
    distance: float = 1.5,
    params: LIFParams | None = None,
    conditions: tuple[str, ...] = CONDITIONS,
    control_dt: float = 0.01,
    progress: bool = False,
) -> LoomingResult:
    """Run the experiment and decide whether the escape pathway is functional.

    The criterion is a *difference*: escape in the majority of looming trials
    and in a minority of control trials.  Each trial uses a different random
    seed, so the escape rate is a real rate and not one lucky run.
    """
    rows = []
    for cond in conditions:
        for trial in range(n_trials):
            body = KinematicBody(_arena(cond, speed, distance), dt=control_dt)
            loop = ClosedLoop(connectome, body, params=params, seed=1000 + trial)
            res = loop.run(duration)
            log = res.log
            escaped = bool(log["escape"].max())
            idx = log.index[log["escape"] > 0]
            latency = float(log["t"].iloc[idx[0]]) if len(idx) else np.nan
            rows.append(
                {
                    "condition": cond,
                    "trial": trial,
                    "escaped": escaped,
                    "latency": latency,
                    "peak_escape": float(
                        max(
                            max(
                                log.get(f"dn_{n}_L", pd.Series([0.0])).max(),
                                log.get(f"dn_{n}_R", pd.Series([0.0])).max(),
                            )
                            for n in ("DNp01", "GF", "DNp02", "DNp04", "DNp09")
                        )
                    ),
                    **{
                        f"peak_{name}": float(
                            max(
                                log.get(f"dn_{name}_L", pd.Series([0.0])).max(),
                                log.get(f"dn_{name}_R", pd.Series([0.0])).max(),
                            )
                        )
                        for name in ("DNp01", "DNp02", "DNp04", "DNp09")
                    },
                    "escape_source": log["escape_source"].dropna().iloc[0]
                    if log["escape_source"].notna().any()
                    else None,
                    "spikes": int(res.spike_counts.sum()),
                }
            )
            if progress:  # pragma: no cover
                print(f"  {cond} trial {trial}: escaped={escaped}", flush=True)

    trials = pd.DataFrame(rows)
    rate = trials.groupby("condition")["escaped"].mean().to_dict()
    loom_rate = rate.get("looming", 0.0)
    controls = [c for c in conditions if c != "looming"]
    control_rate = max((rate.get(c, 0.0) for c in controls), default=0.0)

    notes: list[str] = []
    if trials["spikes"].sum() == 0:
        notes.append(
            "the network never spiked: raise LIFParams.w_syn or check that the "
            "input cell type actually receives the drive"
        )
    if loom_rate == 1.0 and control_rate == 1.0:
        notes.append(
            "escape in every condition, including controls -- the temporal "
            "adaptation stage is probably not working"
        )
    passed = bool(loom_rate > 0.5 and control_rate < 0.5)

    lat = trials.loc[trials["condition"] == "looming", "latency"].dropna()
    return LoomingResult(
        trials=trials,
        escape_rate=rate,
        median_latency=float(lat.median()) if len(lat) else None,
        passed=passed,
        notes=notes,
    )


@dataclass
class ControlComparison:
    """The looming experiment run on the real graph and on control graphs.

    Two different questions are being asked, and conflating them is how a
    connectome demo ends up claiming more than it measured:

    * :func:`looming_experiment` asks whether the response is **stimulus
      selective** -- looming versus a static and a receding object.
    * this comparison asks whether it is **wiring dependent** -- the real graph
      versus graphs with the same degrees, the same topology, or the same
      transmitter counts.

    The primary statistic here is the *peak firing rate of the escape
    populations*, not the fraction of trials that escaped. Binary escape rate
    saturates: on the synthetic fixture every graph except the rewired one
    escapes in 100% of looming trials, so the rate cannot tell them apart even
    though their latencies differ by a factor of three. A continuous statistic
    also matches what a real experiment reports -- ommatid compared a change in
    Hz, not a hit count.
    """

    results: dict[str, LoomingResult]
    table: pd.DataFrame
    wiring_dependent: bool
    ratio: float
    primary: str = "peak_escape"

    def _loom(self, graph: str) -> pd.DataFrame:
        t = self.results[graph].trials
        return t[t["condition"] == "looming"]

    def report(self) -> str:
        lines = [
            "looming experiment vs control graphs",
            "  primary statistic: peak escape-population rate on looming trials",
            "",
            f"  {'graph':18s} {'peak Hz':>9s} {'latency s':>10s} {'escape':>7s}  carried by",
        ]
        for graph in self.results:
            loom = self._loom(graph)
            peak = float(loom[self.primary].median())
            lat = loom["latency"].median()
            rate = float(loom["escaped"].mean())
            src = loom["escape_source"].dropna()
            carried = ", ".join(sorted(src.unique())) if len(src) else "-"
            lat_s = "-" if pd.isna(lat) else f"{lat:.2f}"
            lines.append(
                f"  {graph:18s} {peak:9.1f} {lat_s:>10s} {rate:6.0%}  {carried}"
            )
        lines += [
            "",
            f"  real / best control on {self.primary}: {self.ratio:.2f}x",
            f"  VERDICT: {'wiring-dependent' if self.wiring_dependent else 'NOT established'}",
        ]
        if not self.wiring_dependent:
            lines.append(
                "  The controls do as well as the real graph on the primary "
                "statistic. Whatever the loop is doing, the measured "
                "connectivity is not what makes it happen."
            )
        return "\n".join(lines)


def looming_with_controls(
    connectome: Connectome,
    *,
    n_trials: int = 5,
    seed: int = 0,
    controls: tuple[str, ...] = ("rewired", "relabelled", "signs_scrambled"),
    min_ratio: float = 1.5,
    progress: bool = False,
    **kwargs,
) -> ControlComparison:
    """Run the looming experiment on the real graph and on each control graph.

    This is the experiment that licenses a claim. A looming response on the real
    wiring means nothing on its own: a shuffle of the same graph will also
    produce escape-like activity, because looming is simply the strongest visual
    input. What the real wiring has to add is *where that input lands*.

    ``min_ratio`` is how many times larger the real graph's escape response must
    be than the best control's. The default of 1.5 is deliberately blunt; with
    enough trials, report a permutation test on the underlying rates rather than
    leaning on a threshold.
    """
    from ..connectome.controls import CONTROLS

    graphs: dict[str, Connectome] = {"original": connectome}
    for key in controls:
        if key not in CONTROLS:
            raise KeyError(f"unknown control {key!r}; available: {sorted(CONTROLS)}")
        graphs[key] = CONTROLS[key](connectome, seed=seed)

    results: dict[str, LoomingResult] = {}
    rows = []
    for name, graph in graphs.items():
        if progress:  # pragma: no cover
            print(f"== {name} ==", flush=True)
        res = looming_experiment(graph, n_trials=n_trials, progress=progress, **kwargs)
        results[name] = res
        loom = res.trials[res.trials["condition"] == "looming"]
        rows.append(
            {
                "graph": name,
                "peak_escape": float(loom["peak_escape"].median()),
                "median_latency": res.median_latency,
                **{f"escape_{k}": v for k, v in res.escape_rate.items()},
                "total_spikes": int(res.trials["spikes"].sum()),
            }
        )

    table = pd.DataFrame(rows).set_index("graph")
    real = float(table.loc["original", "peak_escape"])
    controls_peak = [
        float(table.loc[g, "peak_escape"]) for g in graphs if g != "original"
    ]
    best = max(controls_peak, default=0.0)
    ratio = real / best if best > 0 else float("inf") if real > 0 else 0.0
    return ControlComparison(
        results=results,
        table=table,
        wiring_dependent=bool(results["original"].passed and ratio >= min_ratio),
        ratio=ratio,
    )
