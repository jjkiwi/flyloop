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
                    "peak_dnp09": float(
                        max(log["dn_DNp09_L"].max(), log["dn_DNp09_R"].max())
                    ),
                    "peak_gf": float(max(log["dn_GF_L"].max(), log["dn_GF_R"].max())),
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
