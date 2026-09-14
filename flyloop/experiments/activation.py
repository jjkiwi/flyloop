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
