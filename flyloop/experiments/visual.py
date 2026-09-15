"""Showing the fly something, instead of injecting current into its brain.

Runs 1 to 3 drove LC4 directly. That measures what a connectome connects to
LC4, but LC4 activity was imposed, so nothing between the eye and LC4 was ever
exercised and no visual hypothesis was testable.

Here a pattern is painted onto the fly's own hexagonal columns and delivered to
the lamina monopolar cells that receive light decrements. LC4 activity, if any,
has to travel there through the animal's own optic lobe.

The protocol follows ommatid's pre-registered one: a baseline window with no
stimulus, then the stimulus, and the reported quantity is the **change** in
firing rate, not the absolute rate. Their conditions are reproduced too --
looming with a receding and a static control, drifting gratings in both
directions for the optomotor hypothesis, and dark hemifields for phototaxis.

Two things to hold on to when reading the output:

* A stimulus that lights more columns delivers more drive. The gratings and
  hemifields cover about 440 of 892 columns against looming's peak of 169, so
  ``columns_driven`` is reported and conditions are not directly comparable on
  raw magnitude. The control graphs are what make a difference interpretable.
* This model has no photoreceptor stage, so the histaminergic sign inversion is
  applied by hand: a dark object drives L1 and L2 in the columns it covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..brain.lif import LIFBrain, LIFParams
from ..connectome.schema import Connectome
from ..vision.hexstim import HexStimulus, column_neurons, hex_coords, stimulus_set
from .activation import DEFAULT_READOUT, _sided

#: Populations to read. The visual ones matter as much as the descending ones
#: here: if LC4 never fires, nothing downstream of it means anything.
VISUAL_READOUT = ("LC4", "LPLC2", "L1", "L2", "Mi1", "T4a", "T5a", *DEFAULT_READOUT)


@dataclass
class VisualResult:
    """Change in firing rate of each readout population during one stimulus."""

    stimulus: str
    baseline: pd.Series
    during: pd.Series
    delta: pd.Series
    columns_driven: float
    total_spikes: int
    seed: int
    notes: list[str] = field(default_factory=list)

    def report(self, threshold: float = 1.0, top: int = 10) -> str:
        live = self.delta[self.delta.abs() > threshold].sort_values(
            key=np.abs, ascending=False
        )
        lines = [
            f"{self.stimulus}: {self.columns_driven:.0f} columns lit on average, "
            f"{self.total_spikes:,} spikes",
        ]
        if len(live):
            lines += [
                f"    {k:12s} {self.during[k]:7.1f} Hz  (baseline {self.baseline[k]:5.1f}, "
                f"delta {v:+7.1f})"
                for k, v in live.head(top).items()
            ]
        else:
            lines.append("    nothing changed by more than the threshold")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _record_map(c: Connectome, readout: tuple[str, ...]) -> dict[str, np.ndarray]:
    record: dict[str, np.ndarray] = {}
    for t in readout:
        record.update(_sided(c, t))
    return {k: v for k, v in record.items() if len(v)}


def visual_experiment(
    connectome: Connectome,
    stimulus: HexStimulus,
    *,
    side: str = "R",
    cell_types: tuple[str, ...] = ("L1", "L2"),
    rate_hz: float = 100.0,
    frame_duration: float = 0.025,
    baseline_duration: float = 0.1,
    readout: tuple[str, ...] = VISUAL_READOUT,
    params: LIFParams | None = None,
    seed: int = 0,
    columns: dict[str, np.ndarray] | None = None,
    background_hz: float = 0.0,
    background_types: tuple[str, ...] | None = None,
) -> VisualResult:
    """Present one stimulus to one eye and measure the change in firing rates.

    The baseline window runs first with no drive at all. Because this model has
    no basal firing, that baseline is usually exactly zero -- which is itself
    worth seeing, since it means every reported change is evoked rather than
    modulated.
    """
    cols = columns if columns is not None else column_neurons(
        connectome, cell_types=cell_types, side=side
    )
    record = _record_map(connectome, readout)
    brain = LIFBrain(connectome, params, seed=seed)
    if background_hz > 0:
        if background_types is None:
            brain.drive.set_background(None, background_hz)
        else:
            types = connectome.neurons["type"].astype(str)
            idx = np.flatnonzero(types.isin(background_types).to_numpy())
            brain.drive.set_background(idx, background_hz)

    base = brain.run(baseline_duration, record=record)
    baseline = pd.Series(
        {k: base.rate(v) for k, v in record.items()}, dtype=float
    )

    counts = np.zeros(connectome.n, dtype=np.int64)
    lit = []
    for frame in stimulus.frames:
        idx = [cols[label] for label in frame if label in cols]
        driven = np.concatenate(idx) if idx else np.empty(0, dtype=np.int64)
        lit.append(len(frame))
        brain.drive.clear()
        if len(driven):
            brain.drive.set(driven, rate_hz)
        res = brain.run(frame_duration, record=record)
        counts += res.spike_counts

    during_duration = stimulus.n_frames * frame_duration
    during = pd.Series(
        {
            k: float(counts[v].sum() / (len(v) * during_duration))
            for k, v in record.items()
        },
        dtype=float,
    )

    notes = []
    if background_hz == 0 and baseline.max() > 0:
        notes.append(
            "the baseline window is not silent although no background was set; "
            "this model has no basal firing, so check what is driving it"
        )
    if background_hz == 0:
        notes.append(
            "no background activity: inhibitory pathways cannot transmit, so the "
            "ON pathway through L1 is mute by construction (see docs/RESULTS.md)"
        )
    if during.max() == 0:
        notes.append("nothing fired during the stimulus at all")

    return VisualResult(
        stimulus=stimulus.name,
        baseline=baseline,
        during=during,
        delta=during - baseline,
        columns_driven=float(np.mean(lit)),
        total_spikes=int(counts.sum()),
        seed=seed,
        notes=notes,
    )


@dataclass
class VisualProtocol:
    """Every condition, repeated over seeds, as one table of rate changes."""

    deltas: pd.DataFrame  # rows: condition, columns: population (median over seeds)
    spread: pd.Series  # seed-to-seed spread in total spikes, per condition
    spikes: pd.Series
    columns: pd.Series
    graph: str
    n_seeds: int

    def report(self, populations: tuple[str, ...] = (), top: int = 6) -> str:
        cols = [p for p in populations if p in self.deltas.columns]
        if not cols:
            moved = self.deltas.abs().max().sort_values(ascending=False)
            cols = list(moved.head(top).index)
        lines = [
            f"visual protocol on {self.graph}, {self.n_seeds} seeds per condition",
            "  change in firing rate from baseline, Hz",
            "",
            "  "
            + f"{'condition':14s}"
            + "".join(f"{c:>12s}" for c in cols)
            + f"{'spikes':>10s}",
        ]
        for cond in self.deltas.index:
            row = "".join(f"{self.deltas.loc[cond, c]:12.1f}" for c in cols)
            lines.append(f"  {cond:14s}{row}{self.spikes[cond]:10,.0f}")
        noisy = self.spread[self.spread > 2.0]
        if len(noisy):
            lines += [
                "",
                "  too variable across seeds (max/min spikes > 2x): "
                + ", ".join(f"{c} {v:.1f}x" for c, v in noisy.items()),
            ]
        return "\n".join(lines)


def visual_protocol(
    connectome: Connectome,
    *,
    side: str = "R",
    n_time: int = 8,
    seeds: tuple[int, ...] = (0, 1, 2),
    graph_name: str = "original",
    conditions: tuple[str, ...] | None = None,
    progress: bool = False,
    **kwargs,
) -> VisualProtocol:
    """Run the whole stimulus set, repeated over seeds."""
    coords = hex_coords(connectome, cell_type="L1", side=side)
    stimuli = stimulus_set(coords, n_time=n_time)
    if conditions:
        stimuli = {k: v for k, v in stimuli.items() if k in conditions}
    cols = column_neurons(
        connectome, cell_types=kwargs.get("cell_types", ("L1", "L2")), side=side
    )

    deltas, spikes, spread, lit = {}, {}, {}, {}
    for name, stim in stimuli.items():
        per_seed, counts = [], []
        for seed in seeds:
            r = visual_experiment(
                connectome, stim, side=side, seed=seed, columns=cols, **kwargs
            )
            per_seed.append(r.delta)
            counts.append(r.total_spikes)
            lit[name] = r.columns_driven
        deltas[name] = pd.concat(per_seed, axis=1).median(axis=1)
        spikes[name] = float(np.median(counts))
        spread[name] = max(counts) / max(min(counts), 1)
        if progress:  # pragma: no cover
            print(
                f"    {name:12s} {spikes[name]:>9,.0f} spikes (spread {spread[name]:.1f}x)",
                flush=True,
            )

    return VisualProtocol(
        deltas=pd.DataFrame(deltas).T,
        spread=pd.Series(spread),
        spikes=pd.Series(spikes),
        columns=pd.Series(lit),
        graph=graph_name,
        n_seeds=len(seeds),
    )
