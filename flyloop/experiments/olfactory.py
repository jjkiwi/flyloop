"""Approaching an odour source, where dopamine can actually change something.

Run 6 delivered reward to the fly's own PAM cluster, scaled by closeness to a
visual target, and it changed the KC->MBON weights and nothing else. The graph
said why: Kenyon cells receive 99.5% of their input from the olfactory pathway
and almost none from vision, and the mushroom body's output lands on descending
neurons the visual steering loop does not use.

So this moves the same reward to the modality the mushroom body is for. The fly
approaches an odour source; the odour gets stronger as it nears, and so does the
dopamine. Afterwards the mushroom body's response to that odour is measured
against an odour that was never paired with reward.

That is differential conditioning, the standard *Drosophila* paradigm, and it
carries its own control: the unpaired odour went through the same network, the
same number of steps and the same readout, and differs only in having had no
dopamine.

**Kenyon cell gain is calibrated, not guessed.** In the animal roughly 5% of
Kenyon cells respond to a given odour, and that sparse code is what makes odours
separable. At this model's default gain a single glomerulus activates 88% of
them and two different odours overlap by 99.8%, which makes conditioning
impossible by construction -- depressing one odour's cells depresses every
odour's cells. :data:`SPARSE_KC_SLOPE` is the value measured to reproduce the
published sparseness; see the calibration table in docs/RESULTS.md.

The window is narrow. At slope 1.0 the code is 82% dense; at 0.35 it is 4.5%
sparse; at 0.1 nothing fires at all. Report the slope with any result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..brain.dopamine import proximity_reward
from ..brain.rate import RateBrain, steady_state
from ..connectome.schema import Connectome

#: Kenyon cell tanh slope reproducing the ~5% sparse odour code.
SPARSE_KC_SLOPE = 0.35

#: Threshold above which a Kenyon cell counts as responding.
RESPONSE_THRESHOLD = 0.05


def olfactory_receptors(c: Connectome) -> np.ndarray:
    """Rows of the olfactory receptor neurons."""
    rows = np.flatnonzero(
        c.neurons["type"].astype(str).str.startswith("ORN").to_numpy()
    )
    if len(rows) == 0:
        raise KeyError(f"connectome {c.name!r} has no ORN neurons")
    return rows


def mushroom_body(c: Connectome) -> dict[str, np.ndarray]:
    """Kenyon cells and mushroom body output neurons."""
    types = c.neurons["type"].astype(str)
    return {
        "kc": np.flatnonzero(types.str.startswith("KC").to_numpy()),
        "mbon": np.flatnonzero(types.str.startswith("MBON").to_numpy()),
    }


def kc_slopes(c: Connectome, slope: float = SPARSE_KC_SLOPE) -> dict[str, float]:
    """Per-cell-type slope dictionary setting every Kenyon cell type to ``slope``."""
    types = c.neurons["type"].astype(str)
    return {t: slope for t in sorted(types[types.str.startswith("KC")].unique())}


def odour(c: Connectome, glomeruli: tuple[str, ...], receptors: np.ndarray) -> np.ndarray:
    """Input vector over ``receptors`` for an odour activating named glomeruli."""
    want = c.neurons["type"].astype(str).to_numpy()[receptors]
    v = np.zeros(len(receptors), dtype=np.float32)
    for g in glomeruli:
        hit = np.flatnonzero(want == f"ORN_{g}")
        if len(hit) == 0:
            raise KeyError(f"no ORN_{g} neurons; check the glomerulus name")
        v[hit] = 1.0
    return v


@dataclass
class ConditioningResult:
    """Mushroom body response to each odour, before and after pairing."""

    before: pd.DataFrame  # odour x population
    after: pd.DataFrame
    sparseness: dict[str, float]
    overlap: float
    depression: float
    rewards: list[float]
    graph: str
    kc_slope: float
    notes: list[str] = field(default_factory=list)

    @property
    def change(self) -> pd.DataFrame:
        return self.after - self.before

    def learning_index(self, population: str = "MBON") -> float:
        """How much more the paired odour's response fell than the unpaired one's.

        **This statistic is contaminated and should not be reported alone.**
        Measured on MaleCNS it does not flip when the reward is swapped between
        the two odours: one odour falls more than the other whichever is
        rewarded, because the odours drive different numbers of Kenyon cells and
        start from different response magnitudes. The index therefore mixes a
        reward term with an odour-identity term.

        The valid comparison is **the same odour, rewarded versus not**: run the
        experiment twice with the roles swapped and compare that odour's own
        change across the two runs. See :func:`within_odour_effect`.
        """
        ch = self.change[population]
        return float(ch.get("unpaired", 0.0) - ch.get("paired", 0.0))

    def report(self) -> str:
        lines = [
            f"differential conditioning on {self.graph} (KC slope {self.kc_slope})",
            f"  odour code: paired {self.sparseness.get('paired', 0):.1%} of KCs, "
            f"unpaired {self.sparseness.get('unpaired', 0):.1%}, overlap {self.overlap:.1%}",
            f"  reward over training: {self.rewards[0]:.3f} -> {self.rewards[-1]:.3f}"
            f" ({len(self.rewards)} trials)",
            f"  KC->MBON depression: {self.depression:.5f}",
            "",
            f"  {'odour':10s} {'MBON before':>12s} {'MBON after':>11s} {'change':>9s}",
        ]
        for odour_name in self.before.index:
            b = self.before.loc[odour_name, "MBON"]
            a = self.after.loc[odour_name, "MBON"]
            lines.append(f"  {odour_name:10s} {b:12.5f} {a:11.5f} {a - b:+9.5f}")
        lines += ["", f"  learning index: {self.learning_index():+.5f}"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def within_odour_effect(
    rewarded: ConditioningResult,
    unrewarded: ConditioningResult,
    *,
    population: str = "MBON",
) -> float:
    """Extra depression an odour shows when it is the rewarded one.

    Takes two runs with the reward swapped between the odours, and compares one
    odour's own change in the run where it was paired against the run where it
    was not. Because it is the same odour in both, the odour-identity term that
    contaminates :meth:`ConditioningResult.learning_index` cancels.

    Negative means the odour's mushroom body output fell further when it was
    rewarded, which is the direction dopamine-gated depression predicts.
    """
    paired_change = float(rewarded.change[population]["paired"])
    unpaired_change = float(unrewarded.change[population]["unpaired"])
    return paired_change - unpaired_change


def differential_conditioning(
    connectome: Connectome,
    *,
    paired: tuple[str, ...] = ("DM1", "DM4"),
    unpaired: tuple[str, ...] = ("DA1", "VA2"),
    n_trials: int = 8,
    start_distance: float = 1.6,
    step: float = 0.18,
    reward_scale: float = 0.8,
    learning_rate: float = 0.5,
    kc_slope: float = SPARSE_KC_SLOPE,
    hops: int = 6,
    graph_name: str = "original",
    readout: tuple[str, ...] = ("MBON", "KC"),
    progress: bool = False,
) -> ConditioningResult:
    """Approach an odour source, then test what the mushroom body learned.

    Each training trial is one step closer to the source, so both the odour and
    the dopamine grow -- the "closer means more dopamine" rule, moved to the
    circuit that can use it.
    """
    receptors = olfactory_receptors(connectome)
    mb = mushroom_body(connectome)
    brain = RateBrain(
        connectome,
        receptors,
        num_layers=hops,
        default_bias=0.0,
        slope_by_type=kc_slopes(connectome, kc_slope),
    )
    plastic = brain.plasticity(learning_rate=learning_rate)

    odours = {
        "paired": odour(connectome, paired, receptors),
        "unpaired": odour(connectome, unpaired, receptors),
    }
    record = {k: mb[k.lower()] for k in readout}

    def probe() -> pd.DataFrame:
        rows = {}
        for name, v in odours.items():
            res = brain.run(steady_state(v, hops), record=record)
            rows[name] = {k: float(res.populations[k].max()) for k in record}
        return pd.DataFrame(rows).T

    before = probe()

    # Odour-specific Kenyon cell sets, for reporting the code the pairing acts on.
    acts = {}
    for name, v in odours.items():
        a = brain.run(steady_state(v, hops)).activations[mb["kc"]].max(axis=1)
        acts[name] = a > RESPONSE_THRESHOLD
    sparseness = {k: float(v.mean()) for k, v in acts.items()}
    overlap = float(
        (acts["paired"] & acts["unpaired"]).sum() / max(acts["paired"].sum(), 1)
    )

    rewards = []
    for trial in range(n_trials):
        distance = max(start_distance - trial * step, 0.0)
        reward = proximity_reward(distance, scale=reward_scale)
        rewards.append(reward)
        res = brain.run(steady_state(odours["paired"], hops), record=record)
        kc_activity = res.activations[mb["kc"]].max(axis=1)
        depression = plastic.step(kc_activity, reward)
        # The unpaired odour is experienced too, but without dopamine. By the
        # rule that changes nothing, so it is presented once and not stepped --
        # exposure without reinforcement is the control, not a second update.
        brain.run(steady_state(odours["unpaired"], hops))
        if progress:  # pragma: no cover
            print(
                f"    trial {trial}: d={distance:.2f} reward={reward:.3f} "
                f"depression={depression:.5f}",
                flush=True,
            )

    after = probe()

    notes = []
    if overlap > 0.5:
        notes.append(
            f"the two odours share {overlap:.0%} of their Kenyon cells, so nothing "
            "odour specific can be learned; lower kc_slope"
        )
    if sparseness["paired"] == 0:
        notes.append("the paired odour activates no Kenyon cells; raise kc_slope")

    return ConditioningResult(
        before=before,
        after=after,
        sparseness=sparseness,
        overlap=overlap,
        depression=plastic.depression(),
        rewards=rewards,
        graph=graph_name,
        kc_slope=kc_slope,
        notes=notes,
    )
