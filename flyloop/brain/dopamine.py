"""Reward, delivered through the fly's own dopaminergic neurons.

MaleCNS contains the whole machinery: 392 dopaminergic neurons, of which **316
are the PAM cluster** -- the reward-signalling population in *Drosophila* -- plus
4,064 Kenyon cells and 97 mushroom body output neurons. So a reward signal does
not have to be bolted on as an abstract scalar. It can be delivered where the
animal delivers it, and act where the animal's plasticity acts: the KC->MBON
synapse, depressed when dopamine and Kenyon-cell activity coincide.

That rule is the measured one. In the fly, pairing DAN activity with KC activity
**depresses** KC->MBON synapses; MBONs bias approach and avoidance, so
depressing the output of the cells active during a rewarded experience shifts
behaviour toward approach.

**Two honest caveats, both structural.**

*Reward shaping is not biology.* Scaling reward continuously with closeness to a
target is a reinforcement-learning convenience. Real PAM neurons signal reward
delivery and prediction error, not a distance gradient. The graded version is
implemented here because it is what makes a gradient-following task learnable at
all, and it should not be read as a claim about what PAM encodes.

*The learning rate is a free parameter.* Nothing here is fitted to measured
plasticity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..connectome.schema import Connectome

#: Dopaminergic clusters, by the prefix their cell types carry.
REWARD_CLUSTER = "PAM"
PUNISHMENT_CLUSTERS = ("PPL1",)


def dopaminergic(
    c: Connectome, *, cluster: str | None = REWARD_CLUSTER, side: str | None = None
) -> np.ndarray:
    """Neuron rows of the dopaminergic population.

    ``cluster=None`` returns every dopaminergic neuron; otherwise only the
    cluster whose cell types start with that prefix.
    """
    types = c.neurons["type"].astype(str)
    m = c.neurons["nt"].astype(str) == "dopamine"
    if cluster:
        m &= types.str.startswith(cluster)
    if side and "side" in c.neurons.columns:
        m &= c.neurons["side"].astype(str) == side
    rows = np.flatnonzero(m.to_numpy())
    if len(rows) == 0:
        raise KeyError(
            f"connectome {c.name!r} has no dopaminergic neurons"
            + (f" in cluster {cluster!r}" if cluster else "")
        )
    return rows


def proximity_reward(
    distance: float, *, scale: float = 1.0, floor: float = 0.0
) -> float:
    """Reward in [0, 1] that grows as the target gets closer.

    An exponential in distance rather than an inverse, so it is bounded, smooth,
    and has no singularity at contact. ``scale`` is the distance over which it
    falls by a factor of e.
    """
    return float(floor + (1.0 - floor) * np.exp(-max(distance, 0.0) / max(scale, 1e-9)))


@dataclass
class MushroomBodyPlasticity:
    """Dopamine-gated depression of Kenyon cell to MBON synapses.

    Holds the positions of the KC->MBON entries inside the network's weight
    tensor so they can be updated in place, which keeps a learning step cheap
    even though the full matrix has 24 million non-zeros.
    """

    kc: np.ndarray
    mbon: np.ndarray
    entry_index: np.ndarray  # positions in the coalesced values tensor
    entry_pre: np.ndarray  # position of each entry's presynaptic KC within `kc`
    initial: np.ndarray  # starting weights, for reporting drift
    learning_rate: float = 0.05
    min_fraction: float = 0.1
    history: list[float] = field(default_factory=list)

    @property
    def n_synapses(self) -> int:
        return len(self.entry_index)

    def depression(self) -> float:
        """How far the KC->MBON weights have been driven from their start, 0 to 1."""
        if self.n_synapses == 0:
            return 0.0
        return float(1.0 - np.abs(self._current).sum() / np.abs(self.initial).sum())

    def attach(self, values) -> None:
        self._values = values

    @property
    def _current(self) -> np.ndarray:
        return self._values[self.entry_index].detach().cpu().numpy()

    def step(self, kc_activity: np.ndarray, dopamine: float) -> float:
        """Apply one learning step and return the resulting depression.

        ``kc_activity`` is the activation of the Kenyon cells **in the order of
        :attr:`kc`**, not indexed by global neuron id; only the cells active at
        the moment dopamine arrives are depressed, which is what makes the rule
        a coincidence detector rather than a global gain change.
        """
        if len(kc_activity) != len(self.kc):
            raise ValueError(
                f"kc_activity has {len(kc_activity)} entries but there are "
                f"{len(self.kc)} Kenyon cells; pass activations[plastic.kc]"
            )
        if self.n_synapses == 0 or dopamine <= 0:
            self.history.append(self.depression())
            return self.history[-1]
        import torch

        act = np.clip(kc_activity[self.entry_pre], 0.0, 1.0)
        factor = 1.0 - self.learning_rate * dopamine * act
        with torch.no_grad():
            idx = torch.from_numpy(self.entry_index)
            cur = self._values[idx]
            floor = torch.from_numpy(self.initial * self.min_fraction).to(cur.dtype)
            new = cur * torch.from_numpy(factor.astype(np.float32))
            # Depression only, and never past the floor.
            keep = torch.where(
                torch.abs(new) < torch.abs(floor), floor, new
            )
            self._values[idx] = keep
        self.history.append(self.depression())
        return self.history[-1]


def mushroom_body_plasticity(
    c: Connectome,
    values,
    indices,
    *,
    learning_rate: float = 0.05,
    min_fraction: float = 0.1,
) -> MushroomBodyPlasticity:
    """Locate the KC->MBON synapses inside a network's weight tensor.

    ``indices`` must be the tensor's **coalesced** ``(2, nnz)`` index tensor, and
    ``values`` its coalesced values, so that positions line up.

    Taking them from the pre-coalesce COO matrix instead is a silent disaster:
    ``coalesce()`` sorts the entries, so the positions point at unrelated
    synapses and the learning rule quietly rewrites random parts of the
    connectome while reporting a nonsensical depression. That happened; hence
    this paragraph.
    """
    types = c.neurons["type"].astype(str)
    kc = np.flatnonzero(types.str.startswith("KC").to_numpy())
    mbon = np.flatnonzero(types.str.startswith("MBON").to_numpy())

    is_kc = np.zeros(c.n, dtype=bool)
    is_kc[kc] = True
    is_mbon = np.zeros(c.n, dtype=bool)
    is_mbon[mbon] = True

    # The network matrix is (post, pre): rows are targets, columns are sources.
    idx = indices.detach().cpu().numpy()
    post, pre = idx[0], idx[1]
    hit = np.flatnonzero(is_mbon[post] & is_kc[pre])
    # Entries carry global neuron ids; the learning rule is handed activations
    # for the KC population alone, so store the position within it instead.
    kc_position = np.full(c.n, -1, dtype=np.int64)
    kc_position[kc] = np.arange(len(kc), dtype=np.int64)
    plastic = MushroomBodyPlasticity(
        kc=kc,
        mbon=mbon,
        entry_index=hit.astype(np.int64),
        entry_pre=kc_position[pre[hit]],
        initial=values[hit].detach().cpu().numpy().astype(np.float32).copy(),
        learning_rate=learning_rate,
        min_fraction=min_fraction,
    )
    plastic.attach(values)
    return plastic
