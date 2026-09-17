"""Control graphs: the thing that decides whether the connectome did anything.

A closed loop built on a real connectome will produce *some* behaviour. So will
a closed loop built on a graph with the same degree sequence and the wiring
shuffled. Unless both are run under the identical protocol, "the fly brain
steers the robot" is not a finding -- it is a description of a program that
runs.

The ommatid project (MIT, FlyEM male CNS v1.0) ran exactly this comparison on
real data and real hardware in September 2026: looming drove the escape
descending neurons +13.6 Hz on the measured wiring against +4.6 Hz on a
degree-preserving shuffle (permutation p = 0.0001), while their pre-registered
optomotor and phototaxis hypotheses were not met on either graph. That is the
shape of an honest result, and it is only available because the control graph
existed.

Three controls, answering three different questions:

``rewire_degree_preserving``
    Same in- and out-degree for every neuron, different targets. Asks whether
    the *specific* connectivity matters beyond how many partners each cell has.
``relabel_neurons``
    Identical topology, neuron identities permuted. Asks whether it matters
    *which* cell type sits at which position in the graph.
``scramble_signs``
    Identical topology and identities, transmitters permuted. Asks whether the
    excitatory/inhibitory pattern matters, independently of the wiring.

All three are seeded and reproducible. Report the seed with the result.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .schema import Connectome
from .signs import sign_vector


def _collapse(row: np.ndarray, col: np.ndarray, data: np.ndarray, n: int) -> sp.csr_matrix:
    """Sum duplicate (row, col) pairs into a CSR matrix, dropping self-loops."""
    keep = row != col
    W = sp.coo_matrix((data[keep], (row[keep], col[keep])), shape=(n, n), dtype=np.float32)
    return W.tocsr()


def rewire_degree_preserving(
    c: Connectome, *, seed: int = 0, name: str | None = None
) -> Connectome:
    """Shuffle who connects to whom, keeping every neuron's degree.

    Implemented as a configuration-model rewiring: the shuffle permutes the
    postsynaptic endpoints, which leaves both the presynaptic and the
    postsynaptic endpoint multisets exactly as they were and randomises only
    their pairing. Each presynaptic neuron keeps its own weight profile, so the
    sign structure stays tied to the transmitter the neuron actually releases --
    this control randomises the wiring, not the neurochemistry.

    Degrees are therefore preserved by the shuffle itself, but not quite by the
    result: self-loops that the shuffle creates are dropped and duplicate pairs
    are summed, which costs a small fraction of the edges (about 0.5% on the
    synthetic fixture; less on a sparser real connectome). Compare
    :meth:`Connectome.report` on both graphs before reading anything into a
    difference in activity.
    """
    rng = np.random.default_rng(seed)
    coo = c.W.tocoo()
    col = coo.col.copy()
    rng.shuffle(col)
    W = _collapse(coo.row, col, coo.data.copy(), c.n)
    return Connectome(
        neurons=c.neurons.copy(),
        W=W,
        name=name or f"{c.name}[rewired:{seed}]",
        meta=dict(c.meta, control="rewire_degree_preserving", control_seed=seed),
    )


def relabel_neurons(c: Connectome, *, seed: int = 0, name: str | None = None) -> Connectome:
    """Keep the graph exactly, permute which neuron sits at which node.

    Topology, degree sequence and weight distribution are untouched; what
    changes is that LC4 is no longer where LC4 was. If a behaviour survives
    this, it is a property of the graph's shape rather than of the identified
    circuit, and no claim about named cell types is supported.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(c.n)
    neurons = c.neurons.iloc[perm].reset_index(drop=True)
    # The signs travel with the transmitter, which has just moved, so the
    # matrix has to be re-signed from each row's new presynaptic neuron.
    coo = c.W.tocoo()
    old_sign = sign_vector(c.neurons["nt"]).to_numpy()[coo.row]
    new_sign = sign_vector(neurons["nt"]).to_numpy()[coo.row]
    magnitude = np.abs(coo.data)
    data = magnitude * new_sign.astype(np.float32)
    del old_sign
    W = _collapse(coo.row, coo.col, data, c.n)
    return Connectome(
        neurons=neurons,
        W=W,
        name=name or f"{c.name}[relabelled:{seed}]",
        meta=dict(c.meta, control="relabel_neurons", control_seed=seed),
    )


def scramble_signs(c: Connectome, *, seed: int = 0, name: str | None = None) -> Connectome:
    """Keep wiring and identities, permute the transmitters across neurons.

    Excitatory and inhibitory counts are preserved; which neuron is which is
    not. This isolates the contribution of the E/I pattern from the
    contribution of the topology.
    """
    rng = np.random.default_rng(seed)
    neurons = c.neurons.copy()
    neurons["nt"] = rng.permutation(neurons["nt"].to_numpy())
    coo = c.W.tocoo()
    signs = sign_vector(neurons["nt"]).to_numpy()[coo.row]
    data = np.abs(coo.data) * signs.astype(np.float32)
    W = _collapse(coo.row, coo.col, data, c.n)
    return Connectome(
        neurons=neurons,
        W=W,
        name=name or f"{c.name}[signs-scrambled:{seed}]",
        meta=dict(c.meta, control="scramble_signs", control_seed=seed),
    )


#: The controls an experiment should report alongside its result.
CONTROLS = {
    "rewired": rewire_degree_preserving,
    "relabelled": relabel_neurons,
    "signs_scrambled": scramble_signs,
}


def control_graphs(c: Connectome, *, seed: int = 0) -> dict[str, Connectome]:
    """Build every control graph for ``c``, keyed by control name."""
    return {k: fn(c, seed=seed) for k, fn in CONTROLS.items()}
