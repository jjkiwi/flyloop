"""Reading the mushroom body the way the animal does: as a balance.

Run 8 measured dopamine-gated depression at the KC->MBON synapse and found it
did not reach the descending neurons. The reason was structural: the MBONs where
plasticity lands and the MBONs that drive descending neurons are different,
anticorrelated populations (r = -0.42), and averaging 97 cells of opposite sign
and wildly different output weight throws away whatever signal there is.

In the animal the learned valence of an odour is not read off any one MBON. It
is the **balance** across compartments -- some MBONs push the fly toward a
stimulus, others away, and behaviour follows the difference. Depressing the
cells active during a rewarded experience tips that balance.

So this module stops averaging and starts weighting. Each MBON gets a valence:
its net signed influence on the descending neurons that drive *approach*, minus
its influence on those that drive *withdrawal*. The ensemble readout is then

    readout = sum_i  activation_i * valence_i

**Where the valence comes from matters.** It is computed from the signed
connectome by propagating each MBON's influence forward a few synapses -- not
hand-assigned from the literature. The only prior knowledge used is the
behavioural role of four descending neurons, the same four this project has used
since its first run: DNa01 drives forward walking, MDN backward, DNp09 stopping.
Those roles are assumptions, and they are the assumption this readout rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..connectome.schema import Connectome

#: Descending neurons whose activity moves the animal toward things.
APPROACH_OUTPUT = ("DNa01",)
#: Descending neurons whose activity moves it away or stops it.
WITHDRAW_OUTPUT = ("MDN", "DNp09")


def _rows(c: Connectome, types: tuple[str, ...]) -> np.ndarray:
    kinds = c.neurons["type"].astype(str).to_numpy()
    hit = np.flatnonzero(np.isin(kinds, list(types)))
    return hit


def effective_influence(
    c: Connectome,
    sources: np.ndarray,
    targets: np.ndarray,
    *,
    hops: int = 4,
    decay: float = 0.5,
) -> np.ndarray:
    """Net signed influence of each source neuron on a target set.

    A unit impulse is released from each source and propagated forward through
    the signed matrix, accumulating what arrives at the targets after each hop.
    Inhibitory paths subtract, so a source that reaches a target through both an
    excitatory and an inhibitory route nets out -- which is the point, since a
    balance is exactly what is being measured.

    ``decay`` discounts longer paths. It is a modelling choice, not a
    measurement: at 0.5 a four-synapse route counts for a sixteenth of a direct
    one. Report it.
    """
    if hops < 1:
        raise ValueError("hops must be at least 1")
    n = c.n
    x = sp.csr_matrix(
        (
            np.ones(len(sources), dtype=np.float32),
            (sources, np.arange(len(sources))),
        ),
        shape=(n, len(sources)),
    ).toarray()
    wt = c.W.T.tocsr()
    influence = np.zeros(len(sources), dtype=np.float64)
    for h in range(1, hops + 1):
        x = wt @ x
        influence += (decay**h) * x[targets].sum(axis=0)
    return influence


@dataclass
class MBONEnsemble:
    """Per-MBON valence weights and the readout they define."""

    rows: np.ndarray  # neuron rows of the MBONs, in weight order
    valence: np.ndarray  # signed weight per MBON
    types: np.ndarray
    hops: int
    decay: float

    def table(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"type": self.types, "valence": self.valence})
            .groupby("type")["valence"]
            .sum()
            .sort_values(ascending=False)
        )

    def readout(self, activations: np.ndarray) -> float:
        """Weighted balance of MBON activity.

        ``activations`` is indexed by global neuron id, as the rate model
        returns it; only the MBON rows are used.
        """
        return float(np.dot(activations[self.rows], self.valence))

    def report(self, top: int = 6) -> str:
        t = self.table()
        pos = t[t > 0].head(top)
        neg = t[t < 0].tail(top)[::-1]
        lines = [
            f"MBON ensemble: {len(self.rows)} cells, {int((self.valence != 0).sum())} "
            f"with non-zero valence ({self.hops} hops, decay {self.decay})",
            f"  approach-weighted: {', '.join(f'{k} {v:+.4f}' for k, v in pos.items())}"
            if len(pos)
            else "  approach-weighted: none",
            f"  withdraw-weighted: {', '.join(f'{k} {v:+.4f}' for k, v in neg.items())}"
            if len(neg)
            else "  withdraw-weighted: none",
        ]
        return "\n".join(lines)


def mbon_ensemble(
    c: Connectome,
    *,
    approach: tuple[str, ...] = APPROACH_OUTPUT,
    withdraw: tuple[str, ...] = WITHDRAW_OUTPUT,
    hops: int = 4,
    decay: float = 0.5,
) -> MBONEnsemble:
    """Give every MBON a valence from its net effect on the motor output.

    Positive means the cell's activity pushes the animal toward things; negative
    means away. Nothing is hand-labelled: the weights come from propagating each
    MBON's signed influence through the connectome to the named descending
    neurons.
    """
    types = c.neurons["type"].astype(str)
    mbon = np.flatnonzero(types.str.startswith("MBON").to_numpy())
    if len(mbon) == 0:
        raise KeyError(f"connectome {c.name!r} has no MBONs")
    app = _rows(c, approach)
    wdr = _rows(c, withdraw)
    missing = [t for t in approach + withdraw if len(_rows(c, (t,))) == 0]
    if missing:
        raise KeyError(f"output neurons absent from {c.name!r}: {missing}")

    to_approach = effective_influence(c, mbon, app, hops=hops, decay=decay)
    to_withdraw = effective_influence(c, mbon, wdr, hops=hops, decay=decay)
    return MBONEnsemble(
        rows=mbon,
        valence=to_approach - to_withdraw,
        types=types.to_numpy()[mbon],
        hops=hops,
        decay=decay,
    )
