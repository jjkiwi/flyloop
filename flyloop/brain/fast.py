"""The same rate model, in scipy, fast enough to precompute behaviour with.

:class:`~flyloop.brain.rate.RateBrain` wraps ``connectome_interpreter``'s
``MultilayeredNetwork``, which is a PyTorch module built for gradient-based
activation maximisation. Running it forward on MaleCNS costs about **3.5 s per
call** on a CPU. The identical arithmetic in scipy costs about **150 ms** -- a
23x difference that comes entirely from torch's sparse matmul, not from any
change to the model.

That difference is what makes an interactive application possible at all. A
closed-loop control step is 3.5 s of brain plus 1.8 s of physics; drop the brain
to 0.15 s and the physics becomes the bottleneck instead, which is the right
place for it to be.

**This is a reimplementation, so it is pinned by an equivalence test.** Reading
the reference implementation and believing you have understood it is how subtle
divergences get into a model and stay there. ``tests/test_fast.py`` runs both
against the same stimulus on the same graph and requires agreement to float
tolerance. If that test fails, this file is wrong and ``RateBrain`` is right.

Two details of the reference that are easy to miss and are replicated here
deliberately:

* **Biases are used as ``abs(bias)``.** A negative bias in ``bias_by_type`` is
  silently made positive. That is the reference's behaviour, not ours.
* **External input is re-injected at every hop except the last**, and clipped to
  1 after each injection. The time axis is the synaptic-hop axis, so which frame
  lands at which hop changes the answer -- see :func:`flyloop.brain.rate.steady_state`.
"""

from __future__ import annotations

import numpy as np

from ..connectome.schema import Connectome
from .rate import RateResult


class FastRateBrain:
    """Per-cell-type rate dynamics over a connectome, without PyTorch.

    The constructor signature mirrors :class:`~flyloop.brain.rate.RateBrain` so
    the two are interchangeable at a call site.
    """

    def __init__(
        self,
        connectome: Connectome,
        sensory: np.ndarray,
        *,
        num_layers: int = 6,
        default_bias: float = 0.0,
        tanh_steepness: float = 5.0,
        bias_by_type: dict[str, float] | None = None,
        slope_by_type: dict[str, float] | None = None,
        threshold: float = 0.01,
    ):
        if connectome.meta.get("matrix_kind") == "syncount":
            raise ValueError(
                "this model expects input-proportion weights, not synapse counts. "
                "Load with matrix='inprop' -- see flyloop.brain.rate for why."
            )
        self.c = connectome
        self.sensory = np.asarray(sensory, dtype=np.int64)
        self.num_layers = int(num_layers)
        self.threshold = float(threshold)

        # The model wants (post, pre): rows are targets, columns are sources,
        # which is the transpose of Connectome.W. Held as CSR so that a
        # matrix-vector product walks memory in row order.
        self.W = connectome.W.T.tocsr().astype(np.float32)
        self.W.sort_indices()

        types = connectome.neurons["type"].astype(str).to_numpy()
        self.slope = _per_neuron(types, slope_by_type, tanh_steepness)
        # abs() matches the reference, which stores raw biases and exposes
        # their absolute value.
        self.bias = np.abs(_per_neuron(types, bias_by_type, default_bias))

    # ------------------------------------------------------------- plumbing

    @property
    def values(self) -> np.ndarray:
        """The weight values, editable in place (plasticity writes here)."""
        return self.W.data

    @property
    def indices(self) -> np.ndarray:
        """``(2, nnz)`` array of (post, pre), in the order ``values`` uses.

        CSR stores entries sorted by row and then by column, which is the same
        order a coalesced COO tensor uses, so positions found through either
        route agree.
        """
        post = np.repeat(np.arange(self.W.shape[0], dtype=np.int64), np.diff(self.W.indptr))
        return np.vstack([post, self.W.indices.astype(np.int64)])

    def plasticity(self, **kwargs):
        """Attach dopamine-gated KC->MBON plasticity to this network."""
        from .dopamine import mushroom_body_plasticity

        return mushroom_body_plasticity(self.c, self.values, self.indices, **kwargs)

    # -------------------------------------------------------------- forward

    def _activate(self, x: np.ndarray) -> np.ndarray:
        x = self.slope * x + self.bias
        # Thresholded ReLU, then tanh. The threshold is a floor on the
        # *post-bias* value, so a neuron with bias above it is never silent.
        np.putmask(x, x < self.threshold, 0.0)
        return np.tanh(x, out=x)

    def run(self, inputs: np.ndarray, *, record: dict | None = None) -> RateResult:
        """Run a ``(n_sensory, time_steps)`` stimulus through the network."""
        import pandas as pd

        u = np.asarray(inputs, dtype=np.float32)
        if u.ndim != 2 or u.shape[0] != len(self.sensory):
            raise ValueError(
                f"inputs must be (n_sensory={len(self.sensory)}, time_steps), "
                f"got {tuple(u.shape)}"
            )
        u = np.where(u >= self.threshold, u, 0.0)
        np.putmask(u, u > 1.0, 1.0)

        x = np.zeros(self.W.shape[1], dtype=np.float32)
        x[self.sensory] = u[:, 0]

        acts = np.empty((self.W.shape[0], self.num_layers), dtype=np.float32)
        for layer in range(self.num_layers):
            x = self._activate(self.W @ x)
            if layer != self.num_layers - 1:
                x[self.sensory] += u[:, layer + 1]
                np.putmask(x, x > 1.0, 1.0)
            acts[:, layer] = x

        pops = {
            label: acts[rows].mean(axis=0)
            for label, rows in (record or {}).items()
            if len(rows)
        }
        return RateResult(
            activations=acts,
            populations=pd.DataFrame(pops),
            sensory=self.sensory,
        )


def _per_neuron(
    types: np.ndarray, by_type: dict[str, float] | None, default: float
) -> np.ndarray:
    """Expand a per-cell-type value to one number per neuron."""
    out = np.full(len(types), float(default), dtype=np.float32)
    for name, value in (by_type or {}).items():
        out[types == name] = float(value)
    return out
