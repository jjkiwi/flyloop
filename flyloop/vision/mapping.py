"""Assigning ommatidia to input neurons.

With the synthetic fixture this is trivial.  With a real dataset it is one of
the two genuinely hard parts of the project (the other is the descending
readout), because the connectome gives you neurons and synapses but not a
ready-made "column 417 is this one".  Reconstructing retinotopy needs the
dataset's own column annotations -- see ``docs/DATA.md``.

This module deliberately fails loudly rather than quietly assigning ommatidia
to arbitrary neurons: a scrambled retinotopy produces a network that still
spikes, so nothing looks wrong until every motion result is nonsense.
"""

from __future__ import annotations

import numpy as np

from ..connectome.schema import Connectome
from .ommatidia import CompoundEye


class ColumnMap:
    """Maps each ommatidium of each eye to the neuron it drives."""

    def __init__(self, indices: dict[str, np.ndarray]):
        self.indices = indices

    @classmethod
    def from_connectome(
        cls,
        c: Connectome,
        eye: CompoundEye,
        *,
        cell_type: str = "PR",
        order_column: str | None = None,
    ) -> ColumnMap:
        """Match ``eye`` columns to neurons of ``cell_type``, one per ommatidium.

        ``order_column`` names a column of the neuron table holding a
        retinotopic index; when present it is used to sort the neurons so that
        ommatidium *k* drives the neuron the dataset calls column *k*.  Without
        it the neuron table's own order is used, which is only meaningful for
        the synthetic fixture.
        """
        out: dict[str, np.ndarray] = {}
        for side in ("L", "R"):
            m = c.neurons["type"].astype(str) == cell_type
            if "side" in c.neurons.columns:
                m &= c.neurons["side"].astype(str).str.upper().str.startswith(side)
            idx = np.flatnonzero(m.to_numpy())
            if order_column and order_column in c.neurons.columns:
                idx = idx[np.argsort(c.neurons[order_column].to_numpy()[idx])]
            need = eye.eyes[side].n
            if len(idx) < need:
                raise ValueError(
                    f"eye {side} has {need} ommatidia but the connectome has only "
                    f"{len(idx)} neurons of type {cell_type!r} on that side. "
                    "Reduce CompoundEye(n_columns=...) or pick the right input "
                    "cell type for your dataset (see docs/DATA.md)."
                )
            out[side] = idx[:need]
        return cls(out)

    def drive_rates(
        self, per_column: dict[str, np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Pair each side's neuron indices with its per-column rates."""
        return [(self.indices[s], per_column[s]) for s in ("L", "R") if s in per_column]
