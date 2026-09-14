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

    def __init__(
        self, indices: dict[str, np.ndarray], *, assumptions: list[str] | None = None
    ):
        self.indices = indices
        #: Anything about this mapping that was assumed rather than measured.
        #: Report it alongside any result that depends on the mapping.
        self.assumptions = assumptions or []

    @property
    def monocular(self) -> bool:
        """True when only one eye could be mapped from the available data."""
        return len(self.indices) < 2

    @classmethod
    def from_hex_metadata(
        cls,
        c: Connectome,
        eye: CompoundEye,
        *,
        cell_type: str = "L1",
        hex_columns: tuple[str, str] = ("hex1", "hex2"),
    ) -> "ColumnMap":
        """Build a binocular retinotopy from the connectome's own hex columns.

        The prepared MaleCNS metadata carries ``assignedOlHex1``/``assignedOlHex2``
        for 23,720 optic-lobe neurons, on **both** sides: 892 distinct columns for
        the right eye and 875 for the left. That is strictly better than the
        separate published columnar tables, which cover the right optic lobe only
        and force :meth:`from_columnar_table` to return a monocular map.

        Columns are ordered centre-outward in hex coordinates, so truncating to a
        smaller eye crops the periphery instead of reshuffling the field, and the
        same hex column means the same direction in both eyes.
        """
        if not set(hex_columns).issubset(c.neurons.columns):
            raise KeyError(
                f"connectome {c.name!r} has no {hex_columns} columns. This needs the "
                "prepared metadata from connectome_data_prep; see docs/DATA.md."
            )
        h1, h2 = hex_columns
        n = c.neurons
        indices: dict[str, np.ndarray] = {}
        for side in ("L", "R"):
            m = (n["type"].astype(str) == cell_type) & n[h1].notna()
            if "side" in n.columns:
                m &= n["side"].astype(str).str.upper().str.startswith(side)
            rows = np.flatnonzero(m.to_numpy())
            if len(rows) == 0:
                continue
            a = n[h1].to_numpy(dtype=float)[rows]
            b = n[h2].to_numpy(dtype=float)[rows]
            order = np.argsort((a - a.mean()) ** 2 + (b - b.mean()) ** 2)
            indices[side] = rows[order]

        need = eye.n_columns
        short = {s: len(v) for s, v in indices.items() if len(v) < need}
        if short:
            raise ValueError(
                f"eye asks for {need} ommatidia per side but the connectome has "
                f"{short} hex-assigned {cell_type} neurons. Reduce "
                "CompoundEye(n_columns=...) to the smaller eye."
            )
        return cls(
            {s: v[:need] for s, v in indices.items()},
            assumptions=[]
            if len(indices) == 2
            else ["only one eye has hex-assigned neurons in this connectome"],
        )

    @classmethod
    def from_columnar_table(
        cls,
        c: Connectome,
        eye: CompoundEye,
        *,
        dataset: str = "mcns_right",
        cell_type: str = "L1",
        table=None,
        mirror: bool = False,
    ) -> "ColumnMap":
        """Build a *real* retinotopy from a published columnar table.

        This is the mapping :meth:`from_connectome` refuses to guess. The tables
        bundled with connectome-interpreter give, for each hexagonal column of
        the optic lobe, the body ID of that column's L1, L2, Mi1 and so on, so
        ommatidium *k* can be tied to the neuron that actually sits there.

        **The published tables cover the right optic lobe only.** There is no
        left-eye table, and the left eye's column identities cannot be derived
        from the right one. So by default this returns a *monocular* map, and a
        loop built on it sees with one eye. That is a real limitation of the
        data, not an oversight to code around.

        ``mirror=True`` assigns the left eye by taking the same cell type on the
        left side of the connectome in the same rank order as the right. That is
        an assumption about developmental symmetry, not a measurement: it will
        be roughly right near the midline and progressively wrong toward the
        periphery. Anything that depends on precise binocular geometry must not
        use it. It is recorded in ``self.assumptions`` so a result can say so.
        """
        from .columns import columnar_indices, load_columnar_table

        table = load_columnar_table(dataset) if table is None else table
        need = eye.eyes["R"].n
        right = columnar_indices(table, c.neurons["id"], cell_type=cell_type)
        if len(right) < need:
            raise ValueError(
                f"the {dataset!r} table matched only {len(right)} {cell_type} "
                f"neurons in connectome {c.name!r}, but the eye has {need} "
                "ommatidia. Reduce CompoundEye(n_columns=...) or check that the "
                "connectome and the table come from the same dataset."
            )
        indices = {"R": right[:need]}
        assumptions: list[str] = []
        if mirror:
            m = c.neurons["type"].astype(str) == cell_type
            if "side" in c.neurons.columns:
                m &= c.neurons["side"].astype(str).str.upper().str.startswith("L")
            left = np.flatnonzero(m.to_numpy())
            if len(left) < need:
                raise ValueError(
                    f"mirror=True needs {need} left-side {cell_type} neurons; "
                    f"the connectome has {len(left)}."
                )
            indices["L"] = left[:need]
            assumptions.append(
                "left eye mirrored from the right by rank order; not measured"
            )
        return cls(indices, assumptions=assumptions)

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
