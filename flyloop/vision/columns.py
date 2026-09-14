"""Real retinotopy, from published columnar tables.

:class:`~flyloop.vision.mapping.ColumnMap` refuses to invent a mapping from
ommatidia to neurons, because a scrambled retinotopy still spikes and still
looks like it works. This module supplies the real thing.

The ``connectome-interpreter`` package (MIT) bundles the columnar coordinate
tables from two published sources, and they are exactly what is needed:

``mcns_right``  Nern et al. 2024 -- **892 hexagonal columns** of the right
                medulla of MaleCNS, each row giving the body ID of that
                column's L1, L2, L3, L5, Mi1, Mi4, Mi9, C2, C3, Tm1, Tm2, Tm4,
                Tm9, Tm20 and T1. (The file itself has 920 rows; 28 columns are
                listed twice and are collapsed on load.)
``fafb_right``  Matsliah et al. 2024 -- the equivalent for the right optic lobe
                of FlyWire/FAFB, as one row per neuron.

892 is also the number of retinotopic columns the ommatid robot project samples
its camera onto, which is a good sign that this is the table everyone doing
this ends up using.

Pick the input cell type deliberately. L1 and L2 are the two large monopolar
cells carrying the ON and OFF pathways out of the lamina; Mi1 and Tm3 sit in the
ON pathway downstream. Driving L1 is the closest thing to "light lands here"
that a connectome-only model can offer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Tables bundled by connectome-interpreter, and where they live inside it.
BUNDLED = {
    "mcns_right": ("Nern2024", "ME-columnar-cells-hex-location.csv"),
    "fafb_right": ("Matsliah2024", "fafb_right_vis_cols.csv"),
}

_INSTALL_HINT = (
    "The columnar tables ship with connectome-interpreter. Install it with:\n"
    "    pip install 'flyloop[data]'\n"
    "or pass an explicit path to load_columnar_table()."
)


def _bundled_path(dataset: str) -> Path:
    try:
        from importlib.resources import files

        import connectome_interpreter  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_INSTALL_HINT) from exc
    folder, name = BUNDLED[dataset]
    return Path(str(files("connectome_interpreter") / "data" / folder / name))


def load_columnar_table(
    dataset: str = "mcns_right", *, path: str | Path | None = None
) -> pd.DataFrame:
    """Load a columnar table as ``hex1, hex2, x, y`` plus one column per cell type.

    Both bundled tables are normalised to the same shape: one row per
    hexagonal column, with each cell-type column holding the body ID of that
    type's neuron in that column (or NaN where the reconstruction has none).
    """
    if dataset not in BUNDLED and path is None:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(BUNDLED)}")
    src = Path(path) if path is not None else _bundled_path(dataset)
    raw = pd.read_csv(src, low_memory=False)

    if "hex1_id" in raw.columns:  # Nern 2024 layout: already one row per column
        out = raw.rename(columns={"hex1_id": "hex1", "hex2_id": "hex2"})
        # The published file has 920 rows for 892 distinct columns: 28 columns
        # appear twice, as exact duplicates. Left in place they would give two
        # ommatidia the same input neuron, so they are collapsed here and the
        # count reported by len() is the real one.
        out = out.drop_duplicates(subset=["hex1", "hex2"], keep="first")
        return out.reset_index(drop=True)

    # Matsliah 2024 layout: one row per neuron, pivot onto columns.
    needed = {"root_id", "type", "p", "q"}
    if not needed.issubset(raw.columns):
        raise ValueError(
            f"{src.name} has columns {list(raw.columns)[:10]}; expected either a "
            "'hex1_id' column or the FAFB layout with root_id/type/p/q."
        )
    wide = raw.pivot_table(
        index=["p", "q"], columns="type", values="root_id", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    coords = raw.groupby(["p", "q"])[["x", "y"]].first().reset_index()
    out = wide.merge(coords, on=["p", "q"], how="left")
    return out.rename(columns={"p": "hex1", "q": "hex2"}).reset_index(drop=True)


def column_order(table: pd.DataFrame) -> np.ndarray:
    """Row order that walks the hex lattice outward from its centre.

    Ordering matters: it is what makes "ommatidium k" mean the same place in the
    eye as "input neuron k". Centre-outward keeps the mapping stable when a
    smaller eye is used, so truncating the array crops the periphery rather than
    silently reshuffling the whole field.
    """
    x = table["x"].to_numpy(dtype=float)
    y = table["y"].to_numpy(dtype=float)
    return np.argsort((x - x.mean()) ** 2 + (y - y.mean()) ** 2)


def columnar_indices(
    table: pd.DataFrame,
    ids: pd.Series | np.ndarray,
    *,
    cell_type: str = "L1",
    n: int | None = None,
) -> np.ndarray:
    """Row indices into a connectome's neuron table, in retinotopic order.

    ``ids`` is the connectome's ``id`` column. Columns whose ``cell_type`` neuron
    is missing from the connectome are dropped, so the result can be shorter
    than the table; the caller decides whether that is acceptable rather than
    having gaps quietly filled.
    """
    if cell_type not in table.columns:
        raise KeyError(
            f"the columnar table has no {cell_type!r} column. Available cell "
            f"types: {[c for c in table.columns if c not in ('hex1', 'hex2', 'x', 'y')]}"
        )
    ids = np.asarray(ids)
    lookup = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
    if lookup.index.has_duplicates:
        # A connectome should not list a body ID twice, but a subset or a merge
        # can produce one. Keep the first occurrence rather than failing deep
        # inside pandas with an unhelpful reindexing error.
        lookup = lookup[~lookup.index.duplicated(keep="first")]
    ordered = table.iloc[column_order(table)]
    body_ids = ordered[cell_type]
    rows = body_ids.map(lookup)
    rows = rows.dropna().to_numpy(dtype=np.int64)
    return rows if n is None else rows[:n]
