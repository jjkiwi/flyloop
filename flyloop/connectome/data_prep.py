"""Loading the pre-processed connectome matrices from ``connectome_data_prep``.

`YijieYin/connectome_data_prep <https://github.com/YijieYin/connectome_data_prep>`_
publishes the major fly connectomes already reduced to a ``scipy.sparse`` matrix
plus a metadata CSV: MaleCNS, FAFB/FlyWire, BANC, hemibrain, MANC and the larva.
That is the same preparation step :mod:`flyloop.connectome.malecns` and
:mod:`flyloop.connectome.flywire` would otherwise do against neuPrint and Codex,
already done, already checked, and reachable by ``git clone``.

Layout of one dataset folder::

    maleCNS/
        mcns_all_neuron_meta.csv        one row per neuron, in matrix order
        mcns_syncount_all_neuron.npz    (pre, post) synapse counts
        mcns_inprop_all_neuron.npz      the same, normalised per postsynaptic cell
        mcns_ad_*                       axon-dendrite-only variants

**Matrix orientation.** The matrices are ``(pre, post)``: entry ``[i, j]`` is the
connection from neuron ``i`` onto neuron ``j``. That is verifiable rather than
assumed -- the columns of ``inprop`` sum to 1.0 (99.8% of them), because it
normalises by the *recipient's* total input. It matches
:attr:`flyloop.connectome.Connectome.W` directly, so no transpose is needed.

**Signs.** The metadata carries the dataset's own ``sign`` column, and it does
not agree with this project's table for 5% of neurons. See :func:`sign_report`;
the difference is systematic and you should decide it deliberately rather than
inherit it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .schema import Connectome
from .signs import sign_vector

#: Metadata column -> the name this project uses. The two datasets disagree on
#: nearly every name: MaleCNS writes bodyid/type/somaSide, FlyWire writes
#: root_id/cell_type/side. Both spellings map here so a caller never has to know
#: which file it got.
COLUMN_MAP = {
    "bodyid": "id",
    "root_id": "id",
    "type": "type",
    "cell_type": "type",
    "top_nt": "nt",
    "somaSide": "side",
    "side": "side",
    "superclass": "super_class",
    "assignedOlHex1": "hex1",
    "assignedOlHex2": "hex2",
    "sign": "dataset_sign",
    "somaLocation": "soma",
}

#: Soma coordinates arrive as a bracketed string, "[37124 22258 36274]", in the
#: dataset's own voxel space. They are the only spatial information these files
#: carry -- there are no skeletons, so anything drawn from them is a cloud of
#: cell bodies, not morphology.
SOMA_COLUMNS = ("soma_x", "soma_y", "soma_z")

#: Folder name inside ``data/`` for each dataset this loader knows about.
DATASET_FOLDERS = {
    "malecns": "maleCNS",
    "fafb": "fafb_all_neuron",
    "banc": "BANC",
    "hemibrain": "hemibrain",
    "manc": "MANC",
    "malecns_optic": "maleCNS_right_optic_neuron",
    "fafb_optic": "fafb_optic_right_neuron",
}


@dataclass
class PreparedFiles:
    """The two files a dataset folder must provide."""

    meta: Path
    matrix: Path
    axon_dendrite: bool


#: Weight matrices each dataset folder may provide.
#:
#: ``syncount`` is raw synapse counts, which is what a spiking model needs.
#: ``inprop`` is the same connectivity normalised by each *postsynaptic*
#: neuron's total input, so its columns sum to 1. That is the right weighting
#: for a rate model: a connection of 7 synapses is negligible in absolute terms
#: but a target pooling thousands of them still gets a substantial fraction of
#: its input, which is exactly how LC4 is reached.
MATRIX_KINDS = ("syncount", "inprop", "outprop")


def find_files(
    folder: str | Path,
    *,
    matrix: str = "syncount",
    prefer_axon_dendrite: bool = False,
) -> PreparedFiles:
    """Locate the metadata CSV and a weight matrix inside a folder.

    Some datasets ship only the axon-dendrite-split matrix, so the plain file is
    preferred when present and the ``ad_`` variant is used otherwise -- with
    which one was chosen recorded, because the two are not interchangeable: the
    axon-dendrite matrix keeps only connections from an axon onto a dendrite,
    which is a real filtering decision.
    """
    if matrix not in MATRIX_KINDS:
        raise ValueError(f"matrix must be one of {MATRIX_KINDS}, got {matrix!r}")
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a directory")
    metas = sorted(folder.glob("*_meta.csv"))
    if not metas:
        raise FileNotFoundError(f"no *_meta.csv in {folder}")

    plain = sorted(p for p in folder.glob(f"*{matrix}*.npz") if "_ad_" not in p.name)
    split = sorted(folder.glob(f"*_ad_{matrix}*.npz"))
    order = (split, plain) if prefer_axon_dendrite else (plain, split)
    chosen = next((c[0] for c in order if c), None)
    if chosen is None:
        raise FileNotFoundError(
            f"no {matrix!r} matrix in {folder}. Found: {[p.name for p in folder.glob('*.npz')]}"
        )
    return PreparedFiles(meta=metas[0], matrix=chosen, axon_dendrite="_ad_" in chosen.name)


def sign_report(meta: pd.DataFrame) -> pd.DataFrame:
    """Where this project's transmitter signs differ from the dataset's own.

    The datasets apply a simple rule -- glutamate and GABA inhibitory,
    everything else excitatory. This project treats histamine as inhibitory
    (it gates a chloride channel in the fly visual system, exactly as glutamate
    does) and gives neuromodulators no fast sign at all, since the model has no
    mechanism for them.

    Neither is obviously right, but they are different models and the difference
    lands on the photoreceptors, so it must be a choice and not an accident.
    """
    ours = sign_vector(meta["nt"])
    theirs = meta.get("dataset_sign")
    if theirs is None:
        return pd.DataFrame(columns=["nt", "ours", "dataset", "neurons"])
    diff = pd.DataFrame({"nt": meta["nt"].fillna("<missing>"), "ours": ours, "dataset": theirs})
    diff = diff[diff["ours"] != diff["dataset"]]
    out = (
        diff.groupby(["nt", "ours", "dataset"], dropna=False)
        .size()
        .reset_index(name="neurons")
        .sort_values("neurons", ascending=False)
    )
    return out.reset_index(drop=True)


def _in_matrix_order(raw: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Put the metadata in the matrix's row order, using its own ``idx`` column.

    These files do not promise that the CSV rows are in matrix order, and they
    are not. MaleCNS happens to ship sorted, ``idx`` running 0..161428; FlyWire
    does not -- its first row is matrix row 90,908. A loader that trusts CSV
    order therefore reads MaleCNS correctly and mislabels every neuron in
    FlyWire, which is the worst kind of failure because nothing raises. The
    symptom that caught it: "DNa02" appeared to draw 18% of its input from Tm3
    and 11% from Mi1, medulla cells that cannot plausibly feed a descending
    steering neuron, while the same query on MaleCNS returned PFL3 and the LAL
    types the literature names.
    """
    if "idx" not in raw.columns:
        return raw
    idx = raw["idx"].to_numpy()
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{filename}: 'idx' is not unique, so row order is undecidable")
    expected = np.arange(len(raw))
    if np.array_equal(idx, expected):
        return raw
    ordered = raw.sort_values("idx", kind="stable").reset_index(drop=True)
    if not np.array_equal(ordered["idx"].to_numpy(), expected):
        raise ValueError(
            f"{filename}: 'idx' should be a permutation of 0..{len(raw) - 1}, "
            f"but runs {ordered['idx'].min()}..{ordered['idx'].max()}"
        )
    return ordered


def parse_soma(values: pd.Series) -> np.ndarray:
    """Turn ``"[37124 22258 36274]"`` strings into an ``(n, 3)`` float array.

    Neurons without a located soma -- about 14% of MaleCNS, mostly fragments and
    cells whose cell body was never assigned -- come back as NaN rather than
    being dropped, so the array stays aligned with the connectivity matrix.
    """
    out = np.full((len(values), 3), np.nan, dtype=np.float64)
    text = values.astype("string")
    ok = text.notna().to_numpy()
    if not ok.any():
        return out
    cleaned = text[ok].str.strip().str.strip("[]").str.split()
    good = cleaned.str.len() == 3
    rows = np.flatnonzero(ok)[good.to_numpy()]
    out[rows] = np.asarray(cleaned[good].tolist(), dtype=np.float64)
    return out


def load_prepared(
    folder: str | Path,
    *,
    name: str | None = None,
    min_synapses: int = 5,
    sign_source: str = "flyloop",
    prefer_axon_dendrite: bool = False,
    matrix: str = "syncount",
    min_weight: float | None = None,
) -> Connectome:
    """Build a :class:`Connectome` from one prepared dataset folder.

    Parameters
    ----------
    min_synapses:
        Drop connections weaker than this, for count matrices. On MaleCNS a
        threshold of 5 keeps 24% of the connections but 72% of the synapses,
        which is the usual trade: most edges are one or two synapses and mostly
        reconstruction noise.
    matrix:
        Which weighting to load -- see :data:`MATRIX_KINDS`. ``"inprop"``
        weights are already normalised per postsynaptic neuron, so
        ``min_synapses`` does not apply to them; use ``min_weight`` instead.
    min_weight:
        Threshold for non-count matrices. Defaults to keeping everything, since
        the whole point of an input-proportion weighting is that many tiny
        inputs add up.
    sign_source:
        ``"flyloop"`` uses this project's transmitter table;
        ``"dataset"`` uses the metadata's own ``sign`` column. They disagree on
        about 5% of MaleCNS neurons -- run :func:`sign_report` before choosing.
    """
    if sign_source not in ("flyloop", "dataset"):
        raise ValueError(f"sign_source must be 'flyloop' or 'dataset', got {sign_source!r}")

    files = find_files(folder, matrix=matrix, prefer_axon_dendrite=prefer_axon_dendrite)
    raw = pd.read_csv(files.meta, low_memory=False)
    raw = _in_matrix_order(raw, files.meta.name)

    # Several source spellings map to the same name and a file may carry more
    # than one of them -- MaleCNS has both `type` and `cell_type`. Keeping both
    # produces a duplicated column, and `neurons["type"]` then returns a
    # DataFrame instead of a Series, which fails far from here. First spelling
    # in COLUMN_MAP order wins.
    present, taken = {}, set()
    for source, mapped in COLUMN_MAP.items():
        # `mapped`, not `name`: this function already has a `name` parameter --
        # the dataset's -- and shadowing it here renamed every connectome after
        # the last column processed, which was "soma".
        if source in raw.columns and mapped not in taken:
            present[source] = mapped
            taken.add(mapped)
    # One spelling from each group has to be present, not all of them.
    needed = (("bodyid", "root_id"), ("type", "cell_type"), ("top_nt",))
    missing = [
        " or ".join(group) for group in needed if not any(k in raw.columns for k in group)
    ]
    if missing:
        raise ValueError(
            f"{files.meta.name} is missing {missing}; columns are {list(raw.columns)[:20]}"
        )
    neurons = raw[list(present)].rename(columns=present)
    for col in ("type", "nt", "side", "super_class"):
        if col in neurons:
            neurons[col] = neurons[col].fillna("unknown").astype(str)
    if "side" in neurons:
        # FlyWire spells the sides out; everything downstream matches on L and R.
        neurons["side"] = neurons["side"].replace(
            {"left": "L", "right": "R", "center": "M", "middle": "M"}
        )
    if "soma" in neurons:
        xyz = parse_soma(neurons.pop("soma"))
        for i, col in enumerate(SOMA_COLUMNS):
            neurons[col] = xyz[:, i]

    W = sp.load_npz(files.matrix).tocoo()
    if W.shape[0] != len(neurons):
        raise ValueError(
            f"{files.matrix.name} is {W.shape} but {files.meta.name} has "
            f"{len(neurons)} rows. The metadata must be in matrix order."
        )

    from .signs import photoreceptor_check

    pr = photoreceptor_check(neurons)
    if not pr["ok"]:
        warnings.warn(
            f"{name or files.meta.name}: {pr['note']} ({pr['by_nt']})",
            stacklevel=2,
        )

    if sign_source == "dataset" and "dataset_sign" in neurons:
        signs = neurons["dataset_sign"].to_numpy()
    else:
        signs = sign_vector(neurons["nt"]).to_numpy()

    counts = matrix == "syncount"
    cutoff = min_synapses if counts else (min_weight if min_weight is not None else 0.0)
    keep = W.data >= cutoff if counts else W.data > cutoff
    row, col, data = W.row[keep], W.col[keep], W.data[keep].astype(np.float32)
    data *= signs[row].astype(np.float32)
    nonzero = data != 0
    signed = sp.coo_matrix(
        (data[nonzero], (row[nonzero], col[nonzero])), shape=W.shape, dtype=np.float32
    ).tocsr()

    return Connectome(
        neurons=neurons,
        W=signed,
        name=name or Path(folder).name,
        meta={
            "source": str(folder),
            "meta_file": files.meta.name,
            "matrix_file": files.matrix.name,
            "axon_dendrite_only": files.axon_dendrite,
            "matrix_kind": matrix,
            "threshold": float(min_synapses if matrix == "syncount" else (min_weight or 0.0)),
            "min_synapses": min_synapses if matrix == "syncount" else None,
            "sign_source": sign_source,
            "provenance": "YijieYin/connectome_data_prep",
        },
    )


def load_dataset(root: str | Path, dataset: str = "malecns", **kwargs) -> Connectome:
    """Load by dataset name from a clone of ``connectome_data_prep``.

    ``root`` is either the repository root or its ``data`` directory.
    """
    root = Path(root)
    base = root / "data" if (root / "data").is_dir() else root
    if dataset not in DATASET_FOLDERS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASET_FOLDERS)}")
    return load_prepared(base / DATASET_FOLDERS[dataset], name=dataset, **kwargs)
