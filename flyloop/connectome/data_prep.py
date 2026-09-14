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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .schema import Connectome
from .signs import sign_vector

#: Metadata column -> the name this project uses.
COLUMN_MAP = {
    "bodyid": "id",
    "type": "type",
    "top_nt": "nt",
    "somaSide": "side",
    "superclass": "super_class",
    "assignedOlHex1": "hex1",
    "assignedOlHex2": "hex2",
    "sign": "dataset_sign",
}

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


def find_files(folder: str | Path, *, prefer_axon_dendrite: bool = False) -> PreparedFiles:
    """Locate the metadata CSV and the synapse-count matrix inside a folder.

    Some datasets ship only the axon-dendrite-split matrix, so the plain
    ``syncount`` file is preferred when present and the ``ad_`` variant is used
    otherwise -- with which one was chosen recorded, because the two are not
    interchangeable: the axon-dendrite matrix keeps only connections from an
    axon onto a dendrite, which is a real filtering decision.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a directory")
    metas = sorted(folder.glob("*_meta.csv"))
    if not metas:
        raise FileNotFoundError(f"no *_meta.csv in {folder}")

    plain = sorted(p for p in folder.glob("*syncount*.npz") if "_ad_" not in p.name)
    split = sorted(folder.glob("*_ad_syncount*.npz"))
    order = (split, plain) if prefer_axon_dendrite else (plain, split)
    matrix = next((c[0] for c in order if c), None)
    if matrix is None:
        raise FileNotFoundError(
            f"no synapse-count matrix in {folder}. Found: "
            f"{[p.name for p in folder.glob('*.npz')]}"
        )
    return PreparedFiles(meta=metas[0], matrix=matrix, axon_dendrite="_ad_" in matrix.name)


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


def load_prepared(
    folder: str | Path,
    *,
    name: str | None = None,
    min_synapses: int = 5,
    sign_source: str = "flyloop",
    prefer_axon_dendrite: bool = False,
) -> Connectome:
    """Build a :class:`Connectome` from one prepared dataset folder.

    Parameters
    ----------
    min_synapses:
        Drop connections weaker than this. On MaleCNS a threshold of 5 keeps
        24% of the connections but 72% of the synapses, which is the usual
        trade: most edges are one or two synapses and mostly reconstruction
        noise.
    sign_source:
        ``"flyloop"`` uses this project's transmitter table;
        ``"dataset"`` uses the metadata's own ``sign`` column. They disagree on
        about 5% of MaleCNS neurons -- run :func:`sign_report` before choosing.
    """
    if sign_source not in ("flyloop", "dataset"):
        raise ValueError(f"sign_source must be 'flyloop' or 'dataset', got {sign_source!r}")

    files = find_files(folder, prefer_axon_dendrite=prefer_axon_dendrite)
    raw = pd.read_csv(files.meta, low_memory=False)

    present = {k: v for k, v in COLUMN_MAP.items() if k in raw.columns}
    missing = [k for k in ("bodyid", "type", "top_nt") if k not in raw.columns]
    if missing:
        raise ValueError(
            f"{files.meta.name} is missing {missing}; columns are {list(raw.columns)[:20]}"
        )
    neurons = raw[list(present)].rename(columns=present)
    for col in ("type", "nt", "side", "super_class"):
        if col in neurons:
            neurons[col] = neurons[col].fillna("unknown").astype(str)

    W = sp.load_npz(files.matrix).tocoo()
    if W.shape[0] != len(neurons):
        raise ValueError(
            f"{files.matrix.name} is {W.shape} but {files.meta.name} has "
            f"{len(neurons)} rows. The metadata must be in matrix order."
        )

    if sign_source == "dataset" and "dataset_sign" in neurons:
        signs = neurons["dataset_sign"].to_numpy()
    else:
        signs = sign_vector(neurons["nt"]).to_numpy()

    keep = W.data >= min_synapses
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
            "min_synapses": min_synapses,
            "sign_source": sign_source,
            "provenance": "YijieYin/connectome_data_prep",
        },
    )


def load_dataset(
    root: str | Path, dataset: str = "malecns", **kwargs
) -> Connectome:
    """Load by dataset name from a clone of ``connectome_data_prep``.

    ``root`` is either the repository root or its ``data`` directory.
    """
    root = Path(root)
    base = root / "data" if (root / "data").is_dir() else root
    if dataset not in DATASET_FOLDERS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASET_FOLDERS)}")
    return load_prepared(base / DATASET_FOLDERS[dataset], name=dataset, **kwargs)
