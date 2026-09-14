"""Loading the FlyWire (FAFB) adult female brain connectome.

Get the files from Codex (https://codex.flywire.ai/api/download), which exports
the v783 data products as gzipped CSVs.  Two are needed:

    classification.csv.gz   root_id, super_class, cell_type, side, nt_type, ...
    connections.csv.gz      pre_root_id, post_root_id, neuropil, syn_count, nt_type

FlyWire is the female *brain*: it stops at the neck.  It has no ventral nerve
cord and therefore no leg motor neurons, so a FlyWire model can produce
descending commands but cannot close the loop down to the legs on its own.
For that, use MaleCNS -- see :mod:`flyloop.connectome.malecns`.

FlyWire data is CC BY-NC (non-commercial); MaleCNS is CC-BY.  Check the licence
that applies to your use before building anything on top.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import Connectome, build_signed_matrix

#: Filenames looked for, in order of preference, for each role.
_CLASSIFICATION = ("classification.csv.gz", "classification.csv", "flywire_annotations.tsv")
_CONNECTIONS = (
    "connections.csv.gz",
    "connections.csv",
    "connections_princeton_no_threshold.csv.gz",
)


def _find(directory: Path, candidates: tuple[str, ...], role: str) -> Path:
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no {role} file in {directory}. Looked for: {', '.join(candidates)}.\n"
        "Download the v783 data products from https://codex.flywire.ai/api/download "
        "(see docs/DATA.md)."
    )


def load_flywire(
    directory: str | Path,
    *,
    min_synapses: int = 5,
    keep_super_classes: tuple[str, ...] | None = None,
) -> Connectome:
    """Build a :class:`Connectome` from a directory of Codex exports.

    ``min_synapses`` drops the weakest connections.  A threshold of 5 is the
    conventional choice for FlyWire and removes most reconstruction noise; set
    it to 1 only if you know why you want to.
    """
    directory = Path(directory)
    cls_path = _find(directory, _CLASSIFICATION, "classification")
    con_path = _find(directory, _CONNECTIONS, "connections")

    sep = "\t" if cls_path.suffix == ".tsv" else ","
    cls = pd.read_csv(cls_path, sep=sep, low_memory=False)
    cls = cls.rename(
        columns={
            "root_id": "id",
            "cell_type": "type",
            "nt_type": "nt",
            "hemibrain_type": "hemibrain_type",
        }
    )
    if "type" not in cls or cls["type"].isna().all():
        # Codex splits naming across cell_type and hemibrain_type.
        cls["type"] = cls.get("hemibrain_type")
    cls["type"] = cls["type"].fillna("unknown").astype(str)
    cls["nt"] = cls["nt"].fillna("unknown").astype(str)
    if keep_super_classes and "super_class" in cls:
        cls = cls[cls["super_class"].isin(keep_super_classes)]

    cols = [c for c in ("id", "type", "nt", "side", "super_class") if c in cls]
    neurons = cls[cols].drop_duplicates(subset="id").reset_index(drop=True)

    con = pd.read_csv(con_path, low_memory=False)
    con = con.rename(
        columns={"pre_root_id": "pre", "post_root_id": "post", "syn_count": "weight"}
    )
    if "weight" not in con:
        raise ValueError(
            f"{con_path.name} has no syn_count/weight column; columns are {list(con.columns)}"
        )
    # Codex rows are per-neuropil; sum them into one edge per neuron pair.
    con = con.groupby(["pre", "post"], as_index=False)["weight"].sum()

    W = build_signed_matrix(neurons, con, min_synapses=min_synapses)
    return Connectome(
        neurons=neurons,
        W=W,
        name="flywire-v783",
        meta={
            "source": str(directory),
            "min_synapses": min_synapses,
            "licence": "CC BY-NC (FlyWire)",
            "caveat": "brain only; no ventral nerve cord, no leg motor neurons",
        },
    )
