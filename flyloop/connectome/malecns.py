"""Loading the MaleCNS v1.0 connectome (HHMI Janelia FlyEM and collaborators).

MaleCNS is the first *complete central nervous system*: brain plus ventral
nerve cord, 166,700 neurons, released under CC-BY.  The nerve cord is what
makes an embodied project tractable, because it carries the leg motor neurons
that a descending command actually has to reach.

Two ways in:

``from_neuprint``  queries https://neuprint.janelia.org live.  Good for a named
                   subcircuit; impractical for the whole dataset.
``from_dump``      reads the CSV export that backs the neuPrint database.  This
                   is the right path for whole-CNS work.

Both need data this machine may not be able to reach; see ``docs/DATA.md`` for
the download and for the auth token neuPrint requires.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import Connectome, build_signed_matrix

DATASET = "male-cns:v1.0"
SERVER = "neuprint.janelia.org"


def from_neuprint(
    *,
    criteria=None,
    token: str | None = None,
    server: str = SERVER,
    dataset: str = DATASET,
    min_synapses: int = 5,
) -> Connectome:
    """Fetch neurons and their adjacencies from a live neuPrint server.

    ``criteria`` is a ``neuprint.NeuronCriteria``; without one this would try to
    pull the entire CNS over HTTP, so a criteria-less call is refused.
    """
    try:
        from neuprint import Client, fetch_adjacencies
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "neuprint-python is required: pip install 'flyloop[data]'"
        ) from exc

    if criteria is None:
        raise ValueError(
            "refusing to fetch the whole CNS over HTTP. Pass a NeuronCriteria "
            "for the subcircuit you want, or use from_dump() for the full "
            "dataset (see docs/DATA.md)."
        )

    client = Client(server, dataset=dataset, token=token)
    neuron_df, conn_df = fetch_adjacencies(criteria, criteria, client=client)

    neurons = pd.DataFrame(
        {
            "id": neuron_df["bodyId"],
            "type": neuron_df.get("type", pd.Series(dtype=str)).fillna("unknown").astype(str),
            "nt": neuron_df.get("predictedNt", pd.Series(dtype=str))
            .fillna("unknown")
            .astype(str),
            "side": neuron_df.get("somaSide", pd.Series(dtype=str)).fillna("").astype(str),
            "super_class": neuron_df.get("class", pd.Series(dtype=str)).fillna("").astype(str),
        }
    ).drop_duplicates(subset="id").reset_index(drop=True)

    edges = (
        conn_df.rename(columns={"bodyId_pre": "pre", "bodyId_post": "post"})
        .groupby(["pre", "post"], as_index=False)["weight"]
        .sum()
    )
    W = build_signed_matrix(neurons, edges, min_synapses=min_synapses)
    return Connectome(
        neurons=neurons,
        W=W,
        name=f"malecns-{dataset}",
        meta={"source": f"{server}/{dataset}", "licence": "CC-BY", "subset": True},
    )


def from_dump(
    directory: str | Path,
    *,
    neurons_file: str = "Neuprint_Neurons.csv",
    connections_file: str = "Neuprint_Neuron_Connections.csv",
    min_synapses: int = 5,
) -> Connectome:
    """Build the full connectome from the neuPrint CSV export.

    The export's column names have changed between releases, so the loader maps
    what it finds and tells you plainly when a required column is missing rather
    than producing an empty matrix.
    """
    directory = Path(directory)
    npath, cpath = directory / neurons_file, directory / connections_file
    for p in (npath, cpath):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download the male-cns:v1.0 CSV export from "
                "https://male-cns.janelia.org/download/ (see docs/DATA.md)."
            )

    raw = pd.read_csv(npath, low_memory=False)
    rename = {
        ":ID(Body-ID)": "id",
        "bodyId:long": "id",
        "bodyId": "id",
        "type:string": "type",
        "type": "type",
        "predictedNt:string": "nt",
        "predictedNt": "nt",
        "somaSide:string": "side",
        "somaSide": "side",
        "class:string": "super_class",
        "class": "super_class",
    }
    raw = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})
    missing = [c for c in ("id", "type", "nt") if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{npath.name} is missing {missing}. Columns present: {list(raw.columns)[:20]}. "
            "The export format may have changed; map the columns explicitly."
        )
    for c in ("type", "nt", "side", "super_class"):
        if c in raw.columns:
            raw[c] = raw[c].fillna("unknown").astype(str)
    cols = [c for c in ("id", "type", "nt", "side", "super_class") if c in raw.columns]
    neurons = raw[cols].drop_duplicates(subset="id").reset_index(drop=True)

    con = pd.read_csv(cpath, low_memory=False)
    con = con.rename(
        columns={
            ":START_ID(Body-ID)": "pre",
            ":END_ID(Body-ID)": "post",
            "weight:int": "weight",
            "bodyId_pre": "pre",
            "bodyId_post": "post",
        }
    )
    missing = [c for c in ("pre", "post", "weight") if c not in con.columns]
    if missing:
        raise ValueError(
            f"{cpath.name} is missing {missing}. Columns present: {list(con.columns)[:20]}."
        )
    con = con.groupby(["pre", "post"], as_index=False)["weight"].sum()

    W = build_signed_matrix(neurons, con, min_synapses=min_synapses)
    return Connectome(
        neurons=neurons,
        W=W,
        name="malecns-v1.0",
        meta={
            "source": str(directory),
            "min_synapses": min_synapses,
            "licence": "CC-BY",
            "citation": (
                "Sexual dimorphism in the complete Drosophila male CNS "
                "connectome, Cell 2026"
            ),
        },
    )
