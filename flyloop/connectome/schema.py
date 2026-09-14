"""The in-memory representation of a wiring diagram."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .signs import sign_vector, unknown_transmitters

#: Columns every neuron table must provide.
REQUIRED_COLUMNS = ("id", "type", "nt")


@dataclass
class Connectome:
    """A signed, directed, sparse wiring diagram plus its neuron metadata.

    Attributes
    ----------
    neurons:
        One row per neuron, indexed 0..N-1 in the same order as the rows and
        columns of :attr:`W`.  Must carry ``id`` (the dataset's own identifier),
        ``type`` (cell type, e.g. ``"DNp09"``) and ``nt`` (transmitter label).
        May carry ``side`` (``"L"``/``"R"``) and ``super_class``.
    W:
        ``(N, N)`` CSR matrix.  ``W[pre, post]`` is the *signed synapse count*
        from ``pre`` onto ``post``: positive for excitatory presynaptic
        transmitters, negative for inhibitory.  Multiplying by the per-synapse
        weight happens in the brain model, not here, so that the same wiring can
        be run at different synaptic gains.
    """

    neurons: pd.DataFrame
    W: sp.csr_matrix
    name: str = "unnamed"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.neurons.columns]
        if missing:
            raise ValueError(f"neuron table is missing columns: {missing}")
        n = len(self.neurons)
        if self.W.shape != (n, n):
            raise ValueError(f"W has shape {self.W.shape}, expected ({n}, {n})")
        if not isinstance(self.W, sp.csr_matrix):
            self.W = sp.csr_matrix(self.W)
        self.neurons = self.neurons.reset_index(drop=True)
        self._type_index: dict[str, np.ndarray] | None = None

    # ---------------------------------------------------------------- basics

    @property
    def n(self) -> int:
        """Number of neurons."""
        return len(self.neurons)

    @property
    def n_synapses(self) -> float:
        """Total synapse count (absolute value of the signed weights)."""
        return float(np.abs(self.W.data).sum())

    @property
    def n_connections(self) -> int:
        """Number of non-zero neuron-to-neuron connections."""
        return int(self.W.nnz)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<Connectome {self.name!r}: {self.n:,} neurons, "
            f"{self.n_connections:,} connections, {self.n_synapses:,.0f} synapses>"
        )

    # ----------------------------------------------------------- populations

    def _types(self) -> dict[str, np.ndarray]:
        if self._type_index is None:
            groups = self.neurons.groupby(
                self.neurons["type"].fillna("").astype(str)
            ).indices
            self._type_index = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        return self._type_index

    def population(self, *types: str, regex: bool = False) -> np.ndarray:
        """Row indices of neurons whose ``type`` matches any of ``types``.

        With ``regex=False`` the match is exact, which is what you want for
        named cell types like ``DNp09``.  With ``regex=True`` the patterns are
        treated as regular expressions, useful for families such as
        ``r"^LC\\d+$"``.
        """
        idx = self._types()
        if not regex:
            out = [idx[t] for t in types if t in idx]
        else:
            compiled = [re.compile(p) for p in types]
            out = [
                v for k, v in idx.items() if any(c.search(k) for c in compiled)
            ]
        if not out:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(out))

    def require(self, *types: str) -> dict[str, np.ndarray]:
        """Like :meth:`population` per type, raising if any type is absent.

        Use this at the top of an experiment so a missing cell type fails loudly
        instead of silently producing a flat trace.
        """
        idx = self._types()
        absent = [t for t in types if t not in idx or len(idx[t]) == 0]
        if absent:
            raise KeyError(
                f"connectome {self.name!r} has no neurons of type(s): {absent}. "
                f"Available example types: {sorted(idx)[:10]}"
            )
        return {t: idx[t] for t in types}

    def types(self) -> list[str]:
        """Sorted list of all cell-type labels present."""
        return sorted(self._types())

    # ------------------------------------------------------------ diagnostics

    def report(self) -> str:
        """Human-readable sanity report.  Read this before trusting any run."""
        nt_counts = self.neurons["nt"].fillna("unknown").astype(str).str.lower().value_counts()
        signs = sign_vector(self.neurons["nt"])
        unsigned = int((signs == 0).sum())
        exc = int((signs > 0).sum())
        inh = int((signs < 0).sum())
        pos = float(self.W.data[self.W.data > 0].sum())
        neg = float(-self.W.data[self.W.data < 0].sum())
        unknown = unknown_transmitters(self.neurons["nt"])
        lines = [
            f"connectome: {self.name}",
            f"  neurons ............ {self.n:,}",
            f"  connections ........ {self.n_connections:,}",
            f"  synapses ........... {self.n_synapses:,.0f}",
            f"  excitatory neurons . {exc:,} ({exc / max(self.n, 1):.1%})",
            f"  inhibitory neurons . {inh:,} ({inh / max(self.n, 1):.1%})",
            f"  unsigned neurons ... {unsigned:,} "
            f"({unsigned / max(self.n, 1):.1%})  <- their outgoing synapses do nothing",
            f"  E/I synapse ratio .. {pos / max(neg, 1.0):.2f}  (E={pos:,.0f}, I={neg:,.0f})",
            f"  cell types ......... {len(self._types()):,}",
            "  transmitters:      "
            + ", ".join(f"{k}={v:,}" for k, v in nt_counts.head(8).items()),
        ]
        if len(unknown):
            lines.append(
                "  WARNING unrecognised transmitter labels (treated as sign 0): "
                + ", ".join(f"{k}={v:,}" for k, v in unknown.head(5).items())
            )
        return "\n".join(lines)

    # ---------------------------------------------------------------- subsets

    def subset(self, idx: np.ndarray, name: str | None = None) -> Connectome:
        """Return the induced subgraph over ``idx`` (row indices)."""
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        return Connectome(
            neurons=self.neurons.iloc[idx].reset_index(drop=True),
            W=sp.csr_matrix(self.W[idx][:, idx]),
            name=name or f"{self.name}[subset:{len(idx)}]",
            meta=dict(self.meta, parent=self.name),
        )

    # ------------------------------------------------------------------- i/o

    def save(self, path: str | Path) -> Path:
        """Persist to a directory as ``neurons.parquet`` + ``W.npz`` + ``meta.json``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        try:
            self.neurons.to_parquet(path / "neurons.parquet")
        except (ImportError, ValueError):  # no pyarrow, or a column it cannot type
            self.neurons.to_csv(path / "neurons.csv", index=False)
        sp.save_npz(path / "W.npz", self.W)
        (path / "meta.json").write_text(
            json.dumps({"name": self.name, "meta": self.meta}, indent=2)
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Connectome:
        """Inverse of :meth:`save`."""
        path = Path(path)
        if (path / "neurons.parquet").exists():
            neurons = pd.read_parquet(path / "neurons.parquet")
        else:
            neurons = pd.read_csv(path / "neurons.csv")
        W = sp.load_npz(path / "W.npz").tocsr()
        info = json.loads((path / "meta.json").read_text())
        return cls(neurons=neurons, W=W, name=info["name"], meta=info["meta"])


def build_signed_matrix(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    pre_col: str = "pre",
    post_col: str = "post",
    weight_col: str = "weight",
    min_synapses: int = 1,
) -> sp.csr_matrix:
    """Turn an edge list plus a neuron table into a signed CSR matrix.

    ``edges`` refers to neurons by their dataset ``id``; rows whose endpoints are
    not in ``neurons`` are dropped (and counted in the returned matrix's
    provenance only if the caller checks).  The sign comes from the
    *presynaptic* neuron's transmitter, which is how transmitter identity works:
    a neuron releases the same transmitter at all of its outputs.
    """
    pos = pd.Series(np.arange(len(neurons), dtype=np.int64), index=neurons["id"].to_numpy())
    e = edges[[pre_col, post_col, weight_col]].copy()
    e = e[e[weight_col] >= min_synapses]
    e["i"] = e[pre_col].map(pos)
    e["j"] = e[post_col].map(pos)
    e = e.dropna(subset=["i", "j"])
    if e.empty:
        return sp.csr_matrix((len(neurons), len(neurons)), dtype=np.float32)

    signs = sign_vector(neurons["nt"]).to_numpy()
    i = e["i"].to_numpy(dtype=np.int64)
    j = e["j"].to_numpy(dtype=np.int64)
    w = e[weight_col].to_numpy(dtype=np.float32) * signs[i].astype(np.float32)

    keep = w != 0
    W = sp.coo_matrix(
        (w[keep], (i[keep], j[keep])), shape=(len(neurons), len(neurons)), dtype=np.float32
    )
    return W.tocsr()
