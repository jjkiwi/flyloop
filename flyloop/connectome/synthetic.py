"""A synthetic stand-in connectome.

This is **not** a fly.  It is a test fixture: a small wiring diagram with the
same shape as the real thing (retinotopic visual columns, a loom-sensitive
projection-neuron population, a giant fibre, named descending neurons, leg motor
neurons, and a bulk of balanced central interneurons) so that the engine, the
eye, the readout, the body and the experiments can all be exercised end to end
without a 100 GB download and without a GPU.

Use it to prove the *plumbing* works.  Every scientific claim has to be made
against a real dataset -- see ``docs/DATA.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Connectome, build_signed_matrix


def synthetic_connectome(
    *,
    n_columns: int = 128,
    n_lc4_per_side: int = 30,
    n_central: int = 2000,
    central_degree: int = 20,
    seed: int = 0,
) -> Connectome:
    """Build the fixture.

    Parameters mirror the real circuit at reduced scale.  The escape pathway
    ``L4 columns -> LC4 -> GF -> DNp09`` is wired deliberately, because that is
    the chain the acceptance experiment tests; everything else is random but
    E/I balanced so that the network has realistic background activity instead
    of sitting silent.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    nid = 0

    def add(type_: str, nt: str, side: str, super_class: str, k: int = 1) -> list[int]:
        nonlocal nid
        ids = []
        for _ in range(k):
            rows.append(
                {"id": nid, "type": type_, "nt": nt, "side": side, "super_class": super_class}
            )
            ids.append(nid)
            nid += 1
        return ids

    cols: dict[str, list[int]] = {}
    lc4: dict[str, list[int]] = {}
    gf: dict[str, list[int]] = {}
    dn: dict[tuple[str, str], list[int]] = {}
    mn: dict[str, list[int]] = {}
    ttm: dict[str, list[int]] = {}

    for side in ("L", "R"):
        # Photoreceptor-equivalent input columns, one per ommatidium.
        # Real photoreceptors are histaminergic and therefore *inhibit* their
        # targets, with the sign inverted again downstream in the lamina.  The
        # fixture skips that inversion and makes the column output directly
        # excitatory, so that "more light on this patch" means "more drive".
        cols[side] = add("PR", "acetylcholine", side, "sensory", n_columns)
        # Loom-sensitive visual projection neurons.
        lc4[side] = add("LC4", "acetylcholine", side, "optic", n_lc4_per_side)
        # The giant fibre: one enormous cell per side driving escape.
        gf[side] = add("GF", "acetylcholine", side, "descending")
        for name in ("DNp09", "DNa01", "DNa02", "MDN"):
            dn[(name, side)] = add(name, "acetylcholine", side, "descending")
        # Leg motor neurons: 3 joints x 3 legs per side, matching a hexapod.
        mn[side] = add("LegMN", "acetylcholine", side, "motor", 9)
        # Tergotrochanteral ("jump") motor neuron, the giant fibre's target.
        ttm[side] = add("TTMn", "acetylcholine", side, "motor")

    central_e = add("INe", "acetylcholine", "C", "central", int(n_central * 0.7))
    central_i = add("INi", "gaba", "C", "central", n_central - len(central_e))

    neurons = pd.DataFrame(rows)
    edges: list[tuple[int, int, int]] = []

    def connect(pre: list[int], post: list[int], w: int, p: float = 1.0) -> None:
        for a in pre:
            targets = post if p >= 1.0 else [b for b in post if rng.random() < p]
            for b in targets:
                edges.append((a, b, w))

    for side in ("L", "R"):
        # Retinotopy: each LC4 pools a contiguous patch of columns, so a
        # looming stimulus that expands across the eye recruits them in turn.
        patch = max(1, n_columns // n_lc4_per_side * 3)
        for k, cell in enumerate(lc4[side]):
            start = (k * n_columns) // n_lc4_per_side
            idx = [(start + o) % n_columns for o in range(patch)]
            connect([cols[side][i] for i in idx], [cell], w=20)

        # Escape pathway.  Two parallel branches, as in the real animal: the
        # giant fibre drives the jump motor neuron directly and very strongly
        # (a one-shot alarm, not a gentle integrator), while DNp09 is a
        # separate descending neuron recruited by the same visual population.
        # The giant fibre weight is set above synapses_to_threshold() so that a
        # single GF spike fires TTMn -- that is the point of the giant fibre.
        connect(lc4[side], gf[side], w=12)
        connect(gf[side], ttm[side], w=220)
        connect(lc4[side], dn[("DNp09", side)], w=10)
        connect(dn[("DNp09", side)], ttm[side], w=30)

        # Steering: looming on one side drives a turn away from it.  The
        # weight is per-LC4; steering is a population vote, so no single visual
        # neuron comes close to firing DNa02 on its own.
        other = "R" if side == "L" else "L"
        connect(lc4[side], dn[("DNa02", other)], w=90)

        # Descending neurons onto leg motor neurons.  Weights are chosen so
        # that a descending neuron firing at a plausible 20-80 Hz recruits its
        # motor pool, rather than so that one spike does.
        connect(dn[("DNa01", side)], mn[side], w=70)
        connect(dn[("DNa02", side)], mn[side], w=70)
        connect(dn[("MDN", side)], mn[side], w=70)
        connect(dn[("DNp09", side)], mn[side], w=40)

    # Balanced random central pool, plus weak reciprocal coupling with the DNs
    # so that the escape circuit is not an island.
    allc = central_e + central_i
    degree = min(central_degree, max(len(allc) - 1, 0))
    for a in allc:
        targets = rng.choice(allc, size=degree, replace=False)
        for b in targets:
            if b != a:
                edges.append((int(a), int(b), int(rng.integers(1, 4))))
    for side in ("L", "R"):
        srcs = rng.choice(central_e, size=min(40, len(central_e)), replace=False)
        connect([int(s) for s in srcs], dn[("DNa01", side)], w=1)
        srcs = rng.choice(allc, size=min(60, len(allc)), replace=False)
        connect(lc4[side], [int(s) for s in srcs], w=1, p=0.3)

    e = pd.DataFrame(edges, columns=["pre", "post", "weight"])
    e = e.groupby(["pre", "post"], as_index=False)["weight"].sum()
    W = build_signed_matrix(neurons, e)

    return Connectome(
        neurons=neurons,
        W=W,
        name="synthetic",
        meta={
            "synthetic": True,
            "warning": "test fixture, not a real nervous system",
            "n_columns": n_columns,
            "seed": seed,
        },
    )


def eye_columns(c: Connectome) -> dict[str, np.ndarray]:
    """Row indices of the photoreceptor columns of each eye, in retinotopic order."""
    out = {}
    for side in ("L", "R"):
        m = (c.neurons["type"] == "PR") & (c.neurons["side"] == side)
        out[side] = np.flatnonzero(m.to_numpy())
    return out
