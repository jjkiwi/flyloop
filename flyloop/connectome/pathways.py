"""Asking the wiring diagram directly, before asking the simulation.

A spiking model answers "does this population respond". The graph answers "is
there anything for it to respond through". The second question is cheaper, has
no free parameters, and often settles the first outright: a population that
receives zero synapses from the driven cells cannot be recruited by them at any
drive strength, and no amount of sweeping will change that.

Run this before concluding that a silent population is a threshold effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Connectome


def _population(c: Connectome, type_: str, side: str | None) -> np.ndarray:
    m = c.neurons["type"].astype(str) == type_
    if side and "side" in c.neurons.columns:
        m &= c.neurons["side"].astype(str) == side
    return np.flatnonzero(m.to_numpy())


def direct_drive(
    c: Connectome,
    source: str,
    targets: tuple[str, ...],
    *,
    source_side: str | None = "L",
    sides: tuple[str, ...] = ("L", "R"),
) -> pd.DataFrame:
    """Signed synapse counts from one cell type onto each target, per side.

    A zero is the strongest result this function produces: it says the pathway
    does not exist monosynaptically, so any response must be indirect.
    """
    src = _population(c, source, source_side)
    if len(src) == 0:
        raise KeyError(f"no {source!r} neurons on side {source_side!r}")
    W = c.W.tocsr()[src]
    rows = []
    for t in targets:
        row = {"target": t}
        for side in sides:
            tgt = _population(c, t, side)
            row[side] = float(W[:, tgt].sum()) if len(tgt) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).set_index("target")
    out.columns.name = f"synapses from {source}-{source_side}"
    return out


def input_balance(
    c: Connectome, target: str, *, side: str | None = "L", top: int = 8
) -> dict:
    """Excitation, inhibition and the strongest individual inputs onto a population.

    Two neurons with similar excitatory drive can behave completely differently
    if one of them also collects inhibition, which is invisible in a synapse
    count from the source alone.
    """
    tgt = _population(c, target, side)
    if len(tgt) == 0:
        raise KeyError(f"no {target!r} neurons on side {side!r}")
    col = np.asarray(c.W.tocsc()[:, tgt].sum(axis=1)).ravel()
    exc = float(col[col > 0].sum())
    inh = float(-col[col < 0].sum())
    order = np.argsort(-np.abs(col))[:top]
    inputs = pd.DataFrame(
        {
            "type": c.neurons["type"].to_numpy()[order],
            "side": c.neurons.get("side", pd.Series(["?"] * c.n)).to_numpy()[order],
            "synapses": col[order],
        }
    )
    return {
        "target": f"{target}_{side}",
        "excitation": exc,
        "inhibition": inh,
        "ratio": exc / max(inh, 1.0),
        "top_inputs": inputs[inputs["synapses"] != 0].reset_index(drop=True),
    }


def drive_balance(
    c: Connectome,
    source: str,
    targets: tuple[str, ...],
    *,
    source_side: str | None = "L",
    side: str = "L",
) -> pd.DataFrame:
    """Direct drive alongside each target's overall excitation/inhibition balance.

    This is the table to read when the *functional* recruitment order does not
    match the monosynaptic one: a target with more direct synapses can still
    respond later if it sits in a more inhibited part of the network.
    """
    direct = direct_drive(c, source, targets, source_side=source_side, sides=(side,))
    rows = []
    for t in targets:
        bal = input_balance(c, t, side=side, top=1)
        rows.append(
            {
                "target": t,
                f"direct_from_{source}": float(direct.loc[t, side]),
                "excitation": bal["excitation"],
                "inhibition": bal["inhibition"],
                "E/I": bal["ratio"],
            }
        )
    out = pd.DataFrame(rows).set_index("target")
    # A target that receives no excitation at all has no defined share, and
    # that is not the same fact as receiving none *from this source*. The first
    # is NaN, the second is 0.0; collapsing them would hide a population that
    # nothing drives behind one that this particular source does not drive.
    excitation = out["excitation"]
    out["direct_share_of_E"] = np.where(
        excitation > 0,
        out[f"direct_from_{source}"].clip(lower=0) / excitation.where(excitation > 0),
        np.nan,
    )
    return out
