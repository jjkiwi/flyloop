"""Named chemicals as glomerular input, from measured receptor responses.

Until now an odour in this project was a list of glomerulus names -- DM1 and
DM4, picked because the literature uses them. That is fine for asking whether
*a* smell can be learned, and useless for asking whether *this substance* can
be, because nothing connects a chemical to the glomeruli it drives.

DoOR (the Database of Odorant Responses) is that connection: 693 chemicals
against 79 recorded units, plus a receptor-to-glomerulus table. Joined, they
turn a compound into a graded activation pattern over the antennal lobe --
which is what the animal's nose actually produces, and what a hand-picked pair
of names only approximates.

**Only unambiguous units are used.** Many DoOR rows are whole-sensillum
recordings covering several glomeruli at once ("VL1+DP1l+VC5 via ac2"). Giving
each of those glomeruli the same response invents a specificity the recording
does not have, so rows that do not resolve to exactly one glomerulus are
dropped and :func:`door_profile` reports how much signal that cost.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..connectome.schema import Connectome

#: Where DoOR sits inside a connectome_data_prep clone.
DOOR_SUBDIR = "data/DoOR"

#: Response below this is treated as no response.
DOOR_FLOOR = 0.05


@lru_cache(maxsize=4)
def _door(data_root: str) -> tuple[pd.DataFrame, dict, dict]:
    """The response matrix keyed by chemical name, and the glomerulus map."""
    d = Path(data_root) / DOOR_SUBDIR
    matrix = (
        pd.read_csv(d / "door_response_matrix.csv")
        .rename(columns={"Unnamed: 0": "key"})
        .set_index("key")
    )
    meta = pd.read_csv(d / "door_chemical_meta.csv")
    mapping = pd.read_csv(d / "door_receptor_mappings.csv")
    name_by_key = dict(
        zip(meta["InChIKey"].astype(str), meta["Name"].astype(str), strict=True)
    )
    matrix.index = [name_by_key.get(k, k) for k in matrix.index]
    # A unit is addressed either by its OSN code (ab1A) or by its receptor.
    # fillna before astype: recent pandas keeps NaN as NaN through astype(str),
    # so 17 receptors with no glomerulus would arrive as floats and fail the
    # string tests below rather than being skipped.
    unit_to_glomerulus: dict[str, str] = {}
    glom_names = mapping["glomerulus"].fillna("").astype(str)
    for col in ("code.OSN", "receptor"):
        units = mapping[col].fillna("").astype(str)
        for unit, g in zip(units, glom_names, strict=True):
            if unit and g:
                unit_to_glomerulus.setdefault(unit, g)
    return matrix, unit_to_glomerulus, name_by_key


def door_chemicals(data_root: str | Path) -> list[str]:
    """Every chemical DoOR has a response profile for."""
    matrix, _, _ = _door(str(data_root))
    return sorted(str(i) for i in matrix.index if str(i) != "SFR")


def door_profile(
    data_root: str | Path,
    chemical: str,
    *,
    connectome: Connectome | None = None,
    floor: float = DOOR_FLOOR,
) -> pd.Series:
    """Glomerulus -> response for one chemical, as a fraction of maximum.

    Restricted to units that resolve to a single glomerulus, and -- when a
    ``connectome`` is given -- to glomeruli that connectome actually has. The
    returned Series carries ``attrs["dropped"]``: the share of the response that
    fell to ambiguous or absent units, which is the number that says whether
    what remains is the chemical's profile or a corner of it.
    """
    matrix, unit_to_glomerulus, _ = _door(str(data_root))
    if chemical not in matrix.index:
        raise KeyError(
            f"DoOR has no chemical named {chemical!r}; "
            f"{len(matrix)} are available, see door_chemicals()"
        )
    row = matrix.loc[chemical]
    if isinstance(row, pd.DataFrame):  # a duplicated name in the table
        row = row.iloc[0]
    row = row.dropna()
    row = row[row > floor]
    total = float(row.sum())

    have = None
    if connectome is not None:
        from .olfactory import glomeruli

        have = set(glomeruli(connectome).index)

    kept: dict[str, float] = {}
    for unit, value in row.items():
        glom = unit_to_glomerulus.get(str(unit))
        # "+" means a sensillum-level recording spanning several glomeruli.
        if not glom or "+" in glom or glom == "?":
            continue
        if have is not None and glom not in have:
            continue
        kept[glom] = max(kept.get(glom, 0.0), float(value))

    out = pd.Series(kept, dtype=float).sort_values(ascending=False)
    out.index.name = "glomerulus"
    out.name = chemical
    out.attrs["dropped"] = 1.0 - (float(out.sum()) / total if total else 0.0)
    out.attrs["chemical"] = chemical
    return out


def profile_odour(
    c: Connectome,
    profile: pd.Series | dict[str, float],
    receptors: np.ndarray,
    *,
    concentration: float = 1.0,
) -> np.ndarray:
    """A graded ORN input vector: each glomerulus at its measured response.

    The binary :func:`~flyloop.experiments.olfactory.odour` gives every ORN of a
    named glomerulus 1.0. This gives them the response the chemical actually
    evokes, scaled by ``concentration`` -- so the same substance at half the
    concentration is the same pattern at half the drive, which is what makes a
    dose-response experiment mean anything.

    Concentration is a linear scale on receptor drive and nothing more. Real
    ORN dose-response curves are sigmoid in log concentration and saturate;
    this does not model that, and a result that depends on the shape of the
    curve rather than its monotonicity is not supported by it.
    """
    if concentration < 0:
        raise ValueError("concentration cannot be negative")
    want = c.neurons["type"].astype(str).to_numpy()[receptors]
    v = np.zeros(len(receptors), dtype=np.float32)
    items = profile.items() if hasattr(profile, "items") else dict(profile).items()
    hit_any = False
    for glom, response in items:
        hit = np.flatnonzero(want == f"ORN_{glom}")
        if len(hit) == 0:
            continue
        v[hit] = float(response) * concentration
        hit_any = True
    if not hit_any:
        raise KeyError(
            "none of the profile's glomeruli exist in this connectome: "
            f"{sorted(dict(profile))}"
        )
    return v


#: The chemical this project uses to stand in for D-2-hydroxyglutarate.
#:
#: **D-2HG cannot be smelled and this is not a claim that it can.** It is a
#: dicarboxylic acid, ionised at physiological pH and effectively non-volatile;
#: it is measured in blood, urine and tissue, not in air. *Drosophila*
#: olfaction detects volatiles, so no glomerular response to D-2HG exists in
#: DoOR, in Hallem-Carlson, in Dweck, or anywhere else in the prepared data --
#: checked, not assumed.
#:
#: What does exist is its chemical family. D-2HG is the reduced form of
#: 2-oxoglutarate, and DoOR carries the nearest measured homologue of that:
#: 2-oxovaleric acid, the five-carbon alpha-keto acid. Its profile is driven by
#: the ionotropic receptors -- IR64a onto DC4 and DP1m, IR75a onto DP1l, IR84a
#: onto VL2a, IR31a onto VL2p -- which is the acid-sensing pathway, the one an
#: alpha-hydroxy or alpha-keto diacid would engage if it engaged anything.
#:
#: So this is a **proxy stimulus with a stated derivation**, not D-2HG. Every
#: result using it is a result about an acid-class odour that the fly can
#: actually detect, and the substitution has to be reported with it.
D2HG_PROXY = "2-oxovaleric acid"

#: An odour of a different chemical class, for the unrewarded control. Esters
#: go through the odorant receptors rather than the ionotropic ones, so the two
#: engage different receptor families and are separable by construction.
D2HG_CONTROL = "ethyl acetate"


def d2hg_profile(data_root: str | Path, *, connectome: Connectome | None = None):
    """The proxy stimulus for D-2HG, and the control it is measured against.

    Returns ``(proxy, control)``, both glomerulus-indexed response profiles.
    See :data:`D2HG_PROXY` for why the proxy is what it is; nothing here should
    be read as the fly smelling D-2HG.
    """
    return (
        door_profile(data_root, D2HG_PROXY, connectome=connectome),
        door_profile(data_root, D2HG_CONTROL, connectome=connectome),
    )
