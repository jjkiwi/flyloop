"""Neurotransmitter -> synaptic sign.

This is the single most consequential table in the project and the easiest
place to introduce a silent bug.

In *Drosophila*, glutamate is predominantly **inhibitory** (via the GluCl-alpha
chloride channel), unlike in vertebrate cortex.  Several community
reimplementations get this backwards and treat glutamate as excitatory; the
network then runs away into saturation and the authors compensate by tuning the
readout until something moves.  Do not do that.

The measurement is Liu & Wilson, *PNAS* 2013;110:10294-10299, "Glutamate is an
inhibitory neurotransmitter in the Drosophila olfactory system": iontophoresed
glutamate hyperpolarises every major antennal-lobe cell type, and the effect is
abolished by RNAi knockdown of GluCl-alpha.  (Widely miscited as Nature
Neuroscience; it is PNAS.)

**Where the confusion comes from.** At the larval neuromuscular junction
glutamate *is* excitatory, through GluRII receptors.  A reimplementation that
flips the sign is usually importing the neuromuscular fact into the adult
central brain, where it does not hold.  This table is about the central brain.

Neuromodulators (dopamine, octopamine, serotonin) are assigned sign 0: the LIF
model of Shiu et al. has no mechanism for them, so pretending they are fast
excitation would be worse than leaving them out.  Their absence is a stated
limitation of the model, not something to paper over.
"""

from __future__ import annotations

import pandas as pd

#: Sign applied to every synapse made *by* a neuron with the given transmitter.
NT_SIGN: dict[str, int] = {
    "acetylcholine": +1,
    "ach": +1,
    "gaba": -1,
    "glutamate": -1,
    "glut": -1,
    # Neuromodulators: no fast ionotropic effect in this model.
    "dopamine": 0,
    "da": 0,
    "octopamine": 0,
    "oct": 0,
    "serotonin": 0,
    "5ht": 0,
    "histamine": -1,  # photoreceptor transmitter, inhibitory onto LMCs
    "unknown": 0,
    "unclear": 0,
}

#: Transmitters that carry no fast synaptic sign in this model.
UNSIGNED = {nt for nt, s in NT_SIGN.items() if s == 0}


def sign_of(nt: str | float | None) -> int:
    """Return +1, -1 or 0 for a neurotransmitter label (case/space insensitive)."""
    if nt is None or (isinstance(nt, float) and pd.isna(nt)):
        return 0
    return NT_SIGN.get(str(nt).strip().lower(), 0)


def sign_vector(nt: pd.Series, types: pd.Series | None = None) -> pd.Series:
    """Vectorised :func:`sign_of` over a column of transmitter labels.

    Pass ``types`` to let cell identity override the transmitter label where
    identity is the more reliable of the two. Only photoreceptors qualify: a
    cell named R1-6 releases histamine whatever a transmitter classifier says,
    and FlyWire's classifier has no histamine class, so it calls them
    acetylcholine and inverts the visual system's first and largest synapse
    (Run 15). No other type gets this treatment, because no other type has a
    transmitter that is beyond question from the name alone.
    """
    signs = nt.map(sign_of).astype("int8")
    if types is None:
        return signs
    hit = types.astype(str).str.match(PHOTORECEPTOR_PATTERN, na=False)
    if hit.any():
        signs = signs.mask(hit, sign_of(PHOTORECEPTOR_NT)).astype("int8")
    return signs


def unknown_transmitters(nt: pd.Series) -> pd.Series:
    """Labels present in the data that this table does not recognise.

    Call this on every freshly loaded dataset.  A large unrecognised fraction
    means the connections are effectively being deleted, silently.
    """
    seen = nt.dropna().astype(str).str.strip().str.lower()
    return seen[~seen.isin(NT_SIGN)].value_counts()


#: Cell-type patterns for photoreceptors, whose transmitter is not in doubt.
PHOTORECEPTOR_PATTERN = r"^R[1-8]"

#: What they release. Histamine gates a chloride channel (HisCl1/ort), so the
#: first synapse of the visual system is sign-inverting.
PHOTORECEPTOR_NT = "histamine"


def photoreceptor_check(neurons) -> dict:
    """Are this dataset's photoreceptors labelled with the transmitter they use?

    Worth asking of every dataset, because the answer is not always yes and the
    failure is silent. FlyWire's transmitter prediction has no histamine class:
    its 8,325 R1-6 cells come back as acetylcholine and its 1,340 R7 as
    glutamate. MaleCNS labels all 3,145 of its photoreceptors histamine.

    That is not a small discrepancy in a corner of the data. In FlyWire, R1-6
    supplies **48% of L1's input** -- the lamina cell every visual model starts
    from -- so a sign taken from the transmitter label inverts the first synapse
    of the visual system, and inverts the dominant one.

    Returns counts by transmitter plus ``ok``, which is False when any
    photoreceptor is labelled as something other than histamine.
    """

    types = neurons["type"].astype(str)
    hit = types.str.match(PHOTORECEPTOR_PATTERN, na=False)
    if not hit.any():
        return {"n": 0, "ok": True, "by_nt": {}, "wrong": 0}
    nts = neurons.loc[hit, "nt"].astype(str)
    by = nts.value_counts().to_dict()
    wrong = int((nts != PHOTORECEPTOR_NT).sum())
    return {
        "n": int(hit.sum()),
        "ok": wrong == 0,
        "by_nt": by,
        "wrong": wrong,
        "note": (
            ""
            if wrong == 0
            else (
                f"{wrong:,} photoreceptors are not labelled {PHOTORECEPTOR_NT}; "
                "signs derived from their transmitter will be inverted"
            )
        ),
    }
