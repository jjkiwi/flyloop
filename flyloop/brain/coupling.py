"""Where the learned system meets the reflex.

The reflex works: an object in the visual field reaches DNa02 through
LC4/LPLC2, and the left-right difference steers a physical fly (Run 10,
+37.7 deg against +3.0 deg for a rewired control). The learned system also
works, on its own terms: pairing an odour with dopamine depresses KC->MBON
synapses and shifts the MBON balance (Run 9).

They are not the same circuit, and they cannot be made into one, because this
connectome says so:

* **LC4 + LPLC2 supply 0.0000% of Kenyon cell input.** A seen object cannot be
  learned through the mushroom body. Whatever is learned has to be an odour.
* **MBONs supply 0.53% of DNa02's input**, and 0.43% of MDN's, and nothing at
  all to DNa01 or DNp09. So there is exactly one junction where a learned
  valence can reach steering, and this module is built on it.

**The published anatomy agrees, in a way we did not arrange.** Li et al.
(*eLife* 2020;9:e62576) report that DNa02 receives direct mushroom body input
from MBON32 and MBON31. Measured here without reference to that paper, those
two types supply 0.279% and 0.219% of DNa02's input -- 0.498% of the 0.526%
total, so **95% of all MBON input to DNa02 comes from exactly the two cell types
the literature names**.

**And one place we are knowingly short.** The same paper describes the *main*
route from mushroom body to steering as indirect, through DNa03. This
connectome agrees that the indirect leg is the stronger one: MBON -> DNa03 is
1.96%, nearly four times the direct MBON -> DNa02 of 0.53%, with DNa03 -> DNa02
at 1.16%. The coupling here uses only the direct connection, so ``share`` is a
**lower bound** on the true influence, and every "how much gain would it take"
figure derived from it is correspondingly an upper bound. Routing through DNa03
would cut those figures by a factor of a few -- not by the orders of magnitude
that would change the conclusion.

So the hybrid is the arrangement the animal actually has: the odour says
*whether* to approach, the sight says *which way*. The mushroom body does not
generate a steering command; it biases one that already exists.

**The gain is the honest part.** At the measured input share a learned valence
moves steering by about half a percent -- real, and invisible. ``gain`` scales
that, and it defaults to 1.0, meaning the anatomy as measured. Turning it up
makes the effect visible and makes the model less like the animal, in a way the
number reports rather than hides. Anything that displays this must display the
gain next to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..connectome.schema import Connectome

#: The descending neurons a learned valence can actually reach, with the share
#: of their input that MBONs supply. Measured in :func:`measured_share`, not
#: assumed -- DNa01 and DNp09 receive nothing from MBONs at all.
STEERING_OUTPUT = "DNa02"


def measured_share(c: Connectome, target: str = STEERING_OUTPUT, *, hops: int = 1) -> float:
    """Share of ``target``'s input that traces back to MBONs within ``hops``.

    With ``hops=1`` this is the direct connection and nothing else: the fraction
    of the target's synaptic input that MBONs supply. That is the number
    ``gain=1.0`` has meant since this module was written.

    With ``hops>1`` it follows the input backwards. Input proportions make this
    well defined: the column of the matrix for one neuron sums to 1, so it is a
    distribution over where that neuron's input came from, and applying the
    matrix again pushes that distribution one step further back. The share
    landing on MBONs after *k* steps is therefore a genuine fraction, not a
    product of two ratios that happen to be small.

    The distinction matters because the indirect routes are the larger ones.
    On MaleCNS the steps run 0.521%, 0.631%, 0.546%, 0.444% -- the two-step
    share exceeds the direct one -- and the first four together come to 2.14%,
    **4.1x the direct connection alone**.

    (Total ancestry mass falls slightly with each step, from 0.99 to 0.92 over
    four, because some input originates in cells that have no inputs of their
    own. The shares are of the surviving mass and are marginally optimistic in
    the same proportion.)
    """
    types = c.neurons["type"].astype(str)
    mbon = np.flatnonzero(types.str.startswith("MBON").to_numpy())
    tgt = np.flatnonzero((types == target).to_numpy())
    if len(mbon) == 0 or len(tgt) == 0:
        return 0.0
    P = abs(c.W.tocsc()).astype(np.float64)
    w = np.asarray(P[:, tgt].todense()).mean(axis=1)
    is_mbon = np.zeros(c.n, dtype=bool)
    is_mbon[mbon] = True
    total = float(w[is_mbon].sum())
    if hops > 1:
        Pr = P.tocsr()
        for _ in range(hops - 1):
            w = Pr @ w
            total += float(w[is_mbon].sum())
    return total


@dataclass
class LearnedBias:
    """A learned odour valence, biasing the visual steering reflex.

    The readout is the MBON ensemble of Run 9 -- each cell weighted by its net
    signed influence on the descending neurons that drive approach minus those
    that drive withdrawal -- because Run 8 established that averaging the 97
    MBONs throws the signal away.
    """

    rows: np.ndarray  # MBON neuron rows
    valence: np.ndarray  # per-MBON approach-minus-withdraw weight
    share: float  # measured MBON -> DNa02 input share
    gain: float = 1.0
    #: Readout the same odour produced *before* any training, subtracted so the
    #: bias carries what was learned rather than what the odour innately does.
    #: Set per odour by whoever runs the training; 0.0 means "not measured",
    #: which makes the bias the raw readout and is almost never what you want.
    baseline: float = 0.0
    #: Readout scale, so a fully active ensemble gives a readout near 1.
    _norm: float = 1.0

    @property
    def strength(self) -> float:
        """How much of the steering command the learned signal can move."""
        return self.gain * self.share

    def readout(self, activations: np.ndarray) -> float:
        """Valence-weighted MBON activity, roughly in [-1, 1].

        ``activations`` is the full per-neuron vector, indexed by connectome
        row, exactly as the rate model returns it.
        """
        a = np.asarray(activations)
        if a.ndim == 2:  # (neurons, hops) -> peak over hops, as elsewhere
            a = a.max(axis=1)
        return float(np.dot(a[self.rows], self.valence) / self._norm)

    def learned_component(self, activations: np.ndarray) -> float:
        """The part of the readout that training put there.

        Measured on MaleCNS, an odour's raw readout is dominated by which odour
        it is, not by what happened to it: the two odours used here sit at
        -0.00025 and +0.00434 before any training at all, and eight pairing
        trials move each by a few percent of itself. Amplifying the raw readout
        therefore amplifies innate odour identity and calls it learning, which
        is the error this subtraction exists to prevent.

        It also happens to be the better model. In the fly the innate valence of
        an odour is carried by the lateral horn; the mushroom body carries the
        learned modification to it. Subtracting the untrained baseline leaves
        approximately the mushroom body's own contribution.
        """
        return self.readout(activations) - self.baseline

    def modulation(self, activations: np.ndarray) -> float:
        """Factor applied to the approach drive: 1.0 means the reflex, unchanged.

        Positive learned valence pushes the fly toward what it is looking at,
        negative pulls it off. Clipped at zero so a learned aversion can cancel
        approach but never invert the steering -- reversing which way the fly
        turns is the visual pathway's job, not the mushroom body's.
        """
        return float(max(0.0, 1.0 + self.strength * self.learned_component(activations)))

    def report(self) -> str:
        base = "unmeasured" if self.baseline == 0.0 else f"{self.baseline:+.6f}"
        return (
            f"learned bias: {len(self.rows)} MBONs, "
            f"MBON->{STEERING_OUTPUT} input share {self.share:.3%}, "
            f"gain {self.gain:g} -> moves steering by up to {self.strength:.2%}; "
            f"untrained baseline {base}"
        )


def learned_bias(
    c: Connectome,
    *,
    gain: float = 1.0,
    hops: int = 4,
    decay: float = 0.5,
    coupling_hops: int = 1,
) -> LearnedBias:
    """Build the coupling from the connectome, measuring its own strength.

    ``coupling_hops`` is how far back the MBON share of the steering neuron's
    input is traced. It defaults to 1, the direct connection, because that is
    what every number recorded before this option existed used. Set it to 4 to
    include the indirect routes, which are collectively 4.1x larger.
    """
    from ..experiments.ensemble import mbon_ensemble

    ens = mbon_ensemble(c, hops=hops, decay=decay)
    valence = np.asarray(ens.valence, dtype=np.float64)
    # Normalise by the total weight available, so the readout is a fraction of
    # the ensemble rather than a number whose scale depends on hop count.
    norm = float(np.abs(valence).sum()) or 1.0
    return LearnedBias(
        rows=np.asarray(ens.rows),
        valence=valence,
        share=measured_share(c, hops=coupling_hops),
        gain=float(gain),
        _norm=norm,
    )
