"""How straight does this fly steer when the object is dead ahead?

Run 10 measured fixation with mirror pairs -- the same body seed walking once at
a target on its right and once at its mirror image -- and justified that by the
gait's own drift. There is a second reason, and it is in the brain rather than
the legs.

**An object centred in front produces a turn command of +0.0117**, 14% of the
strongest turn anywhere in the tuning curve, to the fly's right. The stimulus is
not to blame: at bearing 0 the projection lights exactly 156 columns in each eye.
The asymmetry is downstream of the retina, in the wiring between the eyes and
DNa02.

**It is not a general lopsidedness in the data.** MaleCNS is symmetric in
aggregate -- 73,467 neurons on the left against 73,638 on the right, and across
373 cell types with 20 or more cells the median left-right asymmetry is exactly
zero. It is this pathway that is uneven: LC4 has 71 cells on the left and 55 on
the right, a 13% imbalance, and it is LC4 that carries object position to the
steering neuron.

So any measurement this project makes from a single side inherits a standing
rightward push worth 14% of full scale, and at the closed loop's turn gain of
3.0 that is a commanded turn of +0.035 held for the whole episode. Mirror pairs
cancel it exactly, which is why every behavioural number in Run 10 and after is
reported as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..brain.rate import population_index, rate_brain, steady_state
from ..connectome.schema import Connectome
from ..vision.hexproject import HexWorldView

#: Angular half-width of the probe object, matching the embodied experiments.
PROBE_HALF_WIDTH = 5.7


def tuning_curve(
    c: Connectome,
    *,
    bearings: tuple[float, ...] | None = None,
    hops: int = 5,
    half_width: float = PROBE_HALF_WIDTH,
    view: HexWorldView | None = None,
) -> pd.DataFrame:
    """Descending turn command as a function of where the object is.

    Bearing is positive to the fly's right. The returned ``turn`` column is
    ``DNa02_R - DNa02_L``, the same quantity the closed loop steers on, so this
    is the open-loop shape of the controller the whole project rests on.
    """
    view = view or HexWorldView(c)
    bearings = bearings or tuple(float(b) for b in range(-90, 91, 10))
    record = {}
    for side in ("L", "R"):
        record.update(population_index(c, ("DNa02", "LC4"), side=side))
    brain = rate_brain(c, view.sensory, num_layers=hops)

    sides = c.neurons["side"].to_numpy()[view.sensory]
    rows = []
    for b in bearings:
        pattern = view.pattern([(b, half_width)])
        lit = pattern > 0
        peak = brain.run(steady_state(pattern, hops), record=record).peak()
        rows.append(
            {
                "bearing": b,
                "turn": float(peak.get("DNa02_R", 0.0) - peak.get("DNa02_L", 0.0)),
                "DNa02_L": float(peak.get("DNa02_L", 0.0)),
                "DNa02_R": float(peak.get("DNa02_R", 0.0)),
                "LC4_diff": float(peak.get("LC4_R", 0.0) - peak.get("LC4_L", 0.0)),
                # Column counts, so a reader can tell a wiring asymmetry from a
                # projection that simply painted more of one eye.
                "lit_L": int((lit & (sides == "L")).sum()),
                "lit_R": int((lit & (sides == "R")).sum()),
            }
        )
    return pd.DataFrame(rows)


def lateral_bias(curve: pd.DataFrame, *, within: float = 60.0) -> dict[str, float]:
    """Summarise how far the controller departs from left-right symmetry.

    ``antisymmetry_residual`` is the mean of ``turn(+b) + turn(-b)`` over the
    mirrored pairs, which is zero for a perfectly balanced controller and
    positive when the fly favours its right. ``centre_bias`` is the command an
    object straight ahead produces, in units of the curve's own peak, which is
    the number that matters for a single-sided experiment.
    """
    by = {round(float(r.bearing), 6): float(r.turn) for r in curve.itertuples()}
    pairs = [(b, by[b] + by[-b]) for b in sorted(by) if b > 0 and -b in by]
    peak = float(np.abs(curve["turn"]).max()) or 1.0
    centre = by.get(0.0, float("nan"))
    m = np.abs(curve["bearing"]) <= within
    slope = float("nan")
    if m.sum() >= 2:
        A = np.vstack([curve["bearing"][m], np.ones(int(m.sum()))]).T
        fit, *_ = np.linalg.lstsq(A, curve["turn"][m].to_numpy(), rcond=None)
        slope = float(fit[0])
        pred = A @ fit
        resid = curve["turn"][m].to_numpy() - pred
        var = ((curve["turn"][m] - curve["turn"][m].mean()) ** 2).sum()
        r2 = float(1 - (resid**2).sum() / var) if var > 0 else float("nan")
    else:
        r2 = float("nan")
    return {
        "antisymmetry_residual": float(np.mean([d for _, d in pairs])) if pairs else 0.0,
        "centre_bias": centre,
        "centre_bias_fraction": abs(centre) / peak,
        "peak_turn": peak,
        "linear_slope_per_deg": slope,
        "linear_r2": r2,
    }


def report(curve: pd.DataFrame) -> str:
    b = lateral_bias(curve)
    return (
        f"turn command peaks at {b['peak_turn']:.5f}; "
        f"an object dead ahead commands {b['centre_bias']:+.5f} "
        f"({b['centre_bias_fraction']:.0%} of peak, to the fly's right).\n"
        f"  linear over |bearing| <= 60 deg: slope {b['linear_slope_per_deg']:.3e}/deg, "
        f"R^2 {b['linear_r2']:.3f}\n"
        f"  antisymmetry residual {b['antisymmetry_residual']:+.5f} (0 would be balanced)"
    )
