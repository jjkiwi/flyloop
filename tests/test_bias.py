"""The steering controller's open-loop shape, and its standing lean.

The statistics are checked against curves whose answer is known by
construction, because a summary that cannot recognise a perfectly balanced
controller is worse than no summary.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.connectome.schema import Connectome
from flyloop.experiments.bias import lateral_bias, report, tuning_curve


def _curve(turns: dict[float, float]) -> pd.DataFrame:
    return pd.DataFrame({"bearing": list(turns), "turn": list(turns.values())})


# ------------------------------------------------------------ the statistics


def test_a_balanced_controller_shows_no_residual():
    curve = _curve({b: 0.001 * b for b in range(-60, 61, 10)})
    out = lateral_bias(curve)
    assert out["antisymmetry_residual"] == pytest.approx(0.0, abs=1e-12)
    assert out["centre_bias"] == pytest.approx(0.0, abs=1e-12)


def test_a_constant_offset_is_reported_as_one():
    """Every bearing pushed right by the same amount is exactly the failure mode."""
    curve = _curve({b: 0.001 * b + 0.02 for b in range(-60, 61, 10)})
    out = lateral_bias(curve)
    assert out["antisymmetry_residual"] == pytest.approx(0.04, rel=1e-9)
    assert out["centre_bias"] == pytest.approx(0.02, rel=1e-9)


def test_centre_bias_is_reported_against_the_curve_s_own_peak():
    curve = _curve({-30.0: -0.1, 0.0: 0.02, 30.0: 0.2})
    out = lateral_bias(curve)
    assert out["peak_turn"] == pytest.approx(0.2)
    assert out["centre_bias_fraction"] == pytest.approx(0.1)


def test_a_straight_line_fits_perfectly():
    curve = _curve({b: 0.0007 * b for b in range(-60, 61, 5)})
    out = lateral_bias(curve)
    assert out["linear_slope_per_deg"] == pytest.approx(0.0007, rel=1e-9)
    assert out["linear_r2"] == pytest.approx(1.0, abs=1e-9)


def test_the_fit_ignores_bearings_outside_the_window():
    """The tuning curve turns over past 30 deg; fitting that would be wrong."""
    turns = {b: 0.001 * b for b in range(-60, 61, 10)}
    turns[90.0], turns[-90.0] = 0.0, 0.0  # far flanks fall back toward zero
    out = lateral_bias(_curve(turns), within=60.0)
    assert out["linear_r2"] == pytest.approx(1.0, abs=1e-9)


def test_an_unpaired_bearing_is_skipped_rather_than_guessed():
    out = lateral_bias(_curve({0.0: 0.0, 30.0: 0.2, -30.0: -0.2, 45.0: 0.3}))
    assert out["antisymmetry_residual"] == pytest.approx(0.0, abs=1e-12)


def test_report_names_the_side(monkeypatch):
    text = report(_curve({-30.0: -0.1, 0.0: 0.02, 30.0: 0.2}))
    assert "right" in text
    assert "10%" in text


# -------------------------------------------------------------- the sweep


def _eye_connectome(radius: int = 3) -> Connectome:
    rows, nid = [], 0
    for side in ("L", "R"):
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                rows.append(
                    {
                        "id": nid,
                        "type": "L1",
                        "nt": "acetylcholine",
                        "side": side,
                        "hex1": x,
                        "hex2": y,
                    }
                )
                nid += 1
    for t in ("DNa02", "LC4"):
        for side in ("L", "R"):
            rows.append(
                {
                    "id": nid,
                    "type": t,
                    "nt": "acetylcholine",
                    "side": side,
                    "hex1": np.nan,
                    "hex2": np.nan,
                }
            )
            nid += 1
    neurons = pd.DataFrame(rows)
    n = len(neurons)
    return Connectome(
        neurons, sp.csr_matrix((n, n)), name="eyes", meta={"matrix_kind": "inprop"}
    )


def test_the_sweep_records_how_much_of_each_eye_was_lit():
    """Without these a wiring asymmetry cannot be told from a lopsided stimulus."""
    pytest.importorskip("connectome_interpreter", reason="optional 'data' extra")
    curve = tuning_curve(_eye_connectome(), bearings=(-30.0, 0.0, 30.0), hops=2)
    assert list(curve["bearing"]) == [-30.0, 0.0, 30.0]
    assert {"lit_L", "lit_R", "turn", "LC4_diff"} <= set(curve.columns)
    centre = curve[curve.bearing == 0].iloc[0]
    assert centre.lit_L == centre.lit_R, "a centred object must light both eyes equally"
    left, right = curve.iloc[0], curve.iloc[2]
    assert left.lit_L == right.lit_R and left.lit_R == right.lit_L
