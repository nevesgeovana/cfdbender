"""The neutral CL, CD and CM are in one standard frame; each solver's native axes map into it.

Stand-in native values in each solver's own axes are known, and the neutral values asserted
are computed independently here. Frame: x aft, y right, z up; CD along the freestream, CL
normal to it in the plane of symmetry, CM nose-up positive.
"""

from __future__ import annotations

import math

import pytest

from cfdbender.core.neutral import neutral_values, wind_directions

ALPHA = 10.0
A = math.radians(ALPHA)


def test_the_wind_directions_are_the_textbook_ones():
    drag, lift = wind_directions(ALPHA, 0.0)
    assert drag == pytest.approx((math.cos(A), 0.0, math.sin(A)))
    assert lift == pytest.approx((-math.sin(A), 0.0, math.cos(A)))
    drag, lift = wind_directions(4.0, 2.0)
    assert sum(d * ell for d, ell in zip(drag, lift, strict=True)) == pytest.approx(0.0)
    assert sum(d * d for d in drag) == pytest.approx(1.0)


def test_su2_wind_coefficients_and_the_three_dimensional_moment_pass_unchanged():
    native = {"CL": 0.8, "CD": 0.05, "CMy": -0.1, "CMz": 0.0}
    out = neutral_values("su2", native, dimensions=3, condition={"ALPHA": ALPHA})
    assert out == {"CL": 0.8, "CD": 0.05, "CM": -0.1}


def test_su2_planar_moment_is_minus_cmz_so_that_nose_up_is_positive():
    # CMz is the counterclockwise moment of the x-y plane; nose-up (the nose at -x turns to +y)
    # is clockwise, so the standard pitching moment is its negative.
    native = {"CL": 0.8, "CD": 0.05, "CMy": 0.0, "CMz": -0.0258}
    out = neutral_values("su2", native, dimensions=2, condition={"ALPHA": ALPHA})
    assert out["CM"] == pytest.approx(0.0258)
    assert neutral_values("su2", {"CMz": 0.3}, dimensions=2)["CM"] == pytest.approx(-0.3)


def test_openfoam_cl_cd_and_cmpitch_about_minus_z_are_already_standard():
    # liftDir (-sin a, cos a, 0), dragDir (cos a, sin a, 0), pitchAxis (0 0 -1): positive
    # CmPitch is nose-up for the flow along +x with y up.
    native = {"Cl": 1.1, "Cd": 0.02, "CmPitch": 0.07}
    out = neutral_values("openfoam", native, dimensions=2, condition={"ALPHA": ALPHA})
    assert out == {"CL": 1.1, "CD": 0.02, "CM": 0.07}


def test_pyfr_rotates_nothing_further_and_reports_no_moment():
    # Cl and Cd are already turned by ALPHA from the raw force components.
    out = neutral_values("pyfr", {"Cl": 0.5, "Cd": 0.01}, condition={"ALPHA": ALPHA})
    assert out == {"CL": 0.5, "CD": 0.01}


def test_xfoil_and_mock_keep_their_values_and_qprop_writes_nothing():
    assert neutral_values("xfoil", {"CL": 0.6, "CD": 0.01, "CM": -0.05}) == {
        "CL": 0.6,
        "CD": 0.01,
        "CM": -0.05,
    }
    assert neutral_values("mock", {"CZ": 1.0, "CX": 0.1, "Cm": 0.2}) == {
        "CL": 1.0,
        "CD": 0.1,
        "CM": 0.2,
    }
    assert neutral_values("qprop", {"CT": 1.0, "CP": 2.0}) == {}


def test_flightstream_body_axis_forces_are_rotated_into_the_wind_axes():
    body = {"CDB": 0.1, "CYB": 0.0, "CLB": 1.0, "CMB25": 0.05}
    out = neutral_values("flightstream", body, condition={"ALPHA": ALPHA, "BETA": 0.0})
    assert out["CL"] == pytest.approx(-0.1 * math.sin(A) + 1.0 * math.cos(A))
    assert out["CD"] == pytest.approx(0.1 * math.cos(A) + 1.0 * math.sin(A))
    assert out["CM"] == 0.05, "the pitching moment about y is not turned by ALPHA"
    # With no angle the body and wind axes coincide.
    flat = neutral_values("flightstream", body, condition={"ALPHA": 0.0})
    assert (flat["CL"], flat["CD"]) == pytest.approx((1.0, 0.1))


def test_flightstream_sideslip_moves_side_force_into_the_drag():
    body = {"CDB": 0.1, "CYB": 0.2, "CLB": 1.0, "CMB25": 0.0}
    out = neutral_values("flightstream", body, condition={"ALPHA": 0.0, "BETA": 5.0})
    b = math.radians(5.0)
    # The air moves toward -y at positive sideslip (frame W): a side force to the right
    # opposes the drag.
    assert out["CD"] == pytest.approx(0.1 * math.cos(b) - 0.2 * math.sin(b))
    assert out["CL"] == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", [2.0, 8.0])
def test_a_known_wind_force_is_recovered_with_positive_lift_at_positive_alpha(alpha):
    # Build the body-axis force from a lift of 1.0 and a drag of 0.1 in the wind axes.
    drag, lift = wind_directions(alpha, 0.0)
    force = [1.0 * ell + 0.1 * d for ell, d in zip(lift, drag, strict=True)]
    body = {"CDB": force[0], "CYB": force[1], "CLB": force[2], "CMB25": 0.0}
    out = neutral_values("flightstream", body, condition={"ALPHA": alpha})
    assert out["CL"] == pytest.approx(1.0) and out["CL"] > 0.0
    assert out["CD"] == pytest.approx(0.1)
