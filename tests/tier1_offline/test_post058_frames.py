"""P058-1: the polar frames W, S, B and R<id>, and the neutral CL and CD agree with W.

Every expected value is derived here from the stated rules: the air moves along
(cos a cos b, -cos a sin b, sin a) in the geometry frame (x aft, y right, z up), the
stability lift direction is (-sin a_s, 0, cos a_s) with a_s = atan2(sin a, cos a cos b), the
body frame is forward, right, down, and a rotor frame's x is its shaft turned so that
thrust is positive forward.
"""

from __future__ import annotations

import math

import pytest

from cfdbender.core.axes import (
    airframe_axis_coefficients,
    frame_names,
    frame_rotations,
    rotor_frame,
    turn,
)
from cfdbender.core.neutral import neutral_values, wind_directions

D5, D10 = math.radians(5.0), math.radians(10.0)


def test_a_tractor_declared_forward_flies_in_body_axes():
    frame = rotor_frame((-1.0, 0.0, 0.0))
    assert (frame.name, frame.pitch_deg, frame.toe_deg, frame.sense) == ("R0", 0.0, 0.0, 1)
    assert frame.declared_aft is False


def test_a_pitched_pusher_declared_aft_is_turned_forward_with_its_pitch():
    axis = (math.cos(D5), 0.0, -math.sin(D5))
    frame = rotor_frame(axis, 1)
    assert frame.name == "R1" and frame.sense == -1 and frame.declared_aft is True
    assert frame.pitch_deg == pytest.approx(5.0) and frame.toe_deg == pytest.approx(0.0)
    # x_R is the shaft turned forward: the declared axis lands on -x_R.
    along = turn(frame_rotations(0.0, 0.0, [frame])["R1"], axis)
    assert along == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)


def test_a_toed_rotor_and_a_vertical_lift_rotor():
    toed = rotor_frame((-math.cos(D10), math.sin(D10), 0.0))
    assert toed.toe_deg == pytest.approx(10.0) and toed.pitch_deg == pytest.approx(0.0)
    up = rotor_frame((0.0, 0.0, 1.0))
    assert (up.pitch_deg, up.toe_deg, up.sense) == (pytest.approx(90.0), 0.0, 1)
    down = rotor_frame((0.0, 0.0, -1.0))
    assert down.sense == -1 and down.pitch_deg == pytest.approx(90.0)
    with pytest.raises(ValueError, match="no direction"):
        rotor_frame((0.0, 0.0, 0.0))


def test_frames_are_written_w_s_b_then_rotors_by_id():
    assert frame_names((2, 0)) == ("W", "S", "B", "R0", "R2")
    turns = frame_rotations(4.0, 2.0, [rotor_frame((-1.0, 0.0, 0.0), 2)])
    assert list(turns) == ["W", "S", "B", "R2"]
    with pytest.raises(ValueError, match="one id"):
        frame_rotations(0.0, 0.0, [rotor_frame((-1, 0, 0)), rotor_frame((0, 0, 1))])


def test_airframe_coefficients_are_aft_up_and_right_in_every_frame():
    a, b = math.radians(4.0), math.radians(2.0)
    force, moment = (3.0, -2.0, 40.0), (1.0, -6.0, 0.5)
    q, sref, cref, bref = 2.0, 5.0, 0.5, 4.0
    rows = airframe_axis_coefficients(
        force,
        moment,
        frame_rotations(4.0, 2.0),
        q_ref_pa=q,
        sref_m2=sref,
        cref_m=cref,
        bref_m=bref,
    )
    stream = (math.cos(a) * math.cos(b), -math.cos(a) * math.sin(b), math.sin(a))
    a_s = math.atan2(math.sin(a), math.cos(a) * math.cos(b))
    lift = (-math.sin(a_s), 0.0, math.cos(a_s))

    def dot(u, v):
        return sum(x * y for x, y in zip(u, v, strict=True))

    assert rows["CDW"] == pytest.approx(dot(force, stream) / (q * sref))
    assert rows["CLW"] == pytest.approx(dot(force, lift) / (q * sref))
    assert rows["CLS"] == pytest.approx(rows["CLW"])
    # Body axes: aft, right and up of the geometry frame.
    assert (rows["CDB"], rows["CYB"], rows["CLB"]) == pytest.approx((0.3, -0.2, 4.0))
    # Moments right-hand about forward, right, down: roll -Mx, pitch My, yaw -Mz.
    assert rows["ClB"] == pytest.approx(-1.0 / (q * sref * bref))
    assert rows["CmB"] == pytest.approx(-6.0 / (q * sref * cref))
    assert rows["CnB"] == pytest.approx(-0.5 / (q * sref * bref))
    assert list(rows)[:6] == ["CDW", "CYW", "CLW", "ClW", "CmW", "CnW"]


def test_an_unknown_component_spoils_only_what_it_enters():
    nan = math.nan
    rows = airframe_axis_coefficients(
        (1.0, nan, 10.0),
        (nan, 2.0, nan),
        frame_rotations(3.0, 0.0),
        q_ref_pa=1.0,
        sref_m2=1.0,
        cref_m=1.0,
        bref_m=None,
    )
    assert isinstance(rows["CDW"], float) and isinstance(rows["CLW"], float)
    assert rows["CYW"] == rows["ClW"] == rows["CnW"] == "NA"
    assert rows["CmW"] == pytest.approx(2.0)


def test_the_neutral_cl_and_cd_are_the_wind_frame_ones():
    body = {"CDB": 0.03, "CYB": 0.2, "CLB": 0.9, "CMB25": 0.0}
    out = neutral_values("flightstream", body, condition={"ALPHA": 4.0, "BETA": 2.0})
    rows = airframe_axis_coefficients(
        (0.03, 0.2, 0.9),
        (0.0, 0.0, 0.0),
        frame_rotations(4.0, 2.0),
        q_ref_pa=1.0,
        sref_m2=1.0,
        cref_m=1.0,
        bref_m=1.0,
    )
    assert out["CD"] == pytest.approx(rows["CDW"]) and out["CL"] == pytest.approx(rows["CLW"])
    drag, lift = wind_directions(10.0, 0.0)
    assert drag == pytest.approx((math.cos(D10), 0.0, math.sin(D10)))
    assert lift == pytest.approx((-math.sin(D10), 0.0, math.cos(D10)))
