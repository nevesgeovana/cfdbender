"""P058-3: the rotor coefficients (RCOEF) in the rotor frame and per frame, with the NA rules.

The hand case: a tractor of diameter 1 m turning at 600 rev/min (n = 10 rev/s) in air of
density 1 kg/m^3 at 10 m/s, thrust 100 N forward and torque 5 N m about the forward shaft.
Then J = 1, CT = 100 / (1 * 100 * 1) = 1, CQ = 5 / 100 = 0.05, CP = 2 pi CQ, P = 2 pi n Q
in kW, ETA = J CT / CP, q = 50 Pa and A = pi / 4.
"""

from __future__ import annotations

import math

import pytest

from cfdbender.core.axes import frame_rotations, rotor_frame
from cfdbender.core.neutral import rotor_loads
from cfdbender.core.rotor_coefficients import ROTOR_FRAME_COLUMNS, rotor_frame_coefficients

NAN = math.nan


def rcoef(force, moment, *, axis=(-1.0, 0.0, 0.0), rpm=600.0, vinf=10.0, alpha=0.0):
    frame = rotor_frame(axis)
    return rotor_frame_coefficients(
        force_n=force,
        moment_hub_nm=moment,
        frame=frame,
        rotations=frame_rotations(alpha, 0.0, [frame]),
        rpm=rpm,
        diameter_m=1.0,
        density_kg_m3=1.0,
        vinf_m_s=vinf,
        sref_m2=2.0,
        cref_m=0.5,
        bref_m=4.0,
    )


def test_the_hand_case_in_the_rotor_frame():
    row = rcoef((-100.0, 0.0, 0.0), (-5.0, 0.0, 0.0))
    cp = 2.0 * math.pi * 0.05
    assert row["T"] == pytest.approx(100.0) and row["Q"] == pytest.approx(5.0)
    assert row["P"] == pytest.approx(2.0 * math.pi * 10.0 * 5.0 / 1000.0)
    assert (row["CT"], row["CQ"], row["CP"]) == pytest.approx((1.0, 0.05, cp))
    assert row["ETA"] == pytest.approx(1.0 / cp)
    q_area = 50.0 * math.pi / 4.0
    assert row["TC"] == pytest.approx(100.0 / q_area)
    assert row["QC"] == pytest.approx(5.0 / q_area)
    assert row["TCA"] == pytest.approx(100.0 / (50.0 * 2.0))
    assert row["QCA"] == pytest.approx(5.0 / (50.0 * 2.0 * 4.0))
    assert row["CTR0"] == row["CT"] and row["CQR0"] == row["CQ"] and row["ETAR0"] == row["ETA"]
    assert row["CTB"] == row["CT"] and row["CTW"] == pytest.approx(row["CT"])
    names = [name for name, *_ in ROTOR_FRAME_COLUMNS]
    assert list(row)[: len(names)] == names
    assert list(row)[len(names) :] == [
        f"{stem}{f}" for f in ("W", "S", "B", "R0") for stem in ("CT", "CQ", "ETA")
    ]


def test_in_plane_loads_follow_the_rotor_axes():
    # A force up (geometry +z) is Z positive; a moment about geometry +z is N = -M_E,z.
    row = rcoef((0.0, 3.0, 2.0), (0.0, 1.0, 4.0))
    assert row["CFZ"] == pytest.approx(2.0 / 100.0) and row["CFY"] == pytest.approx(3.0 / 100.0)
    assert row["CMZ"] == pytest.approx(-4.0 / 100.0) and row["CMY"] == pytest.approx(1.0 / 100.0)
    assert row["ZC"] == pytest.approx(2.0 / (50.0 * math.pi / 4.0))
    assert row["MCA"] == pytest.approx(1.0 / (50.0 * 2.0 * 0.5))


def test_a_rotor_declared_aft_counts_thrust_forward():
    # The same physical rotor, declared along +x (aft) and turning -600 about that axis.
    aft = rcoef((-100.0, 0.0, 0.0), (-5.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), rpm=-600.0)
    forward = rcoef((-100.0, 0.0, 0.0), (-5.0, 0.0, 0.0))
    for name in ("T", "Q", "P", "CT", "CQ", "CP", "ETA"):
        assert aft[name] == pytest.approx(forward[name])


def test_a_static_point_states_no_efficiency_and_no_loading():
    row = rcoef((-100.0, 0.0, 0.0), (-5.0, 0.0, 0.0), vinf=0.0)
    assert row["CT"] == pytest.approx(1.0) and row["CTW"] == pytest.approx(1.0)
    for name in ("ETA", "ETAW", "ETAB", "ETAR0", "TC", "QC", "TCA", "MCA"):
        assert row[name] == "NA", name


def test_a_rotor_at_rest_or_of_unknown_speed():
    still = rcoef((-100.0, 0.0, 0.0), (-5.0, 0.0, 0.0), rpm=0.0)
    assert still["T"] == pytest.approx(100.0) and still["P"] == 0.0
    assert still["CT"] == still["CQ"] == still["CP"] == still["ETA"] == "NA"
    assert still["TC"] == pytest.approx(100.0 / (50.0 * math.pi / 4.0))
    unknown = rcoef((-100.0, 0.0, 0.0), (-5.0, 0.0, 0.0), rpm=None)
    assert unknown["P"] == "NA" and unknown["CT"] == "NA" and unknown["Q"] == pytest.approx(5.0)


def test_qprop_thrust_and_torque_reproduce_its_own_coefficients():
    thrust, torque, rpm, diameter, rho, speed = 12.0, 0.4, 6000.0, 0.4, 1.2, 20.0
    force, moment = rotor_loads("qprop", {"T": thrust, "Q": torque})
    frame = rotor_frame((-1.0, 0.0, 0.0))
    row = rotor_frame_coefficients(
        force_n=force,
        moment_hub_nm=moment,
        frame=frame,
        rotations=frame_rotations(0.0, 0.0, [frame]),
        rpm=rpm,
        diameter_m=diameter,
        density_kg_m3=rho,
        vinf_m_s=speed,
        sref_m2=None,
        cref_m=None,
        bref_m=None,
    )
    n = rpm / 60.0
    assert row["CT"] == pytest.approx(thrust / (rho * n**2 * diameter**4))
    assert row["CQ"] == pytest.approx(torque / (rho * n**2 * diameter**5))
    assert row["P"] == pytest.approx(2.0 * math.pi * n * torque / 1000.0)
    assert row["ETA"] == pytest.approx(speed / (n * diameter) * row["CT"] / row["CP"])
    assert row["CFZ"] == row["CMY"] == row["TCA"] == "NA"
    assert rotor_loads("su2", {"T": 1.0, "Q": 1.0}) is None
    assert rotor_loads("qprop", {"CT": 1.0}) is None
