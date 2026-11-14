"""The rotor frame: x along the thrust (forward), z up, y = z cross x; CT, CQ, CP, J, EFFICIENCY.

QPROP's captured stdout is mapped into it. Thrust forward gives CT > 0; reversing the rotation
(a negative shaft torque, as QPROP prints it) reverses CQ and nothing else.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cfdbender.core.neutral import ROTOR, ROTOR_NAMES, rotor_values
from cfdbender.solvers.qprop import _output

CAPTURED = Path(__file__).resolve().parents[1] / "data" / "qprop_stdout_real.txt"
RHO, RADIUS = 1.2, 0.15


def _row():
    return _output.parse_table(CAPTURED.read_text(encoding="utf-8", errors="replace"))


def _native(row):
    return _output.coefficients(row, rho=RHO, radius=RADIUS)


def test_the_rotor_columns_are_the_documented_set_and_qprop_maps_them():
    assert ROTOR == ("CT", "CQ", "CP", "CN", "CY", "J", "EFFICIENCY")
    values = rotor_values("qprop", _native(_row()))
    assert set(values) == {"CT", "CQ", "CP", "J", "EFFICIENCY"}, "CN and CY are NA for QPROP"
    assert rotor_values("su2", {"CT": 1.0}) == {} and "su2" not in ROTOR_NAMES


def test_normalisations_follow_the_stated_definitions():
    row = _row()
    values = rotor_values("qprop", _native(row))
    n, d = row["rpm"] / 60.0, 2.0 * RADIUS
    assert values["CT"] == pytest.approx(row["T(N)"] / (RHO * n**2 * d**4))
    assert values["CQ"] == pytest.approx(row["Q(N-m)"] / (RHO * n**2 * d**5))
    assert values["CP"] == pytest.approx(row["Pshaft(W)"] / (RHO * n**3 * d**5))
    assert values["J"] == pytest.approx(row["V(m/s)"] / (n * d))
    assert values["EFFICIENCY"] == row["effprop"]


def test_cp_is_two_pi_cq_for_the_positive_rotation_sense():
    row = dict(_row())
    row["Pshaft(W)"] = 2.0 * math.pi * (row["rpm"] / 60.0) * row["Q(N-m)"]
    values = rotor_values("qprop", _native(row))
    assert values["CP"] == pytest.approx(2.0 * math.pi * values["CQ"], rel=1e-9)


def test_thrust_forward_is_a_positive_ct():
    row = dict(_row())
    assert row["T(N)"] > 0 and rotor_values("qprop", _native(row))["CT"] > 0
    row["T(N)"] = -abs(row["T(N)"])
    assert rotor_values("qprop", _native(row))["CT"] < 0, "thrust aft is negative"


def test_reversed_rotation_flips_cq_alone():
    row = dict(_row())
    forward = rotor_values("qprop", _native(row))
    row["Q(N-m)"] = -row["Q(N-m)"]
    reversed_ = rotor_values("qprop", _native(row))
    assert reversed_["CQ"] == pytest.approx(-forward["CQ"])
    for name in ("CT", "CP", "J", "EFFICIENCY"):
        assert reversed_[name] == forward[name]
