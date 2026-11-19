"""The rotor axis reads pyflightstream's two forms: a letter, or { shaft, vector_wrt_global }."""

from __future__ import annotations

import math

import pytest
from test_runtypes_b1_workflows import deck, numbers, workspace

from cfdbender.core.plan import PlanError
from cfdbender.core.rotor import RotorLoads, rotor_block, rotor_row

TILT = [0.996, 0.0, 0.087]
UNIT = tuple(c / math.sqrt(sum(v * v for v in TILT)) for c in TILT)


def block(axis):
    return rotor_block(
        {
            "R": {
                "kind": "rotor",
                "axis": axis,
                "rpm_sign": 1,
                "diameter_m": 1.0,
                "families_blades": ["B1", "B2"],
                "blade1": {"azimuth_deg": 0.0, "zero": "Z"},
            }
        }
    )


def test_an_installed_shaft_is_the_unit_vector_thrust_is_counted_along():
    tilted = block({"shaft": "X", "vector_wrt_global": TILT})
    assert tilted is not None and tilted.axis == pytest.approx(UNIT, rel=1e-12)
    plain = block("X")
    assert plain is not None and plain.axis == (1.0, 0.0, 0.0)
    force = (10.0, 0.0, 2.0)
    loads = RotorLoads(force, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 20.0, 1.2)
    row = rotor_row(tilted, loads, rps=10.0)
    assert row["T"] == pytest.approx(sum(f * a for f, a in zip(force, UNIT, strict=True)))


def test_the_rotation_sent_to_su2_follows_the_installed_shaft(tmp_path):
    rotor = """
[ROTOR]
kind = "rotor"
axis = { shaft = "X", vector_wrt_global = [0.996, 0.0, 0.087] }
rpm_sign = 1
diameter_m = 0.4064
families_blades = ["B1", "B2"]
blade1 = { azimuth_deg = 0.0, zero = "Z" }
"""
    workspace(tmp_path, workflow="qsteady_rotor", condition="MACH:0.0529, RPM:600", rotor=rotor)
    rate = numbers(deck(tmp_path)[1]["ROTATION_RATE"])
    assert rate == pytest.approx([2.0 * math.pi * 10.0 * c for c in UNIT], rel=1e-12)


@pytest.mark.parametrize(
    ("axis", "message"),
    [
        ({"shaft": "X", "vector_wrt_global": [-1.0, 0.0, 0.0]}, "180.00 degrees from shaft X"),
        ({"shaft": "X", "vector_wrt_global": [0.0, 0.0, 0.0]}, "zero length"),
        ([1.0, 0.0, 0.0], 'write axis = { shaft = "X", vector_wrt_global'),
        ("-X", "is not an axis"),
        ({"shaft": "X"}, "states shaft and vector_wrt_global"),
    ],
)
def test_the_forms_pyflightstream_refuses_are_refused(axis, message):
    with pytest.raises(ValueError, match=message):
        block(axis)


def test_a_refusal_reaches_the_plan(tmp_path):
    rotor = """
[ROTOR]
kind = "rotor"
axis = "-X"
diameter_m = 0.4
families_blades = ["B1"]
blade1 = { zero = "Z" }
"""
    workspace(tmp_path, workflow="qsteady_rotor", condition="MACH:0.0529, RPM:600", rotor=rotor)
    with pytest.raises(PlanError, match="is not an axis"):
        deck(tmp_path)
