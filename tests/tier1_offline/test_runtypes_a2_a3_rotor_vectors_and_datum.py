"""The adapter supplies force and moment vectors; the reference's blade one datum is read."""

from __future__ import annotations

import inspect
import math

import pytest
from test_su2_advance_ratio import REFERENCE_BLOCK, build

import cfdbender.core.rotor as rotor
from cfdbender.core.plan import PlanError
from cfdbender.core.rotor import RotorLoads, rotor_block, rotor_row
from cfdbender.core.unsteady_clock import BladeDatum
from cfdbender.solvers.su2._rotor import rotor_loads
from cfdbender.workspace.planning import plan_workspace

BLOCK = {
    "kind": "rotor",
    "axis": "X",
    "rpm_sign": -1,
    "diameter_m": 0.5,
    "families_blades": ["B1", "B2", "B3", "B4"],
    "blade1": {"azimuth_deg": 30.0, "zero": "-Y"},
}


def test_the_agnostic_rotor_module_takes_vectors_and_converts_no_coefficient():
    source = inspect.getsource(rotor)
    for word in ("dynamic_pressure", "force_coefficients", "rotor_quantities"):
        assert word not in source
    assert not hasattr(rotor, "rotor_quantities")


def test_the_su2_adapter_turns_its_coefficients_into_newtons_about_its_moment_point():
    deck = {
        "MACH_NUMBER": 0.1,
        "FLUID_MODEL": "IDEAL_GAS",
        "GAS_CONSTANT": 287.058,
        "GAMMA_VALUE": 1.4,
        "FREESTREAM_TEMPERATURE": 288.15,
        "FREESTREAM_PRESSURE": 101325.0,
        "MACH_MOTION": 0.2,
        "REF_AREA": 2.0,
        "REF_LENGTH": 0.5,
        "REF_ORIGIN_MOMENT_X": 0.1,
        "REF_ORIGIN_MOMENT_Y": 0.0,
        "REF_ORIGIN_MOMENT_Z": -0.2,
    }
    loads = rotor_loads(deck, (0.1, -0.2, 0.3), (0.01, 0.02, -0.03))
    sonic = math.sqrt(1.4 * 287.058 * 288.15)
    rho = 101325.0 / (287.058 * 288.15)
    q = 0.5 * rho * (0.2 * sonic) ** 2  # SU2 scales by MACH_MOTION
    assert loads is not None
    assert loads.force_n == pytest.approx((0.2 * q, -0.4 * q, 0.6 * q), rel=1e-12)
    assert loads.moment_nm == pytest.approx((0.01 * q, 0.02 * q, -0.03 * q), rel=1e-12)
    assert loads.about_m == (0.1, 0.0, -0.2)
    assert loads.speed_m_s == pytest.approx(0.1 * sonic, rel=1e-12)
    assert loads.density_kg_m3 == pytest.approx(rho, rel=1e-12)


def test_the_loads_round_trip_through_a_record():
    loads = RotorLoads((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (0.0, 0.1, 0.2), 20.0, None)
    assert RotorLoads.from_dict(loads.to_dict()) == loads


def test_the_row_comes_from_the_vectors_alone():
    block = rotor_block({"R": BLOCK})
    assert block is not None
    loads = RotorLoads((-10.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0), 20.0, 1.2)
    row = rotor_row(block, loads, rps=40.0)
    assert row["CT"] == pytest.approx(-10.0 / (1.2 * 1600.0 * 0.5**4), rel=1e-12)
    assert row["CP"] == pytest.approx(-2.0 * math.pi * 2.0 / (1.2 * 1600.0 * 0.5**5), rel=1e-12)


def test_blade_one_is_read_and_places_every_blade():
    block = rotor_block({"R": BLOCK})
    assert block is not None and block.blade1 == BladeDatum(30.0, "-Y")
    # rpm_sign -1: the blades turn the negative way about the shaft, 360/8 per step at 8 steps
    assert block.blade_azimuth_deg(1, step=0, steps_per_revolution=8.0) == 30.0
    assert block.blade_azimuth_deg(1, step=1, steps_per_revolution=8.0) == pytest.approx(345.0)
    assert block.blade_azimuth_deg(3, step=0, steps_per_revolution=8.0) == pytest.approx(210.0)
    with pytest.raises(ValueError, match="has 4 blades; blade 5 is not one"):
        block.blade_azimuth_deg(5, step=0, steps_per_revolution=8.0)


@pytest.mark.parametrize(
    ("blade1", "message"),
    [
        ('{ azimuth_deg = 0.0, zero = "X" }', "blade1.zero X equals shaft X"),
        ('{ azimuth_deg = 0.0, zero = "-x" }', "blade1.zero -X equals shaft X"),
        ('{ azimuth_deg = 0.0, zero = "W" }', "zero = 'W', which is not an axis"),
        ('{ azimuth_deg = "a", zero = "Y" }', "azimuth_deg = 'a', which is not a finite"),
        ('{ azimuth_deg = 0.0, zero = "Y", pitch = 1 }', "blade1 has unknown keys: pitch"),
    ],
)
def test_blade_one_is_refused_as_pyflightstream_refuses_it(tmp_path, blade1, message):
    block = REFERENCE_BLOCK.replace(
        'blade1 = { azimuth_deg = 0.0, zero = "Z" }', f"blade1 = {blade1}"
    )
    build(tmp_path, block=block)
    with pytest.raises(PlanError, match=message):
        plan_workspace(tmp_path, "dpw")


def test_a_rotor_without_blade_one_measures_from_x_and_x_is_the_shaft(tmp_path):
    block = REFERENCE_BLOCK.replace('blade1 = { azimuth_deg = 0.0, zero = "Z" }', "")
    build(tmp_path, block=block)
    with pytest.raises(PlanError, match="blade1.zero X equals shaft X"):
        plan_workspace(tmp_path, "dpw")


def test_the_alias_is_the_table_name(tmp_path):
    assert rotor_block({"R": {**BLOCK, "alias": "r"}}) is not None
    build(tmp_path, block=REFERENCE_BLOCK + 'alias = "OTHER"\n')
    with pytest.raises(PlanError, match="alias is 'OTHER'"):
        plan_workspace(tmp_path, "dpw")
