"""The solver-agnostic rotor loads: the periodic sum and the coefficients of a rotor.

Nothing here imports a solver adapter: any adapter that gives a monitored force and moment
about a stated point, a speed and a density gets the same algebra.
"""

from __future__ import annotations

import math

import pytest

from cfdbender.core.rotor import (
    NOT_APPLICABLE,
    PeriodicState,
    RotorBlock,
    RotorLoads,
    periodic_state,
    periodic_sum,
    rotation_rate,
    rotor_coefficients,
    rotor_row,
)

D, RHO, SPEED = 0.5, 1.2, 20.0


def unit(v):
    length = math.sqrt(sum(c * c for c in v))
    return tuple(c / length for c in v)


def basis(axis):
    """Two unit vectors normal to a unit axis."""
    seed = (0.0, 1.0, 0.0) if abs(axis[1]) < 0.9 else (0.0, 0.0, 1.0)
    d = sum(a * s for a, s in zip(axis, seed, strict=True))
    e1 = unit(tuple(s - d * a for a, s in zip(axis, seed, strict=True)))
    e2 = (
        axis[1] * e1[2] - axis[2] * e1[1],
        axis[2] * e1[0] - axis[0] * e1[2],
        axis[0] * e1[1] - axis[1] * e1[0],
    )
    return e1, e2


def combine(axis, along, a, b):
    e1, e2 = basis(axis)
    return tuple(along * x + a * y + b * z for x, y, z in zip(axis, e1, e2, strict=True))


def block(axis, sign=1, origin=(0.0, 0.0, 0.0)):
    return RotorBlock("R", D, None, origin, unit(axis), sign)


def quantities(rotor, force, moment, copies, reference=(0.0, 0.0, 0.0), rps=30.0, rho=RHO):
    return rotor_row(
        rotor,
        RotorLoads(tuple(force), tuple(moment), tuple(reference), SPEED, rho),
        rps=rps,
        copies=copies,
    )


def scales(rps=30.0):
    return RHO * rps**2 * D**4, RHO * rps**2 * D**5


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (1.0, 2.0, 2.0), (-0.3, 0.5, 0.8)])
@pytest.mark.parametrize("copies", [1, 2, 6])
def test_axial_parts_add_up_and_in_plane_parts_cancel(axis, copies):
    rotor = block(axis)
    a = rotor.axis
    force = combine(a, 3.0, 0.7, -1.1)  # all six components non-zero
    moment = combine(a, 0.4, -0.2, 0.9)
    out = quantities(rotor, force, moment, copies)
    forces, moments = scales()
    assert out["T"] == pytest.approx(copies * 3.0, rel=1e-12)
    assert out["Q"] == pytest.approx(copies * 0.4, rel=1e-12)
    in_plane_force = math.hypot(out["CN"], out["CS"]) * forces
    in_plane_moment = math.hypot(out["CMN"], out["CMS"]) * moments
    if copies == 1:
        assert in_plane_force == pytest.approx(math.hypot(0.7, 1.1), rel=1e-12)
        assert in_plane_moment == pytest.approx(math.hypot(0.2, 0.9), rel=1e-12)
    else:
        assert in_plane_force == pytest.approx(0.0, abs=1e-12)
        assert in_plane_moment == pytest.approx(0.0, abs=1e-12)


def test_the_sum_is_a_vector_sum_not_a_multiplication():
    force, moment = periodic_sum((1.0, 2.0, 3.0), (0.5, -1.0, 2.0), (1.0, 0.0, 0.0), 4)
    assert force == pytest.approx((4.0, 0.0, 0.0), abs=1e-12)
    assert moment == pytest.approx((2.0, 0.0, 0.0), abs=1e-12)
    one = periodic_sum((1.0, 2.0, 3.0), (0.5, -1.0, 2.0), (1.0, 0.0, 0.0), 1)
    assert one == ((1.0, 2.0, 3.0), (0.5, -1.0, 2.0))


def test_a_sector_turned_by_the_copy_angle_lands_on_the_next_copy():
    # one copy of a three-copy wheel: +y turns by 120 degrees right-handed about +x
    force, _ = periodic_sum((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 3)
    assert force == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    force, _ = periodic_sum((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1)
    assert force == (0.0, 1.0, 0.0)


def test_a_moment_reference_off_the_axis_is_moved_to_the_hub():
    rotor = block((1.0, 0.0, 0.0), origin=(0.2, 0.0, 0.0))
    force = (-2.0, 0.5, 1.5)
    moment = (0.3, 0.1, -0.2)
    reference = (0.5, 0.04, -0.07)
    arm = (reference[0] - 0.2, reference[1], reference[2])
    transfer_x = arm[1] * force[2] - arm[2] * force[1]  # x component of arm x force
    out = quantities(rotor, force, moment, 6, reference=reference)
    assert out["T"] == pytest.approx(6 * -2.0, rel=1e-12)  # T = F . shaft, pyflightstream's sign
    assert out["Q"] == pytest.approx(6 * (moment[0] + transfer_x), rel=1e-12)
    assert out["CMN"] == pytest.approx(0.0, abs=1e-12)


def test_the_coefficients_follow_the_definitions_with_pyflightstreams_signs():
    rotor = block((1.0, 0.0, 0.0), sign=-1)
    out = quantities(rotor, (-1.5, 0.3, 0.2), (0.25, -0.05, 0.04), 1)
    n = -30.0
    forces, moments = scales()
    assert (out["T"], out["Q"]) == (pytest.approx(-1.5), pytest.approx(0.25))
    assert out["RPM"] == pytest.approx(60.0 * n) and out["DIAMETER"] == D
    assert out["CT"] == pytest.approx(-1.5 / forces)
    assert out["CQ"] == pytest.approx(0.25 / moments)
    assert out["CP"] == pytest.approx(2.0 * math.pi * out["CQ"] * -1.0)
    assert out["J"] == pytest.approx(SPEED / (abs(n) * D))
    assert out["ETA"] == pytest.approx(out["J"] * out["CT"] / out["CP"], rel=1e-12)
    # shaft +x: N is +z and S = N x shaft is +y
    assert out["CN"] == pytest.approx(0.2 / forces) and out["CS"] == pytest.approx(0.3 / forces)
    assert out["CMN"] == pytest.approx(0.04 / moments)
    assert out["CMS"] == pytest.approx(-0.05 / moments)


def test_a_shaft_along_up_has_no_in_plane_coefficients():
    out = quantities(block((0.0, 0.0, 1.0)), (1.0, 2.0, 3.0), (0.1, 0.2, 0.3), 1)
    assert [out[k] for k in ("CN", "CS", "CMN", "CMS")] == [NOT_APPLICABLE] * 4
    assert out["CT"] != NOT_APPLICABLE


def test_no_density_leaves_the_coefficients_unknown():
    out = quantities(block((1.0, 0.0, 0.0)), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 1, rho=None)
    assert out["T"] == pytest.approx(-1.0) and out["CT"] == NOT_APPLICABLE


@pytest.mark.parametrize("sign", [1, -1])
def test_the_rotation_vector_follows_the_hand_about_the_axis(sign):
    rotor = block((1.0, 2.0, 2.0), sign=sign)
    rate, n = rotation_rate(rotor, 0.8, 24.0)
    assert n == pytest.approx(24.0 / (0.8 * D))
    expected = [sign * 2.0 * math.pi * n * c / 3.0 for c in (1.0, 2.0, 2.0)]
    assert rate == pytest.approx(expected, rel=1e-12)


def test_the_symmetry_of_a_row_reads_as_flightstream_reads_it():
    assert periodic_state({}) == PeriodicState("NONE", 1, True)
    state = periodic_state({"SYMMETRY": "PERIODIC", "PERIODIC_COPIES": 6})
    assert (state.copies, state.multiplier, state.loads) == (6, 6, True)
    off = periodic_state({"SYMMETRY": "PERIODIC", "PERIODIC_COPIES": 6, "SYMMETRY_LOADS": False})
    assert off.multiplier == 1
    assert (
        periodic_state(
            {"SYMMETRY": "PERIODIC", "PERIODIC_COPIES": 6, "symmetry_loads": "false"}
        ).loads
        is False
    )


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ({"SYMMETRY": "PERIODIC"}, "needs PERIODIC_COPIES"),
        ({"SYMMETRY": "PERIODIC", "PERIODIC_COPIES": 0}, "positive whole number"),
        ({"SYMMETRY": "PERIODIC", "PERIODIC_COPIES": -2}, "positive whole number"),
        ({"SYMMETRY": "PERIODIC", "PERIODIC_COPIES": 2.5}, "positive whole number"),
        ({"SYMMETRY": "PERIODIC", "PERIODIC_COPIES": True}, "positive whole number"),
        ({"PERIODIC_COPIES": 6}, "needs SYMMETRY: PERIODIC"),
        ({"SYMMETRY": "NONE", "PERIODIC_COPIES": 6}, "needs SYMMETRY: PERIODIC"),
        ({"SYMMETRY": "MIRROR"}, "reads NONE or PERIODIC"),
        ({"SYMMETRY": "PERIODIC", "PERIODIC_COPIES": 6, "SYMMETRY_LOADS": 3}, "true or false"),
    ],
)
def test_the_symmetry_refusals(setup, message):
    with pytest.raises(ValueError, match=message):
        periodic_state(setup)


# --- the coefficient definitions, with the expected numbers of their original tests -------------


@pytest.mark.parametrize(
    ("rps", "diameter", "field"),
    [(0.0, 2.0, "rps is zero"), (20.0, 0.0, "diameter_m"), (20.0, -1.0, "diameter_m")],
)
def test_coefficients_refuse_a_missing_speed_or_diameter(rps, diameter, field):
    with pytest.raises(ValueError, match=field):
        rotor_coefficients(
            thrust_n=500.0,
            torque_nm=40.0,
            rps=rps,
            diameter_m=diameter,
            density_kg_m3=1.225,
            speed_m_s=30.0,
        )


def test_the_coefficients_are_the_standard_definitions():
    rho, n, diameter, thrust, torque, speed = 1.225, 20.0, 2.0, 500.0, 40.0, 30.0
    got = rotor_coefficients(
        thrust_n=thrust,
        torque_nm=torque,
        rps=n,
        diameter_m=diameter,
        density_kg_m3=rho,
        speed_m_s=speed,
    )
    assert math.isclose(got["J"], speed / (n * diameter), rel_tol=1e-12)
    assert math.isclose(got["CT"], thrust / (rho * n**2 * diameter**4), rel_tol=1e-12)
    assert math.isclose(got["CQ"], torque / (rho * n**2 * diameter**5), rel_tol=1e-12)
    assert math.isclose(got["CP"], 2 * math.pi * got["CQ"], rel_tol=1e-12)
    assert math.isclose(got["ETA"], got["J"] * got["CT"] / got["CP"], rel_tol=1e-12)


def test_a_static_point_has_no_efficiency_and_keeps_the_rest():
    got = rotor_coefficients(
        thrust_n=500.0,
        torque_nm=40.0,
        rps=20.0,
        diameter_m=2.0,
        density_kg_m3=1.225,
        speed_m_s=0.0,
    )
    assert got["ETA"] == NOT_APPLICABLE and got["J"] == 0.0
    assert isinstance(got["CT"], float) and got["CT"] > 0
    assert isinstance(got["CQ"], float) and got["CQ"] > 0


def test_reversing_the_rotation_and_the_torque_keeps_the_power_coefficient():
    base = dict(thrust_n=500.0, density_kg_m3=1.225, speed_m_s=30.0, diameter_m=2.0)
    forward = rotor_coefficients(torque_nm=40.0, rps=20.0, **base)
    reverse = rotor_coefficients(torque_nm=-40.0, rps=-20.0, **base)
    assert reverse["CP"] == pytest.approx(forward["CP"], rel=1e-12)
    assert reverse["CQ"] == pytest.approx(-forward["CQ"], rel=1e-12)
    assert reverse["J"] == forward["J"]
