"""Forces and moments to coefficients: known answer, moment transfer, signs, axes, groups."""

from __future__ import annotations

import math

import pytest
from cfdpp_support import builder

from cfdbender.core.neutral import wind_directions
from cfdbender.core.results import References
from cfdbender.solvers.cfdpp import _forces, _history

KNOWN = builder().KNOWN
REFS = References(
    KNOWN["area"],
    KNOWN["length"],
    KNOWN["origin"],
    "geometry",
    ("aft", "right", "up"),
)
Q = KNOWN["dynamic_pressure"]


def _known_step() -> _history.StepForces:
    return _history.parse_info1(builder().info1_known_text(1, 1))[0]


def _known_forces(part: str = "total") -> _forces.Forces:
    return _history.step_forces(_known_step(), (1, 2), part)


def test_sign_constants_are_plus_one() -> None:
    assert _forces.FORCE_SIGN == 1
    assert _forces.MOMENT_SIGN == 1


def test_known_answer_fixture_gives_cl_cd_cm() -> None:
    forces = _known_forces()
    coefficients = _forces.coefficients(forces, Q, REFS)
    cf = (coefficients["CFx"].value, coefficients["CFy"].value, coefficients["CFz"].value)
    cm = (coefficients["CMx"].value, coefficients["CMy"].value, coefficients["CMz"].value)
    neutral = _forces.neutral(cf, cm, 0.0, 0.0, 3)
    assert neutral["CL"] == pytest.approx(0.5, abs=1e-12)
    assert neutral["CD"] == pytest.approx(0.01, abs=1e-12)
    assert neutral["CM"] == pytest.approx(-0.02, abs=1e-12)


def test_raw_sums_stay_in_the_origin_frame() -> None:
    coefficients = _forces.coefficients(_known_forces(), Q, REFS)
    assert coefficients["Fx"].value == pytest.approx(10.0)
    assert coefficients["Fz"].value == pytest.approx(500.0)
    assert coefficients["My"].value == pytest.approx(-145.0)
    assert coefficients["My"].kind == "moment" and coefficients["My"].axis == "y"
    assert coefficients["CFz"].kind == "force" and coefficients["CFz"].axis == "z"


def test_transfer_moment_against_a_hand_value() -> None:
    # c x F = (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3); M - c x F = (4, -4, 6)
    result = _forces.transfer_moment((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (1.0, 2.0, 3.0))
    assert result == pytest.approx((4.0, -4.0, 6.0))


def test_transfer_to_the_origin_changes_nothing() -> None:
    assert _forces.transfer_moment((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (0.0, 0.0, 0.0)) == (
        1.0,
        2.0,
        3.0,
    )


def test_force_sign_flips_force_and_moment_sign_flips_moment() -> None:
    base = _forces.coefficients(_known_forces(), Q, REFS)
    force = _forces.coefficients(_known_forces(), Q, REFS, force_sign=-1)
    assert force["CFz"].value == pytest.approx(-base["CFz"].value)
    assert force["Fx"].value == pytest.approx(-base["Fx"].value)
    moment = _forces.coefficients(_known_forces(), Q, REFS, moment_sign=-1)
    assert moment["My"].value == pytest.approx(-base["My"].value)
    assert moment["CFz"].value == pytest.approx(base["CFz"].value)


def test_inviscid_plus_viscous_equals_total() -> None:
    coefficients = _forces.coefficients(
        _known_forces(),
        Q,
        REFS,
        inviscid=_known_forces("inviscid"),
        viscous=_known_forces("viscous"),
    )
    for axis in "xyz":
        for stem in ("CF", "CM"):
            total = coefficients[f"{stem}{axis}"].value
            parts = (
                coefficients[f"{stem}{axis}_inv"].value + coefficients[f"{stem}{axis}_vis"].value
            )
            assert parts == pytest.approx(total, abs=1e-12)
    assert coefficients["CFx_vis"].value == pytest.approx(3.0 / 1000.0)


def test_groups_that_partition_the_walls_sum_to_total() -> None:
    step = _known_step()
    total = _forces.coefficients(_history.step_forces(step, (1, 2)), Q, REFS)
    parts = [_forces.coefficients(_history.step_forces(step, (s,)), Q, REFS) for s in (1, 2)]
    for name in ("CFx", "CFz", "CMy"):
        assert sum(part[name].value for part in parts) == pytest.approx(
            total[name].value, abs=1e-12
        )


def test_two_dimensional_neutral_projection_against_hand_values() -> None:
    alpha = 10.0
    a = math.radians(alpha)
    cf, cm = (0.02, 0.4, 0.0), (0.0, 0.0, 0.03)
    neutral = _forces.neutral(cf, cm, alpha, 0.0, 2)
    assert neutral["CD"] == pytest.approx(0.02 * math.cos(a) + 0.4 * math.sin(a))
    assert neutral["CL"] == pytest.approx(-0.02 * math.sin(a) + 0.4 * math.cos(a))
    assert neutral["CM"] == pytest.approx(-0.03)


def test_two_dimensional_known_answer() -> None:
    # F = (10, 500, 0); M_c = (0, 0, 20); origin moment = M_c + c x F with c = (0.25, 0, 0)
    forces = _forces.Forces((10.0, 500.0, 0.0), (0.0, 0.0, 145.0))
    coefficients = _forces.coefficients(forces, Q, REFS)
    cf = tuple(coefficients[f"CF{a}"].value for a in "xyz")
    cm = tuple(coefficients[f"CM{a}"].value for a in "xyz")
    neutral = _forces.neutral(cf, cm, 0.0, 0.0, 2)  # type: ignore[arg-type]
    assert neutral["CL"] == pytest.approx(0.5)
    assert neutral["CD"] == pytest.approx(0.01)
    assert neutral["CM"] == pytest.approx(-0.02)


def test_three_dimensional_neutral_projection_follows_the_wind_directions() -> None:
    alpha, beta = 7.0, 3.0
    drag, lift = wind_directions(alpha, beta)
    cf, cm = (0.1, 0.2, 0.3), (0.01, 0.04, 0.05)
    neutral = _forces.neutral(cf, cm, alpha, beta, 3)
    assert neutral["CD"] == pytest.approx(sum(c * d for c, d in zip(cf, drag, strict=True)))
    assert neutral["CL"] == pytest.approx(sum(c * d for c, d in zip(cf, lift, strict=True)))
    assert neutral["CM"] == pytest.approx(0.04)


def test_neutral_refuses_an_unknown_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        _forces.neutral((0, 0, 0), (0, 0, 0), 0.0, 0.0, 4)


@pytest.mark.parametrize("q", [0.0, -1.0, math.nan, math.inf])
def test_coefficients_refuse_a_bad_dynamic_pressure(q: float) -> None:
    with pytest.raises(ValueError, match="dynamic pressure"):
        _forces.coefficients(_known_forces(), q, REFS)
