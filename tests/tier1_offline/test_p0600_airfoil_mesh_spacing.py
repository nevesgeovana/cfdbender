"""P0600 airfoil-2d: the wall-normal spacing of the 2D airfoil mesh.

The first cell comes from a target y+ and the Reynolds number through the turbulent flat-plate
skin friction of Schlichting, Cf = (2 log10 Re - 0.65)^-2.3, so y1 / c = y+ / (Re sqrt(Cf / 2)).
The normal cells grow geometrically from that cell to the far field, the ratio solved for the
level's cell count; a ratio above the cap is refused, naming the count that would satisfy it.
"""

from __future__ import annotations

import math

import pytest

from cfdbender._errors import InputValueError
from cfdbender.mesh import _airfoil2d as af


@pytest.mark.parametrize(
    ("reynolds", "friction", "height"),
    [(1e6, 0.00375, 2.311e-05), (3e6, 0.00311, 8.452e-06), (6e6, 0.00279, 4.465e-06)],
)
def test_p0600_first_cell_follows_the_flat_plate_correlation(reynolds, friction, height):
    assert af.skin_friction(reynolds) == pytest.approx(friction, rel=2e-3)
    assert af.first_cell(1.0, reynolds) == pytest.approx(height, rel=1e-3)
    assert af.first_cell(2.0, reynolds) == pytest.approx(2 * height, rel=1e-3)


def test_p0600_first_cell_is_the_closed_form():
    reynolds, yplus = 2.5e6, 0.7
    friction = (2 * math.log10(reynolds) - 0.65) ** -2.3
    assert af.first_cell(yplus, reynolds) == pytest.approx(
        yplus / (reynolds * math.sqrt(friction / 2)), rel=1e-12
    )


@pytest.mark.parametrize(
    ("level", "wall", "normal", "wake", "around"),
    [("coarse", 64, 64, 48, 225), ("medium", 128, 128, 96, 449), ("fine", 256, 256, 192, 897)],
)
def test_p0600_levels_carry_the_tmr_counts(level, wall, normal, wake, around):
    counts = af.LEVELS[level]
    assert (counts.wall_cells, counts.normal_cells, counts.wake_cells) == (wall, normal, wake)
    assert 2 * counts.wake_cells + 2 * counts.wall_cells + 1 == around


@pytest.mark.parametrize(
    ("level", "ratio"), [("coarse", 1.3118), ("medium", 1.1381), ("fine", 1.0636)]
)
def test_p0600_growth_is_solved_to_the_far_field(level, ratio):
    plan = af.plan_spacing(reynolds=6e6, yplus=1.0, level=level, farfield=500.0)
    assert plan["first_cell"] == pytest.approx(4.465e-06, rel=1e-3)
    assert plan["normal_growth"] == pytest.approx(ratio, abs=2e-4)
    cells, first, growth = plan["normal_cells"], plan["first_cell"], plan["normal_growth"]
    assert first * (growth**cells - 1) / (growth - 1) == pytest.approx(500.0, rel=1e-9)


def test_p0600_defaults_are_yplus_one_far_field_500_and_coarse():
    assert af.DEFAULT_YPLUS == 1.0
    assert af.DEFAULT_FARFIELD == 500.0
    assert af.DEFAULT_LEVEL == "coarse"
    assert af.DEFAULT_TRAILING_EDGE == "sharp"


def test_p0600_growth_above_the_cap_is_refused_naming_the_count():
    with pytest.raises(InputValueError, match=r"1\.35") as caught:
        af.plan_spacing(reynolds=1e9, yplus=0.05, level="coarse", farfield=500.0)
    message = str(caught.value)
    needed = int(message.split("needs at least ")[1].split()[0])
    plan_cells = af.LEVELS["coarse"].normal_cells
    assert needed > plan_cells
    first = af.first_cell(0.05, 1e9)
    assert first * (1.35**needed - 1) / 0.35 >= 500.0
    assert first * (1.35 ** (needed - 1) - 1) / 0.35 < 500.0


@pytest.mark.parametrize(
    ("keywords", "named"),
    [
        ({"reynolds": 0.0}, "reynolds"),
        ({"reynolds": -1e6}, "reynolds"),
        ({"reynolds": math.inf}, "reynolds"),
        ({"yplus": 0.0}, "y+"),
        ({"farfield": 1.0}, "far field"),
        ({"level": "huge"}, "level"),
    ],
)
def test_p0600_spacing_refusals_name_the_value(keywords, named):
    arguments = {"reynolds": 6e6, "yplus": 1.0, "level": "coarse", "farfield": 500.0, **keywords}
    with pytest.raises(InputValueError, match=named):
        af.plan_spacing(**arguments)


def test_p0600_two_sided_distribution_meets_both_end_spacings():
    points = af.two_sided(1.0, 64, 4.2740e-04, 4.2599e-03)
    steps = [b - a for a, b in zip(points, points[1:], strict=False)]
    assert len(points) == 65
    assert points[0] == 0.0 and points[-1] == pytest.approx(1.0, abs=1e-15)
    assert all(step > 0 for step in steps)
    assert steps[0] == pytest.approx(4.2740e-04, rel=0.01)
    assert steps[-1] == pytest.approx(4.2599e-03, rel=0.01)


def test_ra1_low_reynolds_is_refused_with_action():
    with pytest.raises(InputValueError, match="Reynolds.*increase"):
        af.plan_spacing(reynolds=1.0, yplus=1.0, level="coarse", farfield=500.0)
