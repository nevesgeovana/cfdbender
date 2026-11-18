"""An SU2 propeller advance-ratio sweep in a steady rotating frame, from the matrix alone."""

from __future__ import annotations

import math
import re
import warnings

import pytest
import test_su2_matrix_columns as columns

from cfdbender.core.plan import PlanError
from cfdbender.solvers.su2._rotor import rotor_block
from cfdbender.workspace.planning import plan_workspace

GAS, GAMMA, TEMPERATURE, MACH = 287.058, 1.4, 288.15, 0.0529
DIAMETER = 0.4064
REFERENCE_BLOCK = """
[rotors.PROP]
diameter_m = 0.4064
blades = 6
origin = [0.0, 0.0, 0.0]
shaft = [-1.0, 0.0, 0.0]
rpm_sign = 1
"""
FRAME = (
    'GRID_MOVEMENT = "ROTATING_FRAME"\nfreestream_temperature = 288.15\n'
    "GAS_CONSTANT = 287.058\nGAMMA_VALUE = 1.4\n"
)


def row(condition: str, sweep: str = "-", var: str = "-") -> str:
    """A matrix row in the column order of the shared SU2 fixture."""
    return (
        f" 5001 | 0 | 1 | WB | cruise | J sweep | {condition} | {sweep} | wb.su2 "
        f"| 001 | 001 | 001 | - | - | 192 | 4h | 8.3.0 | - | {var} "
    )


def build(root, condition="MACH:0.0529, ALPHA:0, BETA:0, ADVANCE_RATIO:0.4", **kwargs):
    setup = kwargs.pop("setup", FRAME)
    block = kwargs.pop("block", REFERENCE_BLOCK)
    columns.workspace(root, row=row(condition, **kwargs), setup=setup)
    path = root / "inputs/references/r001.toml"
    path.write_text(path.read_text(encoding="ascii") + block, encoding="ascii")
    return root


def numbers(text):
    return [float(item) for item in text.split()]


def expected_rate(advance, mach=MACH, diameter=DIAMETER, sign=-1.0):
    speed = mach * math.sqrt(GAMMA * GAS * TEMPERATURE)
    return sign * 2.0 * math.pi * speed / (advance * diameter)


def test_a_rotor_block_is_read_with_its_defaults(tmp_path):
    block = rotor_block(
        {
            "rotors": {
                "PROP": {
                    "diameter_m": 0.5,
                    "blades": 3,
                    "origin": [1, 2, 3],
                    "shaft": [0, 0, -2],
                }
            }
        }
    )
    assert block.monitored_blades == 3 and block.rpm_sign == 1
    assert block.shaft == (0.0, 0.0, -1.0) and block.origin == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("diameter_m = 0.0", "diameter_m must be positive"),
        ('diameter_m = "big"', "diameter_m must be a finite number"),
        ("diameter_m = true", "diameter_m must be a finite number"),
        ("blades = 6.0", "blades must be a positive integer"),
        ("blades = 0", "blades must be a positive integer"),
        ("monitored_blades = 7", "monitored_blades cannot exceed blades"),
        ("monitored_blades = 0", "monitored_blades must be a positive integer"),
        ("shaft = [0.0, 0.0, 0.0]", "shaft must not be the zero vector"),
        ("shaft = [1.0, 0.0]", "shaft needs three finite coordinates"),
        ("origin = 3", "origin needs three finite coordinates"),
        ("rpm_sign = 2", "rpm_sign must be 1 or -1"),
        ("rpm_sign = 1.0", "rpm_sign must be 1 or -1"),
        ("pitch = 3", "unknown keys: pitch"),
    ],
)
def test_a_bad_rotor_block_refuses_at_planning(tmp_path, change, message):
    key = change.split("=")[0].strip()
    lines = [x for x in REFERENCE_BLOCK.splitlines() if not x.startswith(key + " ")]
    build(tmp_path, block="\n".join([*lines, change]))
    with pytest.raises(PlanError, match=message):
        plan_workspace(tmp_path, "dpw")


def test_a_missing_rotor_key_is_named(tmp_path):
    lines = [x for x in REFERENCE_BLOCK.splitlines() if not x.startswith("blades")]
    build(tmp_path, block="\n".join(lines))
    with pytest.raises(PlanError, match=r"rotors.PROP needs blades"):
        plan_workspace(tmp_path, "dpw")


def test_two_rotors_are_refused_as_not_supported_yet(tmp_path):
    second = REFERENCE_BLOCK.replace("PROP", "OTHER")
    build(tmp_path, block=REFERENCE_BLOCK + second)
    with pytest.raises(PlanError, match="multi-rotor rotating-frame runs are not supported yet"):
        plan_workspace(tmp_path, "dpw")


def test_a_fixed_advance_ratio_writes_the_rotation(tmp_path):
    build(tmp_path)
    deck = columns.deck(tmp_path)
    rate = numbers(deck["ROTATION_RATE"])
    assert rate[1] == 0.0 and rate[2] == 0.0
    assert rate[0] == pytest.approx(expected_rate(0.4), rel=1e-12)
    assert rate[0] < 0.0, "thrust toward minus x with rpm_sign 1 turns about minus x"
    assert numbers(deck["MOTION_ORIGIN"]) == [0.0, 0.0, 0.0]
    assert float(deck["MACH_MOTION"]) == pytest.approx(MACH)
    assert deck["GRID_MOVEMENT"] == "ROTATING_FRAME"
    assert "ADVANCE_RATIO" not in deck


def test_the_rotor_origin_is_written_as_motion_origin(tmp_path):
    build(tmp_path, block=REFERENCE_BLOCK.replace("[0.0, 0.0, 0.0]", "[0.1, -0.2, 0.3]"))
    assert numbers(columns.deck(tmp_path)["MOTION_ORIGIN"]) == [0.1, -0.2, 0.3]


@pytest.mark.parametrize(
    ("shaft", "sign", "direction"),
    [
        ("[-1.0, 0.0, 0.0]", 1, (-1.0, 0.0, 0.0)),
        ("[-1.0, 0.0, 0.0]", -1, (1.0, 0.0, 0.0)),
        ("[1.0, 0.0, 0.0]", 1, (1.0, 0.0, 0.0)),
        ("[0.0, 0.0, -3.0]", 1, (0.0, 0.0, -1.0)),
    ],
)
def test_the_sign_follows_the_shaft_and_rpm_sign(tmp_path, shaft, sign, direction):
    block = REFERENCE_BLOCK.replace("[-1.0, 0.0, 0.0]", shaft).replace(
        "rpm_sign = 1", f"rpm_sign = {sign}"
    )
    build(tmp_path, block=block)
    rate = numbers(columns.deck(tmp_path)["ROTATION_RATE"])
    magnitude = abs(expected_rate(0.4))
    assert rate == pytest.approx([c * magnitude for c in direction], rel=1e-12, abs=1e-9)


def test_the_sweep_gives_each_advance_ratio_its_own_rotation_and_point(tmp_path):
    values = [0.3, 0.45, 0.7]
    build(
        tmp_path,
        "MACH:0.0529, ALPHA:0, BETA:0, ADVANCE_RATIO:sweep",
        sweep=", ".join(map(str, values)),
    )
    plan = plan_workspace(tmp_path, "dpw")
    assert len(plan.points) == 3
    for index, advance in enumerate(values):
        rate = numbers(columns.deck(tmp_path, index)["ROTATION_RATE"])
        assert rate[0] == pytest.approx(expected_rate(advance), rel=1e-12)
    assert len({p.point_id for p in plan.points}) == 3
    assert len({p.directory for p in plan.points}) == 3
    assert [re.search(r"J[+-]\d+", p.directory)[0] for p in plan.points] == [
        "J+030",
        "J+045",
        "J+070",
    ]
    assert all(p.condition["ADVANCE_RATIO"] == v for p, v in zip(plan.points, values, strict=True))


def test_a_velocity_pinned_row_takes_that_speed(tmp_path):
    build(tmp_path, "TASmps:18, TK:288.15, PPA:101325, ADVANCE_RATIO:0.5", setup=FRAME)
    # n = V / (J D) with the speed SU2 itself derives from the MACH_NUMBER written and the
    # gas constants of the deck (the pinned 18 m/s differs from it by the gas constant, 5e-4)
    deck = columns.deck(tmp_path)
    speed = float(deck["MACH_NUMBER"]) * math.sqrt(GAMMA * GAS * TEMPERATURE)
    assert abs(numbers(deck["ROTATION_RATE"])[0]) == pytest.approx(
        2.0 * math.pi * speed / (0.5 * DIAMETER), rel=1e-12
    )
    assert speed == pytest.approx(18.0, rel=1e-3)


def test_the_setups_mach_motion_wins(tmp_path):
    build(tmp_path, setup=FRAME + "MACH_MOTION = 0.06\n")
    assert float(columns.deck(tmp_path)["MACH_MOTION"]) == 0.06


def test_a_stated_origin_that_agrees_is_kept_and_one_that_differs_refuses(tmp_path):
    build(tmp_path, setup=FRAME + 'MOTION_ORIGIN = "0.0 0.0 0.0"\n')
    assert numbers(columns.deck(tmp_path)["MOTION_ORIGIN"]) == [0.0, 0.0, 0.0]
    build(tmp_path / "b", setup=FRAME + 'MOTION_ORIGIN = "0.0 0.0 1.0"\n')
    with pytest.raises(PlanError, match="MOTION_ORIGIN is .* the rotor PROP has origin"):
        plan_workspace(tmp_path / "b", "dpw")


def test_advance_ratio_without_a_rotor_block_refuses_saying_so(tmp_path):
    build(tmp_path, block="")
    with pytest.raises(PlanError, match=r"needs the rotor block of the .*r001.toml.*\[rotors"):
        plan_workspace(tmp_path, "dpw")


def test_advance_ratio_without_a_rotating_frame_refuses_saying_so(tmp_path):
    build(tmp_path, setup="freestream_temperature = 288.15\n")
    with pytest.raises(PlanError, match="needs a steady rotating frame.*states no GRID_MOVEMENT"):
        plan_workspace(tmp_path, "dpw")
    build(tmp_path / "b", setup='GRID_MOVEMENT = "NONE"\n')
    with pytest.raises(PlanError, match="states GRID_MOVEMENT = NONE"):
        plan_workspace(tmp_path / "b", "dpw")


def test_a_second_source_of_the_rotation_refuses_from_the_setup_or_the_row(tmp_path):
    build(tmp_path, setup=FRAME + 'ROTATION_RATE = "-1.0 0.0 0.0"\n')
    with pytest.raises(PlanError, match="both state the rotation"):
        plan_workspace(tmp_path, "dpw")
    build(tmp_path / "b", var='ROTATION_RATE: "-1.0 0.0 0.0"')
    with pytest.raises(PlanError, match="both state the rotation"):
        plan_workspace(tmp_path / "b", "dpw")


@pytest.mark.parametrize("bad", ["0", "-0.3"])
def test_a_non_positive_advance_ratio_refuses(tmp_path, bad):
    build(tmp_path, f"MACH:0.0529, ADVANCE_RATIO:{bad}")
    with pytest.raises(PlanError, match="ADVANCE_RATIO must be positive"):
        plan_workspace(tmp_path, "dpw")


def test_a_zero_speed_has_no_advance_ratio(tmp_path):
    build(tmp_path, "MACH:0, ADVANCE_RATIO:0.4")
    with pytest.raises(PlanError, match="free-stream speed above zero"):
        plan_workspace(tmp_path, "dpw")


def test_rpm_is_still_not_read(tmp_path):
    build(tmp_path, "MACH:0.0529, RPM:6000")
    with pytest.raises(PlanError, match="does not read FLIGHT_CONDITION key RPM"):
        plan_workspace(tmp_path, "dpw")


def test_a_non_zero_angle_with_the_rotating_frame_warns_and_continues(tmp_path):
    build(tmp_path, "MACH:0.0529, ALPHA:2, BETA:0, ADVANCE_RATIO:0.4")
    with pytest.warns(UserWarning, match="ALPHA is not zero with a steady rotating frame"):
        deck = columns.deck(tmp_path)
    assert "ROTATION_RATE" in deck
    build(tmp_path / "b")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        columns.deck(tmp_path / "b")
