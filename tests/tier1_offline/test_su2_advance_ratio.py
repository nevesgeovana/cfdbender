"""An SU2 propeller advance-ratio sweep in a steady rotating frame, from the matrix alone."""

from __future__ import annotations

import math
import re
import warnings

import pytest
import test_su2_matrix_columns as columns

from cfdbender.core.plan import PlanError
from cfdbender.core.rotor import rotor_block
from cfdbender.workspace.planning import plan_workspace

GAS, GAMMA, TEMPERATURE, MACH = 287.058, 1.4, 288.15, 0.0529
DIAMETER = 0.4064
REFERENCE_BLOCK = """
[ROTOR]
kind = "rotor"
x_m = 0.0
y_m = 0.0
z_m = 0.0
axis = "X"
rpm_sign = -1
diameter_m = 0.4064
families_general = ["S"]
families_blades = ["Blade1", "Blade6", "Blade5", "Blade4", "Blade3", "Blade2"]
blade1 = { azimuth_deg = 0.0, zero = "Z" }
"""
FRAME = (
    'GRID_MOVEMENT = "ROTATING_FRAME"\nfreestream_temperature = 288.15\n'
    "GAS_CONSTANT = 287.058\nGAMMA_VALUE = 1.4\n"
)


def row(condition: str, sweep: str = "-", var: str = "-", symmetry="-", loads="-") -> str:
    """A matrix row in the column order of the shared SU2 fixture."""
    return (
        f" 5001 | 0 | 1 | WB | cruise | J sweep | {condition} | {sweep} | wb.su2 "
        f"| 001 | 001 | 001 | {symmetry} | {loads} | 192 | 4h | 8.3.0 | - | {var} "
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


def test_a_rotor_declaration_of_the_flightstream_reference_file_is_read(tmp_path):
    block = rotor_block(
        {
            "area_m2": 50.0,
            "aliases": {"airframe": ["N"]},
            "ROTOR": {
                "kind": "rotor",
                "x_m": 1.0,
                "y_m": 2.0,
                "z_m": 3.0,
                "axis": "X",
                "rpm_sign": -1,
                "diameter_m": 0.5,
                "families_blades": ["Blade1", "Blade2", "Blade3"],
            },
        }
    )
    assert (block.blades, block.rpm_sign, block.origin) == (3, -1, (1.0, 2.0, 3.0))
    assert block.axis == (1.0, 0.0, 0.0) and block.thrust_direction == (-1.0, 0.0, 0.0)
    assert rotor_block({"area": 1.0}) is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("diameter_m = 0.0", "diameter_m must be positive"),
        ('diameter_m = "big"', "diameter_m must be a finite number"),
        ("diameter_m = true", "diameter_m must be a finite number"),
        ('axis = "W"', 'axis must be "X", "Y" or "Z"'),
        ('axis = "Z"', 'reads axis = "X" only'),
        ("x_m = [1.0]", "x_m must be a finite number"),
        ("rpm_sign = 2", "rpm_sign must be 1 or -1"),
        ("rpm_sign = 1.0", "rpm_sign must be 1 or -1"),
        ("families_blades = []", "families_blades must be a list"),
        ("monitored_blades = 1", "unknown keys: monitored_blades"),
        ("pitch = 3", "unknown keys: pitch"),
    ],
)
def test_a_bad_rotor_declaration_refuses_at_planning(tmp_path, change, message):
    key = change.split("=")[0].strip()
    lines = [x for x in REFERENCE_BLOCK.splitlines() if not x.startswith(key + " ")]
    build(tmp_path, block="\n".join([*lines, change]))
    with pytest.raises(PlanError, match=message):
        plan_workspace(tmp_path, "dpw")


@pytest.mark.parametrize("key", ["diameter_m", "axis"])
def test_a_missing_rotor_key_is_named(tmp_path, key):
    lines = [x for x in REFERENCE_BLOCK.splitlines() if not x.startswith(key + " ")]
    build(tmp_path, block="\n".join(lines))
    with pytest.raises(PlanError, match=f"ROTOR needs {key}"):
        plan_workspace(tmp_path, "dpw")


def test_two_rotors_are_refused_as_not_supported_yet(tmp_path):
    second = REFERENCE_BLOCK.replace("ROTOR", "OTHER")
    build(tmp_path, block=REFERENCE_BLOCK + second)
    with pytest.raises(PlanError, match="multi-rotor rotating-frame runs are not supported yet"):
        plan_workspace(tmp_path, "dpw")


def test_a_fixed_advance_ratio_writes_the_rotation(tmp_path):
    build(tmp_path)
    deck = columns.deck(tmp_path)
    rate = numbers(deck["ROTATION_RATE"])
    assert rate[1] == 0.0 and rate[2] == 0.0
    assert rate[0] == pytest.approx(expected_rate(0.4), rel=1e-12)
    assert rate[0] < 0.0, "axis X with rpm_sign -1 turns left-handed about +x: toward minus x"
    assert numbers(deck["MOTION_ORIGIN"]) == [0.0, 0.0, 0.0]
    assert float(deck["MACH_MOTION"]) == pytest.approx(MACH)
    assert deck["GRID_MOVEMENT"] == "ROTATING_FRAME"
    assert "ADVANCE_RATIO" not in deck


def test_the_rotor_origin_is_written_as_motion_origin(tmp_path):
    build(tmp_path, block=REFERENCE_BLOCK.replace("x_m = 0.0", "x_m = 0.1"))
    assert numbers(columns.deck(tmp_path)["MOTION_ORIGIN"]) == [0.1, 0.0, 0.0]


@pytest.mark.parametrize(("sign", "direction"), [(-1, (-1.0, 0.0, 0.0)), (1, (1.0, 0.0, 0.0))])
def test_the_rotation_follows_the_hand_about_the_declared_axis(tmp_path, sign, direction):
    build(tmp_path, block=REFERENCE_BLOCK.replace("rpm_sign = -1", f"rpm_sign = {sign}"))
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
    with pytest.raises(PlanError, match="MOTION_ORIGIN is .* the rotor ROTOR has origin"):
        plan_workspace(tmp_path / "b", "dpw")


def test_advance_ratio_without_a_rotor_block_refuses_saying_so(tmp_path):
    build(tmp_path, block="")
    with pytest.raises(PlanError, match="needs the rotor declaration of the .*r001.toml"):
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


# --- what did not change -----------------------------------------------------------------


def _with_advance(root, solver, add):
    from test_matrix_columns_all_solvers import build as build_other

    build_other(root, solver)
    matrix = root / "inputs/matrices/full.fs"
    lines = matrix.read_text(encoding="ascii").splitlines()
    names = [x.strip() for x in lines[0].split("|")]
    cells = lines[2].split("|")
    column = names.index("FLIGHT_CONDITION")
    cells[column] = cells[column].rstrip() + add + " "
    lines[2] = "|".join(cells)
    matrix.write_text("\n".join(lines) + "\n", encoding="ascii")
    return plan_workspace(root, "full")


@pytest.mark.parametrize(
    ("solver", "message"),
    [
        ("openfoam", "unknown key 'advance_ratio'"),
        ("pyfr", "unknown key 'advance_ratio'"),
        ("xfoil", "only scalar alpha or an alpha sweep"),
        ("qprop", "give only one of RPM and J"),
    ],
)
def test_the_other_adapters_keep_their_handling_of_the_key(tmp_path, solver, message):
    with pytest.raises(PlanError, match=message):
        _with_advance(tmp_path, solver, ", ADVANCE_RATIO:0.4")


def test_the_mock_adapter_still_names_the_point_by_the_advance_ratio(tmp_path):
    plan = _with_advance(tmp_path, "mock", ", ADVANCE_RATIO:0.4")
    assert [p.directory.rsplit("/", 1)[1] for p in plan.points] == [
        "DP-M200AL+000J+040",
        "DP-M200AL+020J+040",
    ]


@pytest.mark.parametrize("block", [REFERENCE_BLOCK, ""])
def test_a_matrix_without_advance_ratio_plans_and_writes_what_it_did(tmp_path, block):
    """The rotating frame typed by hand, with or without the rotor block, is unchanged."""
    typed = FRAME + 'ROTATION_RATE = "-674.44 0.0 0.0"\nMACH_MOTION = 0.0529\n'
    build(tmp_path, "MACH:0.0529, ALPHA:0, BETA:0", setup=typed, block=block)
    plan = plan_workspace(tmp_path, "dpw")
    assert [p.directory.rsplit("/", 1)[1] for p in plan.points] == ["DP-M053AL+000BE+000"]
    deck = columns.deck(tmp_path)
    assert deck["ROTATION_RATE"] == "-674.44 0.0 0.0" and deck["MACH_MOTION"] == "0.0529"
    assert "MOTION_ORIGIN" not in deck, (
        "nothing is added to a deck that states ADVANCE_RATIO nowhere"
    )
