"""Datapoint folders are named from the flight condition, as pyflightstream names them."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfdbender.core.naming import PointNameError, name_field, point_name
from cfdbender.core.plan import PlanError, RowRequirements, plan_matrix

# Literal expectations copied from pyflightstream's own tests.
# test_goal024_point_name.py::CASES (one value per code-table key).
TABLE = (
    ("MACH", 0.144, "M144"),
    ("TASmps", 34.0, "V0340"),
    ("REmi", 4.38, "RE438"),
    ("ALTFT", 5000.0, "ALT05000"),
    ("dISA", 10.0, "DT+100"),
    ("RHOkgm3", 1.225, "RHO12250"),
    ("MUPas", 1.8e-5, "MU18000"),
    ("ASMPS", 340.0, "A3400"),
    ("TK", 290.0, "T2900"),
    ("PPA", 100000.0, "PS100000"),
    ("ALPHA", -2.0, "AL-020"),
    ("BETA", 4.0, "BE+040"),
    ("ADVANCE_RATIO", 0.8, "J+080"),
    ("RPM", 1860.0, "RPM01860"),
    ("roll_rate", 2.5, "P+025"),
    ("pitch_rate", -2.5, "Q-025"),
    ("yaw_rate", 0.0, "R+000"),
)


@pytest.mark.parametrize(("key", "value", "written"), TABLE)
def test_name_field_matches_the_pyflightstream_code_table(key, value, written):
    assert name_field(key, value) == written


def test_point_name_matches_pyflightstream_examples():
    # test_naming.py: the worked example of a mixed row.
    order = ["MACH", "REmi", "ALPHA", "BETA", "ADVANCE_RATIO"]
    point = {"MACH": 0.144, "REmi": 4.38, "ALPHA": 0.0, "BETA": 0.0, "ADVANCE_RATIO": 0.8007}
    assert point_name(order, point) == "M144RE438AL+000BE+000J+080"
    # test_goal024_rpm.py: a swept rotor speed, the unsigned magnitude field.
    rpm = {"MACH": 0.144, "REmi": 4.38, "ALPHA": 0.0, "RPM": 600.0}
    assert point_name(["MACH", "REmi", "ALPHA", "RPM"], rpm) == "M144RE438AL+000RPM00600"
    # A real pyflightstream folder: sims/sim_7001/datapoints/DP-M120RE009AL+000BE+000J+100.
    real = {"MACH": 0.12, "REmi": 0.09, "ALPHA": 0.0, "BETA": 0.0, "ADVANCE_RATIO": 1.0}
    assert point_name(list(real), real) == "M120RE009AL+000BE+000J+100"


def test_the_rows_order_decides_the_name():
    # test_goal024_point_name_every_declared_variable_is_written_in_the_cell_s_order.
    point = {"MACH": 0.144, "ALPHA": 0.0}
    assert point_name(["MACH", "ALPHA"], point) == "M144AL+000"
    assert point_name(["ALPHA", "MACH"], point) == "AL+000M144"


def test_a_negative_unsigned_value_and_a_negative_rotor_speed_are_refused():
    # test_goal024_point_name.py and test_goal024_rpm.py refuse both.
    with pytest.raises(PointNameError):
        name_field("MACH", -0.1)
    with pytest.raises(PointNameError):
        name_field("RPM", -800.0)
    with pytest.raises(PointNameError):
        name_field("MACH", float("nan"))


def test_native_solver_keys_are_named_and_j_is_the_advance_ratio():
    order = ["V", "J", "BETA07", "RHO", "MU"]
    point = {"V": 20.0, "J": 1.0, "BETA07": 30.0, "RHO": 1.2, "MU": 1.8e-5}
    assert point_name(order, point) == "V20J+100BETA0730RHO1.2MU1.8e-05"
    assert point_name([], {}) == "NOCOND"


HEADER = "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"


def workspace(root: Path, rows: str) -> None:
    for kind in ("setups", "references", "pproc", "matrices", "geometries"):
        (root / "inputs" / kind).mkdir(parents=True, exist_ok=True)
    (root / "init.toml").write_text('schema_version = 1\nsolver = "mock"\n')
    (root / "inputs/setups/s002.toml").write_text("[solver]\niterations = 10\n")
    (root / "inputs/references/r003.toml").write_text("area = 1.5\n")
    (root / "inputs/pproc/p001.toml").write_text("history = true\n")
    (root / "inputs/matrices/study.fs").write_text(HEADER + rows)


def validate(row, inputs):
    return RowRequirements(
        actions=("solve",), outputs=("sims/sim_{pol}/datapoints/{point}/history.csv",)
    )


def ids(root: Path) -> list[str]:
    plan = plan_matrix(root, "study", validators={"mock": validate})
    return [point.point_id for point in plan.points]


def test_plan_names_each_point_like_pyflightstream(tmp_path):
    workspace(
        tmp_path,
        "7001|1|002|003|001|MACH:0.85, REmi:5, ALPHA:sweep|2.5,16\n"
        "7002|1|002|003|001|MACH:0.2, REmi:5, ALPHA:sweep|16\n"
        "7003|1|002|003|001|V:20, J:sweep, BETA07:30|1.0\n",
    )
    assert ids(tmp_path) == [
        "7001/DP-M850RE500AL+025",
        "7001/DP-M850RE500AL+160",
        "7002/DP-M200RE500AL+160",
        "7003/DP-V20J+100BETA0730",
    ]
    plan = plan_matrix(tmp_path, "study", validators={"mock": validate})
    first = plan.points[0]
    assert first.directory == "sims/sim_7001/datapoints/DP-M850RE500AL+025"
    assert first.outputs == ("sims/sim_7001/datapoints/DP-M850RE500AL+025/history.csv",)


def test_plan_names_are_deterministic_and_follow_the_cells_order(tmp_path):
    workspace(tmp_path, "7001|1|002|003|001|MACH:0.2, ALPHA:sweep|2\n")
    assert ids(tmp_path) == ids(tmp_path) == ["7001/DP-M200AL+020"]
    workspace(tmp_path, "7001|1|002|003|001|ALPHA:sweep, MACH:0.2|2\n")
    assert ids(tmp_path) == ["7001/DP-AL+020M200"]


def test_plan_refuses_two_sweep_values_that_round_to_one_name(tmp_path):
    # pyflightstream refuses at plan time and names both values
    # (test_goal024_point_name.py: "round to the same name").
    workspace(tmp_path, "7001|1|002|003|001|MACH:0.2, ADVANCE_RATIO:sweep|0.8007,0.8049\n")
    with pytest.raises(PlanError) as caught:
        ids(tmp_path)
    assert "round to the same point name" in str(caught.value)
    assert "J+080" in str(caught.value)
