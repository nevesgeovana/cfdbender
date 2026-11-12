"""P0000 item 3B: plan_matrix(root, 'study', validators={'mock': validate})."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


def plan_api():
    if importlib.util.find_spec("cfdbender.core.plan") is None:
        return SimpleNamespace(
            plan_matrix=lambda *a, **kw: SimpleNamespace(points=()),
            PlanError=ValueError,
            RowRequirements=lambda **kw: SimpleNamespace(**kw),
        )
    return importlib.import_module("cfdbender.core.plan")


def study(root: Path) -> Path:
    for kind in ("setups", "references", "pproc", "matrices", "geometries"):
        (root / "inputs" / kind).mkdir(parents=True, exist_ok=True)
    (root / "init.toml").write_text('schema_version = 1\nsolver = "mock"\n')
    (root / "inputs/setups/s002.toml").write_text("[solver]\niterations = 10\n")
    (root / "inputs/references/r003.toml").write_text("area = 1.5\n")
    (root / "inputs/pproc/p001.toml").write_text("history = true\n")
    (root / "inputs/geometries/mesh.dat").write_bytes(b"synthetic mesh")
    path = root / "inputs/matrices/study.fs"
    path.write_text(
        "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES|WALLTIME\n"
        "1001|1|002|003|001|MACH:0.2, ALPHA:sweep|-2,0,2|4h\n"
    )
    return path


def validator(row, inputs):
    assert row.pol == "1001"
    assert inputs.setup["solver"]["iterations"] == 10
    assert inputs.reference["area"] == 1.5
    assert inputs.post["history"] is True
    assert inputs.read_bytes("inputs/geometries/mesh.dat") == b"synthetic mesh"
    return plan_api().RowRequirements(
        actions=("write input", "solve", "read history"),
        outputs=("sims/sim_{pol}/datapoints/{point}/history.csv",),
    )


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def cli_process(args):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    return subprocess.run(
        [sys.executable, "-m", "cfdbender.matrix.cli", *args],
        env=env,
        capture_output=True,
        text=True,
    )


def test_p0000_item3b_plan_expands_points_and_hashes_every_input_without_writing(tmp_path):
    study(tmp_path)
    before = snapshot(tmp_path)
    plan = plan_api().plan_matrix(tmp_path, "study", validators={"mock": validator})
    assert len(plan.points) == 3, "planning must expand every sweep value"
    assert [p.condition["ALPHA"] for p in plan.points] == [-2.0, 0.0, 2.0]
    assert len({p.point_id for p in plan.points}) == 3
    assert plan.solver == "mock"
    for point in plan.points:
        assert (point.setup_id, point.reference_id, point.post_id) == ("s002", "r003", "p001")
        assert point.actions == ("write input", "solve", "read history")
        assert dict(point.inputs_sha256) == {
            k: hashlib.sha256(v).hexdigest() for k, v in before.items()
        }
    assert snapshot(tmp_path) == before


def test_p0000_item3b_point_identity_survives_condition_spelling_not_order(tmp_path):
    path = study(tmp_path)
    api = plan_api()
    first = api.plan_matrix(tmp_path, "study.fs", validators={"mock": validator})
    assert len(first.points) == 3
    path.write_text(
        path.read_text()
        .replace("MACH:0.2, ALPHA:sweep", "mach:0.20, alpha:sweep")
        .replace("-2,0,2", "2.0,-2.0,-0.0")
    )
    second = api.plan_matrix(tmp_path, "study", validators={"mock": validator})
    assert {p.condition["ALPHA"]: p.point_id for p in first.points} == {
        p.condition["ALPHA"]: p.point_id for p in second.points
    }
    # The name follows the row's order, as pyflightstream's does.
    path.write_text(path.read_text().replace("mach:0.20, alpha:sweep", "alpha:sweep, mach:0.20"))
    third = api.plan_matrix(tmp_path, "study", validators={"mock": validator})
    assert {p.point_id for p in third.points} == {
        "1001/DP-AL-020M200",
        "1001/DP-AL+000M200",
        "1001/DP-AL+020M200",
    }


@pytest.mark.parametrize(
    "missing",
    [
        "init.toml",
        "inputs/setups/s002.toml",
        "inputs/references/r003.toml",
        "inputs/pproc/p001.toml",
    ],
)
def test_p0000_item3b_missing_input_names_row_and_file(tmp_path, missing):
    study(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(plan_api().PlanError) as caught:
        plan_api().plan_matrix(tmp_path, "study", validators={"mock": validator})
    assert missing in str(caught.value).replace(chr(92), "/")
    if missing != "init.toml":
        assert "1001" in str(caught.value)


@pytest.mark.parametrize("name", ["../study", "study.xlsx", "absent"])
def test_p0000_item3b_matrix_must_be_named_in_its_workspace(tmp_path, name):
    study(tmp_path)
    with pytest.raises(plan_api().PlanError):
        plan_api().plan_matrix(tmp_path, name, validators={"mock": validator})


def test_p0000_item3b_solver_validation_refusal_names_row(tmp_path):
    study(tmp_path)

    def refuse(row, inputs):
        raise ValueError("setup needs a mesh")

    with pytest.raises(plan_api().PlanError, match="1001.*setup needs a mesh"):
        plan_api().plan_matrix(tmp_path, "study", validators={"mock": refuse})


def test_p0000_item3b_unknown_solver_hook_refused(tmp_path):
    study(tmp_path)
    with pytest.raises(plan_api().PlanError, match="mock"):
        plan_api().plan_matrix(tmp_path, "study", validators={})


def test_p0000_item3b_invalid_toml_names_row_and_file(tmp_path):
    study(tmp_path)
    (tmp_path / "inputs/setups/s002.toml").write_text("broken = [")
    with pytest.raises(plan_api().PlanError, match="1001.*s002.toml.*TOML"):
        plan_api().plan_matrix(tmp_path, "study", validators={"mock": validator})


def test_p0000_item3b_colliding_output_paths_refused_before_writing(tmp_path):
    study(tmp_path)
    before = snapshot(tmp_path)

    def same_path(row, inputs):
        return plan_api().RowRequirements(actions=("solve",), outputs=("post/shared.csv",))

    with pytest.raises(plan_api().PlanError, match="shared.csv.*point"):
        plan_api().plan_matrix(tmp_path, "study", validators={"mock": same_path})
    assert snapshot(tmp_path) == before


def test_p0000_item3b_adapter_cannot_read_outside_workspace(tmp_path):
    study(tmp_path)

    def escape(row, inputs):
        inputs.read_bytes("../private.dat")

    with pytest.raises(plan_api().PlanError, match="outside"):
        plan_api().plan_matrix(tmp_path, "study", validators={"mock": escape})


def test_p0000_item3b_cli_prints_plan_and_writes_nothing(tmp_path, monkeypatch, capsys):
    study(tmp_path)
    before = snapshot(tmp_path)
    try:
        cli = importlib.import_module("cfdbender.matrix.cli")
        composition = importlib.import_module("cfdbender.workspace.planning")
    except ModuleNotFoundError:
        cli = SimpleNamespace(main=lambda *a: 2)
        composition = SimpleNamespace(load_solver=None)
    monkeypatch.setattr(
        composition, "load_solver", lambda name: SimpleNamespace(validate_row=validator)
    )
    assert cli.main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    for token in ("study.fs", "mock", "3 point", "1001", "s002", "r003", "p001", "solve", "4h"):
        assert token in output
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("args", [["--help"], ["plan", "--help"]])
def test_p0000_item3b_cli_help(args):
    result = cli_process(args)
    assert result.returncode == 0, result.stderr
    assert "cfd-matrix" in result.stdout


def test_p0000_item3b_cli_outside_workspace_names_init(tmp_path):
    result = cli_process(["plan", "study", "--workspace", str(tmp_path)])
    assert result.returncode == 2
    assert "init.toml" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_p0000_item3b_matrix_console_script_declared():
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    scripts = tomllib.loads(project.read_text())["project"]["scripts"]
    assert scripts.get("cfd-matrix") == "cfdbender.matrix.cli:main"
