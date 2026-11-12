"""P0000: mock.translate(point), execute(point), classify(result), collect(point)."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cfdbender.core.plan import Point, RowRequirements, plan_matrix


def api():
    module = importlib.import_module("cfdbender.solvers.mock")
    if hasattr(module, "execute"):
        return module
    return SimpleNamespace(
        validate_row=lambda row, workspace: RowRequirements(
            actions=("solve",), outputs=("sims/sim_{pol}/datapoints/{point}/history.csv",)
        ),
        translate=lambda point: "",
        execute=lambda point, **kw: SimpleNamespace(exit_code=0, history=()),
        classify=lambda result: None,
        collect=lambda point: None,
    )


def workspace(root: Path, outcome: str = "converged") -> Point:
    files = {
        "init.toml": "schema_version = 1" + chr(10) + 'solver = "mock"',
        "inputs/setups/s001.toml": chr(10).join(
            ["[mock]", f'outcome = "{outcome}"', "iterations = 4"]
        ),
        "inputs/references/r001.toml": chr(10).join(
            [
                "area = 2.0",
                "length = 3.0",
                "moment_origin = [1.0, 2.0, 3.0]",
                'axis_system = "body"',
                'axis_directions = ["aft", "right", "up"]',
            ]
        ),
        "inputs/pproc/p001.toml": "history = true",
        "inputs/matrices/study.fs": "POL|RUN|SET|REF|PPROC" + chr(10) + "1001|1|001|001|001",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((text + chr(10)).encode())
    return Point(
        "1001/DP-test",
        "1001",
        "s001",
        "r001",
        "p001",
        {"ALPHA": 2.0},
        None,
        "sims/sim_1001/datapoints/DP-test",
        ("write input", "solve", "read history"),
        tuple(
            f"sims/sim_1001/datapoints/DP-test/{name}"
            for name in ("mock.in", "history.csv", "execution.json")
        ),
        {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
    )


def prepared(root: Path, outcome: str = "converged"):
    point = workspace(root, outcome)
    try:
        module = importlib.import_module("cfdbender.core.adapter")
    except ModuleNotFoundError:
        return SimpleNamespace(root=root, point=point)
    return module.load_point(root, point, solver="mock")


@pytest.mark.parametrize("outcome", ["converged", "iteration_limit", "failed"])
def test_p0000_item3c_mock_roundtrip(tmp_path, outcome):
    point = prepared(tmp_path, outcome)
    adapter = api()
    translated = adapter.translate(point)
    assert translated, "translation must describe the s/r/p inputs"
    assert translated.isascii()
    assert adapter.translate(point) == translated
    calls = []
    result = adapter.execute(point, check_ram=lambda: calls.append(1))
    assert calls, "execute must check RAM before and during work"
    assert adapter.classify(result) == outcome
    collected = adapter.collect(point)
    assert collected.classification == outcome
    assert collected.history == result.history
    assert collected.solver_version == "mock-1"
    assert collected.references.area == 2
    assert collected.references.length == 3
    assert collected.references.moment_origin == (1, 2, 3)
    assert collected.references.axis_system == "body"
    assert collected.references.axis_directions == ("aft", "right", "up")
    assert set(collected.coefficients) == {"CX", "CY", "CZ", "Cl", "Cm", "Cn"}
    assert {c.axis for c in collected.coefficients.values()} == {"x", "y", "z"}
    assert (tmp_path / point.point.directory / "mock.in").read_text() == translated
    assert (tmp_path / point.point.directory / "history.csv").is_file()
    assert json.loads((tmp_path / point.point.directory / "execution.json").read_text())


def test_p0000_item3c_exit_success_is_not_convergence(tmp_path):
    adapter = api()
    result = adapter.execute(prepared(tmp_path, "iteration_limit"), check_ram=lambda: None)
    assert adapter.classify(result) == "iteration_limit"
    assert result.exit_code == 0
    assert adapter.classify(replace(result, history=())) == "failed"


@pytest.mark.parametrize("problem", ["outcome", "iterations", "area", "axis_system", "history"])
def test_p0000_item3c_plan_hook_names_invalid_input(tmp_path, problem):
    workspace(tmp_path)
    names = {
        "outcome": "setups/s001",
        "iterations": "setups/s001",
        "area": "references/r001",
        "axis_system": "references/r001",
        "history": "pproc/p001",
    }
    path = tmp_path / f"inputs/{names[problem]}.toml"
    lines = path.read_text().splitlines()
    lines = [line for line in lines if not line.startswith(problem + " =")]
    path.write_text(chr(10).join(lines) + chr(10))
    with pytest.raises(ValueError, match=problem):
        plan_matrix(tmp_path, "study", validators={"mock": api().validate_row})


def test_p0000_item3c_plan_uses_real_mock_hook_without_writes(tmp_path):
    workspace(tmp_path)
    from cfdbender.matrix.cli import main

    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_p0000_item3c_init_and_drift_refused(tmp_path):
    point = prepared(tmp_path)
    (tmp_path / "init.toml").write_text("schema_version = 1" + chr(10) + 'solver = "other"')
    with pytest.raises(ValueError, match="init.toml"):
        api().translate(point)


def test_p0000_item3c_direct_execute_never_overwrites(tmp_path):
    point = prepared(tmp_path)
    adapter = api()
    adapter.execute(point, check_ram=lambda: None)
    with pytest.raises((ValueError, FileExistsError), match="exist|record"):
        adapter.execute(point, check_ram=lambda: None)
