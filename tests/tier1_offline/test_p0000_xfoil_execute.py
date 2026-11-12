"""P0000: run, collect and post XFOIL points using an independent polar oracle."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_p0000_xfoil_plan import FAKE, adapter, prepared, tree, workspace

from cfdbender.core.adapter import load_point
from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main as matrix_cli
from cfdbender.workspace.planning import plan_workspace


def oracle(path, alpha):
    """Read space-delimited XFOIL output independently of production fixed widths."""
    with path.open(encoding="ascii", newline="") as stream:
        lines = list(csv.reader(stream, delimiter=" ", skipinitialspace=True))
    header = next(i for i, row in enumerate(lines) if row and row[0] == "alpha")
    for cells in lines[header + 2 :]:
        cells = [cell for cell in cells if cell]
        if cells and float(cells[0]) == alpha:
            return dict(zip(("CL", "CD", "CDp", "CM"), map(float, cells[1:5]), strict=True))
    raise AssertionError(f"no oracle row for alpha {alpha}")


def test_p0000_xfoil_run_collect_post_sweep(tmp_path):
    workspace(tmp_path)
    plan = plan_workspace(tmp_path, "study")
    assert matrix_cli(["run", "study", "--workspace", str(tmp_path), "--local"]) == 0
    records = RunRegistry(tmp_path).read()
    assert len(records) == 3
    assert all(r.status == "COMPLETED" and r.classification == "converged" for r in records)
    assert matrix_cli(["collect", "study", "--workspace", str(tmp_path)]) == 0
    summary_path = tmp_path / "post/study/campaign_summary.json"
    summary = json.loads(summary_path.read_bytes())
    assert summary["solver"] == "xfoil"
    assert len(summary["points"]) == 3
    for point, row in zip(plan.points, summary["points"], strict=True):
        folder = tmp_path / point.directory
        expected = oracle(folder / "polar.txt", point.condition["ALPHA"])
        assert {k: v["value"] for k, v in row["result"]["coefficients"].items()} == expected
        assert row["classification"] == "converged" and row["error"] is None
        result = adapter().collect(load_point(tmp_path, point, solver="xfoil"))
        assert result.references.length == 1 and result.references.area == 1
        assert result.references.moment_origin == (0.25, 0, 0)
        assert result.coefficients["CL"].axis == "z"
        assert result.coefficients["CD"].axis == result.coefficients["CDp"].axis == "x"
        assert result.coefficients["CM"].axis == "y" and result.coefficients["CM"].kind == "moment"
        assert result.solver_version == "6.99"
        assert "stand-in diagnostic" in (folder / "stderr.log").read_text()
        metadata = json.loads((folder / "execution.json").read_bytes())
        assert metadata["argv"] == [sys.executable, str(FAKE)]
        assert metadata["exit_code"] == 0 and metadata["duration_s"] >= 0
        assert metadata["version_line"] == "XFOIL Version 6.99"
    before = tree(tmp_path)
    assert matrix_cli(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert tree(tmp_path) == before
    assert matrix_cli(["run", "study", "--workspace", str(tmp_path), "--local"]) == 2
    assert tree(tmp_path) == before


def test_p0000_xfoil_failed_cli_collect_post(tmp_path, monkeypatch):
    workspace(tmp_path, condition="alpha:4", sweep="")
    monkeypatch.setenv("CFDBENDER_XFOIL_FAKE_MODE", "failed")
    assert matrix_cli(["run", "study", "--workspace", str(tmp_path), "--local"]) == 2
    assert matrix_cli(["collect", "study", "--workspace", str(tmp_path)]) == 1
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    row = summary["points"][0]
    assert row["status"] == "COMPLETED" and row["classification"] == "failed"
    assert row["result"]["coefficients"] == {}


@pytest.mark.parametrize("mode", ["failed", "exit_only", "wrong_alpha", "nonzero"])
def test_p0000_xfoil_no_requested_polar_row_is_failed(tmp_path, monkeypatch, mode):
    workspace(tmp_path, condition="alpha:4", sweep="")
    point = prepared(tmp_path)
    monkeypatch.setenv("CFDBENDER_XFOIL_FAKE_MODE", mode)
    checks = []
    result = adapter().execute(point, check_ram=lambda: checks.append(1))
    assert len(checks) >= 3
    assert adapter().classify(result) == "failed"
    collected = adapter().collect(point)
    assert collected.classification == "failed"
    if mode == "failed":
        assert result.exit_code == 0
        assert (
            "VISCAL:  Convergence failed"
            in (tmp_path / point.point.directory / "stdout.log").read_text()
        )


def test_p0000_xfoil_timeout_and_ram_cleanup(tmp_path, monkeypatch):
    workspace(tmp_path, condition="alpha:4", sweep="", options="timeout_s = 0.4\n")
    point = prepared(tmp_path)
    monkeypatch.setenv("CFDBENDER_XFOIL_FAKE_MODE", "timeout")
    result = adapter().execute(point, check_ram=lambda: None)
    assert adapter().classify(result) == "failed" and result.exit_code == 124
    folder = tmp_path / point.point.directory
    meta = json.loads((folder / "execution.json").read_bytes())
    assert meta["timed_out"] is True
    assert "XFOIL" in (folder / "stdout.log").read_text()
    assert (folder / "stderr.log").exists()
    with pytest.raises(ValueError, match="existing"):
        adapter().execute(point, check_ram=lambda: None)


def test_p0000_xfoil_ram_guard_during_process(tmp_path, monkeypatch):
    workspace(tmp_path, condition="alpha:4", sweep="")
    point = prepared(tmp_path)
    monkeypatch.setenv("CFDBENDER_XFOIL_FAKE_MODE", "timeout")
    checks = []

    def guard():
        checks.append(1)
        if len(checks) == 4:
            raise ValueError("RAM floor threatened")

    with pytest.raises(ValueError, match="RAM"):
        adapter().execute(point, check_ram=guard)
    meta = json.loads((tmp_path / point.point.directory / "execution.json").read_bytes())
    assert meta["exit_code"] != 0


def test_p0000_xfoil_low_ram_refuses_before_output(tmp_path, monkeypatch):
    workspace(tmp_path)
    point = prepared(tmp_path)
    from cfdbender.solvers.su2 import _ram

    monkeypatch.setattr(_ram, "free_ram_gib", lambda: 0.6)
    with pytest.raises(ValueError, match="RAM"):
        adapter().execute(point, check_ram=lambda: None)
    assert not (tmp_path / point.point.directory).exists()


def test_p0000_xfoil_file_airfoil_staged_without_rewriting(tmp_path):
    workspace(
        tmp_path,
        setup='airfoil_file="inputs/airfoils/profile.dat"\n',
        condition="alpha:4",
        sweep="",
    )
    source = tmp_path / "inputs/airfoils/profile.dat"
    source.parent.mkdir()
    source.write_bytes(b"NACA 0012\n1 0\n0 0\n1 0\n")
    point = prepared(tmp_path)
    result = adapter().execute(point, check_ram=lambda: None)
    assert adapter().classify(result) == "converged"
    assert (tmp_path / point.point.directory / "airfoil.dat").read_bytes() == source.read_bytes()


def test_p0000_xfoil_classification_does_not_use_iteration_limit(tmp_path):
    workspace(tmp_path, condition="alpha:4", sweep="")
    result = adapter().execute(prepared(tmp_path), check_ram=lambda: None)
    assert adapter().classify(replace(result, iteration_limit=1)) == "converged"
    assert adapter().classify(replace(result, history=())) == "failed"
    assert adapter().classify(replace(result, exit_code=1)) == "failed"


def test_p0000_xfoil_mutant_reader_wrong_cl_column_fails(tmp_path):
    module = adapter()
    import cfdbender.solvers.xfoil._polar as reader

    fixture = FAKE.with_name("polar.txt")
    expected = oracle(fixture, 4)
    assert {key: value.value for key, value in reader.read_polar(fixture, 4).items()} == expected
    source = Path(reader.__file__).read_text(encoding="utf-8")
    assert '"CL": 1' in source, "mutation must touch the production CL column mapping"
    scratch = tmp_path / "mutant_polar.py"
    scratch.write_text(source.replace('"CL": 1', '"CL": 2', 1), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("xfoil_mutant", scratch)
    mutant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutant)
    with pytest.raises(AssertionError):
        actual = {key: value.value for key, value in mutant.read_polar(fixture, 4).items()}
        assert actual == expected
    assert module.SOLVER_NAME == "xfoil"


@pytest.mark.parametrize("mode", ["truncated", "nan", "wrong_alpha"])
def test_p0000_xfoil_invalid_polar_cannot_converge(tmp_path, mode):
    workspace(tmp_path, condition="alpha:4", sweep="")
    point = prepared(tmp_path)
    adapter().execute(point, check_ram=lambda: None)
    polar = tmp_path / point.point.directory / "polar.txt"
    text = polar.read_text(encoding="ascii")
    if mode == "truncated":
        text = text.rsplit(chr(10), 2)[0] + "\n   4.000   0.4278\n"
    elif mode == "nan":
        text = text.replace("0.4278", "   nan")
    else:
        text = text.replace("4.000", "6.000")
    polar.write_text(text, encoding="ascii")
    assert adapter().collect(point).classification == "failed"
