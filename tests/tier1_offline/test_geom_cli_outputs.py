"""cfd-geom writes into inputs/geometry/<name>/ with geometry.json, and refuses abbreviations."""

from __future__ import annotations

import json

import pytest

from cfdbender.core.mesh import GEOMETRY_SIDECAR, generated_geometry_of
from cfdbender.geometry.cli import main

BODY = '[components.FUS]\ntype = "body"\nlength_mm = 2000.0\n'


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / "demo.geom.toml"
    path.write_text(BODY, encoding="utf-8")
    return path


def _run(args, capsys):
    capsys.readouterr()
    try:
        code = main([str(a) for a in args])
    except SystemExit as error:
        code = error.code
    return code, capsys.readouterr()


def test_geom_cli_outputs_run_writes_into_the_workspace_geometry_folder(demo, tmp_path, capsys):
    workspace = tmp_path / "ws"
    code, _ = _run(["--workspace", workspace, "run", demo, "--until", "S0"], capsys)
    assert code == 0
    folder = workspace / "inputs" / "geometry" / "demo.geom"
    assert (folder / "report.json").is_file()
    sidecar = json.loads((folder / GEOMETRY_SIDECAR).read_text(encoding="utf-8"))
    assert sidecar["command"] == "cfd-geom run"
    assert sidecar["status"] == "accepted" and sidecar["until"] == "S0"
    assert generated_geometry_of(workspace, "inputs/geometry/demo.geom/report.json") is not None


def test_geom_cli_outputs_name_and_out_choose_the_folder(demo, tmp_path, capsys):
    workspace = tmp_path / "ws"
    _run(["--workspace", workspace, "run", demo, "--until", "S0", "--name", "wing-a"], capsys)
    assert (workspace / "inputs" / "geometry" / "wing-a" / GEOMETRY_SIDECAR).is_file()
    _run(["run", demo, "--until", "S0", "--out", tmp_path / "elsewhere"], capsys)
    assert (tmp_path / "elsewhere" / GEOMETRY_SIDECAR).is_file()


def test_geom_cli_outputs_a_refused_run_writes_no_sidecar(tmp_path, capsys):
    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    code, _ = _run(["run", empty, "--out", tmp_path / "o"], capsys)
    assert code == 2
    assert not (tmp_path / "o" / GEOMETRY_SIDECAR).exists()


def test_geom_cli_outputs_doe_writes_one_geometry_per_named_set(demo, tmp_path, capsys):
    workspace = tmp_path / "ws"
    sets = tmp_path / "sets.toml"
    sets.write_text(
        "[[sets]]\nname = '1001'\n'FUS.length_mm' = 2500.0\n"
        "[[sets]]\nname = '1002'\n'FUS.length_mm' = 3000.0\n",
        encoding="utf-8",
    )
    code, _ = _run(["--workspace", workspace, "doe", demo, sets, "--until", "S0"], capsys)
    assert code == 0
    for name, length in (("1001", 2500.0), ("1002", 3000.0)):
        record = json.loads(
            (workspace / "inputs" / "geometry" / name / GEOMETRY_SIDECAR).read_text("utf-8")
        )
        assert record["command"] == "cfd-geom doe"
        assert record["overrides"] == {"FUS.length_mm": length}


def test_geom_cli_outputs_doe_refuses_a_set_name_that_is_not_one_folder(demo, tmp_path, capsys):
    sets = tmp_path / "sets.toml"
    sets.write_text("[[sets]]\nname = 'a/b'\n'FUS.length_mm' = 2500.0\n", encoding="utf-8")
    code, captured = _run(["doe", demo, sets, "--out", tmp_path / "d"], capsys)
    assert code == 2 and "folder name" in captured.err


@pytest.mark.parametrize(
    "args", [["run", "x.toml", "--unt", "S0"], ["--works", ".", "stages"], ["valid", "x"]]
)
def test_geom_cli_outputs_abbreviations_are_refused(args, capsys):
    code, _ = _run(args, capsys)
    assert code == 2
