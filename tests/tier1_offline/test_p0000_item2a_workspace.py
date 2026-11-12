"""Initialize a mock solver workspace with the command-line interface."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KINDS = (
    "geometries",
    "references",
    "setups",
    "pproc",
    "profiles",
    "freestreams",
    "calibrations",
    "hpc",
    "fsi",
    "matrices",
    "management",
)


def _cli(*args, cwd=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "cfdbender.workspace.cli", *map(str, args)],
        cwd=cwd or ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("args", [("--help",), ("init", "--help")])
def test_p0000_item2a_help(args):
    result = _cli(*args)
    assert result.returncode == 0, result.stderr
    assert "cfd-workspace" in result.stdout
    assert "init" in result.stdout
    assert "pyflightstream" not in result.stdout + result.stderr


def test_p0000_item2a_entry_point_is_declared():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"].get("scripts", {}).get("cfd-workspace") == (
        "cfdbender.workspace.cli:main"
    )


@pytest.mark.parametrize("style", ["positional", "workspace", "cwd", "both", "equivalent"])
def test_p0000_item2a_init_creates_root_declaration_and_empty_layout(tmp_path, style):
    target = tmp_path / "campaign"
    args = ["init", "--solver", "mock"]
    cwd = None
    if style == "positional":
        args += [target]
    elif style == "workspace":
        args += ["--workspace", target]
    elif style == "cwd":
        target.mkdir()
        cwd = target
    elif style == "both":
        args += [target, "--workspace", target]
    else:
        args += [target / ".", "--workspace", target]
    result = _cli(*args, cwd=cwd)
    assert result.returncode == 0, result.stderr
    assert tomllib.loads((target / "init.toml").read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "solver": "mock",
    }
    for kind in KINDS:
        assert (target / "inputs" / kind).is_dir(), kind
        assert list((target / "inputs" / kind).iterdir()) == []
        assert f"inputs/{kind}/" in result.stdout
    assert {p.name for p in (target / "inputs").iterdir()} == {*KINDS, ".gitignore"}
    assert (target / "inputs/.gitignore").read_text() == "executables.local.toml\n"
    assert {p.name for p in target.iterdir()} == {"init.toml", "inputs", "sims", "post", "archive"}
    assert "campaign workspace ready at" in result.stdout
    assert str(target.resolve()) in result.stdout
    assert "init.toml" in result.stdout and "mock" in result.stdout
    assert result.stderr == ""
    assert "pyflightstream" not in result.stdout


def test_p0000_item2a_nonempty_target_preserves_existing_inputs(tmp_path):
    target = tmp_path / "campaign"
    setups = target / "inputs" / "setups"
    setups.mkdir(parents=True)
    existing = setups / "s001.toml"
    existing.write_bytes(b"existing setup")
    notes = target / "notes.txt"
    notes.write_bytes(b"keep")
    result = _cli("init", target, "--solver", "mock")
    assert result.returncode == 0, result.stderr
    assert existing.read_bytes() == b"existing setup"
    assert notes.read_bytes() == b"keep"
    assert (target / "init.toml").is_file()


@pytest.mark.parametrize("solver", ["no-such-solver", "missing", "../mock"])
def test_p0000_item2a_unknown_solver_refused_before_writing(tmp_path, solver):
    target = tmp_path / "campaign"
    result = _cli("init", target, "--solver", solver)
    assert result.returncode == 2
    assert f"unknown solver {solver!r}" in result.stderr
    assert (
        "available solvers: flightstream, mock, openfoam, pyfr, qprop, su2, xfoil" in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert "pyflightstream" not in result.stderr
    assert not target.exists()


@pytest.mark.parametrize("original", [b"existing declaration", b"solver = ["])
def test_p0000_item2a_existing_init_refused_without_changes(tmp_path, original):
    marker = tmp_path / "init.toml"
    marker.write_bytes(original)
    result = _cli("init", tmp_path, "--solver", "mock")
    assert result.returncode == 2
    assert "init.toml" in result.stderr and "already exists" in result.stderr
    assert marker.read_bytes() == original
    assert list(tmp_path.iterdir()) == [marker]


@pytest.mark.parametrize("collision", ["root", "inputs", "setups", "post"])
def test_p0000_item2a_bad_layout_refused_with_exit_2(tmp_path, collision):
    target = tmp_path / "campaign"
    paths = {
        "root": target,
        "inputs": target / "inputs",
        "setups": target / "inputs/setups",
        "post": target / "post",
    }
    taken = paths[collision]
    taken.parent.mkdir(parents=True, exist_ok=True)
    taken.write_bytes(b"not a folder")
    result = _cli("init", target, "--solver", "mock")
    assert result.returncode == 2
    assert str(taken) in result.stderr
    assert "Traceback" not in result.stderr
    assert taken.read_bytes() == b"not a folder"
    assert not (target / "init.toml").exists()


def test_p0000_item2a_conflicting_workspace_names_refused(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    result = _cli("init", first, "--workspace", second, "--solver", "mock")
    assert result.returncode == 2
    assert "the campaign is named twice" in result.stderr
    assert "Name it once." in result.stderr
    assert not first.exists() and not second.exists()


@pytest.mark.parametrize("args", [(), ("init",), ("init", "--sol", "mock")])
def test_p0000_item2a_parser_refuses_missing_arguments_and_abbreviations(args):
    result = _cli(*args)
    assert result.returncode == 2
    assert "usage: cfd-workspace" in result.stderr
    assert "Traceback" not in result.stderr
