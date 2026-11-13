"""Adapter contract, executable selection, staging, timeout and archive clauses.

Six adapters, offline stand-ins; each test docstring names its clause of
docs/adapter-contract.md. The mock adapter is skipped where it launches nothing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_conformance_workspace import SOLVERS, case, local, prepared, state_no_build  # noqa: F401

from cfdbender.core.adapter import load_point
from cfdbender.core.executables import bind_executable
from cfdbender.matrix.cli import main
from cfdbender.solvers import load_solver

HANG = "import time\ntime.sleep(60)\n"


def setup_path(root: Path) -> Path:
    return root / "inputs/setups/s001.toml"


def declared_executable(text: str) -> str:
    match = re.search(r'^executable = "([^"]+)"', text, re.M)
    assert match, text
    return match.group(1)


def point_folder(root: Path, point) -> Path:
    return root / point.point.directory


def staged_copy(root: Path, point, solver: str, tmp_path_factory) -> tuple[Path, set[str]]:
    """Stage the point in a spare copy of the workspace; return it and the staged names."""
    spare = tmp_path_factory.mktemp("staged") / "ws"
    shutil.copytree(root, spare)
    load_solver(solver).stage(load_point(spare, point.point, solver=solver))
    folder = spare / point.point.directory
    return spare, {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def test_c25_a_solver_past_its_timeout_is_stopped_and_failed(case, tmp_path):  # noqa: F811
    """C-25: a solver that never finishes is stopped at timeout_s and classified failed."""
    solver, root = case
    if solver == "mock":
        pytest.skip("the mock adapter launches no solver process")
    hang = tmp_path / "hang.py"
    hang.write_text(HANG, encoding="ascii")
    path = setup_path(root)
    text = path.read_text(encoding="ascii")
    old = declared_executable(text)
    path.write_text(
        text.replace(f'executable = "{old}"', f'executable = "{hang.as_posix()}"\ntimeout_s = 0.5'),
        encoding="ascii",
    )
    adapter, point = prepared(root, solver)
    try:
        result = adapter.execute(point, check_ram=lambda: None)
    except (ValueError, OSError) as error:
        assert "timed out" in str(error).lower() or "timeout" in str(error).lower(), error
        return
    assert adapter.classify(result) == "failed", solver
    record = point_folder(root, point) / "execution.json"
    assert json.loads(record.read_bytes()).get("timed_out") is True, solver


def test_c29_a_missing_executable_is_refused_before_any_write(case):  # noqa: F811
    """C-29: an executable that does not exist is refused and no point folder is written."""
    solver, root = case
    if solver == "mock":
        pytest.skip("the mock adapter has no executable")
    path = setup_path(root)
    text = path.read_text(encoding="ascii")
    old = declared_executable(text)
    path.write_text(
        text.replace(f'executable = "{old}"', 'executable = "absent-solver-87"'), encoding="ascii"
    )
    adapter, point = prepared(root, solver)
    with pytest.raises((ValueError, OSError)):
        adapter.execute(point, check_ram=lambda: None)
    assert not point_folder(root, point).exists(), solver


def test_c29_c49_a_row_build_selects_the_declared_executable(case):  # noqa: F811
    """C-29 and C-49: SOLVER_BUILD binds the declared path to the option the adapter reads."""
    solver, root = case
    if solver == "mock":
        pytest.skip("the mock adapter has no executable")
    adapter, point = prepared(root, solver)
    (root / "inputs/executables.toml").write_text(
        f'[{solver}]\n"1.0" = "/declared/by/build/solver"\n', encoding="ascii"
    )
    keys = getattr(adapter, "EXECUTABLE_KEYS", None) or {"path": "executable"}
    bound = bind_executable(point, adapter)
    assert (
        Path(bound.setup["solver"][keys["path"]]).as_posix().endswith("declared/by/build/solver")
    ), solver


def test_c46_stage_refuses_an_existing_point_folder(case):  # noqa: F811
    """C-46: a second stage of the same point is refused and changes nothing."""
    solver, root = case
    adapter, point = prepared(root, solver)
    adapter.stage(point)
    assert point_folder(root, point).is_dir()
    before = {p.name: p.read_bytes() for p in point_folder(root, point).rglob("*") if p.is_file()}
    with pytest.raises((ValueError, OSError)):
        adapter.stage(point)
    after = {p.name: p.read_bytes() for p in point_folder(root, point).rglob("*") if p.is_file()}
    assert after == before


def test_c46_stage_runs_no_solver(case, monkeypatch):  # noqa: F811
    """C-46: staging launches no process."""
    solver, root = case
    adapter, point = prepared(root, solver)

    def refuse(*args, **kwargs):
        raise AssertionError("stage launched a process")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    adapter.stage(point)


def test_c45_the_staged_config_is_among_the_staged_files(case, tmp_path_factory):  # noqa: F811
    """C-45: stage writes the file the adapter names as STAGED_CONFIG."""
    solver, root = case
    adapter, point = prepared(root, solver)
    _, names = staged_copy(root, point, solver, tmp_path_factory)
    assert adapter.STAGED_CONFIG in names, (solver, sorted(names))


def test_c20_a_cluster_folder_holds_the_bytes_a_local_run_reads(case, tmp_path_factory):  # noqa: F811
    """C-20: every staged input equals the file a local run of the same point starts from."""
    solver, root = case
    adapter, point = prepared(root, solver)
    spare, names = staged_copy(root, point, solver, tmp_path_factory)
    staged = spare / point.point.directory
    adapter.execute(point, check_ram=lambda: None)
    local_folder = point_folder(root, point)
    differing = [
        name
        for name in sorted(names)
        if not name.endswith("execution.json")
        and not (local_folder / name).exists()
        or not name.endswith("execution.json")
        and (local_folder / name).read_bytes() != (staged / name).read_bytes()
    ]
    assert not differing, (solver, differing)


def test_c18_c20_staged_text_is_the_same_bytes_on_every_platform(case, tmp_path_factory):  # noqa: F811
    """C-18 and C-20: no staged file carries a carriage return the translation did not write."""
    solver, root = case
    adapter, point = prepared(root, solver)
    spare, names = staged_copy(root, point, solver, tmp_path_factory)
    folder = spare / point.point.directory
    carriage = [
        name
        for name in sorted(names)
        if (folder / name).is_file()
        and (folder / name).suffix not in {".pyfrm", ".su2", ".msh", ".cgns"}
        and b"\r" in (folder / name).read_bytes()
    ]
    assert not carriage, (solver, carriage)


def test_c52_an_archived_point_keeps_its_linked_mesh(case, tmp_path):  # noqa: F811
    """C-52: after a forced rerun the archived point folder's mesh still resolves to the input."""
    solver, root = case
    if solver not in ("su2", "openfoam", "pyfr"):
        pytest.skip("only su2, openfoam and pyfr link a mesh into the point folder")
    adapter, point = prepared(root, solver)
    mesh = root / point.geometry
    state_no_build(root)
    assert main(["run", "full", "--workspace", str(root), "--local"]) == 0
    assert main(["run", "full", "--workspace", str(root), "--local", "--force-rerun-all"]) == 0
    archived = [
        p for p in (root / "archive").rglob("*") if p.name == f"mesh{mesh.suffix}"
    ]
    assert archived, solver
    assert all(p.is_file() and p.samefile(mesh) for p in archived), solver


def test_c52_probe_rejects_a_replaced_mesh_of_different_size(case, tmp_path, monkeypatch):  # noqa: F811
    from cfdbender.core.registry import RunRegistry

    solver, root = case
    if solver not in ("su2", "openfoam", "pyfr"):
        pytest.skip("this adapter does not stage a mesh link")
    _, point = prepared(root, solver)
    mesh = root / point.geometry
    original = RunRegistry.archive

    def corrupt(self, *args, **kwargs):
        folder = original(self, *args, **kwargs)
        for path in folder.rglob("*"):
            if path.is_file() and path.samefile(mesh):
                path.unlink()
                path.write_bytes(b"broken mesh with a different size")
        return folder

    monkeypatch.setattr(RunRegistry, "archive", corrupt)
    with pytest.raises(AssertionError):
        test_c52_an_archived_point_keeps_its_linked_mesh(case, tmp_path)
