"""P0000: route='pysu2' isolates the serial driver and preserves its evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_p0000_item3d_config import api, plan, prepared, workspace
from test_p0000_item3d_execute import FAKE

from cfdbender.core.run import execute_plan
from cfdbender.solvers.su2 import routes

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/su2"
SOURCE = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    directory = tmp_path / "wrapper install"
    directory.mkdir()
    shutil.copyfile(FIXTURES / "fake_pysu2.py", directory / "pysu2.py")
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(directory), str(SOURCE))))
    monkeypatch.delenv("CFDBENDER_PYSU2_PATH", raising=False)
    return directory


def case(root, outcome="converged", *, route="pysu2", extra="", timeout=5):
    workspace(
        root,
        extra=chr(10).join(
            [
                f'FAKE_CASE = "{outcome}"',
                "[solver]",
                *([f'route = "{route}"'] if route else []),
                "threads = 1",
                f"timeout_s = {timeout}",
                extra,
            ]
        ),
    )
    return prepared(root)


def execute(point):
    failure = None
    result = None
    try:
        result = api().execute(point, check_ram=lambda: None)
    except ValueError as error:
        failure = str(error)
    assert failure is None, f"pysu2 must execute in its worker: {failure}"
    return result


@pytest.mark.parametrize("outcome", ["converged", "iteration_limit", "failed", "missing"])
def test_p0000_item3g_child_lifecycle(tmp_path, wrapper, outcome):
    point = case(tmp_path / "workspace", outcome)
    result = execute(point)
    expected = "failed" if outcome == "missing" else outcome
    assert api().classify(result) == expected
    folder = point.root / point.point.directory
    metadata = json.loads((folder / "execution.json").read_text())
    assert metadata["route"] == "pysu2"
    assert result.exit_code == (1 if outcome == "failed" else 0)
    stdout = (folder / "stdout.log").read_text()
    assert f"worker pid={os.getpid()}" not in stdout
    assert "worker pid=" in stdout and "OMP_NUM_THREADS=1" in stdout
    assert "diagnostic channel" in (folder / "stderr.log").read_text()
    assert (folder / "finalized.txt").exists() is (outcome != "failed")
    collected = api().collect(point)
    assert collected.classification == expected
    assert collected.to_dict()["route"] == "pysu2"
    if outcome == "failed":
        assert "hard exit" in (folder / "stderr.log").read_text()
    elif outcome != "missing":
        assert collected.coefficients["CL"].value == 0.3
        assert collected.coefficients["CD"].value == 0.02


def test_p0000_item3g_parent_never_imports_binding(tmp_path, monkeypatch):
    directory = tmp_path / "dangerous"
    directory.mkdir()
    sentinel = tmp_path / "imported-in-parent"
    (directory / "pysu2.py").write_text(
        f"from pathlib import Path; Path({str(sentinel)!r}).touch(); raise SystemExit(7)",
        encoding="utf-8",
    )
    code = "from cfdbender.solvers.su2.routes import select_route; print(select_route('pysu2'))"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(directory), str(SOURCE))))
    child = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert child.returncode == 0 and child.stdout.strip() == b"pysu2", child.stderr
    assert not sentinel.exists(), "availability imported the native binding in the parent"


@pytest.mark.parametrize("scope", ["row", "workspace"])
def test_p0000_item3g_explicit_refusal_has_remedy(tmp_path, monkeypatch, scope):
    monkeypatch.setitem(sys.modules, "pysu2", None)
    point = case(tmp_path, route="pysu2" if scope == "row" else None)
    if scope == "workspace":
        (tmp_path / "init.toml").write_text(
            'schema_version=1\n[solver]\nname="su2"\nroute="pysu2"\n', encoding="ascii"
        )
        point = prepared(tmp_path)
    with pytest.raises(ValueError) as failure:
        api().execute(point, check_ram=lambda: None)
    text = str(failure.value)
    assert all(word in text for word in ("Python wrapper", "Linux", "WSL", "Windows", "executable"))
    assert not (tmp_path / point.point.directory).exists()


@pytest.mark.parametrize("option_source", ["row", "workspace", "environment"])
def test_p0000_item3g_wrapper_path(tmp_path, monkeypatch, option_source):
    directory = tmp_path / "wrapper install"
    directory.mkdir()
    shutil.copyfile(FIXTURES / "fake_pysu2.py", directory / "pysu2.py")
    path_option = "pysu2_path = " + json.dumps(str(directory))
    error = None
    try:
        point = case(tmp_path / "workspace", extra=path_option if option_source == "row" else "")
        if option_source == "workspace":
            (point.root / "init.toml").write_text(
                'schema_version=1\n[solver]\nname="su2"\n' + path_option, encoding="utf-8"
            )
            point = prepared(point.root)
        if option_source == "environment":
            monkeypatch.setenv("CFDBENDER_PYSU2_PATH", str(directory))
        monkeypatch.setenv("PYTHONPATH", str(SOURCE))
        result = api().execute(point, check_ram=lambda: None)
    except ValueError as failure:
        error = str(failure)
    assert error is None, f"wrapper install directory must work without parent import: {error}"
    assert api().classify(result) == "converged"
    assert "pysu2" not in sys.modules


def test_p0000_item3g_fallback_record_and_neutral_route(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pysu2", None)
    point = case(tmp_path, route=None, extra="executable = " + json.dumps(FAKE.as_posix()))
    with pytest.warns(RuntimeWarning, match="pysu2.*falling back.*executable"):
        records = execute_plan(plan(tmp_path), api())
    name = point.point.directory + "/execution.json"
    evidence = (tmp_path / name).read_bytes()
    assert records[0].outputs_sha256[name] == hashlib.sha256(evidence).hexdigest()
    assert json.loads(evidence)["route"] == "executable"
    assert api().collect(point).to_dict().get("route") == "executable"


def test_p0000_item3g_timeout_kills_worker_tree(tmp_path, wrapper):
    marker = tmp_path / "survived"
    point = case(tmp_path / "workspace", "timeout", timeout=0.8)
    setup = point.root / "inputs/setups/s001.toml"
    setup.write_text(
        setup.read_text().replace(
            "[solver]", "FAKE_MARKER = " + json.dumps(marker.as_posix()) + chr(10) + "[solver]"
        ),
        encoding="ascii",
    )
    point = prepared(point.root)
    result = execute(point)
    assert api().classify(result) == "failed"
    folder = point.root / point.point.directory
    assert json.loads((folder / "execution.json").read_text())["timed_out"] is True
    assert "worker pid=" in (folder / "stdout.log").read_text()
    time.sleep(2.2)
    assert not marker.exists(), "worker descendant survived timeout"


def test_p0000_item3g_worker_missing_import(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(SOURCE))
    child = subprocess.run(
        [sys.executable, "-S", "-m", "cfdbender.solvers.su2._pysu2_worker", "missing.cfg"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert child.returncode != 0
    assert b"No module named 'pysu2'" in child.stderr, child.stderr


def test_p0000_item3g_find_spec_missing_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysu2", None)
    with pytest.warns(RuntimeWarning, match="pysu2.*not found.*executable"):
        assert routes.select_route() == "executable"


def test_p0000_item3g_broken_binding_stays_in_child(tmp_path, monkeypatch):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "pysu2.py").write_text("raise ImportError('native dependency missing')")
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(directory), str(SOURCE))))
    point = case(tmp_path / "workspace", route=None)
    result = execute(point)
    assert result.exit_code != 0 and api().classify(result) == "failed"
    folder = point.root / point.point.directory
    assert "native dependency missing" in (folder / "stderr.log").read_text()
    assert json.loads((folder / "execution.json").read_text())["route"] == "pysu2"


def test_p0000_item3g_low_ram_before_worker(tmp_path, wrapper, monkeypatch):
    from cfdbender.solvers.su2 import _ram

    point = case(tmp_path / "workspace")
    monkeypatch.setattr(_ram, "free_ram_gib", lambda: 0.6)
    with pytest.raises(ValueError, match="RAM"):
        api().execute(point, check_ram=lambda: None)
    assert not (point.root / point.point.directory).exists()


def test_p0000_item3g_relative_row_path_overrides_workspace_and_environment(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    directory = root / "wrapper install"
    directory.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "fake_pysu2.py", directory / "pysu2.py")
    case(root, extra='pysu2_path = "wrapper install"')
    (root / "init.toml").write_text(
        'schema_version=1\n[solver]\nname="su2"\npysu2_path="absent-workspace"',
        encoding="ascii",
    )
    monkeypatch.setenv("CFDBENDER_PYSU2_PATH", "absent-environment")
    monkeypatch.setenv("PYTHONPATH", str(SOURCE))
    assert api().classify(execute(prepared(root))) == "converged"


def test_p0000_item3g_worker_does_not_inherit_unrelated_pythonpath(tmp_path, monkeypatch):
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()
    shutil.copyfile(FIXTURES / "fake_pysu2.py", wrapper_dir / "pysu2.py")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "sitecustomize.py").write_text("raise RuntimeError('leaked parent path')")
    monkeypatch.setenv("PYTHONPATH", str(unrelated))
    point = case(tmp_path / "workspace", extra="pysu2_path = " + json.dumps(str(wrapper_dir)))
    result = execute(point)
    assert api().classify(result) == "converged"
    stderr = (point.root / point.point.directory / "stderr.log").read_text()
    assert "leaked parent path" not in stderr
