"""With the large-input check off, no planned input above the threshold is ever opened.

A threshold of one KiB stands in for the 8 MiB one. An audit hook counts every open of the mesh
and of its links in the point folders, by any code path; the HDF5 boundary-name reader is
replaced by a function that fails, so a call to it is seen too.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pytest
from test_p0000_item3f_plan import profile
from test_runfast_digest_cache import build

from cfdbender.core import plan as plan_module
from cfdbender.matrix import cli as cli_module
from cfdbender.matrix import receipt
from cfdbender.matrix.cli import main
from cfdbender.solvers._host import _groups

MESHES = {"large.su2", "mesh.su2", "large.cgns", "mesh.cgns"}
_SEEN: list[str] = []
_ACTIVE = [False]


def _hook(event: str, args: tuple) -> None:
    if _ACTIVE[0] and event == "open" and args and Path(str(args[0])).name in MESHES:
        _SEEN.append(str(args[0]))


sys.addaudithook(_hook)


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch):
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1024)
    monkeypatch.setattr(plan_module, "_BLOCK", 512)
    monkeypatch.setattr(plan_module, "_clock_ns", lambda: time.time_ns() + 60 * 10**9)
    monkeypatch.delenv(plan_module.NO_READ_ENV, raising=False)


@pytest.fixture
def opens(monkeypatch):
    def no_hdf5(*args, **kwargs):
        _SEEN.append("h5py")
        raise AssertionError("the HDF5 boundary-name reader was called")

    monkeypatch.setattr(_groups, "_cgns_names", no_hdf5)
    _SEEN.clear()
    _ACTIVE[0] = True
    yield _SEEN
    _ACTIVE[0] = False


def workspace(root: Path, kind: str, *, hpc: bool = False) -> Path:
    build(root)
    if kind == "cgns":
        matrix = root / "inputs/matrices/study.fs"
        matrix.write_text(matrix.read_text().replace("large.su2", "large.cgns"), encoding="ascii")
        mesh = root / "inputs/geometries/large.su2"
        mesh.with_suffix(".cgns").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 4000)
        mesh.unlink()
    if hpc:
        path = profile(root, fields={"working_directory": "{workdir}", "config": "{config_path}"})
        path.write_text(path.read_text().split("[builds]")[0], encoding="utf-8")
    os.utime(next((root / "inputs/geometries").iterdir()), ns=(10**18, 10**18))
    _SEEN.clear()  # writing the workspace is not a read by a command
    return root.resolve()


def cli(root: Path, *words: str) -> int:
    return main([words[0], "study", "--workspace", str(root), *words[1:]])


def switch(root: Path, how: str, monkeypatch) -> tuple[str, ...]:
    """Turn the check off in one of the three ways; return the words to add to a command."""
    if how == "option":
        return ("--no-read-large-inputs",)
    if how == "environment":
        monkeypatch.setenv(plan_module.NO_READ_ENV, "1")
    else:
        init = root / "init.toml"
        init.write_text('large_input_check = "identity"\n' + init.read_text(), encoding="utf-8")
    return ()


def mesh_name(kind: str) -> str:
    return f"inputs/geometries/large.{kind}"


@pytest.mark.parametrize("kind", ["su2", "cgns"])
@pytest.mark.parametrize("how", ["option", "environment", "setting"])
def test_no_open_on_plan_local_run_and_collect(tmp_path, monkeypatch, opens, kind, how):
    root = workspace(tmp_path / "ws", kind)
    words = switch(root, how, monkeypatch)
    assert cli(root, "plan", *words) == 0
    assert cli(root, "run", "--local", *words) == 0
    assert cli(root, "collect", *words) in (0, 1)
    assert opens == []
    receipt = json.loads((root / "post/study/plan.json").read_text(encoding="utf-8"))
    for point in receipt["points"]:
        value = point["inputs_sha256"][mesh_name(kind)]
        assert re.fullmatch(r"identity:size=[0-9]+;mtime_ns=[0-9]+", value)
    records = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    text = json.dumps(records)
    assert "identity:size=" in text


@pytest.mark.parametrize("kind", ["su2", "cgns"])
def test_no_open_on_scheduler_submission(tmp_path, opens, kind):
    root = workspace(tmp_path / "ws", kind, hpc=True)
    assert cli(root, "plan", "--no-read-large-inputs") == 0
    assert cli(root, "run", "--hpc", "h001", "--no-read-large-inputs") == 0
    assert cli(root, "collect", "--no-read-large-inputs") in (0, 1, 3)
    assert opens == []


def test_value_format_and_key_order(tmp_path, monkeypatch):
    plain = workspace(tmp_path / "a", "su2")
    assert cli(plain, "plan") == 0
    off = workspace(tmp_path / "b", "su2")
    assert cli(off, "plan", "--no-read-large-inputs") == 0
    read = json.loads((plain / "post/study/plan.json").read_text(encoding="utf-8"))
    unread = json.loads((off / "post/study/plan.json").read_text(encoding="utf-8"))
    a = read["points"][0]["inputs_sha256"]
    b = unread["points"][0]["inputs_sha256"]
    assert list(a) == list(b)
    stat = (off / mesh_name("su2")).stat()
    assert b[mesh_name("su2")] == f"identity:size={stat.st_size};mtime_ns={stat.st_mtime_ns}"
    assert re.fullmatch("[0-9a-f]{64}", a[mesh_name("su2")])
    assert {k: v for k, v in a.items() if k != mesh_name("su2")} == {
        k: v for k, v in b.items() if k != mesh_name("su2")
    }


def change_mesh(root: Path, change: str) -> None:
    mesh = root / mesh_name("su2")
    if change == "size":
        with mesh.open("ab") as stream:
            stream.write(b"% more\n")
        os.utime(mesh, ns=(10**18, 10**18))
    else:
        os.utime(mesh, ns=(10**18 + 10**9, 10**18 + 10**9))


@pytest.mark.parametrize("change", ["size", "mtime"])
def test_mesh_changed_before_the_run_is_refused(tmp_path, capsys, monkeypatch, change):
    monkeypatch.setattr(cli_module, "require_receipt", receipt.require_receipt)
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan", "--no-read-large-inputs") == 0
    change_mesh(root, change)
    capsys.readouterr()
    assert cli(root, "run", "--local", "--no-read-large-inputs") == 2
    assert f"input changed since it was planned: {mesh_name('su2')}" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["size", "mtime"])
def test_mesh_changed_after_the_run_is_refused_by_collect(tmp_path, capsys, change):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan", "--no-read-large-inputs") == 0
    assert cli(root, "run", "--local", "--no-read-large-inputs") == 0
    change_mesh(root, change)
    capsys.readouterr()
    assert cli(root, "collect", "--no-read-large-inputs") != 0
    captured = capsys.readouterr()
    assert f"input changed after planning: {mesh_name('su2')}; plan again" in (
        captured.out + captured.err
    )


def test_unchanged_mesh_is_accepted_by_a_second_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_module, "require_receipt", receipt.require_receipt)
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan", "--no-read-large-inputs") == 0
    assert cli(root, "run", "--local", "--no-read-large-inputs") == 0


def test_skipped_marker_check_warns_once_and_does_not_refuse(tmp_path, capsys):
    root = workspace(tmp_path / "ws", "su2")
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text().replace('["wall"]', '["absent"]'), encoding="ascii")
    assert cli(root, "plan", "--no-read-large-inputs") == 0
    text = capsys.readouterr()
    output = text.out + text.err
    assert output.count("[groups] marker names not checked") == 1


def test_rehash_and_no_read_refuse_together(tmp_path, capsys):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan", "--no-read-large-inputs", "--rehash-inputs") == 2
    assert "contradict each other" in capsys.readouterr().err


def test_bad_setting_is_refused(tmp_path, capsys):
    root = workspace(tmp_path / "ws", "su2")
    init = root / "init.toml"
    init.write_text('large_input_check = "never"\n' + init.read_text(), encoding="utf-8")
    assert cli(root, "plan") == 2
    assert "large_input_check must be" in capsys.readouterr().err
