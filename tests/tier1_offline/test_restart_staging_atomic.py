"""The in-job staging is all-or-nothing, quotes any source path and ignores a failing digest."""

from __future__ import annotations

import os
import shutil
import subprocess
from types import SimpleNamespace

from test_restart_in_job import needs_sh

from cfdbender.matrix.batch import _staging


def _staged(root, pairs, extra_path=None):
    """Run the staging snippet of ``pairs`` under sh in ``root``; return (rc, stderr text)."""
    snippet = _staging(SimpleNamespace(restart_from="SRC"), "dep", pairs)  # type: ignore[arg-type]
    (root / "dep").mkdir(exist_ok=True)
    env = dict(os.environ)
    if extra_path is not None:
        env["PATH"] = str(extra_path) + os.pathsep + env["PATH"]
    program = f"rc=0\n{snippet}\nprintf '%s' \"$rc\" > rc.txt\n"
    (root / "stage.sh").write_text(program, newline="\n")
    sh = shutil.which("sh")
    assert sh
    subprocess.run([sh, "stage.sh"], cwd=root, env=env, capture_output=True, check=False)
    err = root / "dep" / "stderr.log"
    return (root / "rc.txt").read_text(), err.read_text() if err.exists() else ""


@needs_sh
def test_a_missing_second_source_stages_nothing_and_names_the_file(tmp_path):
    (tmp_path / "a.dat").write_text("a")
    rc, err = _staged(tmp_path, {"first.dat": "a.dat", "second.dat": "absent.dat"})
    assert rc == "98"
    assert not (tmp_path / "dep" / "first.dat").exists()
    assert "absent.dat" in err and err.count("\n") == 1


@needs_sh
def test_a_failing_second_copy_removes_the_first_staged_file(tmp_path):
    (tmp_path / "a.dat").write_text("a")
    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "inner.dat").write_text("b")
    rc, err = _staged(tmp_path, {"first.dat": "a.dat", "second.dat": "folder"})
    assert rc == "98" and not (tmp_path / "dep" / "first.dat").exists()
    assert "folder" in err


@needs_sh
def test_a_source_path_with_a_space_a_quote_and_a_dollar_is_staged_by_name(tmp_path):
    names = ["with space.dat", "it's.dat", "cost$HOME.dat"]
    for name in names:
        (tmp_path / name).write_text(name)
    rc, _ = _staged(tmp_path, {f"s{i}.dat": name for i, name in enumerate(names)})
    assert rc == "0"
    for i, name in enumerate(names):
        assert (tmp_path / "dep" / f"s{i}.dat").read_text() == name


@needs_sh
def test_a_failing_sha256sum_leaves_the_run_at_status_zero(tmp_path):
    (tmp_path / "a.dat").write_text("a")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "sha256sum").write_text("#!/bin/sh\nexit 1\n", newline="\n")
    (stub / "sha256sum").chmod(0o755)
    rc, _ = _staged(tmp_path, {"first.dat": "a.dat"}, stub)
    assert rc == "0" and (tmp_path / "dep" / "first.dat").read_text() == "a"
