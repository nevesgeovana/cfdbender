"""P060: a skipped cluster request gives an executable local command and checks."""

import shlex
import subprocess
import sys

from test_p0000_item3f_plan import prepared

from cfdbender.matrix import cli
from cfdbender.solvers import mock


def test_p060_cluster_skip_prints_the_users_complete_local_command(tmp_path, monkeypatch, capsys):
    prepared(tmp_path)
    monkeypatch.setattr(mock, "CAPABILITIES", frozenset())
    argv = ["run", "study", "--workspace", str(tmp_path), "--sims", "1001", "--no-local"]
    assert cli.main(argv) == 0
    local = [word for word in argv if word != "--no-local"] + ["--local"]
    typed = subprocess.list2cmdline(local) if sys.platform == "win32" else shlex.join(local)
    assert f"Type: cfd-matrix {typed}" in capsys.readouterr().err
    assert not (tmp_path / "runs.json").exists()
