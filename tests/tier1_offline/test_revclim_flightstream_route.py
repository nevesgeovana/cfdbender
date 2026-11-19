"""RC-1: a FlightStream workspace answers the moved verbs under cfd-workspace.

pyflightstream owns these verbs under pyfs-matrix (pyfs-workspace has none of them), so
cfd-workspace must forward them there. cfd-matrix keeps forwarding every pyfs-matrix
command unchanged in such a workspace (the FlightStream parity contract).
"""

from __future__ import annotations

import pytest

from cfdbender import _pyfs
from cfdbender.matrix.cli import MOVED
from cfdbender.matrix.cli import main as matrix_main
from cfdbender.workspace.cli import main as workspace_main


@pytest.fixture
def forwarded(tmp_path, monkeypatch):
    (tmp_path / "init.toml").write_text(
        'schema_version = 1\nsolver = "flightstream"\n', encoding="utf-8"
    )
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(_pyfs, "matrix_main", lambda argv: calls.append(("matrix", argv)) or 0)
    monkeypatch.setattr(
        _pyfs, "workspace_main", lambda argv: calls.append(("workspace", argv)) or 0
    )
    return tmp_path, calls


# setups is cfdbender's own verb: pyflightstream has none, so it is answered locally
# (test_setups_command.py), never forwarded.
@pytest.mark.parametrize("verb", [verb for verb in MOVED if verb != "setups"])
def test_rc1_cfd_workspace_forwards_a_moved_verb_to_pyfs_matrix(verb, forwarded):
    root, calls = forwarded
    argv = [verb, "--workspace", str(root)]
    assert workspace_main(argv) == 0
    assert calls == [("matrix", argv)]


@pytest.mark.parametrize("verb", ["archive", "migrate"])
def test_rc1_the_workspace_own_verbs_still_go_to_pyfs_workspace(verb, forwarded):
    root, calls = forwarded
    argv = [verb, "--workspace", str(root)]
    workspace_main(argv)
    assert calls == [("workspace", argv)]


@pytest.mark.parametrize("verb", MOVED)
def test_rc1_cfd_matrix_still_forwards_the_old_form_on_a_flightstream_workspace(verb, forwarded):
    root, calls = forwarded
    argv = [verb, "--workspace", str(root)]
    assert matrix_main(argv) == 0
    assert calls == [("matrix", argv)]


def test_rc2_a_matrix_verb_still_goes_to_pyfs_matrix(forwarded):
    root, calls = forwarded
    argv = ["status", "--workspace", str(root)]
    assert matrix_main(argv) == 0
    assert calls == [("matrix", argv)]


def test_setups_is_answered_locally_on_a_flightstream_workspace(forwarded, capsys):
    root, calls = forwarded
    assert workspace_main(["setups", "--workspace", str(root)]) == 0
    assert calls == []
    assert capsys.readouterr().out == "no setup templates for flightstream\n"


def test_setups_as_a_workspace_name_does_not_disable_forwarding(forwarded, monkeypatch):
    root, calls = forwarded
    target = root / "setups"
    target.mkdir()
    (target / "init.toml").write_bytes((root / "init.toml").read_bytes())
    monkeypatch.chdir(root)
    argv = ["show", "all", "--workspace", "setups"]
    assert workspace_main(argv) == 0
    assert calls == [("matrix", argv)]
