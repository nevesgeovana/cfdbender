"""--rehash-inputs is warned about and skipped on a FlightStream workspace."""

from __future__ import annotations

import pytest

from cfdbender import _pyfs
from cfdbender.matrix.cli import main as matrix_main
from cfdbender.workspace.cli import main as workspace_main

WARNING = (
    "warning: --rehash-inputs is ignored on a FlightStream workspace "
    "(the FlightStream engine keeps no digest cache)"
)


@pytest.fixture
def forwarded(tmp_path, monkeypatch):
    (tmp_path / "init.toml").write_text(
        'schema_version = 1\nsolver = "flightstream"\n', encoding="utf-8"
    )
    calls: list[list[str]] = []

    def stub(argv):
        calls.append(list(argv))
        return 7

    monkeypatch.setattr(_pyfs, "matrix_main", stub)
    monkeypatch.setattr(_pyfs, "workspace_main", stub)
    return tmp_path, calls


@pytest.mark.parametrize("verb", ["plan", "run", "collect"])
@pytest.mark.parametrize("main", [matrix_main, workspace_main])
def test_p0000_fts_rehash_flag_is_dropped_with_a_warning(verb, main, forwarded, capsys):
    root, calls = forwarded
    code = main([verb, "m", "--rehash-inputs", "--workspace", str(root)])
    assert code == 7
    assert calls == [[verb, "m", "--workspace", str(root)]]
    assert capsys.readouterr().err.splitlines().count(WARNING) == 1


def test_p0000_fts_without_the_flag_nothing_changes(forwarded, capsys):
    root, calls = forwarded
    argv = ["plan", "m", "--workspace", str(root)]
    assert matrix_main(argv) == 7
    assert calls == [argv]
    assert "rehash" not in capsys.readouterr().err
