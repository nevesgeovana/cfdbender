"""F8-004: a run this machine refuses archives nothing, also under --force-rerun."""

from __future__ import annotations

import pytest
from test_cli_verbs import FIRST, ran, snapshot

from cfdbender.core import run as core_run
from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main


def _kept(root):
    # logs/ records the refused command; the suite's plan-first fixture rewrites the receipt.
    skipped = ("logs/", "post/study/plan.json")
    return {k: v for k, v in snapshot(root).items() if not k.startswith(skipped)}


@pytest.mark.parametrize("option", [["--force-rerun", FIRST], ["--force-rerun-all"]])
def test_a_refused_environment_leaves_the_point_recorded(tmp_path, monkeypatch, capsys, option):
    ran(tmp_path)
    before = _kept(tmp_path)
    monkeypatch.setattr(core_run, "free_ram_gib", lambda: 0.1)
    assert main(["run", "study", "--workspace", str(tmp_path), *option]) == 2
    assert "below the 0.5 GiB floor" in capsys.readouterr().err
    assert _kept(tmp_path) == before, "the refusal changed the records or moved a point folder"
    assert not (tmp_path / "archive").exists()
    assert {r.status for r in RunRegistry(tmp_path).read()} == {"COMPLETED"}
    monkeypatch.setattr(core_run, "free_ram_gib", lambda: 8.0)
    assert main(["run", "study", "--workspace", str(tmp_path), *option]) == 0
