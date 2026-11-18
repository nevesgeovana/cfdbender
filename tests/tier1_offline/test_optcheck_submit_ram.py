"""A scheduler submission warns about low free RAM on the submitting host and goes on.

The job runs on the cluster, so the memory of the host that submits it (often a shared login
node) is no reason to refuse; a local run keeps its refusal unchanged.
"""

from __future__ import annotations

import time

import pytest
from test_optcheck_identity import cli, workspace

from cfdbender.core import plan as plan_module
from cfdbender.core import run as run_module
from cfdbender.matrix import submission

FLOOR_TEXT = "below the 0.5 GiB floor"


@pytest.fixture(autouse=True)
def settled(monkeypatch):
    monkeypatch.setattr(plan_module, "_clock_ns", lambda: time.time_ns() + 60 * 10**9)


def low() -> float:
    return 0.1


def unmeasured() -> float:
    raise OSError("no measurement")


@pytest.mark.parametrize(
    ("measure", "text"),
    [
        (low, "free RAM 0.100 GiB on this host is below the 0.5 GiB floor; submitting anyway"),
        (unmeasured, "cannot measure free RAM on this host; submitting anyway"),
    ],
)
def test_submission_warns_once_and_submits(tmp_path, capsys, monkeypatch, measure, text):
    root = workspace(tmp_path / "ws", "su2", hpc=True)
    assert cli(root, "plan") == 0
    monkeypatch.setattr(submission, "free_ram_gib", measure)
    monkeypatch.setattr(run_module, "free_ram_gib", measure)
    capsys.readouterr()
    assert cli(root, "run", "--hpc", "h001") == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert output.count(text) == 1
    assert "SUBMITTED" in (root / "runs.json").read_text(encoding="utf-8")


def test_local_run_still_refuses_below_the_floor(tmp_path, capsys, monkeypatch):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan") == 0
    monkeypatch.setattr(run_module, "free_ram_gib", low)
    capsys.readouterr()
    assert cli(root, "run", "--local") == 2
    assert f"free RAM 0.100 GiB is {FLOOR_TEXT}" in capsys.readouterr().err


def test_local_run_still_refuses_an_unmeasured_host(tmp_path, capsys, monkeypatch):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan") == 0
    monkeypatch.setattr(run_module, "free_ram_gib", unmeasured)
    capsys.readouterr()
    assert cli(root, "run", "--local") == 2
    assert "cannot measure free RAM; local run refused" in capsys.readouterr().err
