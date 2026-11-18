"""Every reader of recorded input values copes with a file identity, and the modes mix.

A plan made with the large-input check off and a later command with it on (or the reverse)
works; a digest cache entry makes the off mode record the real SHA-256 without a read; reports
show an unread file as not hashed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_optcheck_identity import (  # noqa: F401  (fixtures used by name)
    cli,
    mesh_name,
    opens,
    small_threshold,
    workspace,
)

from cfdbender.core import plan as plan_module
from cfdbender.core.registry import RunRegistry
from cfdbender.matrix import cli as cli_module
from cfdbender.matrix import receipt
from cfdbender.matrix.polar2d import _mesh
from cfdbender.workspace.cli import main as workspace_cli

MESH = mesh_name("su2")
IDENTITY = "identity:size=5171;mtime_ns=1000000000000000000"


@pytest.fixture(autouse=True)
def real_receipt(monkeypatch):
    monkeypatch.setattr(cli_module, "require_receipt", receipt.require_receipt)


def recorded(root: Path) -> set[str]:
    data = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    rows = data["runs"] if isinstance(data, dict) and "runs" in data else data
    text = json.dumps(rows)
    return {value for value in (IDENTITY,) if value in text}


def test_default_mode_records_the_sha256_of_the_bytes(tmp_path):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan") == 0
    data = json.loads((root / "post/study/plan.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((root / MESH).read_bytes()).hexdigest()
    assert {p["inputs_sha256"][MESH] for p in data["points"]} == {digest}


def test_a_cache_entry_gives_the_real_digest_without_a_read(tmp_path, opens):  # noqa: F811
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan") == 0  # reads once and fills the digest cache
    digest = hashlib.sha256((root / MESH).read_bytes()).hexdigest()
    opens.clear()
    assert cli(root, "plan", "--no-read-large-inputs") == 0
    assert cli(root, "run", "--local", "--no-read-large-inputs") == 0
    assert opens == []
    data = json.loads((root / "post/study/plan.json").read_text(encoding="utf-8"))
    assert {p["inputs_sha256"][MESH] for p in data["points"]} == {digest}


def test_identity_plan_then_default_run_and_collect(tmp_path, capsys):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan", "--no-read-large-inputs") == 0
    assert cli(root, "run", "--local") == 0
    assert cli(root, "collect") in (0, 1)
    assert "input changed" not in capsys.readouterr().err


def test_default_plan_then_identity_run_then_default_collect(tmp_path, capsys):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan") == 0
    (root / plan_module.DigestCache.NAME).unlink()
    assert cli(root, "run", "--local", "--no-read-large-inputs") == 0
    assert IDENTITY in recorded(root)
    capsys.readouterr()
    assert cli(root, "collect") in (0, 1)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "input changed" not in output
    assert output.count(f"{MESH}: checked by size and modification time only") == 1


def test_identity_command_meeting_a_planned_digest_warns_it_is_unverified(tmp_path, capsys):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan") == 0
    assert cli(root, "run", "--local") == 0
    (root / plan_module.DigestCache.NAME).unlink()
    capsys.readouterr()
    assert cli(root, "collect", "--no-read-large-inputs") in (0, 1)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "input changed" not in output
    assert output.count(f"{MESH}: not verified against its planned SHA-256") == 1


def test_records_with_identity_values_are_read_back(tmp_path):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan", "--no-read-large-inputs") == 0
    assert cli(root, "run", "--local", "--no-read-large-inputs") == 0
    records = RunRegistry(root).read()
    assert records
    for record in records:
        assert record.point.inputs_sha256[MESH] == IDENTITY
        staged = [name for name in record.outputs_sha256 if name.endswith("/mesh.su2")]
        assert staged and record.outputs_sha256[staged[0]] == IDENTITY


def test_trace_and_diff_show_not_hashed(tmp_path, capsys):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan", "--no-read-large-inputs") == 0
    assert cli(root, "run", "--local", "--no-read-large-inputs") == 0
    record = RunRegistry(root).read()[0]
    product = next(name for name in record.point.outputs if name.endswith("/mesh.su2"))
    capsys.readouterr()
    assert workspace_cli(["trace", product, "--workspace", str(root), "--json"]) == 0
    text = capsys.readouterr().out
    assert "not hashed" in text and "identity:" not in text


def test_an_unverified_file_is_warned_about_once_per_command(tmp_path):
    root = workspace(tmp_path / "ws", "su2")
    with plan_module.large_input_check("identity"), pytest.warns(UserWarning) as caught:
        session = plan_module.InputSession(root)
        for _ in range(3):
            assert session.files().agrees(MESH, "a" * 64)
    texts = [str(w.message) for w in caught if "not verified" in str(w.message)]
    assert len(texts) == 1


def test_a_polar_header_shows_not_hashed():
    members = [{"inputs_sha256": {"inputs/mesh/wing.su2": IDENTITY}}]
    assert _mesh("inputs/mesh/wing.su2", members) == ("wing.su2", "not hashed")


@pytest.mark.parametrize(
    ("recorded_value", "current", "same"),
    [
        (IDENTITY, IDENTITY, True),
        (IDENTITY, "a" * 64, True),
        ("a" * 64, IDENTITY, True),
        ("a" * 64, "b" * 64, False),
        ("identity:size=1;mtime_ns=1", IDENTITY, False),
        (IDENTITY, "identity:size=1;mtime_ns=1", False),
    ],
)
def test_receipt_values_agree_only_for_the_file_on_disk(tmp_path, recorded_value, current, same):
    root = workspace(tmp_path / "ws", "su2")
    assert plan_module.recorded_values_agree(root, MESH, recorded_value, current) is same


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("a" * 64, True),
        (IDENTITY, True),
        ("identity:size=1;mtime_ns=-5", True),
        ("identity:size=x;mtime_ns=1", False),
        ("A" * 64, False),
        ("a" * 63, False),
        (None, False),
    ],
)
def test_recorded_value_shapes(value, valid):
    assert plan_module.recorded_value_valid(value) is valid
    assert plan_module.is_identity_value(value) is (valid and str(value).startswith("identity:"))
