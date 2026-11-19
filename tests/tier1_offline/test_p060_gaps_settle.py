"""P060: collect waits for stable files and tolerates a writer replacing one."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_p0000_item3f_collect import outputs
from test_p0000_item3f_registry import submitted

from cfdbender.matrix import collection


def test_p060_collect_waits_when_size_or_time_changes(tmp_path, monkeypatch):
    registry, point, _ = submitted(tmp_path)
    outputs(tmp_path, point)
    path = tmp_path / point.directory / "history.csv"
    monkeypatch.setattr(
        collection.time, "sleep", lambda seconds: path.write_text(path.read_text() + "\n")
    )
    report = collection.collect_workspace(tmp_path, "study", settle=0)
    assert report.outstanding == 1 and report.collected == 0
    assert registry.read()[0].status == "SUBMITTED"


def test_p060_settle_observes_once_and_handles_a_disappearing_file(tmp_path, monkeypatch):
    path = tmp_path / "out"
    path.write_text("a")
    record = SimpleNamespace(
        status="SUBMITTED",
        solver="mock",
        point_id="1/DP-A",
        submission=None,
        point=SimpleNamespace(outputs=("out",)),
    )
    original = Path.stat
    calls = []

    def stat(target, *args, **kwargs):
        if target == path:
            calls.append(target)
            if len(calls) >= 2:
                raise FileNotFoundError("writer replaced output")
        return original(target, *args, **kwargs)

    # Isolate output observations from containment resolution, which also stats paths.
    monkeypatch.setattr(collection, "workspace_path", lambda root, name: root / name)
    monkeypatch.setattr(Path, "stat", stat)
    try:
        unsettled = collection._unsettled(tmp_path, [record], 0)
    except OSError as error:
        pytest.fail(f"a disappearing output must remain outstanding: {error}")
    assert unsettled == {"1/DP-A"}
    assert calls == [path, path]
