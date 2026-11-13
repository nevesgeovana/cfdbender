"""Sync archive generations remain immutable."""

from argparse import Namespace

from cfdbender.matrix.storage import _copy_files


def test_sync_keeps_every_overwritten_generation(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from cfdbender.matrix import storage

    class FixedClock:
        @staticmethod
        def now(zone):
            return datetime(2026, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(storage, "datetime", FixedClock)
    root, other = tmp_path / "main", tmp_path / "other"
    for folder in (root, other):
        (folder / "post").mkdir(parents=True)
    target, source = root / "post/result.csv", other / "post/result.csv"
    target.write_text("first")
    args = Namespace(include_archives=False, overwrite=True, apply=True)
    for value in ("second", "third"):
        source.write_text(value)
        _copy_files(root, other, "post", args, [])
    assert sorted(p.read_text() for p in (root / "archive").rglob("result.csv")) == [
        "first",
        "second",
    ]
