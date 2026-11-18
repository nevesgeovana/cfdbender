"""Adversarial cache parsing: corrupt optimization data must not stop input reads."""

from cfdbender.core.plan import DigestCache


def test_deeply_nested_cache_is_ignored(tmp_path):
    (tmp_path / DigestCache.NAME).write_text("[" * 100000 + "]" * 100000)
    assert DigestCache(tmp_path)._all() == {}


def test_oversized_cache_is_not_parsed(tmp_path, monkeypatch):
    from cfdbender.core import plan

    path = tmp_path / DigestCache.NAME
    path.write_bytes(b" " * (9 << 20))

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized cache reached JSON parser")

    monkeypatch.setattr(plan.json, "loads", forbidden)
    assert DigestCache(tmp_path)._all() == {}
