"""A timestamp-preserving replacement must not inherit another file's digest."""

from types import SimpleNamespace

from cfdbender.core.plan import DigestCache


def identity(**changes):
    values = dict(st_size=9000000, st_mtime_ns=1, st_ctime_ns=1, st_dev=3, st_ino=40)
    return SimpleNamespace(**(values | changes))


def test_new_inode_with_preserved_times_is_not_cached(tmp_path):
    cache = DigestCache(tmp_path)
    cache.put("mesh", identity(), "a" * 64, {})
    assert cache.get("mesh", identity(st_ino=41)) is None


def test_cache_copied_to_another_workspace_is_not_used(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    DigestCache(source).put("mesh", identity(), "a" * 64, {})
    (target / DigestCache.NAME).write_bytes((source / DigestCache.NAME).read_bytes())
    # A real copy has new inodes, so the copied entry is not served.
    assert DigestCache(target).get("mesh", identity(st_ino=99)) is None


def test_missing_file_identity_disables_cache(tmp_path):
    cache = DigestCache(tmp_path)
    cache.put("mesh", identity(st_ino=0), "a" * 64, {})
    assert cache.get("mesh", identity(st_ino=0)) is None
