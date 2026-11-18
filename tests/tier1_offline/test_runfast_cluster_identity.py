"""The persistent digest cache must survive moving between the nodes of a shared file system."""

import json
import platform
from types import SimpleNamespace

from cfdbender.core.plan import DigestCache


def identity(**changes):
    values = dict(st_size=9000000, st_mtime_ns=1, st_ctime_ns=1, st_dev=3, st_ino=40)
    return SimpleNamespace(**(values | changes))


def test_cache_written_on_one_host_is_served_on_another(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "node", lambda: "login1")
    DigestCache(tmp_path).put("mesh", identity(), "a" * 64, {})
    monkeypatch.setattr(platform, "node", lambda: "compute7")
    entry = DigestCache(tmp_path).get("mesh", identity())
    assert entry is not None and entry["sha256"] == "a" * 64


def test_a_different_device_number_is_still_served(tmp_path):
    DigestCache(tmp_path).put("mesh", identity(st_dev=3), "a" * 64, {})
    assert DigestCache(tmp_path).get("mesh", identity(st_dev=77)) is not None


def test_replacement_with_a_new_inode_is_not_served(tmp_path):
    DigestCache(tmp_path).put("mesh", identity(), "a" * 64, {})
    assert DigestCache(tmp_path).get("mesh", identity(st_ino=41)) is None


def test_copied_workspace_is_hashed_again_without_error(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    DigestCache(source).put("mesh", identity(), "a" * 64, {})
    (target / DigestCache.NAME).write_bytes((source / DigestCache.NAME).read_bytes())
    assert DigestCache(target).get("mesh", identity(st_ino=99)) is None


def test_a_cache_file_with_an_old_scope_field_is_still_read(tmp_path):
    DigestCache(tmp_path).put("mesh", identity(), "a" * 64, {})
    path = tmp_path / DigestCache.NAME
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scope"] = ["old-host", "/old/root"]
    for entry in document["entries"].values():
        entry["dev"] = 5
    path.write_text(json.dumps(document), encoding="utf-8")
    assert DigestCache(tmp_path).get("mesh", identity()) is not None


def test_no_host_or_root_is_written_to_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "node", lambda: "login1")
    DigestCache(tmp_path).put("mesh", identity(), "a" * 64, {})
    text = (tmp_path / DigestCache.NAME).read_text(encoding="utf-8")
    assert "login1" not in text and str(tmp_path.resolve()) not in text
