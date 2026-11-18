"""A large input streamed through its digest is dropped from the page cache as it is read.

On Linux the kernel keeps every byte read in its page cache; a mesh of several GiB read once
would otherwise stay there, counted against the memory of a shared host. The advice is given
with ``os.posix_fadvise``; on a platform without it nothing is asked. A threshold of one KiB
stands in for the 8 MiB one.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from cfdbender.core import plan as plan_module


@pytest.fixture
def advice(monkeypatch):
    calls: list[tuple[int, int, int]] = []

    def record(fd, offset, length, flag):
        calls.append((offset, length, flag))

    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1024)
    monkeypatch.setattr(plan_module, "_BLOCK", 256)
    monkeypatch.setattr(plan_module, "_DROP_EVERY", 4)
    monkeypatch.setattr(os, "posix_fadvise", record, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    return calls


def test_a_large_input_is_dropped_from_the_page_cache_while_hashed(tmp_path, advice):
    data = os.urandom(4500)
    (tmp_path / "mesh.su2").write_bytes(data)
    files = plan_module.InputFiles(tmp_path)
    with plan_module.rehash_inputs():
        assert files.digest("mesh.su2") == hashlib.sha256(data).hexdigest()
    assert advice, "no page-cache advice was given"
    assert advice[-1] == (0, 4500, 4)  # 18 blocks: the last drop is the one at the end
    assert all(offset == 0 and length <= 4500 for offset, length, _ in advice)
    assert len(advice) >= 2  # during the read, not only at its end


def test_a_streamed_link_and_output_are_dropped_too(tmp_path, advice):
    data = os.urandom(2900)
    (tmp_path / "out.bin").write_bytes(data)
    files = plan_module.InputFiles(tmp_path)
    assert files.digest_output(tmp_path / "out.bin", ()) == hashlib.sha256(data).hexdigest()
    assert advice[-1] == (0, 2900, 4)


def test_a_small_input_is_left_alone(tmp_path, advice):
    (tmp_path / "small.toml").write_bytes(b"a = 1\n")
    plan_module.InputFiles(tmp_path).digest("small.toml")
    assert advice == []


def test_a_failing_advice_never_fails_the_read(tmp_path, monkeypatch, advice):
    def refuse(*args):
        raise OSError("not supported here")

    monkeypatch.setattr(os, "posix_fadvise", refuse, raising=False)
    data = os.urandom(5000)
    (tmp_path / "mesh.su2").write_bytes(data)
    with plan_module.rehash_inputs():
        digest = plan_module.InputFiles(tmp_path).digest("mesh.su2")
    assert digest == hashlib.sha256(data).hexdigest()
