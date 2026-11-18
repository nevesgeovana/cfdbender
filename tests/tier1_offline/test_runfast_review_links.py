"""Hard-link staging must not hash an unchanged input once per point."""

import hashlib
import os

import pytest

from cfdbender.core import plan
from cfdbender.solvers._host import _mesh


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard links change ctime")
def test_hardlink_staging_reads_once(tmp_path, monkeypatch):
    source = tmp_path / "mesh"
    source.write_bytes(b"mesh bytes" * 100)
    monkeypatch.setattr(plan, "LARGE_INPUT_BYTES", 1)

    def no_symlink(*args):
        raise OSError("forced fallback")

    monkeypatch.setattr(_mesh, "_symlink", no_symlink)
    calls = []
    real = plan._open_input

    def counted(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(plan, "_open_input", counted)
    files = plan.InputSession(tmp_path).files()
    with plan.rehash_inputs():
        digest = files.digest("mesh")
        for index in range(5):
            assert files.digest("mesh") == digest
            target = tmp_path / f"output{index}"
            _mesh.link_mesh(source, target, digest, inputs=files, name="mesh")
            assert files.digest_output(target, ["mesh"]) == digest
    assert len(calls) == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard-link metadata")
def test_hardlink_does_not_acknowledge_a_content_change(tmp_path, monkeypatch):
    source = tmp_path / "mesh"
    source.write_bytes(b"original" * 100)
    monkeypatch.setattr(plan, "LARGE_INPUT_BYTES", 1)

    def no_symlink(*args):
        raise OSError("forced fallback")

    monkeypatch.setattr(_mesh, "_symlink", no_symlink)
    real = _mesh._hardlink

    def changed(target, path):
        real(target, path)
        path.write_bytes(b"different" * 100)

    monkeypatch.setattr(_mesh, "_hardlink", changed)
    files = plan.InputSession(tmp_path).files()
    digest = files.digest("mesh")
    with pytest.raises(ValueError, match="mesh changed after planning"):
        _mesh.link_mesh(source, tmp_path / "output", digest, inputs=files, name="mesh")
    assert hashlib.sha256(source.read_bytes()).hexdigest() != digest


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard links change ctime")
def test_local_run_hardlinks_keep_read_count_constant(tmp_path, monkeypatch):
    from test_runfast_digest_cache import build

    from cfdbender.matrix.cli import main
    from cfdbender.matrix.receipt import write_receipt
    from cfdbender.workspace.planning import plan_workspace

    monkeypatch.setattr(plan, "LARGE_INPUT_BYTES", 1024)
    root = build(tmp_path / "ws", groups=False).resolve()
    write_receipt(plan_workspace(root, "study"))

    def no_symlink(*args):
        raise OSError("forced fallback")

    monkeypatch.setattr(_mesh, "_symlink", no_symlink)
    calls = []
    real = plan._open_input

    def counted(path):
        if path.suffix == ".su2":
            calls.append(path)
        return real(path)

    monkeypatch.setattr(plan, "_open_input", counted)
    assert main(["run", "study", "--workspace", str(root), "--local", "--rehash-inputs"]) == 0
    assert len(calls) <= 4
