"""A PyFR restart checks the saved mesh against its planned digest without reading it whole."""

from __future__ import annotations

from pathlib import Path

from test_p060_gaps_restart import call, prepared, solution


def test_a_pyfr_restart_checks_the_saved_mesh_without_reading_it_whole(tmp_path, monkeypatch):
    source, argv = prepared(tmp_path, "pyfr")
    solution(tmp_path, source, "pyfr")
    real_read = Path.read_bytes

    def refuse(self):
        assert self.name != "mesh.msh", "a restart read the saved mesh whole"
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", refuse)
    assert call([*argv, "--restart"]) == 0


def test_saved_mesh_is_checked_when_workspace_mesh_is_missing(tmp_path, monkeypatch):
    import hashlib

    from cfdbender.core import plan as plan_module
    from cfdbender.core.plan import InputFiles

    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 0)
    saved = tmp_path / "saved.msh"
    saved.write_bytes(b"old mesh")
    assert (
        InputFiles(tmp_path).digest_via("missing.msh", saved)
        == hashlib.sha256(b"old mesh").hexdigest()
    )
