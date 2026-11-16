"""An existing provenance file requires an explicit overwrite request."""

from cfdbender.mesh.cli import main


def test_scale_preserves_existing_provenance(tmp_path, capsys):
    source = tmp_path / "source.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    target = tmp_path / "scaled.obj"
    sidecar = tmp_path / "scaled.obj.provenance.json"
    sidecar.write_bytes(b"user file\n")
    assert main(["scale", "2", "--mesh", str(source), "--out", str(target)]) == 2
    assert sidecar.read_bytes() == b"user file\n"
    assert "provenance" in capsys.readouterr().err

    assert not target.exists()
    assert not target.with_name(target.name + ".scale.json").exists()
