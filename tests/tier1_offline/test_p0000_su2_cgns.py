"""P0000 HPC-1: mesh='inputs/meshes/tiny.cgns'; staged meshes are links, never copies."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from test_p0000_item3d_config import api, prepared, workspace

from cfdbender.solvers._host import _mesh

FAKE = Path(__file__).resolve().parents[1] / "fixtures/su2/fake_su2.py"


def mesh_point(root, suffix, link=None):
    extra = ("" if link is None else f"mesh_link = {str(link).lower()}\n") + (
        '[solver]\nroute = "executable"\nexecutable = ' + json.dumps(FAKE.as_posix())
    )
    workspace(root, extra=extra)
    setup = root / "inputs/setups/s001.toml"
    setup.write_bytes(setup.read_bytes().replace(b"tiny.su2", f"tiny{suffix}".encode()))
    (root / f"inputs/meshes/tiny{suffix}").write_bytes(b"synthetic mesh bytes")
    try:
        return prepared(root)
    except ValueError as error:
        pytest.fail(f"P0000-HPC-1: supported mesh setup refused: {error}")


@pytest.mark.parametrize("suffix,format_name", [(".su2", "SU2"), (".cgns", "CGNS")])
def test_p0000_su2_cgns_tracking_render_and_link(tmp_path, suffix, format_name):
    point = mesh_point(tmp_path, suffix)
    name = f"inputs/meshes/tiny{suffix}"
    assert point.point.inputs_sha256[name] == hashlib.sha256(b"synthetic mesh bytes").hexdigest()
    text = api().translate(point)
    assert f"MESH_FILENAME= mesh{suffix}\n" in text
    assert f"MESH_FORMAT= {format_name}\n" in text
    assert f"{point.point.directory}/mesh{suffix}" in point.point.outputs
    api().execute(point, check_ram=lambda: None)
    staged = tmp_path / point.point.directory / f"mesh{suffix}"
    assert staged.read_bytes() == b"synthetic mesh bytes"
    assert staged.samefile(tmp_path / name), "a staged mesh is a link, never a copy"


@pytest.mark.parametrize("suffix", [".su2", ".cgns"])
@pytest.mark.parametrize("link", [None, True])
def test_p0000_su2_mesh_symlink_is_relative(tmp_path, monkeypatch, suffix, link):
    point = mesh_point(tmp_path, suffix, link)
    calls = []
    original = _mesh._symlink

    def spy(target, relative):
        calls.append(relative)
        original(target, relative)

    monkeypatch.setattr(_mesh, "_symlink", spy)
    api().execute(point, check_ram=lambda: None)
    staged = tmp_path / point.point.directory / f"mesh{suffix}"
    assert calls == [os.path.join("..", "..", "..", "..", "inputs", "meshes", f"tiny{suffix}")]
    assert staged.samefile(tmp_path / f"inputs/meshes/tiny{suffix}")
    if staged.is_symlink():
        assert os.readlink(staged) == calls[0]


def test_p0000_su2_mesh_hard_link_when_symlinks_are_refused(tmp_path, monkeypatch):
    point = mesh_point(tmp_path, ".su2")

    def unavailable(*args, **kwargs):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(_mesh, "_symlink", unavailable)
    api().execute(point, check_ram=lambda: None)
    staged = tmp_path / point.point.directory / "mesh.su2"
    assert not staged.is_symlink()
    assert staged.samefile(tmp_path / "inputs/meshes/tiny.su2")
    assert os.stat(staged).st_nlink == 2


def test_p0000_su2_mesh_refuses_without_copy(tmp_path, monkeypatch):
    point = mesh_point(tmp_path, ".su2")

    def unavailable(*args, **kwargs):
        raise OSError("links unavailable")

    monkeypatch.setattr(_mesh, "_symlink", unavailable)
    monkeypatch.setattr(_mesh, "_hardlink", unavailable)
    with pytest.raises(ValueError, match="never copied"):
        api().execute(point, check_ram=lambda: None)
    assert not (tmp_path / point.point.directory / "mesh.su2").exists()


def test_p0000_su2_mesh_link_false_is_refused(tmp_path):
    workspace(tmp_path, extra="mesh_link = false")
    with pytest.raises(ValueError, match="never copied"):
        prepared(tmp_path)


def test_p0000_su2_cgns_changed_mesh_refused(tmp_path):
    point = mesh_point(tmp_path, ".cgns", True)
    (tmp_path / "inputs/meshes/tiny.cgns").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        api().execute(point, check_ram=lambda: None)


@pytest.mark.parametrize("value", [1, "true", [], None])
def test_p0000_su2_cgns_mesh_link_requires_boolean(value):
    from cfdbender.solvers.su2._config import config

    with pytest.raises(ValueError, match="mesh_link.*boolean"):
        config({"mesh_link": value}, {}, {}, {})
