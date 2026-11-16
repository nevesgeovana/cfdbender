"""Generated geometry in inputs/geometry/<name>/ is told from a legacy mesh folder by a sidecar."""

from __future__ import annotations

import warnings

import pytest

from cfdbender.core import mesh as mesh_module


@pytest.fixture(autouse=True)
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(mesh_module, "_warned", set())


def _generated(root, name="wing", leaf="wing.obj"):
    folder = root / "inputs/geometry" / name
    folder.mkdir(parents=True)
    (folder / leaf).write_text("v 0 0 0\n", encoding="ascii")
    (folder / mesh_module.GEOMETRY_SIDECAR).write_text("{}", encoding="ascii")
    return f"inputs/geometry/{name}/{leaf}"


def test_geom_layout_generated_geometry_resolves_as_named_without_warning(tmp_path):
    path = _generated(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mesh_module.resolve_mesh_path(tmp_path, path) == path
    assert mesh_module.generated_geometry_of(tmp_path, path) == "inputs/geometry/wing"


def test_geom_layout_generated_geometry_is_not_replaced_by_a_mesh_of_the_same_name(tmp_path):
    path = _generated(tmp_path)
    twin = tmp_path / "inputs/mesh/wing/wing.obj"
    twin.parent.mkdir(parents=True)
    twin.write_text("v 1 1 1\n", encoding="ascii")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mesh_module.resolve_mesh_path(tmp_path, path) == path


def test_geom_layout_a_geometry_folder_without_sidecar_stays_a_legacy_mesh_folder(tmp_path):
    folder = tmp_path / "inputs/geometry/wing"
    folder.mkdir(parents=True)
    (folder / "wing.obj").write_text("v 0 0 0\n", encoding="ascii")
    with pytest.warns(UserWarning, match="legacy mesh folder"):
        assert (
            mesh_module.resolve_mesh_path(tmp_path, "inputs/geometry/wing/wing.obj")
            == "inputs/geometry/wing/wing.obj"
        )
    assert mesh_module.generated_geometry_of(tmp_path, "inputs/geometry/wing/wing.obj") is None


def test_geom_layout_only_the_geometry_folder_holds_generated_geometry(tmp_path):
    folder = tmp_path / "inputs/meshes/wing"
    folder.mkdir(parents=True)
    (folder / mesh_module.GEOMETRY_SIDECAR).write_text("{}", encoding="ascii")
    assert mesh_module.generated_geometry_of(tmp_path, "inputs/meshes/wing/wing.obj") is None
    assert mesh_module.generated_geometry_of(tmp_path, "inputs/geometry/wing.obj") is None
