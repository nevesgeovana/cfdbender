"""QPROP takes its blade from the row's GEOMETRY through the one mesh resolver."""

from __future__ import annotations

import shutil

import pytest
from qprop_workspace import CONDITION, adapter, prepared, setup_text, workspace

from cfdbender.core import mesh as mesh_module
from cfdbender.workspace.planning import plan_workspace

MATRIX = (
    "POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION|GEOMETRY\n1001|1|001|001|001|"
    + CONDITION
    + "|blade.obj\n"
)


@pytest.mark.parametrize("folder", ["mesh", "geometries", "geometry", "meshes"])
def test_layout_qprop_blade_comes_from_geometry_without_a_setup_key(tmp_path, folder):
    mesh_module._warned.clear()
    workspace(tmp_path, setup=setup_text(blade=None))
    source = tmp_path / "inputs/geometry/blade.obj"
    target = tmp_path / "inputs" / folder / "blade.obj"
    target.parent.mkdir(exist_ok=True)
    if target != source:
        shutil.move(source, target)
    (tmp_path / "inputs/matrices/study.fs").write_text(MATRIX, encoding="ascii")
    plan = plan_workspace(tmp_path, "study")
    assert plan.points[0].geometry == f"inputs/{folder}/blade.obj"
    point = prepared(tmp_path)
    assert adapter().translate(point)


def test_layout_qprop_setup_blade_in_a_legacy_folder_still_works_beside_mesh(tmp_path):
    workspace(tmp_path)
    (tmp_path / "inputs/mesh").mkdir(exist_ok=True)
    shutil.copy(tmp_path / "inputs/geometry/blade.obj", tmp_path / "inputs/mesh/blade.obj")
    point = prepared(tmp_path)
    # The setup names the legacy file and it exists, so it is the one used.
    assert point.setup["blade"] == "inputs/geometry/blade.obj"
