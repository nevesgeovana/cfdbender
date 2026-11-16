"""Meshing a generated configuration into all-quad patches (needs gmsh).

The kernel-bound checks of the mesher. The refusal and the unknown-name checks run everywhere.
"""

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernel_support import kernel_or_skip

from cfdbender._errors import InputValueError
from cfdbender.geometry._geometry_mesh import junction_bays, mesh_generated
from cfdbender.geometry._mesh_topology import read_obj
from cfdbender.mesh._extras import MissingExtraError

WB = {
    "components": {
        "FUS": {"type": "body"},
        "WING": {
            "type": "lifting_surface",
            "attach": {"parent": "FUS", "station": 0.4},
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
        },
        "FAIRING": {
            "type": "wing_fuselage_fairing",
            "hosts": ["FUS", "WING"],
            "attach": {"parent": "FUS", "station": 0.0},
        },
    },
}


def need_gmsh():
    return kernel_or_skip()


def test_geom_mesh_stages_unknown_component_is_refused_with_the_names_found():
    need_gmsh()
    with pytest.raises(InputValueError, match="unknown; available"):
        mesh_generated(WB, component="NOPE")


def test_geom_mesh_stages_a_fairing_cannot_be_meshed_alone():
    need_gmsh()
    with pytest.raises(InputValueError, match="only body and lifting_surface"):
        mesh_generated(WB, component="FAIRING")


def test_geom_mesh_stages_assembly_needs_one_fairing():
    need_gmsh()
    raw = {"components": {"FUS": {"type": "body"}}}
    with pytest.raises(InputValueError, match="exactly one wing_fuselage_fairing"):
        mesh_generated(raw)


def test_geom_mesh_stages_missing_gmsh_is_refused_with_the_install_command(monkeypatch):
    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "gmsh" or name.startswith("gmsh."):
            raise ImportError("no gmsh")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.delitem(sys.modules, "gmsh", raising=False)
    with pytest.raises(MissingExtraError, match="gmsh"):
        mesh_generated(WB, component="FUS")


def test_geom_mesh_stages_junction_bays_follow_the_family_order_and_skip_the_cap():
    patches = [SimpleNamespace(family=f) for f in ("a", "b", "c", "d", "tip_cap")]
    expected = ["inner", "outer", "outboard", "inner", None]
    assert junction_bays(patches) == expected  # type: ignore[arg-type]


def test_geom_mesh_stages_the_body_meshes_all_quad_and_is_accepted(tmp_path):
    need_gmsh()
    out = tmp_path / "fus.obj"
    result = mesh_generated(WB, component="FUS", obj=out)
    record = result.record
    assert record["accepted"] is True
    assert record["non_quad_count"] == 0
    assert record["inverted_count"] == 0
    assert record["degenerate_count"] == 0
    assert record["non_manifold_count"] == 0
    assert record["free_edges_off_y0_count"] == 0
    assert record["spacing_rules"]["passed"] is True
    again = read_obj(out)
    assert len(again.cells) == len(result.mesh.cells)


def test_geom_mesh_stages_the_wing_meshes_all_quad_with_its_tip_cap():
    need_gmsh()
    record = mesh_generated(WB, component="WING").record
    assert record["accepted"] is True
    assert record["non_quad_count"] == 0
    assert record["cap_cells"] > 0
    assert record["spacing_rules"]["passed"] is True


def test_geom_mesh_stages_the_assembly_has_no_open_or_nonmanifold_edge_and_triangles_only_in_the():
    need_gmsh()
    result = mesh_generated(WB)
    record = result.record
    assert record["non_manifold_count"] == 0
    assert record["free_edges_off_y0_count"] == 0
    assert record["degenerate_count"] == 0
    assert record["forbidden_non_quad_cells"] == []
    assert record["inconsistent_shared_edge_directions"] == 0
    assert set(result.mesh.components) >= {"FUS", "WING"}
    assert record["accepted"] is (record["inverted_count"] == 0)


def test_geom_mesh_stages_the_module_names_no_research_path():
    text = Path(sys.modules["cfdbender.geometry._geometry_mesh"].__file__).read_text(
        encoding="utf-8"
    )
    assert "sys.path" not in text
    assert "GeoverseResearch" not in text
