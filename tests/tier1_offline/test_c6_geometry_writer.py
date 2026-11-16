"""Geometry and mesh share a point writer below their peer layers."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module", ["cfdbender.geometry.trailing_edges", "cfdbender.mesh._te_points"]
)
def test_point_writer_is_shared_below_geometry_and_mesh(module, tmp_path):
    writer = importlib.import_module(module).write_trailing_edge_points
    assert writer.__module__.startswith("cfdbender.")
    assert not writer.__module__.startswith(("cfdbender.geometry.", "cfdbender.mesh."))
    target = tmp_path / "edge.txt"
    writer(target, [(1.0, 2.0, 3.0)], unit="METER")
    assert target.read_text().splitlines() == ["METER", "1.0,2.0,3.0"]
