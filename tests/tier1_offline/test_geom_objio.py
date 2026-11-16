"""OBJ input and welding: the group reader, the weld and the plane cut of a surface mesh."""

from __future__ import annotations

import numpy as np
import pytest

from cfdbender._errors import InputValueError
from cfdbender.mesh.objio import obj_group_surface, plane_cut_pieces, read_mesh, weld

TWO_GROUPS = """\
# two groups
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 5 5 5
g hub
f 1 2 5
g blade
f 1 2 3 4
f -2 -4 -3
"""


def _cylinder(sides: int = 12, rings: int = 3):
    """Open triangulated tube of radius 1 about z, from z = 0 to z = rings - 1."""
    angles = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    points = np.array([[np.cos(a), np.sin(a), float(k)] for k in range(rings) for a in angles])
    faces = []
    for k in range(rings - 1):
        for i in range(sides):
            a, b = k * sides + i, k * sides + (i + 1) % sides
            c, d = a + sides, b + sides
            faces += [[a, b, d], [a, d, c]]
    return points, np.array(faces)


def test_geom_objio_group_reader_fans_polygons_in_file_order(tmp_path):
    obj = tmp_path / "rotor.obj"
    obj.write_text(TWO_GROUPS, encoding="utf-8")
    vertices, triangles = obj_group_surface(obj, "blade")
    # vertex 5 belongs to the hub only, so the blade keeps the first four, in file order
    assert vertices.tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    # the quad becomes the fan (1,2,3) (1,3,4); the negative indices count from the end
    assert triangles.tolist() == [[0, 1, 2], [0, 2, 3], [3, 1, 2]]


def test_geom_objio_group_reader_refuses_an_absent_group(tmp_path):
    obj = tmp_path / "rotor.obj"
    obj.write_text(TWO_GROUPS, encoding="utf-8")
    with pytest.raises(InputValueError, match="no face of 'tip'"):
        obj_group_surface(obj, "tip")


def test_geom_objio_weld_merges_a_doubled_seam_and_drops_collapsed_faces():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 2], [1, 3, 4]])
    merged, triangles = weld(points, faces)
    assert len(merged) == 4
    # the third face used vertices 1 and 3, now one vertex: it collapses and is dropped
    assert len(triangles) == 2
    np.testing.assert_allclose(merged[triangles[1]], [[1, 0, 0], [1, 1, 0], [0, 1, 0]])


def test_geom_objio_plane_cut_closes_around_a_tube():
    points, faces = _cylinder()
    pieces = plane_cut_pieces(points, faces, 0.5)
    assert pieces is not None and len(pieces) == 1
    xy, closed = pieces[0]
    assert closed
    radius = np.hypot(xy[:, 0], xy[:, 1])
    # every cut point lies on a facet of the 12-sided tube: between its apothem and its radius
    assert np.all(radius >= np.cos(np.pi / 12) - 1e-12) and np.all(radius <= 1.0 + 1e-12)
    assert len(xy) == 24


def test_geom_objio_plane_cut_is_open_on_a_strip_and_empty_off_the_mesh():
    points, faces = _cylinder()
    strip = faces[:4]  # four triangles of the first ring: an open strip
    pieces = plane_cut_pieces(points, strip, 0.5)
    assert pieces is not None and len(pieces) == 1 and not pieces[0][1]
    assert plane_cut_pieces(points, faces, 10.0) == []


def test_geom_objio_plane_cut_reports_a_branch_as_none():
    # three triangles share the edge (0, 1) that the plane z = 0.5 crosses
    points = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0.5], [-1, 0, 0.5], [0, 1, 0.5]], float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    assert plane_cut_pieces(points, faces, 0.25) is None


def test_geom_objio_read_mesh_loads_through_trimesh(tmp_path):
    obj = tmp_path / "quad.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n", encoding="utf-8")
    mesh = read_mesh(obj)
    assert mesh.faces.shape == (2, 3)
