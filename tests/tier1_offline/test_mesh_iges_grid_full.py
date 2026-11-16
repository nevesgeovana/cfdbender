"""Grid the named components of IGES files into one OBJ.

Usage: cfd-mesh iges-grid a.igs b.igs --grid grid.toml --out model.obj

Every IGES file here is written by the test from the fixed eighty-column layout: bilinear
B-spline patches (128), a surface of revolution (120) on a line axis (110), a free line (110),
and 406 form 15 NAME entities on levels. The gmsh tests skip where the optional extra is absent;
the name reader and the missing-extra refusal run everywhere.
"""

import json
import math
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest
from iges_support import (
    Iges,
    cli,
    gmsh_or_skip,
    grid_file,
    plate_and_flap,
    square,
    wedge,
)

from cfdbender._errors import GeometryWarning, NotApplicableWarning
from cfdbender.mesh._refine._obj import read_obj, read_te
from cfdbender.mesh._sidecars import read_boundary_declarations

# ---------------------------------------------------------------- names, without gmsh


def test_iges_grid_a_grid_value_that_cannot_be_read_is_refused_before_import(tmp_path, capsys):
    source = plate_and_flap(tmp_path)
    grid = grid_file(tmp_path, '[component.Flap]\nchord = {count = 4, law = "spline"}\n')
    out = tmp_path / "model.obj"
    code, _, stderr = cli(
        ["geometry", "iges-grid", str(source), "--grid", str(grid), "--out", str(out)], capsys
    )
    assert code == 2
    assert "law must be one of cosine, geometric, tanh, uniform" in stderr
    assert not out.exists()


# ---------------------------------------------------------------- grid, with gmsh

PLATE_GRID = """
[defaults]
size = 0.25
[component.Flap]
size = 0.25
chord = {count = 6, law = "cosine"}
span = {count = 4, law = "geometric", ratio = 1.2}
[component.Plate]
size = 0.25
"""


def run_grid(tmp_path, capsys, sources, text, out="model.obj"):
    gmsh_or_skip()
    grid = grid_file(tmp_path, text)
    target = tmp_path / out
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code, stdout, stderr = cli(
            [
                "geometry",
                "iges-grid",
                *map(str, sources),
                "--grid",
                str(grid),
                "--out",
                str(target),
            ],
            capsys,
        )
    assert code == 0, stderr
    record = json.loads((tmp_path / f"{out}.json").read_text(encoding="utf-8"))
    return target, record, caught


def test_iges_grid_families_are_the_named_components_in_the_grid_order(tmp_path, capsys):
    target, record, _ = run_grid(tmp_path, capsys, [plate_and_flap(tmp_path)], PLATE_GRID)
    mesh = read_obj(target)
    assert list(mesh.families) == ["Flap", "Plate"]
    assert list(record["components"]) == ["Flap", "Plate"]
    assert record["unit"] == "MILLIMETER"


def test_iges_grid_meshes_a_file_of_one_face(tmp_path, capsys):
    """One face has nothing to be joined to: it is meshed as it is, not stopped by the join."""
    iges = Iges()
    iges.name(1, "Plate")
    iges.patch(square(0, 0, 2, 1), level=1)
    path = iges.write(tmp_path / "one.igs")
    target, record, _ = run_grid(tmp_path, capsys, [path], "[defaults]\nsize = 0.25\n")
    assert list(read_obj(target).families) == ["Plate"]
    assert record["transfinite_patches"]["Plate"] == 1
    assert record["components"]["Plate"]["faces"] == 8 * 4
    assert record["fragment"]["faces_before"] == record["fragment"]["faces_after"] == 1
    assert record["fragment"]["split"] == 0


def test_iges_grid_shared_edges_are_conformal(tmp_path, capsys):
    target, record, _ = run_grid(tmp_path, capsys, [plate_and_flap(tmp_path)], PLATE_GRID)
    mesh = read_obj(target)
    flap = {v for f in mesh.families["Flap"] for v in f}
    plate = {v for f in mesh.families["Plate"] for v in f}
    shared = flap & plate
    # The flap's lower side, cut into 6 chordwise cells, is 7 nodes both families use.
    assert len(shared) == 7
    assert np.allclose(mesh.verts[sorted(shared)][:, 1], 1.0)
    assert record["conformal"]["conformal_mismatches"] == 0
    assert record["conformal"]["open_edges"] == record["conformal"]["boundary_edges"] > 0


def test_iges_grid_counts_an_unjoined_edge_as_a_mismatch(tmp_path, capsys):
    """The control of the conformal check: a gap wider than the join is not a boundary."""
    source = plate_and_flap(tmp_path, flap_gap=0.0005)
    text = PLATE_GRID.replace("[defaults]\n", "[defaults]\ntolerance = 0.0001\n")
    _, record, caught = run_grid(tmp_path, capsys, [source], text)
    assert record["conformal"]["conformal_mismatches"] > 0
    assert any("not boundaries of the body" in str(w.message) for w in caught)
    # A stated tolerance is the user's: the near miss is reported, not closed.
    assert record["closed_gaps"] == []


def shared_nodes(target, a, b):
    mesh = read_obj(target)
    return {v for f in mesh.families[a] for v in f} & {v for f in mesh.families[b] for v in f}


def test_iges_grid_closes_a_near_miss_join_from_its_measured_gap(tmp_path, capsys):
    """A gap far below the component size is a join the file missed: measured, then closed."""
    # 0.0005 is a fifth of a percent of the size 0.25, and far above the default join.
    source = plate_and_flap(tmp_path, flap_gap=0.0005)
    target, record, _ = run_grid(tmp_path, capsys, [source], PLATE_GRID)
    assert record["conformal"]["conformal_mismatches"] == 0
    assert len(shared_nodes(target, "Flap", "Plate")) == 7
    (closed,) = record["closed_gaps"]
    assert sorted(closed["components"]) == ["Flap", "Plate"]
    assert closed["gap"] == pytest.approx(0.0005, rel=1e-3)
    assert record["tolerance"] == pytest.approx(1.5 * closed["gap"], rel=1e-6)


def test_iges_grid_keeps_a_gap_of_the_component_scale_as_an_open_boundary(tmp_path, capsys):
    """A gap of 4 percent of the size is a real opening, such as a blunt trailing edge left open."""
    source = plate_and_flap(tmp_path, flap_gap=0.01)
    target, record, _ = run_grid(tmp_path, capsys, [source], PLATE_GRID)
    assert record["conformal"]["conformal_mismatches"] == 0
    assert record["conformal"]["boundary_edges"] == record["conformal"]["open_edges"]
    assert not shared_nodes(target, "Flap", "Plate")
    assert record["closed_gaps"] == []


def test_iges_grid_a_point_projected_past_a_curves_end_does_not_lie_on_it():
    """The numbers the no-cap tailplane root gave: the cut root's next piece, on the extension."""
    from cfdbender.mesh._iges_mesh import lies_on

    gaps = np.array([2.19e-07, 7.49e-07, 3.73e-07, 4.69e-07, 5.3e-08])
    beyond = np.array([1679.06967, 1679.6661, 1680.262529, 1680.858959, 1681.455389])
    assert not lies_on(beyond, (0.0, 1678.47323995), gaps, 0.5)
    # The control: the same gaps on the curve's own span lie on it, and a wide gap does not.
    within = np.array([10.0, 400.0, 800.0, 1200.0, 1678.47323995])
    assert lies_on(within, (0.0, 1678.47323995), gaps, 0.5)
    assert not lies_on(within, (0.0, 1678.47323995), gaps + 0.6, 0.5)


def curved_patch(iges, low, high, *, level):
    """A patch quadratic along u (three control points per row) and linear along v."""
    values = ["2", "1", "2", "1", "0", "0", "0", "0", "0"]
    values += ["0.0", "0.0", "0.0", "1.0", "1.0", "1.0", "0.0", "0.0", "1.0", "1.0"]
    values += ["1.0"] * 6
    for point in (*low, *high):
        values += [f"{c:.6f}" for c in point]
    values += ["0.0", "1.0", "0.0", "1.0"]
    return iges.add(128, values, level=level)


def plate_and_fin(folder):
    """A plate whose free lower edge is a parabola, and a fin on x = 1 that the join cuts it at."""
    iges = Iges()
    iges.name(1, "Plate")
    iges.name(2, "Fin")
    curved_patch(
        iges, [(0, 0, 0), (1, -0.4, 0), (2, 0, 0)], [(0, 1, 0), (1, 1, 0), (2, 1, 0)], level=1
    )
    iges.patch([(1, -0.5, 0), (1, 1.2, 0), (1, -0.5, 1), (1, 1.2, 1)], level=2)
    return iges.write(folder / "fin.igs")


def test_iges_grid_two_pieces_of_one_cut_edge_are_both_boundaries(tmp_path, capsys):
    """The pieces of a free edge the join cut lie on one curve's extension, never on each other."""
    text = "[defaults]\nsize = 0.25\n[component.Plate]\n[component.Fin]\n"
    target, record, _ = run_grid(tmp_path, capsys, [plate_and_fin(tmp_path)], text)
    assert record["fragment"]["split"] >= 1
    assert record["unsewn_curves"] == 0
    assert record["conformal"]["conformal_mismatches"] == 0
    assert record["conformal"]["boundary_edges"] == record["conformal"]["open_edges"] > 0


def test_iges_grid_four_sided_patches_follow_the_laws_and_are_quads(tmp_path, capsys):
    target, record, _ = run_grid(tmp_path, capsys, [plate_and_flap(tmp_path)], PLATE_GRID)
    mesh = read_obj(target)
    flap = mesh.families["Flap"]
    assert len(flap) == 6 * 4 and all(len(f) == 4 for f in flap)
    assert record["transfinite_patches"]["Flap"] == 1
    nodes = mesh.verts[sorted({v for f in flap for v in f})]
    # Span (y): geometric toward the tip (y = 2), each cell 1.2 times the next one outward.
    ys = np.unique(np.round(nodes[np.isclose(nodes[:, 0], 0.0)][:, 1], 9))
    cells = np.diff(ys)
    assert len(cells) == 4
    assert np.allclose(cells[:-1] / cells[1:], 1.2, rtol=1e-6)
    # Chord (x): cosine, symmetric, the end cells smaller than the middle ones.
    xs = np.unique(np.round(nodes[np.isclose(nodes[:, 1], 2.0)][:, 0], 9))
    chord = np.diff(xs)
    assert len(chord) == 6
    assert np.allclose(chord, chord[::-1], rtol=1e-6)
    assert chord[0] < chord[2]


def test_iges_grid_other_patches_are_quad_dominant_at_the_component_size(tmp_path, capsys):
    """A patch with three corners is no block: it is meshed quad-dominant at its size."""
    text = "[defaults]\nsize = 0.25\n[component.Tri]\n[component.Tab]\n"
    target, record, _ = run_grid(tmp_path, capsys, [triangle_and_tab(tmp_path)], text)
    assert record["transfinite_patches"]["Tri"] == 0
    assert record["components"]["Tri"]["quad_share"] >= 0.8
    mesh = read_obj(target)
    edges = [
        np.linalg.norm(mesh.verts[a] - mesh.verts[b])
        for f in mesh.families["Tri"]
        for a, b in zip(f, f[1:] + f[:1], strict=True)
    ]
    assert 0.1 < float(np.median(edges)) < 0.4


def _intervals_on(mesh, family, y):
    """Count the mesh edges of a family that lie on the line of constant ``y``."""
    edges = {
        tuple(sorted((a, b)))
        for f in mesh.families[family]
        for a, b in zip(f, f[1:] + f[:1], strict=True)
        if abs(mesh.verts[a][1] - y) < 1e-9 and abs(mesh.verts[b][1] - y) < 1e-9
    }
    return len(edges)


def test_iges_grid_a_shared_curve_override_keeps_opposite_composite_totals_equal(tmp_path, capsys):
    """The plate's top side is cut in two where the flap ends, and the flap's count wins on the
    shared half. The plate's bottom side follows the new total, so the plate is one structured
    block of quads only, and the record counts it because the mesh is structured."""
    target, record, caught = run_grid(tmp_path, capsys, [plate_and_flap(tmp_path)], PLATE_GRID)
    mesh = read_obj(target)
    bottom, top = _intervals_on(mesh, "Plate", 0.0), _intervals_on(mesh, "Plate", 1.0)
    assert bottom == top
    plate = mesh.families["Plate"]
    assert all(len(f) == 4 for f in plate)
    assert record["transfinite_patches"]["Plate"] == 1
    said = [str(w.message) for w in caught if issubclass(w.category, GeometryWarning)]
    assert any("Plate" in s and "opposite sides" in s for s in said), said


def test_iges_grid_a_block_the_mesh_does_not_structure_is_warned_and_not_counted():
    """Control: the planned plate block came out as the unequal-count mesh the review measured,
    42 quads and 2 triangles; it is not counted and a warning names the face. A block meshed
    as exactly its planned quads is counted. A patch that cannot hold equal totals is dropped
    from the blocks with a warning."""
    from types import SimpleNamespace

    from cfdbender.mesh._iges_mesh import drop_unbalanced, structured_blocks

    elements = {11: ([3, 2], [list(range(42)), [0, 1]]), 12: ([3], [list(range(80))])}
    gmsh = SimpleNamespace(
        model=SimpleNamespace(
            mesh=SimpleNamespace(getElements=lambda dim, face: (*elements[face], None))
        )
    )
    warned: list[str] = []
    counted = structured_blocks(
        gmsh, {"Plate": [11], "Flap": [12]}, {11: (10, 4), 12: (10, 8)}, warned
    )
    assert counted == {"Plate": 0, "Flap": 1}
    assert len(warned) == 1 and "Plate: face 11" in warned[0]
    assert "not counted as transfinite" in warned[0]
    patches, corners = {"Plate": [11, 13]}, {11: [1, 2, 3, 4]}
    drop_unbalanced(patches, corners, {11}, warned)
    assert patches == {"Plate": [13]} and corners == {}
    assert "Plate: face 11 cannot hold equal cell totals" in warned[-1]


def test_iges_grid_unequal_totals_that_no_raise_can_fix_are_reported():
    """A side whose curves are all tied to the other side cannot be raised alone; a side with
    a free curve is raised until the totals are equal, sparing a curve another component asks
    for while another curve can take the raise."""
    from cfdbender.mesh._iges_mesh import balance_counts

    class Groups:
        def __init__(self, tied):
            self.tied = tied

        def find(self, curve):
            return self.tied.get(curve, curve)

    lengths = dict.fromkeys((1, 2, 3, 4, 5), 1.0)
    tied = Groups({4: 1})
    counts = dict.fromkeys((1, 2, 3, 5), 3)
    assert balance_counts(tied, counts, {7: [[1, 2], [3], [4], [5]]}, lengths) == {7}
    free = Groups({})
    counts = {1: 3, 2: 3, 3: 3, 4: 2, 5: 2}
    loops = {9: [[1, 2], [3], [4, 5], [3]]}
    assert balance_counts(free, counts, loops, lengths, pinned=frozenset({4})) == set()
    assert counts == {1: 3, 2: 3, 3: 3, 4: 2, 5: 4}


def test_iges_grid_structured_false_meshes_a_four_sided_patch_unstructured(tmp_path, capsys):
    text = PLATE_GRID.replace("[component.Flap]\n", "[component.Flap]\nstructured = false\n")
    _, record, _ = run_grid(tmp_path, capsys, [plate_and_flap(tmp_path)], text)
    assert record["transfinite_patches"]["Flap"] == 0
    assert record["components"]["Flap"]["faces"] != 24


def test_iges_grid_warns_when_the_strip_rule_overrides_a_stated_count(tmp_path, capsys):
    """A chord side 0.1 long under a size of 0.25 is a strip of one cell: the chord count of 6 the
    grid file states is not silently dropped, it is warned with the component, side and counts."""
    iges = Iges()
    iges.name(1, "Plate")
    iges.name(2, "Flap")
    iges.patch(square(0, 0, 2, 1), level=1)
    iges.patch(square(0, 1, 0.1, 2), level=2)
    path = iges.write(tmp_path / "strip.igs")
    _, record, caught = run_grid(tmp_path, capsys, [path], PLATE_GRID)
    assert record["transfinite_patches"]["Flap"] == 1
    said = [str(w.message) for w in caught if issubclass(w.category, GeometryWarning)]
    expected = (
        "Flap: 1 patch(es) with a chord side shorter than the size 0.25 take 1 cell across it, "
        "not the chord count of 6"
    )
    assert any(expected in s for s in said), said
    assert any(expected in w for w in record["warnings"])
    # A law with no stated count takes its cells from the size and is not warned.
    _, again, caught = run_grid(
        tmp_path, capsys, [path], PLATE_GRID.replace("count = 6, ", ""), out="again.obj"
    )
    said = [str(w.message) for w in caught if issubclass(w.category, GeometryWarning)]
    assert not any("take 1 cell across it" in s for s in said), said


def triangle_and_tab(folder):
    """A three-cornered patch on level 1, its top edge collapsed to the apex (1, 1), and a tab on
    level 2 that shares the left half of its base, so the join cuts the base in two at (1, 0)."""
    iges = Iges()
    iges.name(1, "Tri")
    iges.name(2, "Tab")
    iges.patch([(0, 0, 0), (2, 0, 0), (1, 1, 0), (1, 1, 0)], level=1)
    iges.patch(square(0, -1, 1, 0), level=2)
    return iges.write(folder / "tri.igs")


def test_iges_grid_a_three_cornered_patch_with_a_cut_side_is_not_transfinite(tmp_path, capsys):
    """Four curves, three corners: the 180 degree point on the cut base is no corner."""
    text = "[defaults]\nsize = 0.25\n[component.Tri]\n[component.Tab]\n"
    target, record, _ = run_grid(tmp_path, capsys, [triangle_and_tab(tmp_path)], text)
    assert record["transfinite_patches"]["Tab"] == 1
    assert record["transfinite_patches"]["Tri"] == 0
    tri = record["components"]["Tri"]
    assert tri["least_scaled_jacobian"] > 0
    assert tri["quad_share"] > 0.5  # quad-dominant
    assert record["conformal"]["conformal_mismatches"] == 0


def corners(mesh, family):
    """Every corner angle of the faces of one family, in degrees."""
    out = []
    for face in mesh.families[family]:
        p = mesh.verts[face]
        for k in range(len(p)):
            a, b = p[k - 1] - p[k], p[(k + 1) % len(p)] - p[k]
            cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
            out.append(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))
    return np.asarray(out)


def thin_wing_on_body(folder):
    """A thin wedge (chord 1, span 0.03) whose root lies on a body plate at y = 0. The body comes
    first in the grid file, so the join leaves the wing's root chord with the body's orientation
    and its tip chord with its own: the two run against each other."""
    iges = Iges()
    iges.name(1, "Wing")
    iges.name(2, "Body")
    iges.patch([(0, 0, 0.01), (1, 0, 0), (0, 0.03, 0.01), (1, 0.03, 0)], level=1)
    iges.patch([(0, 0, -0.01), (1, 0, 0), (0, 0.03, -0.01), (1, 0.03, 0)], level=1)
    iges.patch([(-0.5, 0, -0.5), (1.5, 0, -0.5), (-0.5, 0, 0.5), (1.5, 0, 0.5)], level=2)
    return iges.write(folder / "thin.igs")


def test_iges_grid_a_one_sided_law_clusters_alike_on_opposite_sides_of_a_patch(tmp_path, capsys):
    """Both chord sides cluster toward the larger x whichever way the join left each curve: the
    cells are not sheared into slivers."""
    text = (
        "[defaults]\nsize = 0.25\n[component.Body]\n[component.Wing]\n"
        'chord = {count = 8, law = "geometric", ratio = 1.4}\n'
    )
    target, record, _ = run_grid(tmp_path, capsys, [thin_wing_on_body(tmp_path)], text)
    assert record["transfinite_patches"]["Wing"] == 2
    angles = corners(read_obj(target), "Wing")
    assert angles.min() > 80 and angles.max() < 100, (angles.min(), angles.max())
    wing = record["components"]["Wing"]
    assert wing["smallest_corner_deg"] == pytest.approx(angles.min(), abs=1e-6)
    assert wing["largest_corner_deg"] == pytest.approx(angles.max(), abs=1e-6)
    assert wing["sliver_faces"] == 0


def test_iges_grid_the_heal_fixes_degenerate_edges_and_leaves_small_edges():
    """Fixing small edges took the zero-length apex edge off two half cones of revolution, and the
    join gave both halves one half circle; the heal leaves small edges and the note says so."""
    from types import SimpleNamespace

    from cfdbender.mesh._iges_mesh import _heal

    calls = []

    def heal_shapes(tags, tolerance, degenerated, small_edges, small_faces, sew, solid):
        calls.append((degenerated, small_edges, small_faces, sew, solid))
        return tags

    occ = SimpleNamespace(healShapes=heal_shapes, synchronize=lambda: None)
    gmsh = SimpleNamespace(model=SimpleNamespace(occ=occ))
    note = _heal(gmsh, SimpleNamespace(faces=[1, 2]), 1e-6)
    assert calls == [(True, False, False, True, False)]
    assert note == "healed (degenerate edges fixed, faces sewn)"


def test_iges_grid_a_short_curve_the_join_leaves_makes_no_sliver(tmp_path, capsys):
    """The flap ends 0.01 short of the plate's corner, so the join leaves a curve of 0.01 on the
    plate's top edge, 25 times shorter than the size: the cells grow from it, none a sliver."""
    iges = Iges()
    iges.name(1, "Plate")
    iges.name(2, "Flap")
    iges.patch(square(0, 0, 2, 1), level=1)
    iges.patch(square(0, 1, 1.99, 2), level=2)
    path = iges.write(tmp_path / "short.igs")
    text = "[defaults]\nsize = 0.25\n[component.Flap]\n[component.Plate]\n"
    target, record, caught = run_grid(tmp_path, capsys, [path], text)
    angles = corners(read_obj(target), "Plate")
    assert angles.min() >= 5 and angles.max() <= 175, (angles.min(), angles.max())
    assert record["components"]["Plate"]["sliver_faces"] == 0
    assert record["conformal"]["conformal_mismatches"] == 0
    said = [str(w.message) for w in caught if issubclass(w.category, GeometryWarning)]
    assert not any("sliver" in s for s in said), said


def test_iges_grid_warns_a_sliver_the_geometry_forces_with_its_angles(tmp_path, capsys):
    """A three-cornered patch whose apex is 1.7 degrees cannot be meshed without a corner under
    5 degrees: warned with the count and the angles, recorded, and the grid is still written."""
    iges = Iges()
    iges.name(1, "Spike")
    iges.name(2, "Tab")
    iges.patch([(0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0.03, 0)], level=1)
    iges.patch(square(0, -1, 1, 0), level=2)
    path = iges.write(tmp_path / "spike.igs")
    text = "[defaults]\nsize = 0.25\n[component.Tab]\n[component.Spike]\n"
    target, record, caught = run_grid(tmp_path, capsys, [path], text)
    assert target.is_file()
    spike = record["components"]["Spike"]
    assert spike["sliver_faces"] >= 1
    assert spike["smallest_corner_deg"] < 5
    said = [str(w.message) for w in caught if issubclass(w.category, GeometryWarning)]
    expected = (
        f"Spike: {spike['sliver_faces']} sliver face(s) with a corner under 5 or over 175 "
        f"degrees (corners from {spike['smallest_corner_deg']:.3g} to "
        f"{spike['largest_corner_deg']:.4g})"
    )
    assert any(expected in s for s in said), said
    assert any(expected in w for w in record["warnings"])


def test_iges_grid_finds_a_sharp_trailing_edge_and_marks_it(tmp_path, capsys):
    text = (
        '[component.Wing]\nsize = 0.5\nchord = {count = 8, law = "cosine"}\n'
        'span = {count = 5, law = "uniform"}\ntrailing_edge = "auto"\n'
    )
    target, record, _ = run_grid(tmp_path, capsys, [wedge(tmp_path)], text)
    assert record["trailing_edges"]["Wing"]["edges"] == 5
    data = tomllib.loads(Path(str(target) + ".boundaries.toml").read_text(encoding="utf-8"))
    entry = data["trailing_edges"]["Wing"]
    assert entry["type"] == "STANDARD"
    unit, points = read_te(tmp_path / entry["file"])
    assert unit == "MILLIMETER"
    assert len(points) == 5
    assert np.allclose(points[:, 0], 1.0) and np.allclose(points[:, 2], 0.0)
    read_boundary_declarations(Path(str(target) + ".boundaries.toml"))


def blunt_wedge(folder, name="blunt.igs"):
    """A wing wedge whose trailing edge at x = 1 is a base 0.02 tall, closed at the nose."""
    iges = Iges()
    iges.name(1, "Wing")
    iges.patch([(0, 0, 0.1), (1, 0, 0.01), (0, 1, 0.1), (1, 1, 0.01)], level=1)
    iges.patch([(0, 0, -0.1), (1, 0, -0.01), (0, 1, -0.1), (1, 1, -0.01)], level=1)
    iges.patch([(1, 0, -0.01), (1, 0, 0.01), (1, 1, -0.01), (1, 1, 0.01)], level=1)
    iges.patch([(0, 0, -0.1), (0, 0, 0.1), (0, 1, -0.1), (0, 1, 0.1)], level=1)
    return iges.write(folder / name)


@pytest.mark.parametrize("outboard", ["sharp", "tip base"])
def test_iges_grid_one_rim_keeps_a_sharp_line_where_a_blunt_edge_closes(outboard):
    """A blunt base closing to a point inboard of a sharp trailing edge: the sharp line meets
    both rims at one node and is kept. A tip's base edge joins the dropped rim to the kept one at
    two different nodes and goes with the dropped rim."""
    from cfdbender.mesh.iges_grid import one_rim

    at = {
        "U0": (1.0, 0.0, 0.05),
        "U1": (1.0, 1.0, 0.05),
        "L0": (1.0, 0.0, -0.05),
        "L1": (1.0, 1.0, -0.05),
        "T": (1.0, 2.0, 0.0),
        "S1": (1.0, 3.0, 0.0),
        "S2": (1.0, 4.0, 0.0),
    }
    index = {name: i for i, name in enumerate(at)}
    verts = np.array(list(at.values()))

    def edge(a, b):
        return (min(index[a], index[b]), max(index[a], index[b]))

    rims = {edge("U0", "U1"), edge("L0", "L1")}
    faces = [[index[n] for n in ("U0", "U1", "L1", "L0")]]
    if outboard == "sharp":
        faces.append([index[n] for n in ("U1", "T", "L1")])
        rims |= {edge("U1", "T"), edge("L1", "T")}
        extra = {edge("T", "S1"), edge("S1", "S2")}
    else:
        extra = {edge("U1", "L1")}
    kept, dropped = one_rim(verts, faces, rims | extra)
    if outboard == "sharp":
        assert extra <= kept
        assert dropped == 2
    else:
        assert not extra & kept
        assert dropped == 2
    # One rim survives whole: upper or lower, never a mix.
    sides = {float(np.sign(verts[list(e)][:, 2].sum())) for e in kept - extra}
    assert len(sides) == 1


def test_iges_grid_marks_one_rim_of_a_blunt_trailing_edge(tmp_path, capsys):
    """Both rims of a blunt trailing edge are marked lines a base cell apart; one is kept."""
    text = (
        '[component.Wing]\nsize = 0.5\nchord = {count = 8, law = "cosine"}\n'
        'span = {count = 5, law = "uniform"}\ntrailing_edge = "auto"\n'
    )
    target, record, _ = run_grid(tmp_path, capsys, [blunt_wedge(tmp_path)], text)
    entry = record["trailing_edges"]["Wing"]
    _, points = read_te(tmp_path / entry["file"])
    assert len(points) == entry["edges"] == 5
    assert entry["dropped_rim_edges"] == 5
    assert np.allclose(points[:, 0], 1.0)
    # One rim: every marked edge lies on the same side of the base.
    assert np.allclose(np.abs(points[:, 2]), 0.01) and len(set(np.sign(points[:, 2]))) == 1
    read_boundary_declarations(Path(str(target) + ".boundaries.toml"))


def test_iges_grid_names_a_trailing_edge_file_the_file_system_takes(tmp_path, capsys):
    """a component name a file name cannot hold still gets its points file."""
    name = "Wing/Left: A"
    text = (
        f'[component."{name}"]\nsize = 0.5\nchord = {{count = 8, law = "cosine"}}\n'
        'span = {count = 5, law = "uniform"}\ntrailing_edge = "auto"\n'
    )
    target, record, _ = run_grid(tmp_path, capsys, [wedge(tmp_path, component=name)], text)
    data = tomllib.loads(Path(str(target) + ".boundaries.toml").read_text(encoding="utf-8"))
    file = data["trailing_edges"][name]["file"]
    assert file == record["trailing_edges"][name]["file"] == "model.Wing_Left_ A.te.txt"
    unit, points = read_te(tmp_path / file)
    assert unit == "MILLIMETER" and len(points) == 5


def test_iges_grid_two_names_that_differ_in_case_get_two_trailing_edge_files(tmp_path, capsys):
    """R2-002: ``Wing`` and ``wing`` are one file on Windows and macOS, so the second
    component's trailing-edge file took the first's and overwrote its points."""
    iges = Iges()
    for level, name, y in ((1, "Wing", 0.0), (2, "wing", 2.0)):
        iges.name(level, name)
        iges.patch([(0, y, 0.1), (1, y, 0), (0, y + 1, 0.1), (1, y + 1, 0)], level=level)
        iges.patch([(0, y, -0.1), (1, y, 0), (0, y + 1, -0.1), (1, y + 1, 0)], level=level)
        iges.patch([(0, y, -0.1), (0, y, 0.1), (0, y + 1, -0.1), (0, y + 1, 0.1)], level=level)
    text = "".join(
        f'[component.{name}]\nsize = 0.5\nchord = {{count = 8, law = "cosine"}}\n'
        'span = {count = 5, law = "uniform"}\ntrailing_edge = "auto"\n'
        for name in ("Wing", "wing")
    )
    _, record, _ = run_grid(tmp_path, capsys, [iges.write(tmp_path / "two.igs")], text)
    files = [record["trailing_edges"][name]["file"] for name in ("Wing", "wing")]
    assert files[0].casefold() != files[1].casefold(), files
    for file, y in zip(files, (0.5, 2.5), strict=True):
        points = read_te(tmp_path / file)[1]
        assert len(points) == 5 and all(y - 0.6 < p[1] < y + 0.6 for p in points), (file, points)


def test_iges_grid_takes_a_trailing_edge_from_a_named_iges_curve(tmp_path, capsys):
    text = (
        '[component.Wing]\nsize = 0.5\nchord = {count = 8, law = "cosine"}\n'
        'span = {count = 5, law = "uniform"}\ntrailing_edge = ["WingTE"]\n'
    )
    _, record, _ = run_grid(tmp_path, capsys, [wedge(tmp_path, te_curve=True)], text)
    assert record["trailing_edges"]["Wing"]["edges"] == 5


def box(iges, low, high, *, level):
    """The six faces of the box from ``low`` to ``high``, as bilinear patches."""
    (x0, y0, z0), (x1, y1, z1) = low, high
    iges.patch([(x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0)], level=level)
    iges.patch([(x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1)], level=level)
    iges.patch([(x0, y0, z0), (x1, y0, z0), (x0, y0, z1), (x1, y0, z1)], level=level)
    iges.patch([(x0, y1, z0), (x1, y1, z0), (x0, y1, z1), (x1, y1, z1)], level=level)
    iges.patch([(x0, y0, z0), (x0, y1, z0), (x0, y0, z1), (x0, y1, z1)], level=level)
    iges.patch([(x1, y0, z0), (x1, y1, z0), (x1, y0, z1), (x1, y1, z1)], level=level)


def body_and_slab(folder):
    """A closed body box and a closed wing slab that runs through it, wall to wall, untrimmed."""
    iges = Iges()
    iges.name(1, "Body")
    iges.name(2, "Wing")
    box(iges, (0, 0, 0), (2, 2, 2), level=1)
    box(iges, (0.5, -1, 0.7), (1.5, 3, 1.3), level=2)
    return iges.write(folder / "crossing.igs")


CROSSING_GRID = "[defaults]\nsize = 0.25\n[component.Body]\n[component.Wing]\nsize = 0.1\n"


def centres(target, name):
    mesh = read_obj(target)
    return np.array([mesh.verts[f].mean(axis=0) for f in mesh.families[name]])


def non_manifold_edges(target):
    mesh = read_obj(target)
    uses = {}
    for faces in mesh.families.values():
        for f in faces:
            for a, b in zip(f, list(f[1:]) + [f[0]], strict=True):
                key = (min(a, b), max(a, b))
                uses[key] = uses.get(key, 0) + 1
    return sum(1 for n in uses.values() if n > 2)


def crossing_mesh(*, half=False):
    """A conformal quad mesh of a body box [0,2]^3 and a wing slab x 0.5..1.5, z 0.7..1.3 that
    runs through it along y from -1 to 3, untrimmed: the four faces along each crossing edge
    are the body's inside and outside the slab and the slab's inside and outside the body.

    ``half`` cuts both at the plane y = 0 and keeps y >= 0 open there, as a half model is.
    """
    nodes, families = {}, {"Body": [], "Wing": []}

    def node(p):
        return nodes.setdefault(tuple(round(c, 9) for c in p), len(nodes))

    def wall(name, axis, value, us, vs):
        others = [a for a in range(3) if a != axis]
        for i in range(len(us) - 1):
            for j in range(len(vs) - 1):
                quad = []
                for u, v in (
                    (us[i], vs[j]),
                    (us[i + 1], vs[j]),
                    (us[i + 1], vs[j + 1]),
                    (us[i], vs[j + 1]),
                ):
                    p = [0.0, 0.0, 0.0]
                    p[axis], p[others[0]], p[others[1]] = value, u, v
                    quad.append(node(p))
                families[name].append(quad)

    xs, zs = [0, 0.5, 1, 1.5, 2], [0, 0.7, 1.3, 2]
    ys = [0, 1, 2]
    for value in (0, 2):
        wall("Body", 0, value, ys, zs)
        wall("Body", 2, value, xs, ys)
    for value in [2] if half else [0, 2]:
        wall("Body", 1, value, xs, zs)
    slab_y = [0, 1, 2, 3] if half else [-1, 0, 1, 2, 3]
    for value in (0.5, 1.5):
        wall("Wing", 0, value, slab_y, [0.7, 1.3])
    for value in (0.7, 1.3):
        wall("Wing", 2, value, [0.5, 1, 1.5], slab_y)
    for value in [3] if half else [-1, 3]:
        wall("Wing", 1, value, [0.5, 1, 1.5], [0.7, 1.3])
    verts = np.zeros((len(nodes), 3))
    for p, i in nodes.items():
        verts[i] = p
    return verts, families


def mesh_edges_used_more_than_twice(families):
    uses = {}
    for faces in families.values():
        for f in faces:
            for a, b in zip(f, f[1:] + f[:1], strict=True):
                uses[(min(a, b), max(a, b))] = uses.get((min(a, b), max(a, b)), 0) + 1
    return sum(1 for n in uses.values() if n > 2)


@pytest.mark.parametrize("half", [False, True])
def test_iges_grid_the_faces_inside_where_components_cross_are_removed(half):
    """The slab inside the body and the body wall inside the slab are not the outer skin; a half
    model, open on its plane of symmetry, loses the same faces."""
    from cfdbender.geometry._iges_skin import outer_skin

    verts, families = crossing_mesh(half=half)
    assert mesh_edges_used_more_than_twice(families) > 0
    kept, removed, note = outer_skin(verts, families)
    assert note == "removed"
    wing = np.array([verts[f].mean(axis=0) for f in kept["Wing"]])
    assert not ((wing[:, 1] > 0) & (wing[:, 1] < 2)).any()
    body = np.array([verts[f].mean(axis=0) for f in kept["Body"]])
    footprint = (body[:, 0] > 0.5) & (body[:, 0] < 1.5) & (body[:, 2] > 0.7) & (body[:, 2] < 1.3)
    assert not footprint.any()
    assert mesh_edges_used_more_than_twice(kept) == 0
    # Inside the body the slab is 2 long: upper and lower faces 1 wide, sides 0.6 high.
    assert removed["Wing"]["area"] == pytest.approx(2 * (2 * 1.0 + 2 * 0.6))
    assert removed["Wing"]["faces"] == 12
    assert removed["Body"]["faces"] == (2 if half else 4)
    # The outer skin keeps every face that is not inside.
    total = sum(len(f) for f in families.values())
    assert sum(len(f) for f in kept.values()) == total - 12 - removed["Body"]["faces"]


def test_iges_grid_inner_faces_kept_whole_are_warned():
    """An open sheet through a closed body leaves no selection that closes every crossing: every
    face is kept, and a warning says so (the crossing warning before the mesh said they would be
    removed)."""
    from types import SimpleNamespace

    from cfdbender.mesh._iges_grid_config import GridConfig
    from cfdbender.mesh.iges_grid import _inner

    verts, families = crossing_mesh()
    sheet = [f for f in families["Wing"] if np.allclose(verts[f][:, 2], 0.7)]
    meshed = SimpleNamespace(
        nodes=verts,
        faces={"Body": families["Body"], "Wing": sheet},
        record={"fragment": {"crossings": [{"components": ["Body", "Wing"], "curves": 2}]}},
    )
    config = GridConfig(size=None, tolerance=None, te_angle_deg=60.0)
    said: list[str] = []
    kept, removed, note = _inner(config, meshed, ["Body", "Wing"], said)
    assert note.startswith("kept whole: no selection")
    assert removed == {}
    assert sum(map(len, kept.values())) == len(families["Body"]) + len(sheet)
    assert any("inside the assembly" in w and "are kept" in w for w in said)
    config.remove_inner_faces = False
    said = []
    _inner(config, meshed, ["Body", "Wing"], said)
    assert any("remove_inner_faces = false" in w and "are kept" in w for w in said)


def test_iges_grid_keeps_the_inner_faces_when_told_to(tmp_path, capsys):
    text = CROSSING_GRID.replace("[defaults]\n", "[defaults]\nremove_inner_faces = false\n")
    target, record, _ = run_grid(tmp_path, capsys, [body_and_slab(tmp_path)], text)
    wing = centres(target, "Wing")
    assert ((wing[:, 1] > 0.01) & (wing[:, 1] < 1.99)).any()
    assert non_manifold_edges(target) > 0
    assert record["inner_faces"] == "kept whole: remove_inner_faces = false"
    assert record["components"]["Wing"]["inner_faces_removed"] == 0
    bad = CROSSING_GRID.replace("[defaults]\n", '[defaults]\nremove_inner_faces = "no"\n')
    grid = grid_file(tmp_path, bad)
    code, _, stderr = cli(
        [
            "geometry",
            "iges-grid",
            str(body_and_slab(tmp_path)),
            "--grid",
            str(grid),
            "--out",
            str(tmp_path / "bad.obj"),
        ],
        capsys,
    )
    assert code == 2
    assert "defaults.remove_inner_faces must be true or false" in stderr


def test_iges_grid_quality_is_warned_with_its_numbers_and_never_refused(tmp_path, capsys):
    text = (
        "[defaults]\nsize = 0.25\n[component.Flap]\n"
        'chord = {count = 1, law = "uniform"}\nspan = {count = 40, law = "uniform"}\n'
    )
    target, record, caught = run_grid(tmp_path, capsys, [plate_and_flap(tmp_path)], text)
    assert target.is_file()
    aspect = record["components"]["Flap"]["largest_aspect"]
    assert aspect > 20
    said = [str(w.message) for w in caught if issubclass(w.category, GeometryWarning)]
    assert any("Flap: largest aspect" in s and f"{aspect:.3g}" in s for s in said), said
    assert any("Flap: largest aspect" in w for w in record["warnings"])


def test_iges_grid_warns_a_component_with_no_table_and_a_table_with_no_component(tmp_path, capsys):
    text = (
        "[defaults]\nsize = 0.3\n[component.Rudder]\nsize = 0.1\n[component.Plate]\nsize = 0.25\n"
    )
    target, record, caught = run_grid(tmp_path, capsys, [plate_and_flap(tmp_path)], text)
    said = [str(w.message) for w in caught if issubclass(w.category, NotApplicableWarning)]
    assert any("component.Rudder names no component" in s for s in said), said
    assert any("component Flap has no [component.Flap] table" in s for s in said), said
    assert list(read_obj(target).families) == ["Plate", "Flap"]
    assert record["components"]["Flap"]["size"] == 0.3


def test_iges_grid_scales_every_file_to_the_first_files_unit(tmp_path, capsys):
    first = plate_and_flap(tmp_path, name="mm.igs")
    iges = Iges("IN")
    iges.name(5, "Tab")
    iges.patch(square(0, -1, 0.05, 0), level=5)
    second = iges.write(tmp_path / "in.igs")
    text = "[defaults]\nsize = 0.25\n"
    target, record, _ = run_grid(tmp_path, capsys, [first, second], text)
    mesh = read_obj(target)
    tab = mesh.verts[sorted({v for f in mesh.families["Tab"] for v in f})]
    assert np.isclose(tab[:, 1].min(), -25.4) and np.isclose(tab[:, 0].max(), 1.27)
    assert record["unit"] == "MILLIMETER"
    assert any("in.igs is in INCH and was scaled to MILLIMETER" in w for w in record["warnings"])


# ---------------------------------------------------------------- convert, with gmsh


def directory_kinds(path):
    lines = [line for line in path.read_text(encoding="latin-1").splitlines() if len(line) >= 73]
    rows = [line for line in lines if line[72] == "D"]
    return [int(rows[i][0:8]) for i in range(0, len(rows), 2)]


def directory_rows(path):
    """Each directory entry's type, status field and first parameter record, by sequence number."""
    lines = [line for line in path.read_text(encoding="latin-1").splitlines() if len(line) >= 73]
    rows = [line for line in lines if line[72] == "D"]
    params = {}
    for line in lines:
        if line[72] == "P":
            params.setdefault(int(line[64:72]), "")
            params[int(line[64:72])] += line[:64]
    return {
        i + 1: (int(rows[i][0:8]), rows[i][64:72], params.get(i + 1, "").split(";")[0].split(","))
        for i in range(0, len(rows), 2)
    }


def occ_like_trimmed_patch(folder):
    """A trimmed patch as the OpenCASCADE writer leaves it: the 142 flagged as parameter-space
    geometry (use 05) with PREF 3, its 102 boundary in parameter space, the members flagged 00."""
    iges = Iges()
    iges.name(1, "Body")
    surface = iges.patch(square(0, 0, 1, 1), level=1, status="00010000")
    curve = ["1", "1", "0", "0", "1", "0", "0.0", "0.0", "1.0", "1.0", "1.0", "1.0"]
    a = iges.add(
        126, curve + ["0.0", "0.0", "0.0", "1.0", "0.0", "0.0", "0.0", "1.0"], status="00010000"
    )
    b = iges.add(
        126, curve + ["1.0", "0.0", "0.0", "1.0", "1.0", "0.0", "0.0", "1.0"], status="00010000"
    )
    loop = iges.add(102, ["2", str(a), str(b)], status="00010000")
    model = iges.add(
        126, curve + ["0.0", "0.0", "0.0", "1.0", "1.0", "0.0", "0.0", "1.0"], status="00010000"
    )
    on = iges.add(142, ["0", str(surface), str(loop), str(model), "3"], status="00010500")
    iges.add(144, [str(surface), "1", "0", str(on)], level=1)
    return iges.write(folder / "occ.igs"), {
        "surface": surface,
        "members": (a, b),
        "loop": loop,
        "model": model,
        "on": on,
    }


def test_iges_grid_a_bare_output_name_lands_in_the_workspace_with_its_provenance(tmp_path, capsys):
    gmsh_or_skip()
    source = plate_and_flap(tmp_path)
    grid = grid_file(tmp_path, PLATE_GRID)
    ws = tmp_path / "ws"
    code, stdout, stderr = cli(
        ["--workspace", str(ws), "iges-grid", str(source), "--grid", str(grid), "--out", "m.obj"],
        capsys,
    )
    assert code == 0, stderr
    assert json.loads(stdout)["faces"] > 0
    mesh = ws / "inputs" / "mesh"
    for name in ("m.obj", "m.obj.json", "m.obj.boundaries.toml", "m.obj.provenance.json"):
        assert (mesh / name).is_file(), name


def test_iges_grid_a_grid_file_with_a_build_key_warns_and_skips_it(tmp_path):
    from cfdbender.mesh._iges_grid_config import read_grid

    path = grid_file(tmp_path, 'fs_build = "26.127"\n[defaults]\nsize = 0.25\n')
    with pytest.warns(NotApplicableWarning, match="fs_build"):
        read_grid(path)
