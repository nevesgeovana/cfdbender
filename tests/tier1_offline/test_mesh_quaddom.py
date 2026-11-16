"""Tier 1, 0.38.0 S1: the quad-dominant remesh of a remeshed family.

A remeshed family whose element mode is ``"quad-dominant"`` is remeshed into
triangles as any other, and its triangles are then paired into
quadrilaterals across edges that are no curve, each quadrilateral convex,
within the angle band and the warp of ``_geometry``, and its free nodes are
smoothed back onto the source surface. Every fixture is synthetic: a flat
plate, a cylindrical strip, a plate cut into two families, a plate with an
interior curve, and the sheet with a triangle strip of
``tests/p0380_mesh_fixtures.py``. Each measurement is also run on an input
it must reject, or with the feature off, so a check that accepts everything
cannot pass.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import numpy
import pytest
import trimesh
from mesh_fixtures import (  # noqa: F401
    HAS_INSPECT,
    diag,
    plate_cells,
    plate_nodes,
    sheet_with_strips,
    two_families,
    write_source,
)

from cfdbender._errors import InputArtifactError
from cfdbender.mesh._refine import audit_mesh, refine_mesh
from cfdbender.mesh._refine._geometry import (
    QUAD_ANGLE_BAND,
    QUAD_WARP_DEGREES,
    edge_faces,
    normals_and_centroids,
    quad_shapes,
)
from cfdbender.mesh._refine._obj import ObjMesh, read_obj, write_obj
from cfdbender.mesh._refine._remesh import (
    BOUNDARY,
    TRAILING_EDGE,
    Patch,
    Remesher,
    refine_group,
)

QD = "quad-dominant"
#: The share of quadrilaterals a remeshed flat plate must reach (measured: 0.99).
FLAT_SHARE = 0.9
#: The share a cylindrical strip must reach (measured above 0.9).
CURVED_SHARE = 0.8
STRIP = (0.03, 0.04, 0.05, 0.065, 0.08)

# ---- fixtures and measurements


def _flat_plate(n: int = 12) -> ObjMesh:
    nodes = plate_nodes(n, n)
    nodes[:, 2] = 0.0
    return ObjMesh(nodes, {"P": plate_cells(n, [(i, j) for j in range(n) for i in range(n)])})


def _cylinder_strip(radius: float = 1.0, step_degrees: float = 5.0, n: int = 12) -> ObjMesh:
    """Return a strip of the cylinder of the given radius about z, n by n cells."""
    theta = numpy.radians(step_degrees) * numpy.arange(n + 1)
    t, z = numpy.meshgrid(theta, numpy.linspace(0.0, 1.0, n + 1))
    nodes = numpy.column_stack(
        [radius * numpy.cos(t.ravel()), radius * numpy.sin(t.ravel()), z.ravel()]
    )
    return ObjMesh(nodes, {"C": plate_cells(n, [(i, j) for j in range(n) for i in range(n)])})


def _quads(points: numpy.ndarray, faces: list[list[int]]) -> numpy.ndarray:
    return points[numpy.array([f for f in faces if len(f) == 4])].reshape(-1, 4, 3)


def _share(faces: list[list[int]]) -> float:
    return sum(len(f) == 4 for f in faces) / len(faces)


def _edges(faces: list[list[int]]) -> set[tuple[int, int]]:
    return set(edge_faces(faces))


def _missing_edges(points, faces, a: numpy.ndarray, b: numpy.ndarray) -> int:
    """Return how many segments a-b are not an edge of the faces (matched by end positions)."""
    index = {tuple(numpy.round(p, 12)): v for v, p in enumerate(points)}
    edges = _edges(faces)
    missing = 0
    for p, q in zip(a, b, strict=True):
        u, w = index.get(tuple(numpy.round(p, 12))), index.get(tuple(numpy.round(q, 12)))
        if u is None or w is None or (min(u, w), max(u, w)) not in edges:
            missing += 1
    return missing


def _merge_across(faces: list[list[int]], edge: tuple[int, int]) -> list[list[int]]:
    """Return the faces with the two across an edge replaced by one polygon (a control)."""
    i, j = edge_faces(faces)[edge]
    f, g = faces[i], faces[j]
    x, y = edge if f[(f.index(edge[0]) + 1) % len(f)] == edge[1] else edge[::-1]
    f = f[f.index(y) :] + f[: f.index(y)]
    g = g[g.index(x) :] + g[: g.index(x)]
    merged = f + g[1:-1]
    return [h for k, h in enumerate(faces) if k not in (i, j)] + [merged]


# ---- the pairing


def _pair(verts, faces, labels, curve=None) -> tuple[Remesher, int]:
    """Return a remesher of the given triangles and how many pairs it merged."""
    points = numpy.asarray(verts, dtype=float)
    remesher = Remesher(Patch(points, faces, labels, numpy.ones(len(points)), dict(curve or {})))
    return remesher, remesher.pair_quads({"P"})


SQUARE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
TWO = [[0, 1, 2], [0, 2, 3]]


@pytest.mark.parametrize(
    ("verts", "labels", "curve"),
    [
        (SQUARE, ["P", "P"], {(0, 2): TRAILING_EDGE}),
        (SQUARE, ["P", "Q"], {}),
        ([[0.0, 0.0, 0.0], [1.0, -0.15, 0.0], [2.0, 0.0, 0.0], [1.0, 0.15, 0.0]], ["P", "P"], {}),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.6], [0.0, 1.0, 0.0]], ["P", "P"], {}),
        (
            [[0.0, 0.0, 0.0], [0.866, -0.5, 0.0], [0.633, 0.0, 0.0], [0.866, 0.5, 0.0]],
            ["P", "P"],
            {},
        ),
    ],
    ids=[
        "across-a-curve",
        "across-two-families",
        "angle-under-the-band",
        "warp-over-the-limit",
        "not-convex",
    ],
)
def test_a_pair_is_merged_only_into_a_valid_quadrilateral_of_one_family(verts, labels, curve):
    """no merge across a curve or two families, or out of bounds.

    Two triangles sharing the edge 0-2 are not merged when the edge is a
    curve, when they belong to two families, when the quadrilateral has an
    angle of 17 degrees (under QUAD_ANGLE_BAND), when it folds by 43
    degrees across its diagonal (over QUAD_WARP_DEGREES), or when it is not
    convex (a dart whose reflex corner of 230 degrees measures 130, inside
    the band; across its other diagonal it folds by 180 degrees, so the warp
    refuses it as well). The control is the flat unit square of one family,
    which is merged into one quadrilateral that runs the way its triangles
    did.
    """
    remesher, merged = _pair(verts, TWO, labels, curve)
    assert merged == 0 and sorted(len(f) for f in remesher.F.values()) == [3, 3]
    remesher, merged = _pair(SQUARE, TWO, ["P", "P"])
    assert merged == 1 and list(remesher.F.values()) == [[2, 3, 0, 1]]
    assert normals_and_centroids(numpy.asarray(SQUARE), [[2, 3, 0, 1]])[0][0, 2] > 0


def _fan(far: float) -> tuple[Remesher, int]:
    """Return the 2 by 2 rectangles around node 4, the right column at x = far, paired."""
    xs, ys = (-1.0, 0.0, far), (-1.0, 0.0, 1.0)
    verts = [[x, y, 0.0] for y in ys for x in xs]
    faces = []
    for j in range(2):
        for i in range(2):
            a, b, c, d = 3 * j + i, 3 * j + i + 1, 3 * j + i + 4, 3 * j + i + 3
            faces += [[a, b, c], [a, c, d]]
    ring = [0, 1, 2, 5, 8, 7, 6, 3]
    curve = {
        (min(u, w), max(u, w)): BOUNDARY for u, w in zip(ring, ring[1:] + ring[:1], strict=True)
    }
    return _pair(verts, faces, ["P"] * len(faces), curve)


def test_a_smoothing_move_that_breaks_a_quadrilateral_is_undone():
    """smoothing never leaves a quadrilateral outside the bounds.

    With the right column far away (x = 30), moving the free node halfway to
    its edge neighbors' centroid would give its upper-left quadrilateral an
    angle of 165 degrees; the move is undone and the node stays. The control
    puts the column at x = 1.5, and the node moves.
    """
    remesher, merged = _fan(30.0)
    assert merged == 4
    assert remesher.smooth_quads({"P"}) == 0
    assert numpy.array_equal(remesher.V[4], numpy.zeros(3))
    remesher, merged = _fan(1.5)
    assert merged == 4 and remesher.smooth_quads({"P"}) == 1
    assert remesher.V[4][0] > 0.01


def test_a_flat_plate_is_mostly_quadrilaterals_each_convex_within_the_band_and_planar():
    """a flat plate refined quad-dominant is above FLAT_SHARE quads.

    Every quadrilateral is convex, its angles lie in QUAD_ANGLE_BAND, its warp
    is within QUAD_WARP_DEGREES, its nodes on the plane z = 0, and it faces
    the way the source does. The control is the same plate in the default
    mode, whose share is zero.
    """
    mesh = _flat_plate()
    result = refine_group(mesh, ["P"], (), 2.0, elements=QD)
    faces = result.families["P"]
    assert _share(faces) > FLAT_SHARE, _share(faces)
    angle, warp, convex = quad_shapes(_quads(result.points, faces))
    assert convex.all()
    assert QUAD_ANGLE_BAND[0] <= angle.min() and angle.max() <= QUAD_ANGLE_BAND[1]
    assert warp.max() <= QUAD_WARP_DEGREES
    assert numpy.abs(result.points[:, 2]).max() < 1e-12
    source_up = numpy.sign(normals_and_centroids(mesh.verts, mesh.families["P"])[0][0, 2])
    normals, _ = normals_and_centroids(result.points, faces)
    assert (numpy.sign(normals[:, 2]) == source_up).all()
    assert result.info["edge_ratio_median"] > 0
    plain = refine_group(mesh, ["P"], (), 2.0)
    assert _share(plain.families["P"]) == 0.0
    assert {len(f) for f in faces} == {3, 4} or {len(f) for f in faces} == {4}


@pytest.mark.parametrize("factor", [2.0, 0.5])
def test_a_curved_strip_keeps_every_node_on_the_surface(factor):
    """smoothing projects onto the source; nodes stay on the cylinder.

    Every node lies on the source triangulation within 1e-9 of the size, so
    on the cylinder within the source's own chordal error (the sagitta of its
    5-degree chords). The control moves one node outward by 1e-3, and the
    same check sees it off both.
    """
    mesh = _cylinder_strip()
    result = refine_group(mesh, ["C"], (), factor, elements=QD)
    faces = result.families["C"]
    assert _share(faces) > CURVED_SHARE, _share(faces)
    size = float(numpy.linalg.norm(numpy.ptp(mesh.verts, axis=0)))
    source = trimesh.Trimesh(mesh.verts, numpy.asarray(mesh.families["C"]), process=False)
    sagitta = 1.0 - numpy.cos(numpy.radians(2.5))

    def off(points: numpy.ndarray) -> tuple[float, float, float]:
        _, distance, _ = trimesh.proximity.closest_point(source, points)
        radius = numpy.linalg.norm(points[:, :2], axis=1)
        return float(distance.max()), float((1.0 - radius).max()), float((radius - 1.0).max())

    surface, inside, outside = off(result.points)
    assert surface < 1e-9 * size and inside <= sagitta + 1e-12 and outside < 1e-12
    moved = result.points.copy()
    moved[len(moved) // 2, :2] *= 1.001
    surface, _, outside = off(moved)
    assert surface > 1e-4 and outside > 1e-4


def test_no_quadrilateral_spans_an_interior_curve():
    """a curve inside a family is never merged across.

    The row y = 0.5 of a flat plate is declared the trailing edge: every
    trailing-edge edge of the level is an edge of its faces, so no
    quadrilateral spans it, though pairs across it would be planar and
    convex. The control merges the two faces across one such edge, and the
    same count finds it missing.
    """
    mesh = _flat_plate()
    row = [6 * 13 + i for i in range(13)]
    result = refine_group(mesh, ["P"], row, 2.0, elements=QD)
    faces = result.families["P"]
    assert _share(faces) > FLAT_SHARE
    te = [
        e
        for e, fs in edge_faces(faces).items()
        if len(fs) == 2 and numpy.abs(result.points[list(e), 1] - 0.5).max() < 1e-12
    ]
    mids = {tuple(numpy.round(0.5 * result.points[list(e)].sum(axis=0), 12)) for e in te}
    returned = {tuple(numpy.round(m, 12)) for m in result.trailing_edge}
    assert len(te) > 12 and returned == mids
    assert all(len(edge_faces(faces)[e]) == 2 for e in te)
    control = _merge_across(faces, te[0])
    assert te[0] not in _edges(control)


def test_no_quadrilateral_crosses_a_family_boundary_or_a_frozen_interface():
    """the cut between A and B keeps every edge.

    A remeshed alone (B unchanged, so the cut is frozen) keeps every source
    edge of the cut. A and B remeshed together, both quad-dominant, hold one
    set of cut nodes, every cut edge is an edge of both families, and no face
    has nodes on both sides of the cut. The control is a face built across
    the cut, which the side test finds.
    """
    mesh = two_families()
    a_ids = set(mesh.family_vertices("A").tolist())
    cut = sorted(a_ids & set(mesh.family_vertices("B").tolist()), key=lambda v: mesh.verts[v, 1])
    a, b = mesh.verts[cut[:-1]], mesh.verts[cut[1:]]
    alone = refine_group(mesh, ["A"], (), 2.0, elements=QD)
    assert _share(alone.families["A"]) > 0.5
    assert _missing_edges(alone.points, alone.families["A"], a, b) == 0
    assert _missing_edges(alone.points, alone.families["A"], a[:, [0, 1, 2]] + 0.01, b) == len(a)
    both = refine_group(mesh, ["A", "B"], (), 2.0, elements=QD)
    fa, fb = both.families["A"], both.families["B"]
    assert _share(fa) > 0.5 and _share(fb) > 0.5
    shared = sorted({v for f in fa for v in f} & {v for f in fb for v in f})
    pts = both.points

    def crossing(fs: list[list[int]]) -> int:
        x = [pts[f, 0] for f in fs]
        return sum(int(v.min() < 1.0 - 1e-9 and v.max() > 1.0 + 1e-9) for v in x)

    assert crossing(fa + fb) == 0
    on_cut = sorted(shared, key=lambda v: pts[v, 1])
    pairs = list(zip(on_cut[:-1], on_cut[1:], strict=True))
    assert all((min(p), max(p)) in _edges(fa) and (min(p), max(p)) in _edges(fb) for p in pairs)
    left = next(f for f in fa if len(f) == 3)
    right = next(f for f in fb if len(f) == 3)
    assert crossing([left + right]) == 1


# ---- a level written quad-dominant


def _two_source(tmp_path: Path, toml: str | None) -> Path:
    folder = tmp_path / "src"
    folder.mkdir(parents=True)
    write_obj(two_families(), folder / "two.obj", "synthetic")
    if toml is not None:
        (folder / "two.refine.toml").write_bytes(toml.encode("utf-8"))
    return folder / "two.obj"


def test_a_quad_dominant_level_passes_the_gates_and_reports_its_share(tmp_path):
    """G1 to G5 pass and refine.json states the share.

    Both families are remeshed together quad-dominant at factor 2. The audit's
    gates G1 to G5 pass (G2: no two neighbors of opposite orientation), and
    refine.json gives each family its mode, quadrilaterals, triangles and
    share, equal to the OBJ's. The control flips one quadrilateral of the
    level, and G2 fails on it.
    """
    src = _two_source(tmp_path, f'[refine]\nelements = "{QD}"\n')
    level = refine_mesh(src, 2.0)
    assert level.audit is not None
    gates = {i.name for i in level.audit.gates}
    assert {"G1", "G2", "G3", "G4", "G5"} <= gates
    assert all(i.passed for i in level.audit.gates if i.name in {"G1", "G2", "G3", "G4", "G5"})
    obj = read_obj(level.obj)
    for name in ("A", "B"):
        faces = obj.families[name]
        quads = sum(len(f) == 4 for f in faces)
        entry = level.report[name]
        assert entry["elements"] == QD and entry["method"] == "remesh"
        assert (entry["quads"], entry["triangles"]) == (quads, len(faces) - quads)
        assert entry["quad_share"] == round(quads / len(faces), 6) and entry["quad_share"] > 0.5
    record = json.loads(level.files[-2].read_text(encoding="utf-8"))
    assert record["families"] == level.report
    k = next(i for i, f in enumerate(obj.families["A"]) if len(f) == 4)
    obj.families["A"][k] = obj.families["A"][k][::-1]
    (tmp_path / "flipped").mkdir()
    flipped = write_obj(obj, tmp_path / "flipped" / "flipped.obj", "control")
    g2 = [i for i in audit_mesh(flipped).gates if i.name == "G2" and i.family == "(whole mesh)"]
    assert len(g2) == 1 and not g2[0].passed


def test_the_default_stays_triangles_byte_for_byte(tmp_path):
    """without the key, or with "triangles", nothing changes.

    The level of a source with no refinement file, of one stating
    ``[refine] elements = "triangles"``, and of one stating it on the
    family, are the same OBJ byte for byte, every remeshed face a triangle,
    and refine.json states the mode with a share of zero; the remesh of the
    group gives the same nodes and faces with and without the argument. The
    control is the same source quad-dominant, whose OBJ differs.
    """
    names = []
    for k, toml in enumerate(
        [None, '[refine]\nelements = "triangles"\n', '[families.A]\nelements = "triangles"\n']
    ):
        src = _two_source(tmp_path / str(k), toml)
        level = refine_mesh(src, 2.0)
        names.append(level.obj.read_bytes())
        assert all(len(f) == 3 for fs in read_obj(level.obj).families.values() for f in fs)
        assert level.report["A"]["elements"] == "triangles" and level.report["A"]["quads"] == 0
    assert names[0] == names[1] == names[2]
    quad = refine_mesh(_two_source(tmp_path / "qd", f'[refine]\nelements = "{QD}"\n'), 2.0)
    assert quad.obj.read_bytes() != names[0]
    mesh = two_families()
    plain = refine_group(mesh, ["A", "B"], (), 2.0)
    stated = refine_group(mesh, ["A", "B"], (), 2.0, elements={"A": "triangles", "B": "triangles"})
    assert numpy.array_equal(plain.points, stated.points) and plain.families == stated.families


def test_a_family_mode_replaces_the_default_and_a_grid_ignores_it(tmp_path):
    """per-family modes, and a grid keeps its cells.

    ``[refine] elements`` is quad-dominant and ``[families.T]`` says
    triangles: T stays triangles. The grid G keeps its cells, equal to its
    level in the default mode, and refine.json says the mode was ignored.
    The control states nothing on T, which then gets quadrilaterals.
    """
    mesh = sheet_with_strips((("T", "tri", STRIP),))
    toml = f'[refine]\nelements = "{QD}"\n\n[families.T]\nelements = "triangles"\n'
    src = write_source(tmp_path / "a" / "src", "wing", mesh, refine_toml=toml)
    level = refine_mesh(src, 2.0)
    assert level.report["T"]["elements"] == "triangles" and level.report["T"]["quads"] == 0
    assert level.report["G"]["method"] == "grid"
    assert level.report["G"]["elements"].startswith(f"{QD} ignored")
    plain = refine_mesh(write_source(tmp_path / "b" / "src", "wing", mesh), 2.0)
    assert read_obj(level.obj).families["G"] == read_obj(plain.obj).families["G"]
    assert "elements" not in plain.report["G"]
    toml = f'[refine]\nelements = "{QD}"\n'
    control = refine_mesh(write_source(tmp_path / "c" / "src", "wing", mesh, refine_toml=toml), 2.0)
    assert control.report["T"]["elements"] == QD and control.report["T"]["quads"] > 0


# ---- refusals


@pytest.mark.parametrize(
    ("toml", "where", "said"),
    [
        (
            '[refine]\nelements = "quads"\n',
            "[refine] elements",
            "'quads' is not an element mode; the modes are triangles, quad-dominant",
        ),
        (
            '[families.A]\nfactor = 2\nelements = "hex"\n',
            "[families.A] elements",
            "'hex' is not an element mode; the modes are triangles, quad-dominant",
        ),
        (
            f'[families.A]\nfactor = 2\nmethod = "grid"\nelements = "{QD}"\n',
            "[families.A] elements",
            "elements applies to a remeshed family; a grid keeps its cells",
        ),
        (
            '[refine]\nelement = "triangles"\n',
            "[refine]",
            "unknown key 'element'; the known keys are tag, filename, elements",
        ),
    ],
    ids=["unknown-default", "unknown-family", "on-a-grid", "unknown-refine-key"],
)
def test_an_unknown_mode_is_refused_naming_the_file_table_and_modes(tmp_path, toml, where, said):
    """each bad value is refused before anything is written.

    The message names the file, the table and the key, lists the allowed
    values, and ends with the diagnostic PDF path; no level folder is left. The
    control is the same source with a valid mode, which writes its level.
    """
    src = _two_source(tmp_path, toml)
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, 2.0)
    text = str(caught.value)
    assert text.startswith(f"{src.with_name('two.refine.toml')} {where}: {said}"), text
    pdf = src.with_suffix(".refine-refusal.pdf")
    assert text.endswith(diag(pdf).lstrip())
    assert not HAS_INSPECT or pdf.read_bytes().startswith(b"%PDF")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src"]
    src.with_name("two.refine.toml").write_bytes(f'[refine]\nelements = "{QD}"\n'.encode())
    assert refine_mesh(src, 2.0).report["A"]["elements"] == QD
