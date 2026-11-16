"""a family rebuilt at factor 1 beside a refined side takes its spacing.

Every source is synthetic: the cambered sheet grid ``G`` of
tests/p0380_mesh_fixtures.py and ``C``, a curved cap of triangles closing G's
tip station, its rows bent a quarter turn about the chord (a quarter
cylinder). G is refined at factor 2 and C is rebuilt at factor 1, stated
``factor = 1``, ``method = "remesh"``, or as the band of an unselected
neighbour.

The measurement the box asks for first was made on this fixture and on its
variants (a quarter or half cylinder, a dome of shrinking rows, a flat zipper
cap and a dome closing a wing tube, a section thinned to 0.5 percent, 2 to 6
cap rows, 24 and 48 nodes, factors 2, 3 and 4, stated and band): the 0.38.0
target gave no face under ``ZERO_AREA_FRACTION * size**2`` on any of them,
so the defect of the multi-component mesh was not reproduced synthetically.
What 0.38.0 did give on every one is the mechanism: the faces of C beside the
curve kept C's own length, two to five times the new shared edges, so the
tests below hold the rule itself. The edges of C crossing from the curve are
as long as the shared edges (a ratio of at most 1.5, against 1.9 to 5.3 on
0.38.0), the targets at and near the curve are the shared spacing graded
over two face layers, and ``refine.json`` records the target taken.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from mesh_fixtures import (
    Composite,
    face_coordinates,
    read_mesh,
    sheet_with_strips,
    write_source,
)

from cfdbender._errors import GeometryWarning
from cfdbender.mesh._refine import refine_mesh
from cfdbender.mesh._refine._geometry import SHARED_LAYERS, ZERO_AREA_FRACTION
from cfdbender.mesh._refine._obj import read_obj
from cfdbender.mesh._refine._remesh import refine_group
from cfdbender.mesh._refine._remesh_group import shared_targets

pytest.importorskip("trimesh")

STATED = '[families.G]\nfactor = 2\n[families.C]\nfactor = 1\nmethod = "remesh"\n'
#: The largest median ratio of a crossing edge of C to a shared edge the rule allows.
CROSS_RATIO = 1.5


def sheet_with_cap(rows: int = 3, sweep_deg: float = 90.0, radius: float = 0.1) -> Composite:
    """Return the sheet grid ``G`` and ``C``, a cap of triangles bent about the tip's chord.

    Row ``j`` of the cap is the tip section moved by ``radius`` along a
    quarter circle in the y-z plane, ``j / rows`` of ``sweep_deg``; each cell
    is split along a diagonal that alternates.
    """
    base = sheet_with_strips(())
    tip = list(base.rows["G"][-1])
    section = base.verts[tip]
    verts = [base.verts]
    count = len(base.verts)
    cap_rows = [tip]
    y0 = float(section[0, 1])
    for j in range(1, rows + 1):
        angle = np.radians(sweep_deg) * j / rows
        row = section.copy()
        row[:, 1] = y0 + radius * np.sin(angle)
        row[:, 2] = section[:, 2] + radius * (1.0 - np.cos(angle))
        verts.append(row)
        cap_rows.append(list(range(count, count + len(row))))
        count += len(row)
    faces = []
    for j in range(rows):
        lo, hi = cap_rows[j], cap_rows[j + 1]
        for i in range(len(lo) - 1):
            a, b, c, d = lo[i], lo[i + 1], hi[i + 1], hi[i]
            faces += [[a, b, c], [a, c, d]] if (i + j) % 2 else [[a, b, d], [b, c, d]]
    families = {**base.families, "C": faces}
    return Composite(np.vstack(verts), families, base.te, {**base.rows, "C": cap_rows})


def _edges(faces) -> set[tuple[int, int]]:
    return {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1], strict=True)}


def _record(level) -> dict:
    return json.loads(next(Path(level.folder).glob("*.refine.json")).read_text(encoding="utf-8"))


def _source(tmp_path: Path, name: str, mesh: Composite, toml: str | None = None) -> Path:
    return write_source(tmp_path / name, "wing", mesh, refine_toml=toml)


def _polyline_spacing(points: np.ndarray) -> np.ndarray:
    """Return, at each node of an ordered polyline, the mean length of its one or two segments."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    out = np.empty(len(points))
    out[0], out[-1] = seg[0], seg[-1]
    out[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    return out


def _own(points: np.ndarray, faces) -> np.ndarray:
    """Return each node's own target at factor 1: the mean length of its incident edges."""
    total, count = np.zeros(len(points)), np.zeros(len(points))
    for a, b in _edges(faces):
        total[[a, b]] += float(np.linalg.norm(points[a] - points[b]))
        count[[a, b]] += 1
    return total / np.maximum(count, 1)


def _midpoints_inserted(points: np.ndarray) -> np.ndarray:
    """Return an ordered polyline with the mid-point of each segment inserted (a refined side)."""
    new = np.empty((2 * len(points) - 1, 3))
    new[0::2] = points
    new[1::2] = 0.5 * (points[:-1] + points[1:])
    return new


#: The refined side's curve of the strip below, and the source's nodes of the rows above it.
CURVE_X = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0)
ROW_X = (0.0, 1.0, 3.0, 7.0)


def _refined_strip():
    """Return a strip whose bottom row is the curve of a refined side, and its rows and chain.

    Rows 1 to 3 hold the nodes ``ROW_X`` at y 1, 2 and 3. Row 1 is joined to
    the curve by fans and the rows above by split cells, so a node of row j is
    j face layers from the curve.
    """
    xs = [CURVE_X, ROW_X, ROW_X, ROW_X]
    rows, pts = [], []
    for y, row in enumerate(xs):
        rows.append(list(range(len(pts), len(pts) + len(row))))
        pts += [(x, float(y), 0.0) for x in row]
    curve, first = rows[0], rows[1]
    under = [CURVE_X.index(x) for x in ROW_X]
    faces = []
    for i in range(len(ROW_X) - 1):
        a, b = first[i], first[i + 1]
        faces += [[curve[k], curve[k + 1], a] for k in range(under[i], under[i + 1])]
        faces.append([curve[under[i + 1]], b, a])
    for lo, hi in zip(rows[1:-1], rows[2:], strict=True):
        for i in range(len(ROW_X) - 1):
            faces += [[lo[i], lo[i + 1], hi[i + 1]], [lo[i], hi[i + 1], hi[i]]]
    chain = {(curve[k], curve[k + 1]) for k in range(len(CURVE_X) - 1)}
    return np.array(pts, dtype=float), faces, rows, chain


def test_mesh_zeroarea_zeroarea_targets_at_the_curve_and_its_layers_follow_r1_and_r2():
    """every curve node, then the layers, exactly.

    The curve's nodes are spaced 0.5, 0.5, 1, 1, 2 and 2, so each takes the
    mean of its one or two incident shared edges, an inserted node as any
    other. A node one face layer away takes the target of the curve node
    nearest it, found here from the coordinates, half way to its own (``SHARED_LAYERS`` 2); the node at x 1 takes 0.75 from the curve node
    below it, not an average of the curve nodes its faces touch. Nodes two
    layers away and farther keep their own target (not returned). The record
    is the least, the mean and the greatest curve target.
    """
    assert SHARED_LAYERS == 2
    points, faces, rows, chain = _refined_strip()
    own = _own(points, faces)
    targets, record = shared_targets(
        points, faces, ["C"] * len(faces), own=own, held=set(), chains=[chain]
    )
    r1 = [0.5, 0.5, 0.75, 1.0, 1.5, 2.0, 2.0]
    assert [targets[v] for v in rows[0]] == r1
    curve = points[rows[0]]
    for v in rows[1]:
        s = r1[int(np.argmin(np.linalg.norm(curve - points[v], axis=1)))]
        assert targets[v] == s + (own[v] - s) / 2, v
    assert targets[rows[1][1]] == 0.75 + (own[rows[1][1]] - 0.75) / 2
    assert set(targets) == set(rows[0]) | set(rows[1])
    mean = round(sum(r1) / len(r1), 12)
    assert record == {"C": {"min": 0.5, "mean": mean, "max": 2.0}}


def test_mesh_zeroarea_zeroarea_the_curve_spacing_reads_the_shared_edges_not_proximity():
    """a bent curve: each node averages the shared edges it ends.

    The curve runs (0, 0), (1, 0), (1, 1), (0, 0.1): its last node is 0.1
    from its first, which is no edge of it, so a spacing read from the
    nearest nodes would give both 0.1. Each node takes the mean of the curve
    edges it ends. Control: the two ends are indeed that close.
    """
    curve = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.1, 0.0)])
    apex = np.array([(0.5, -0.5, 0.0), (1.5, 0.5, 0.0), (0.5, 1.2, 0.0)])
    points = np.vstack([curve, apex])
    faces = [[k, k + 1, 4 + k] for k in range(3)]
    chain = {(0, 1), (1, 2), (2, 3)}
    targets, _ = shared_targets(
        points, faces, ["C"] * 3, own=_own(points, faces), held=set(), chains=[chain]
    )
    last = float(np.linalg.norm(curve[3] - curve[2]))
    assert [targets[v] for v in range(4)] == [1.0, 1.0, (1.0 + last) / 2, last]
    assert float(np.linalg.norm(curve[3] - curve[0])) == pytest.approx(0.1)


def test_mesh_zeroarea_zeroarea_no_shared_curve_sets_no_target():
    """with no shared curve no target changes and nothing is recorded."""
    points, faces, _, _ = _refined_strip()
    own = _own(points, faces)
    assert shared_targets(points, faces, ["C"] * len(faces), own=own, held=set(), chains=[]) == (
        {},
        {},
    )


def _spacing_record(spacing: np.ndarray) -> dict:
    return {
        "min": round(float(spacing.min()), 12),
        "mean": round(float(spacing.mean()), 12),
        "max": round(float(spacing.max()), 12),
    }


def test_mesh_zeroarea_zeroarea_inserted_curve_nodes_are_recorded_with_their_spacing(tmp_path):
    """a refined side's inserted nodes take and record their
     spacing.

    The cap C is remeshed at factor 1 beside G's tip row refined with a
    mid-point inserted in each segment (the row's spacing is nonuniform).
    The targets are set once the interface holds the new nodes, so the
    record is over every new curve node, the inserted ones included: the
    least, mean and greatest of the mean incident new shared edge. Measured
    first: C's own targets at the curve are over 1.5 times the new spacing.
    """
    mesh = sheet_with_cap()
    tip = mesh.rows["G"][-1]
    obj = read_obj(write_source(tmp_path, "wing", mesh))
    new = _midpoints_inserted(obj.verts[tip])
    spacing = _polyline_spacing(new)
    own = _own(mesh.verts, mesh.families["C"])[tip]
    assert np.all(own > 1.5 * spacing[0::2]), "the fixture must make the 0.38.0 target differ"
    done = refine_group(obj, ["C"], (), {"C": 1.0}, conform=[(tip, new)], shared=[(tip, new)])
    assert done.info["shared_edge_target_mesh_units"] == {"C": _spacing_record(spacing)}


def test_mesh_zeroarea_zeroarea_a_family_at_2_beside_a_kept_curve_takes_its_spacing(tmp_path):
    """a cap refined alone at 2 takes the spacing its grid keeps.

    G is not selected, so it keeps its tip row; C, refined at factor 2,
    takes at that curve the mean of the kept shared edges incident to each
    node, recorded under C, and its level passes G1 with no face of zero
    area. Control: C's own target at the curve, half its incident edges, is
    not the kept spacing.
    """
    mesh = sheet_with_cap()
    tip = mesh.rows["G"][-1]
    obj = read_obj(write_source(tmp_path / "unit", "wing", mesh))
    spacing = _polyline_spacing(obj.verts[tip])
    own = _own(mesh.verts, mesh.families["C"])[tip] / 2.0
    assert not np.allclose(own, spacing)
    done = refine_group(obj, ["C"], (), {"C": 2.0})
    assert done.info["shared_edge_target_mesh_units"] == {"C": _spacing_record(spacing)}
    src = _source(tmp_path, "alone", mesh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        level = refine_mesh(src, 2, families=["C"], out_dir=tmp_path / "o")
    entry = _record(level)["families"]["C"]
    assert entry["shared_edge_target_mesh_units"] == _spacing_record(spacing)
    gate = next(item for item in level.audit.gates if item.name == "G1")
    assert gate.values["faces_of_zero_area"] == 0 and gate.verdict == "pass"


def _square_in_frame() -> Composite:
    """Return ``C``, one unit-square quadrilateral, and ``F``, four quadrilaterals framing it.

    Every node of C is a node of F, so the four sides of C are the edges C
    shares with F, each of length 1; the group fans C into two triangles, and
    their diagonal (length sqrt 2) joins two shared nodes without being an
    edge of F.
    """
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [-1, -1, 0],
            [2, -1, 0],
            [2, 2, 0],
            [-1, 2, 0],
        ],
        dtype=float,
    )
    frame = [[0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
    return Composite(verts, {"C": [[0, 1, 2, 3]], "F": frame}, np.zeros((0, 3)), {})


def test_mesh_zeroarea_zeroarea_a_diagonal_between_shared_nodes_is_not_a_kept_edge(tmp_path):
    """only an edge of the unchanged neighbour is a kept shared edge.

    The unit-square cap C, remeshed alone at factor 2 inside the unchanged
    frame F, takes at each corner the mean of its kept shared edges, all of
    length 1, so ``shared_edge_target_m`` is 1.0 at its least, mean and
    greatest. Taking every edge frozen between two shared nodes takes the
    diagonal C's triangulation draws too (length sqrt 2), and records a mean
    of about 1.069 and a greatest of about 1.138.
    """
    obj = read_obj(write_source(tmp_path / "square", "wing", _square_in_frame()))
    done = refine_group(obj, ["C"], (), {"C": 2.0})
    assert done.info["shared_edge_target_mesh_units"] == {
        "C": {"min": 1.0, "mean": 1.0, "max": 1.0}
    }


@pytest.mark.parametrize("how", ["stated", "rebuilt", "band"])
def test_mesh_zeroarea_zeroarea_level_takes_the_spacing_and_records_it(tmp_path, how):
    """C beside G at 2 meshes to the shared spacing; G1 passes.

    C is stated at factor 1 (``stated``), rebuilt whole by as an
    unselected cap two rows deep (``rebuilt``), or the band of an unselected
    cap three rows deep (``band``). A rebuilt family records the target taken
    at the curve; an unchanged neighbour keeps the record of.
    """
    mesh = sheet_with_cap(rows=2 if how == "rebuilt" else 3)
    if how == "stated":
        src = _source(tmp_path, how, mesh, STATED)
        level = refine_mesh(src, config=src.with_suffix(".refine.toml"), out_dir=tmp_path / "o")
    else:
        src = _source(tmp_path, how, mesh)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GeometryWarning)
            level = refine_mesh(src, 2, families=["G"], out_dir=tmp_path / "o")
    verts, fams = read_mesh(level.obj)
    curve = {v for f in fams["G"] for v in f} & {v for f in fams["C"] for v in f}
    edges = _edges(fams["C"])
    shared = [(a, b) for a, b in edges if a in curve and b in curve]
    cross = [(a, b) for a, b in edges if (a in curve) != (b in curve)]

    def length(e):
        return float(np.linalg.norm(verts[e[0]] - verts[e[1]]))

    ratio = np.median([length(e) for e in cross]) / np.median([length(e) for e in shared])
    assert ratio <= CROSS_RATIO, f"crossing edges {ratio:.2f} times the shared ones"
    per_node: dict[int, list[float]] = {}
    for e in shared:
        for v in e:
            per_node.setdefault(v, []).append(length(e))
    means = [sum(x) / len(x) for x in per_node.values()]
    entry = _record(level)["families"]["C"]
    if how == "band":
        assert entry["method"] == "unchanged" and {"band", "faces"} <= set(entry)
    else:
        assert entry["method"] == ("remesh" if how == "stated" else "rebuilt_whole")
    taken = entry["shared_edge_target_mesh_units"]
    assert taken["min"] == pytest.approx(min(means), rel=1e-6)
    assert taken["max"] == pytest.approx(max(means), rel=1e-6)
    assert taken["mean"] == pytest.approx(sum(means) / len(means), rel=1e-9)
    gate = next(item for item in level.audit.gates if item.name == "G1")
    assert gate.verdict == "pass"
    assert gate.values["faces_of_zero_area"] == 0
    used = sorted({v for fs in fams.values() for f in fs for v in f})
    size = float(np.linalg.norm(np.ptp(verts[used], axis=0)))
    p = verts[np.array(fams["C"])]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    assert int((area < ZERO_AREA_FRACTION * size**2).sum()) == 0


def test_mesh_zeroarea_zeroarea_control_whole_mesh_at_factor_2(tmp_path):
    """( control) the whole-mesh refine at 2 writes no zero-area face."""
    src = _source(tmp_path, "whole", sheet_with_cap())
    level = refine_mesh(src, 2, out_dir=tmp_path / "o")
    gate = next(item for item in level.audit.gates if item.name == "G1")
    assert gate.values["faces_of_zero_area"] == 0
    assert gate.verdict == "pass"
    assert "shared_edge_target_m" not in _record(level)["families"]["C"], "C is at factor 2"


def test_mesh_zeroarea_zeroarea_family_sharing_no_curve_is_unchanged(tmp_path):
    """a factor-1 family with no curve on a refined side is unchanged."""
    strips = (("T", "tri", (0.03, 0.04, 0.05, 0.065, 0.08)), ("U", "tri", (0.1, 0.1)))
    mesh = sheet_with_strips(strips)
    tail = '[families.U]\nfactor = 1\nmethod = "remesh"\n'
    beside = _source(tmp_path, "beside", mesh, "[families.G]\nfactor = 2\n" + tail)
    alone = _source(tmp_path, "alone", mesh, "[families.G]\nfactor = 1\n" + tail)
    refined = refine_mesh(beside, config=beside.with_suffix(".refine.toml"), out_dir=tmp_path / "a")
    control = refine_mesh(alone, config=alone.with_suffix(".refine.toml"), out_dir=tmp_path / "b")
    assert "shared_edge_target_m" not in _record(refined)["families"]["U"]
    v1, f1 = read_mesh(refined.obj)
    v2, f2 = read_mesh(control.obj)
    assert face_coordinates(v1, f1["U"]) == face_coordinates(v2, f2["U"])


# ---- a remesh writes no face of zero or near-zero area

#:'s thresholds, restated so that a change of the constants is a change of this test.
AREA_FRACTION, ANGLE_DEGREES = 1e-3, 1.0
#: How far a node of a finer neighbour stands off the mid-point of a side of a cell.
SLIVER = 1e-6


def _shapes(points: np.ndarray, faces) -> tuple[np.ndarray, np.ndarray]:
    """Return each polygon's area (Newell) and its smallest interior angle in degrees."""
    area, angle = [], []
    for f in faces:
        p = points[f]
        area.append(0.5 * np.linalg.norm(np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)))
        side = np.roll(p, -1, axis=0) - p
        length = np.linalg.norm(side, axis=1)
        cosine = -(side * np.roll(side, 1, axis=0)).sum(axis=1) / (length * np.roll(length, 1))
        angle.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).min())
    return np.array(area), np.array(angle)


def _degenerate(points: np.ndarray, faces) -> int:
    """Return how many faces are below the area fraction of the median, or the angle."""
    area, angle = _shapes(points, faces)
    return int(((area < AREA_FRACTION * np.median(area)) | (angle < ANGLE_DEGREES)).sum())


def _hanging_cells(fanned: bool):
    """Return ``F``, 4 by 2 cells 1 wide and 2 tall, over ``G``, a finer strip below it.

    G holds a node ``SLIVER`` below the mid-point of each bottom side of F, so
    each bottom cell of F is a convex pentagon whose fan from its first corner
    is a sliver; ``fanned`` gives F that fan itself, as triangles.
    """
    from cfdbender.mesh._refine._obj import ObjMesh

    pts: list[tuple[float, float, float]] = []

    def node(x: float, y: float) -> int:
        pts.append((x, y, 0.0))
        return len(pts) - 1

    grid = {(i, j): node(float(i), 2.0 * j) for j in range(3) for i in range(5)}
    mid = [node(i + 0.5, -SLIVER) for i in range(4)]
    low = [node(float(i), -1.0) for i in range(5)]
    cells = []
    for j in range(2):
        for i in range(4):
            a, b = grid[(i, j)], grid[(i + 1, j)]
            c, d = grid[(i + 1, j + 1)], grid[(i, j + 1)]
            cells.append([a, mid[i], b, c, d] if j == 0 else [a, b, c, d])
    if fanned:
        cells = [[f[0], f[k], f[k + 1]] for f in cells for k in range(1, len(f) - 1)]
    strip = []
    for i in range(4):
        a, m, b = grid[(i, 0)], mid[i], grid[(i + 1, 0)]
        strip += [[m, a, low[i]], [b, m, low[i + 1]], [m, low[i], low[i + 1]]]
    return ObjMesh(np.array(pts), {"F": cells, "G": strip})


@pytest.mark.parametrize("fanned", [False, True], ids=["pentagons", "sliver-triangles"])
def test_fr3403_remesh_leaves_no_degenerate_face(fanned):
    """a polygon whose fan is a sliver, or a sliver itself, leaves no degenerate face.

    The pentagons are cut where the sliver is not a triangle (prevention); the
    slivers given as triangles are flipped away (repair). The nodes F shares
    with G never move.
    """
    mesh = _hanging_cells(fanned)
    sliced = _hanging_cells(True).families["F"]
    assert _degenerate(mesh.verts, sliced) > 0, "the measure must see the source's slivers"
    done = refine_group(mesh, ["F"], set(), 1.0)
    assert _degenerate(done.points, done.families["F"]) == 0
    shared = np.intersect1d(mesh.family_vertices("F"), mesh.family_vertices("G"))
    for v in shared:
        assert np.min(np.linalg.norm(done.points - mesh.verts[v], axis=1)) == 0.0


def test_fr3403_triangulation_cuts_no_sliver():
    """a pentagon and a quadrilateral with a corner on a side are cut with no sliver.

    Their fan from the first corner holds the sliver; the pentagon takes the
    fan of its best corner, the quadrilateral its other diagonal.
    """
    from cfdbender.mesh._refine._remesh_group import triangulate_polygon

    pentagon = np.array([(0, 0, 0), (0.5, -SLIVER, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0)], float)
    quad = np.array([(0, 0, 0), (0.5, -SLIVER, 0), (1, 0, 0), (0.5, 2, 0)], float)
    for verts in (pentagon, quad):
        cut = triangulate_polygon(verts, list(range(len(verts))))
        assert len(cut) == len(verts) - 2
        assert _degenerate(verts, cut) == 0
        _, fan_angle = _shapes(verts, [[0, k, k + 1] for k in range(1, len(verts) - 1)])
        assert fan_angle.min() < ANGLE_DEGREES, "the fan of the first corner is the sliver"


def test_fr3403_clean_family_comes_out_identical(monkeypatch):
    """control: a clean quadrilateral family is cut and remeshed as before the repair.

    Each of its quadrilaterals keeps the fan of its first corner, and its
    remesh with the repair is the remesh without it, node for node and face
    for face.
    """
    from mesh_fixtures import plate_nodes

    from cfdbender.mesh._refine import _remesh
    from cfdbender.mesh._refine._obj import ObjMesh
    from cfdbender.mesh._refine._remesh_group import triangulate_polygon

    nodes = plate_nodes(8, 8)
    quads = [
        [j * 9 + i, j * 9 + i + 1, (j + 1) * 9 + i + 1, (j + 1) * 9 + i]
        for j in range(8)
        for i in range(8)
    ]
    for f in quads:
        assert triangulate_polygon(nodes, f) == [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]
    mesh = ObjMesh(nodes, {"Q": quads})
    repaired = refine_group(mesh, ["Q"], set(), 2.0)
    assert _degenerate(repaired.points, repaired.families["Q"]) == 0
    monkeypatch.setattr(_remesh.Remesher, "repair_degenerate", lambda self: [])
    plain = refine_group(mesh, ["Q"], set(), 2.0)
    assert np.array_equal(repaired.points, plain.points)
    assert repaired.families == plain.families


def _pinched_strip() -> Composite:
    """Return ``F``, a strip between ``G`` and ``H`` pinched to ``SLIVER`` at its end.

    Every node of F is shared with G or H, so a remesh may move none of them;
    its last cell is the sliver b3, b4, p, its longest side b3-b4 on G.
    """
    pts: list[tuple[float, float, float]] = []

    def node(x: float, y: float) -> int:
        pts.append((x, y, 0.0))
        return len(pts) - 1

    g = [node(float(i), -1.0) for i in range(5)]
    b = [node(float(i), 0.0) for i in range(5)]
    t = [node(float(i), 1.0) for i in range(4)] + [node(3.5, SLIVER)]
    h = [node(float(i), 2.0) for i in range(4)] + [node(3.5, 2.0)]
    families = {
        "G": [[g[i], g[i + 1], b[i + 1], b[i]] for i in range(4)],
        "F": [[b[i], b[i + 1], t[i + 1], t[i]] for i in range(3)]
        + [[b[3], b[4], t[4]], [b[3], t[4], t[3]]],
        "H": [[t[i], t[i + 1], h[i + 1], h[i]] for i in range(4)],
    }
    return Composite(np.array(pts), families, np.zeros((0, 3)), {})


def test_fr3403_held_sliver_is_left_listed_and_warned(tmp_path):
    """a sliver on nodes no remesh may move is left, listed in audit.json and warned.

    G1 counts it against F's own median area: against the whole mesh's size
    it would pass, as it did before.
    """
    mesh = _pinched_strip()
    src = _source(tmp_path, "strip", mesh, '[families.F]\nfactor = 1\nmethod = "remesh"\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        level = refine_mesh(src, config=src.with_suffix(".refine.toml"), out_dir=tmp_path / "o")
    verts, fams = read_mesh(level.obj)
    b3, b4, p = 8, 9, 14  # in _pinched_strip's numbering
    corner_points = mesh.verts[[b3, b4, p]]
    held = [np.flatnonzero(np.all(np.abs(verts - q) < 1e-12, axis=1)) for q in corner_points]
    assert all(len(found) == 1 for found in held), "the sliver's nodes did not move"
    corners = {int(found[0]) for found in held}
    index = next(i for i, f in enumerate(fams["F"]) if set(f) == corners)
    record = json.loads(level.audit.path.read_text(encoding="utf-8"))
    assert record["degenerate_faces"] == [
        {
            "family": "F",
            "face": index,
            "centroid": pytest.approx([3.5, SLIVER / 3, 0.0], abs=1e-9),
            "area_ratio": pytest.approx(record["degenerate_faces"][0]["area_ratio"]),
            "min_angle_degrees": pytest.approx(np.degrees(np.arctan(2 * SLIVER)), rel=1e-3),
        }
    ]
    assert record["degenerate_faces"][0]["area_ratio"] < AREA_FRACTION
    gate = next(item for item in level.audit.gates if item.name == "G1")
    assert gate.verdict == "fail" and gate.values["faces_of_zero_area"] == 1
    said = [str(w.message) for w in caught if issubclass(w.category, GeometryWarning)]
    assert any("1 degenerate face(s)" in s and str(level.audit.path) in s for s in said), said
    assert any("re-cut the source near them" in s for s in said), said


# ---- the nodes a repair holds, and the thresholds bracketed


def _sliver_remesher(delta: float, held: str | None = None, cluster: float = 0.0):
    """Return a Remesher of ``F``: the sliver A, C, B over the open side A-C, fanned to the top.

    A (0, 0), C (1, 0) and B (0.5, ``delta``); D, P, E along y = 1. ``held``
    makes B a node no repair may move: on an open ``boundary`` curve, on a
    listed ``trailing_edge`` line (B-P), a ``corner`` or ``fixed`` (shared
    with a family outside the group). ``cluster`` adds to F, apart, nine
    squares cut into eighteen triangles of that area, so F's median face area
    is ``cluster`` and the sliver's area ratio is set by it alone.
    """
    from cfdbender.mesh._refine._geometry import edge_faces
    from cfdbender.mesh._refine._remesh import (
        BOUNDARY,
        TRAILING_EDGE,
        Patch,
        Remesher,
    )

    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, delta), (0.0, 1.0), (1.0, 1.0), (0.5, 1.0)]
    a, c, b, d, e, p = range(6)
    faces = [[a, c, b], [a, b, d], [b, c, e], [b, e, p], [b, p, d]]
    if cluster:
        side = float(np.sqrt(2.0 * cluster))
        base = len(pts)
        pts += [(10.0 + i * side, j * side) for j in range(4) for i in range(4)]
        for j in range(3):
            for i in range(3):
                q = base + 4 * j + i
                faces += [[q, q + 1, q + 5], [q, q + 5, q + 4]]
    curve = {k: BOUNDARY for k, fs in edge_faces(faces).items() if len(fs) == 1}
    if held in ("boundary", "trailing_edge"):
        curve[(b, p)] = BOUNDARY if held == "boundary" else TRAILING_EDGE
    verts = np.array([(x, y, 0.0) for x, y in pts])
    patch = Patch(verts, faces, ["F"] * len(faces), np.ones(len(verts)), curve)
    if held == "fixed":
        patch.fixed = {b}
    remesher = Remesher(patch)
    if held == "corner":
        remesher.corner.add(b)
    return remesher


@pytest.mark.parametrize("held", ["boundary", "trailing_edge", "corner", "fixed"])
def test_fr3403_repair_never_moves_a_held_node(held):
    """a sliver on a curve, trailing-edge, corner or fixed node is left as it is.

    The repair would collapse B onto A (the control, B free); held, B and
    every other node keep their place and number, every face is kept, and
    the sliver is what the repair returns as left.
    """
    free = _sliver_remesher(SLIVER)
    assert free.repair_degenerate() == []
    assert all(2 not in f for f in free.F.values()), "control: the free B is collapsed"
    remesher = _sliver_remesher(SLIVER, held)
    faces, points = dict(remesher.F), [v.copy() for v in remesher.V]
    left = remesher.repair_degenerate()
    assert [remesher.F[i] for i in left] == [[0, 1, 2]]
    assert remesher.F == faces
    assert all(np.array_equal(v, w) for v, w in zip(remesher.V, points, strict=True))


#: The bracket of each threshold: a factor either side of it.
BRACKET = (0.8, 1.2)


@pytest.mark.parametrize("scale", BRACKET, ids=["below", "above"])
def test_fr3403_area_threshold_is_bracketed(scale, tmp_path):
    """a well-shaped sliver 0.8 of 1e-3 of F's median area is repaired, 1.2 is not.

    Its least angle is 5.7 degrees, so its area alone decides. The audit of
    the same faces counts it (G1) and lists it below the threshold, not above.
    """
    from mesh_fixtures import obj_text

    from cfdbender.mesh._refine import audit_mesh
    from cfdbender.mesh._refine._obj import ObjMesh

    delta = 0.05  # the sliver's area is delta / 2
    remesher = _sliver_remesher(delta, cluster=delta / 2 / (scale * AREA_FRACTION))
    faces = dict(remesher.F)
    obj = tmp_path / "f.obj"
    mesh = ObjMesh(np.array(remesher.V), {"F": list(faces.values())})
    obj.write_bytes(obj_text(mesh.verts, mesh.families).encode("utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        audit = audit_mesh(obj)
    g1 = next(item for item in audit.gates if item.name == "G1")
    listed = [face["face"] for face in audit.degenerate_faces]
    left = remesher.repair_degenerate()
    if scale < 1.0:
        assert left == [] and all(2 not in f for f in remesher.F.values())
        assert g1.values["faces_of_zero_area"] == 1 and listed == [0]
    else:
        assert left == [] and remesher.F == faces
        assert g1.values["faces_of_zero_area"] == 0 and listed == []


@pytest.mark.parametrize("held", [None, "fixed"], ids=["free", "held"])
@pytest.mark.parametrize("scale", BRACKET, ids=["below", "above"])
def test_fr3403_angle_threshold_is_bracketed(scale, held, tmp_path):
    """a sliver of 0.8 degree is repaired, or held and listed; 1.2 degrees is neither.

    Its area is 2 percent of F's median, so its angle alone decides; the
    audit counts it under G1's faces_under_1_degree and lists it with its
    least angle.
    """
    from mesh_fixtures import obj_text

    from cfdbender.mesh._refine import audit_mesh
    from cfdbender.mesh._refine._obj import ObjMesh

    angle = scale * ANGLE_DEGREES
    remesher = _sliver_remesher(0.5 * np.tan(np.radians(angle)), held)
    faces = dict(remesher.F)
    obj = tmp_path / "f.obj"
    mesh = ObjMesh(np.array(remesher.V), {"F": list(faces.values())})
    obj.write_bytes(obj_text(mesh.verts, mesh.families).encode("utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        audit = audit_mesh(obj)
    g1 = next(item for item in audit.gates if item.name == "G1")
    left = remesher.repair_degenerate()
    if scale > 1.0:
        assert left == [] and remesher.F == faces
        assert g1.values["faces_under_1_degree"] == 0 and audit.degenerate_faces == ()
        return
    assert g1.values["faces_of_zero_area"] == 0 and g1.values["faces_under_1_degree"] == 1
    assert g1.passed, "an angle sliver is counted and listed, never a failure of G1"
    assert [face["face"] for face in audit.degenerate_faces] == [0]
    assert audit.degenerate_faces[0]["min_angle_degrees"] == pytest.approx(angle, rel=1e-3)
    if held is None:
        assert left == [] and all(2 not in f for f in remesher.F.values())
    else:
        assert [remesher.F[i] for i in left] == [[0, 1, 2]] and remesher.F == faces
