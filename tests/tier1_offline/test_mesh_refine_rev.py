"""refinement structure and geometry regression coverage.

Retain module boundaries, import paths and tuning-constant locations, plus
coarse-source surface bounds, pole-fan limits and split-sheet recovery.
All geometry fixtures are synthetic.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from mesh_fixtures import Composite, Fixture, build_grid, write_source
from test_mesh_grid import _distance_to_surface

import cfdbender
from cfdbender.mesh._refine import refine_mesh
from cfdbender.mesh._refine._geometry import GRID_SURFACE_DISTANCE
from cfdbender.mesh._refine._grid import Grid, recover_grid, refine_grid

REPO = Path(__file__).resolve().parents[2]
PACKAGE = Path(cfdbender.__file__).resolve().parent
REFINE_DIR = PACKAGE / "mesh" / "_refine"
REFINE = "cfdbender.mesh._refine"
REMESH = f"{REFINE}._remesh"

# ------------------------------------------------------------ PUSH-W3-ARCH-1


def _remesh_reads() -> set[str]:
    """Return every name another module or test reads from the remesher, by syntax tree."""
    found: set[str] = set()
    files = [*PACKAGE.rglob("*.py"), *(REPO / "tests").rglob("*.py")]
    for path in files:
        if path.name.startswith("_remesh") and path.parent == REFINE_DIR:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == REMESH:
                found.update(a.name for a in node.names)
            if isinstance(node, ast.ImportFrom) and node.module == REFINE:
                aliases.update(a.asname or a.name for a in node.names if a.name == "_remesh")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                found.add(node.attr)
    return found


def _defined(path: Path) -> set[str]:
    """Return the names a module defines at its top level (assignments, functions, classes)."""
    out = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


def test_push_w3_arch_1_the_remesher_split_keeps_every_import_path():
    """(PUSH-W3-ARCH-1): the remesher is split and its paths hold.

    The group a remesh works on and the pairing of triangles into
    quadrilaterals are modules of their own; each name defined there is
    defined nowhere else in the package. Every name another module or a test
    reads from ``_remesh`` (read from the syntax tree of the package and of
    the tests, so a new reader is covered without editing a list) resolves,
    and a name defined in a section is the very object of its section.
    Control: a name no reader uses is not in the set read.
    """
    remesh = importlib.import_module(REMESH)
    sections = {f"{REFINE}.{m}": REFINE_DIR / f"{m}.py" for m in ("_remesh_group", "_remesh_pairs")}
    homes = {}
    for module, path in sections.items():
        names = _defined(path)
        assert names, module
        assert not names & _defined(REFINE_DIR / "_remesh.py"), module
        homes.update(dict.fromkeys(names, importlib.import_module(module)))
    read = _remesh_reads()
    assert {"refine_group", "element_report", "Patch"} <= read
    assert "INDEX_MODULES" not in read
    for name in sorted(read):
        assert hasattr(remesh, name), name
        if name in homes:
            assert getattr(remesh, name) is getattr(homes[name], name), name


# ------------------------------------------------------------ PUSH-W12-ARCH-1


def _constant_comments(path: Path) -> dict[str, str]:
    """Return each module-level numeric constant of a source with its ``#:`` comment."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    module = importlib.import_module(f"{REFINE}.{path.stem}")
    out = {}
    for node in ast.parse(text).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id.isupper()):
            continue
        value = getattr(module, target.id)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        comment: list[str] = []
        row = node.lineno - 2
        while row >= 0 and lines[row].startswith("#:"):
            comment.insert(0, lines[row][2:].strip())
            row -= 1
        out[target.id] = " ".join(comment)
    return out


# ------------------------------------------------------------------- RC-QA-1


def _dense_surface(fx: Fixture, per: int) -> tuple[np.ndarray, list[list[int]]]:
    """Return the fixture's analytic surface as triangles ``per`` times finer than its cells."""
    cols = fx.nodes if fx.wrap else fx.nodes - 1
    s = np.linspace(0.0, cols, cols * per + 1)
    t = np.linspace(0.0, fx.stations - 1, (fx.stations - 1) * per + 1)
    pts = fx.surface(s, t)
    rows, width = pts.shape[:2]
    gid = np.arange(rows * width).reshape(rows, width)
    faces = []
    for k in range(rows - 1):
        for i in range(width - 1):
            a, b, c, d = gid[k, i], gid[k, i + 1], gid[k + 1, i + 1], gid[k + 1, i]
            faces += [[int(a), int(b), int(c)], [int(a), int(c), int(d)]]
    return pts.reshape(-1, 3), faces


def test_rc_qa_1_a_coarse_source_bounds_the_level_by_the_spline_surface():
    """(RC-QA-1;): the bound is met on the spline surface.

    A coarse cambered sheet of 4 stations by 4 chordwise nodes (three
    intervals each way, the fewest the not-a-knot spline takes) is refined
    at factor 2. Its coordinates are cubic in the grid indices, so the spline
    through the source's nodes is the analytic surface; that surface is
    sampled 16 times finer than the cells, which places it within a tenth of
    the bound (checked). Every level node lies within
    ``GRID_SURFACE_DISTANCE`` of the size from that surface, while the
    source's panels lie farther than the bound from the level's nodes: a
    distance measured to the panels fails on this fixture. Control: the
    linear interpolation of the same nodes (a resampling that stays on the
    panels) is farther than the bound from the surface.
    """
    fx = build_grid(Fixture("coarse-sheet", False, 4, 4, ("edge", "edge"), None))
    grid = recover_grid(fx.verts, fx.faces, fx.te)
    assert isinstance(grid, Grid), grid
    level = refine_grid(fx.verts, fx.faces, grid, chordwise=2.0, spanwise=2.0)
    assert len(level.points) == 7 * 7
    bound = GRID_SURFACE_DISTANCE * fx.size
    dense, faces = _dense_surface(fx, 16)
    between = fx.surface(np.arange(0.25, 3.0, 0.5), np.arange(0.25, 3.0, 0.5)).reshape(-1, 3)
    assert _distance_to_surface(between, dense, faces).max() < 0.1 * bound
    assert _distance_to_surface(level.points, dense, faces).max() <= bound
    assert _distance_to_surface(level.points, fx.verts, fx.faces).max() > bound
    at = {place: v for v, place in fx.place.items()}
    corners = [
        fx.verts[[at[(k, i)], at[(k, i + 1)], at[(k + 1, i)], at[(k + 1, i + 1)]]]
        for k in range(3)
        for i in range(3)
    ]
    linear = np.array([0.25 * (a + b + c + d) for a, b, c, d in corners])
    assert _distance_to_surface(linear, dense, faces).max() > bound


# ------------------------------------------------- IND4-01 and PUSH-IND3-QA-3


def _volume(verts: np.ndarray, faces: list[list[int]]) -> float:
    total = 0.0
    for f in faces:
        for j in range(1, len(f) - 1):
            a, b, c = verts[f[0]], verts[f[j]], verts[f[j + 1]]
            total += float(np.dot(a, np.cross(b, c))) / 6.0
    return total


def _two_pole_tube(around: int, split: bool) -> tuple[Composite, Fixture]:
    """Return a wing tube of 5 stations closed by a pole fan at each end, ``around`` nodes round.

    The lateral cells are quadrilaterals, or split into triangles with
    mirrored diagonals; the tip fan is the fixture's own and the root fan is
    wound like it around a pole 0.05 of the span inboard of the root section.
    """
    diag = {i: (0 if i < around // 2 else 1) for i in range(around)} if split else None
    fx = build_grid(Fixture(f"tube-{around}", True, 5, around, ("open", "pole"), diag))
    at = {place: v for v, place in fx.place.items()}
    ring = [at[(0, i)] for i in range(around)]
    center = fx.verts[ring].mean(axis=0) - [0.0, 0.05, 0.0]
    verts = np.vstack([fx.verts, center[None, :]])
    pole = len(verts) - 1
    faces = [list(f) for f in fx.faces]
    faces += [[pole, ring[(i + 1) % around], ring[i]] for i in range(around)]
    if _volume(verts, faces) < 0:
        faces = [f[::-1] for f in faces]
    column = [at[(k, 0)] for k in range(fx.stations)]
    te = 0.5 * (verts[column[:-1]] + verts[column[1:]])
    return Composite(verts, {"W": faces}, te, {}), fx


@pytest.mark.parametrize(
    ("around", "split", "shared"),
    [(4, False, 20), (5, True, 6), (6, True, 20)],
    ids=["4-quadrilaterals", "5-split-cells", "6-split-cells"],
)
def test_ind4_01_a_pole_fan_size_shared_by_other_nodes_is_remeshed_never_a_wrong_grid(
    tmp_path, around, split, shared
):
    """(IND4-01, PUSH-IND3-QA-3;): the limit of 0.38.0 is pinned.

    A two-pole tube is recovered only when its two poles are the only nodes
    of their fan size. Here the fan size is shared: 4 around a tube of
    quadrilaterals (every lateral node has 4 faces), 5 around a tube of
    split cells (the near-pole ring's valence, PUSH-IND3-QA-3) and 6 (the
    split cells' interior valence). No grid is recovered, the reason is
    named, and ``refine_mesh`` under ``auto`` remeshes the family, never
    writing a wrong grid. Control: 8 around, a size only the poles have, is
    recovered as the tube's grid of 5 stations by 8 nodes.
    """
    pytest.importorskip("trimesh")
    mesh, fx = _two_pole_tube(around, split)
    faces = mesh.families["W"]
    sizes = np.bincount(np.array([v for f in faces for v in f]))
    assert int((sizes == around).sum()) == shared + 2
    te = {v for v, (k, i) in fx.place.items() if i == 0}
    found = recover_grid(mesh.verts, faces, te)
    assert isinstance(found, str) and found, found
    src = write_source(tmp_path / "src", "wing", mesh)
    entry = refine_mesh(src, 2.0).report["W"]
    assert (entry["method"], entry["reason"]) == ("remesh", f"no grid recovered: {found}")
    control, cfx = _two_pole_tube(8, split)
    cte = {v for v, (k, i) in cfx.place.items() if i == 0}
    grid = recover_grid(control.verts, control.families["W"], cte)
    assert isinstance(grid, Grid) and grid.ends == ("pole", "pole") and grid.ids.shape == (5, 8)


# ------------------------------------------------------------ PUSH-IND2-QA-1


def _split_faces(fx: Fixture, diagonal) -> list[list[int]]:
    """Return a fixture's quadrilaterals each split into two triangles along ``diagonal(k, i)``."""
    out = []
    for quad in fx.faces:
        k, i = (min(fx.place[v][j] for v in quad) for j in (0, 1))
        a, b, c, d = quad
        out += [[a, b, c], [a, c, d]] if diagonal(k, i) == 0 else [[a, b, d], [b, c, d]]
    return out


def _corner_faces(faces: list[list[int]], grid: Grid) -> list[int]:
    """Return how many faces hold each corner of the grid."""
    corners = [grid.ids[0, 0], grid.ids[0, -1], grid.ids[-1, 0], grid.ids[-1, -1]]
    return [sum(int(c) in f for f in faces) for c in corners]


def test_push_ind2_qa_1_a_square_split_sheet_is_recovered_with_its_corners():
    """(PUSH-IND2-QA-1; GRID FAMILY, IND-08): a square sheet.

    A cambered sheet of 9 by 9 nodes (8 by 8 cells, so both sides have the
    same count and the corner search cannot tell them apart by length) is
    split along a-c, which puts two corners in two triangles: the boundary
    reading misses it and the corners come from the cells. The grid is the
    unsplit sheet's, node for node. Control: the unsplit sheet's corners
    each lie in one face.
    """
    fx = build_grid(Fixture("square", False, 9, 9, ("edge", "edge"), None))
    quads = recover_grid(fx.verts, fx.faces, fx.te)
    assert isinstance(quads, Grid) and _corner_faces(fx.faces, quads) == [1, 1, 1, 1]
    faces = _split_faces(fx, lambda k, i: 0)
    grid = recover_grid(fx.verts, faces, fx.te)
    assert isinstance(grid, Grid), grid
    assert sorted(_corner_faces(faces, grid)) == [1, 1, 2, 2]
    assert grid.layout == "sheet" and np.array_equal(grid.ids, quads.ids)
    assert grid.cell_diag is not None and grid.ids.shape == (9, 9)


def test_push_ind2_qa_1_a_one_by_n_split_strip_is_recovered_with_its_corners():
    """(PUSH-IND2-QA-1; GRID FAMILY, IND-08): a strip of one cell.

    A cambered strip of 2 stations by 19 chordwise nodes (one cell by 18) is
    split along a-c: every node lies on the boundary, and the corner search
    must place the short sides at the strip's ends. The grid is the unsplit
    strip's, node for node, with the trailing edge at chordwise index 0, for
    the strip split along a-c and along b-d. Control: the unsplit strip.
    """
    fx = build_grid(Fixture("strip", False, 2, 19, ("edge", "edge"), None))
    quads = recover_grid(fx.verts, fx.faces, fx.te)
    assert isinstance(quads, Grid) and quads.ids.shape == (2, 19)
    for rule in (0, 1):
        faces = _split_faces(fx, lambda k, i, rule=rule: rule)
        grid = recover_grid(fx.verts, faces, fx.te)
        assert isinstance(grid, Grid), grid
        assert np.array_equal(grid.ids, quads.ids), rule
        assert {int(v) for v in grid.ids[:, 0]} == fx.te


def _pinwheel() -> tuple[np.ndarray, list[list[int]]]:
    """Return a cambered 2 by 2 sheet whose four diagonals all meet at its center node.

    Node ``3 k + i`` is at station k, chordwise i. Every corner then lies in
    two triangles and so does every side's middle node, and both sets of four
    are corners of a 2 by 2 sheet whose cells are these triangles.
    """
    k, i = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    verts = np.stack([0.5 * i, 0.6 * k, 0.04 * np.sin(np.pi * i / 2.0)], axis=-1)
    a_c = lambda a, b, c, d: [[a, b, c], [a, c, d]]  # noqa: E731
    b_d = lambda a, b, c, d: [[a, b, d], [b, c, d]]  # noqa: E731
    faces = a_c(0, 1, 4, 3) + b_d(1, 2, 5, 4) + b_d(3, 4, 7, 6) + a_c(4, 5, 8, 7)
    return verts.reshape(-1, 3).astype(float), faces


def test_push_ind2_qa_1_two_sheets_fitting_the_cells_are_told_apart_by_the_perimeter_corner(
    tmp_path,
):
    """(PUSH-IND2-QA-1;, IND-08): the ambiguity and its cut.

    The pinwheel sheet's triangles are the cells of two 2 by 2 sheets, one
    cornered at the corners and one at the sides' middle nodes, and the
    trailing edge (the chordwise index 0 line and the edge from its first
    node to the next) is a grid line of both: the grid recovery names the
    ambiguity. A [[perimeter]] list holding those nodes is cut at its corner,
    and the chordwise line alone gives the one sheet cornered at the corners,
    so ``method = "grid"`` refines it as a grid. Control: the same line given
    whole to the recovery is still refused as two sheets.
    """
    pytest.importorskip("trimesh")
    verts, faces = _pinwheel()
    alone = recover_grid(verts, faces, {0, 3, 6})
    assert isinstance(alone, Grid) and sorted(alone.ids[:, 0].tolist()) == [0, 3, 6]
    reason = "2 sheets of split quadrilaterals fit its cells"
    assert recover_grid(verts, faces, {0, 1, 3, 6}) == reason
    te = 0.5 * (verts[[0, 3, 0]] + verts[[3, 6, 1]])
    toml = '[families.G]\nfactor = 2\nmethod = "grid"\n'
    src = write_source(
        tmp_path / "src", "wing", Composite(verts, {"G": faces}, te, {}), refine_toml=toml
    )
    entry = refine_mesh(src).report["G"]
    assert entry["method"] == "grid", entry
