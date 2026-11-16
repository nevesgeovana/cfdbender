"""smooth tubes and multiblock grids, recovered and
 resampled.

A body of revolution (a spinner, a nacelle barrel) holds no trailing edge: its
seam is the lowest source vertex index of its first station, and its sections
are resampled by a periodic spline, so the closed section has no knot at the
seam. An all-quadrilateral family that is not one grid (an L-shaped sheet, a
sheet with a hole) is cut along the grid lines leaving its singular vertices
into four-sided patches; each patch side is resampled once and shared.

The fixtures are synthetic (tests/p0380_mesh_fixtures.py), the numbering
shuffled: circles about the x axis whose radius varies along it, and sheets
cut from a grid on a bicubic surface, which the index-space splines
reproduce exactly, so every level node is known to lie on the surface.
"""
# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from mesh_fixtures import (
    Blocky,
    Body,
    Composite,
    body_of_revolution,
    face_coordinates,
    l_sheet,
    open_loop_count,
    read_mesh,
    square_hole_sheet,
    tube_pole_quads,
    write_source,
)

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh._refine import refine_mesh
from cfdbender.mesh._refine._blocks import (
    Blocks,
    recover_family,
    recover_joined,
    refine_blocks,
)
from cfdbender.mesh._refine._grid import (
    Grid,
    not_a_knot,
    periodic,
    recover_grid,
    refine_grid,
)

#: The radial error a periodic spline of 24 nodes per circle stays within (the open
#: not-a-knot spline through the same closed chain misses by about 6e-5 at the seam).
CIRCLE_TOLERANCE = 2e-5


def _radial_error(points: np.ndarray) -> np.ndarray:
    return np.abs(np.hypot(points[:, 1], points[:, 2]) - Body.radius(points[:, 0]))


def _smooth(body: Body) -> Grid:
    grid = recover_grid(body.verts, body.faces, set())
    assert isinstance(grid, Grid), grid
    return grid


@pytest.mark.parametrize("pole", [False, True], ids=["open", "pole"])
def test_a_smooth_tube_is_recovered_with_its_seam_at_the_lowest_source_index(pole):
    """no trailing edge and no ridge give a smooth tube.

    The seam is the lowest source vertex index of the first station. Control:
    a wing tube without its trailing-edge vertices has a sharp trailing edge,
    and is still refused with today's reason.
    """
    body = body_of_revolution(pole=pole)
    grid = _smooth(body)
    assert grid.layout == "tube" and grid.smooth
    assert grid.ends == (("open", "pole") if pole else ("open", "open"))
    assert int(grid.ids[0, 0]) == int(grid.ids[0].min())
    assert grid.describe().startswith("a smooth tube without trailing edge of 7 stations")
    wing = tube_pole_quads()
    assert recover_grid(wing.verts, wing.faces, set()) == "no trailing-edge vertex on the end loop"


@pytest.mark.parametrize("pole", [False, True], ids=["open", "pole"])
def test_a_smooth_tube_at_factor_one_is_the_source_in_coordinates_and_order(pole):
    """at factor 1 a smooth tube's faces equal the source's, in order.

    Control: the level's faces rotated by one vertex, and reversed, fail.
    """
    body = body_of_revolution(pole=pole)
    level = refine_grid(body.verts, body.faces, _smooth(body), chordwise=1.0, spanwise=1.0)
    want = face_coordinates(body.verts, body.faces)
    assert face_coordinates(level.points, level.faces) == want
    assert face_coordinates(level.points, [f[1:] + f[:1] for f in level.faces]) != want
    assert face_coordinates(level.points, level.faces[::-1]) != want
    assert len(level.te_midpoints) == 0
    assert level.report["chordwise_spline"] == "periodic"


@pytest.mark.parametrize("pole", [False, True], ids=["open", "pole"])
def test_factor_two_puts_the_sections_on_the_circle_and_the_seam_has_no_kink(pole):
    """the periodic spline keeps the new nodes on the circle.

    At chordwise 2 every level node lies within 2e-5 of the analytic circle
    of its station, the source nodes are kept bit for bit, and the count
    doubles around. Control: the open not-a-knot spline through the same
    closed chain (today's tube spline) misses the circle by more at the seam
    and fails the tolerance.
    """
    body = body_of_revolution(pole=pole)
    grid = _smooth(body)
    level = refine_grid(body.verts, body.faces, grid, chordwise=2.0, spanwise=1.0)
    ring = level.points[:-1] if pole else level.points
    assert len(ring) == 2 * body.nodes * body.stations
    assert float(_radial_error(ring).max()) < CIRCLE_TOLERANCE
    held = {tuple(p) for p in level.points}
    assert all(tuple(p) in held for p in body.verts)
    assert level.report["intervals"]["chordwise"] == [body.nodes, 2 * body.nodes]

    station = body.verts[grid.ids[0]]
    s = np.linspace(0.0, body.nodes, 2 * body.nodes + 1)[:-1]
    closed = periodic(station, s)
    opened = not_a_knot(np.vstack([station, station[:1]]), s)
    seam = [1, len(s) - 1]  # the new nodes either side of the seam
    assert _radial_error(closed)[seam].max() < 0.2 * _radial_error(opened)[seam].max()
    assert float(_radial_error(opened).max()) > CIRCLE_TOLERANCE


def test_the_periodic_spline_keeps_its_knots_and_is_smooth_across_the_seam():
    """the periodic spline is exact at its knots and closes smoothly.

    Its slope just before the seam equals its slope just after it. Control:
    the open not-a-knot spline through the same closed chain has a kink there.
    """
    n = 16
    angle = 2.0 * np.pi * np.arange(n) / n
    loop = np.stack([np.cos(angle) + 0.3 * np.cos(3 * angle), np.sin(2 * angle)], axis=1)
    assert np.array_equal(periodic(loop, np.arange(n)), loop)
    eps = 1e-6

    def kink(f):
        before = (f(np.array([n - eps])) - f(np.array([n - 2 * eps]))) / eps
        after = (f(np.array([eps])) - f(np.array([0.0]))) / eps
        return float(np.abs(before - after).max())

    assert kink(lambda p: periodic(loop, p % n)) < 1e-4
    closed = np.vstack([loop, loop[:1]])
    assert kink(lambda p: not_a_knot(closed, p)) > 1e-2


def test_the_periodic_spline_equals_scipys_where_scipy_is_installed():
    """the numpy periodic spline is scipy's to 1e-12.

    Control: scipy's not-a-knot spline through the same closed chain differs.
    """
    from scipy import interpolate

    rng = np.random.default_rng(5)
    for n in (3, 4, 7, 30):
        y = rng.normal(size=(n, 3))
        closed = np.vstack([y, y[:1]])
        p = np.sort(rng.uniform(0, n, 200))
        spline = interpolate.CubicSpline(np.arange(n + 1), closed, bc_type="periodic")
        assert float(np.abs(periodic(y, p) - spline(p)).max()) < 1e-12
        if n > 3:
            other = interpolate.CubicSpline(np.arange(n + 1), closed, bc_type="not-a-knot")
            assert float(np.abs(periodic(y, p) - other(p)).max()) > 1e-6


def test_a_smooth_tube_is_refined_as_a_grid_by_refine_mesh(tmp_path):
    """refine.json names the smooth tube.

    At factor 1 the level's faces are the source's (G6 judged and passed); at
    2 and 0.5 the gates G1 to G5 pass and no trailing-edge points are written.
    Control: the level at factor 2 is not the source.
    """
    body = body_of_revolution(pole=True)
    mesh = Composite(body.verts, {"S": body.faces}, np.zeros((0, 3)), {})
    src = write_source(tmp_path / "src", "body", mesh)
    source = face_coordinates(*_family(src, "S"))
    for factor in (1.0, 2.0, 0.5):
        level = refine_mesh(src, factor, out_dir=tmp_path / f"f{factor}")
        entry = level.report["S"]
        assert entry["method"] == "grid", entry
        assert entry["reason"].startswith("a smooth tube without trailing edge")
        assert entry["chordwise_spline"] == "periodic"
        gates = {g.name: g.verdict for g in level.audit.gates}
        assert all(g.passed for g in level.audit.gates), gates
        assert not any(p.name.endswith(".te.txt") for p in level.files)
        written = face_coordinates(*_family(level.obj, "S"))
        if factor == 1.0:
            assert gates.get("G6") == "pass", gates
            assert written == source
        else:
            assert written != source


def _family(path, name):
    verts, families = read_mesh(path)
    return verts, families[name]


# ------------------------------------------------------------ multiblock grids

SHEETS = {
    "l-sheet": (l_sheet, [(4, 7), (4, 5), (6, 7)], 1),
    "square-hole": (square_hole_sheet, [(4, 4)] * 8, 2),
}


def _blocks(sheet: Blocky) -> Blocks:
    found = recover_family(sheet.verts, sheet.faces, set())
    assert isinstance(found, Blocks), found
    return found


def _edge_uses(faces) -> Counter:
    return Counter(
        (min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1], strict=True)
    )


def _conforming(points, faces, loops: int) -> bool:
    """Return whether a level is one welded all-quadrilateral surface with the source's loops.

    No edge has more than two faces, no two nodes share a position, and the
    open boundary loops are the source's in number: a shared curve whose two
    sides held different nodes would leave a slit, and the count would grow.
    """
    uses = _edge_uses(faces)
    distinct = len({tuple(np.round(p, 12)) for p in points}) == len(points)
    return (
        all(len(f) == 4 for f in faces)
        and max(uses.values()) <= 2
        and distinct
        and open_loop_count(faces) == loops
    )


def _off_surface(points) -> float:
    return float(np.abs(points[:, 2] - Blocky.height(points[:, 0], points[:, 1])).max())


@pytest.mark.parametrize("name", SHEETS)
def test_the_layout_is_cut_along_the_grid_lines_leaving_the_singular_vertices(name):
    """the separatrices cut the family into four-sided patches.

    The L-shaped sheet (six corners) is cut by the two grid lines leaving its
    re-entrant corner into three patches; the sheet with a square hole by the
    eight lines leaving the hole's corners into eight. Neither is one grid
    (the control: its sheet or tube recovery names why).
    """
    make, dims, _ = SHEETS[name]
    sheet = make()
    blocks = _blocks(sheet)
    assert sorted(tuple(int(n) - 1 for n in ids.shape) for ids in blocks.patches) == sorted(dims)
    assert sum(cells.size for cells in blocks.cells) == len(sheet.faces)
    assert blocks.describe() == f"a multiblock of {len(dims)} four-sided patches of quadrilaterals"
    assert isinstance(recover_grid(sheet.verts, sheet.faces, set()), str)


@pytest.mark.parametrize("shuffled", [False, True], ids=["rows", "shuffled"])
@pytest.mark.parametrize("name", SHEETS)
def test_a_multiblock_family_at_factor_one_is_the_source_in_coordinates_and_order(name, shuffled):
    """at factor 1 the faces equal the source's, in order.

    The source's faces are written row by row, or shuffled. Control: the
    level's faces rotated by one vertex, and reversed, fail.
    """
    sheet = SHEETS[name][0]()
    if shuffled:
        order = np.random.default_rng(2).permutation(len(sheet.faces))
        sheet.faces = [sheet.faces[k] for k in order]
    level = refine_blocks(sheet.verts, sheet.faces, _blocks(sheet), factor=1.0)
    want = face_coordinates(sheet.verts, sheet.faces)
    assert face_coordinates(level.points, level.faces) == want
    assert face_coordinates(level.points, [f[1:] + f[:1] for f in level.faces]) != want
    assert face_coordinates(level.points, level.faces[::-1]) != want
    assert len(level.points) == len(sheet.verts)


@pytest.mark.parametrize("factor", [2.0, 0.5])
@pytest.mark.parametrize("name", SHEETS)
def test_factor_two_and_half_give_a_conforming_all_quad_family_on_the_surface(name, factor):
    """one welded quad surface, every node on the surface.

    Every edge has at most two faces, no two nodes share a position, the open
    boundary loops are the source's, the counts are round(f m) per patch, and
    every node lies on the bicubic surface within 1e-9. At factor 2 every
    source node is kept bit for bit. Control: the same level with the faces of
    one corner given their own nodes (cut loose along a curve) fails the
    conformity check.
    """
    make, dims, loops = SHEETS[name]
    sheet = make()
    level = refine_blocks(sheet.verts, sheet.faces, _blocks(sheet), factor=factor)
    assert _conforming(level.points, level.faces, loops)
    rows = sorted(tuple(b["rows"]) + tuple(b["columns"]) for b in level.report["blocks"])
    assert rows == sorted(
        (r, max(1, round(factor * r)), c, max(1, round(factor * c))) for r, c in dims
    )
    assert len(level.faces) == sum(b["rows"][1] * b["columns"][1] for b in level.report["blocks"])
    assert _off_surface(level.points) < 1e-9
    if factor == 2.0:
        held = {tuple(p) for p in level.points}
        assert all(tuple(p) in held for p in sheet.verts)
    assert not _conforming(*_unweld(level.points, level.faces), loops)


def _unweld(points, faces):
    """Control: give the faces of the sheet's corner at x, y < 0.26 their own copy of each node."""
    centroids = np.array([points[f].mean(axis=0) for f in faces])
    mine = set(np.nonzero((centroids[:, 0] < 0.26) & (centroids[:, 1] < 0.26))[0].tolist())
    assert 0 < len(mine) < len(faces)
    extra = {}
    out = []
    for k, f in enumerate(faces):
        if k in mine:
            f = [extra.setdefault(v, len(points) + len(extra)) for v in f]
        out.append(f)
    copies = np.array([points[v] for v in extra], dtype=float).reshape(-1, 3)
    shifted = copies + 1e-6
    return np.vstack([points, shifted]), out


@pytest.mark.parametrize("name", SHEETS)
def test_a_shared_patch_side_is_the_same_nodes_from_both_sides(name):
    """each arc is resampled once; both patches hold its nodes.

    For every source edge two patches share, the level's nodes between its
    two ends, run straight through 2 m level edges, each of them used by two
    faces, one of each patch. A side resampled once per patch would leave
    those edges with one face each (the mutation that names an arc's nodes
    per patch fails here).
    """
    sheet = SHEETS[name][0]()
    blocks = _blocks(sheet)
    level = refine_blocks(sheet.verts, sheet.faces, blocks, factor=2.0)
    users = Counter(arc for sides in blocks.sides for arc, _ in sides)
    shared = [a for a, n in users.items() if n == 2]
    assert shared
    uses = _edge_uses(level.faces)
    for arc in shared:
        listed = level.points.tolist()
        ends = [listed.index(sheet.verts[v].tolist()) for v in blocks.arcs[arc]]
        chain = _walk(level.points, uses, ends[0], ends[-1], 2 * (len(blocks.arcs[arc]) - 1))
        assert chain is not None and len(chain) == 2 * (len(blocks.arcs[arc]) - 1) + 1
        assert all(
            uses[(min(a, b), max(a, b))] == 2 for a, b in zip(chain, chain[1:], strict=False)
        )


def _walk(points, uses, start, stop, steps):
    """Return the straight chain of level nodes from start to stop in the given number of edges."""
    adj = {}
    for a, b in uses:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    chain = [start]
    for _ in range(steps):
        here = points[chain[-1]]
        aim = points[stop] - here
        best = max(
            adj[chain[-1]] - set(chain),
            key=lambda u: float(np.dot(points[u] - here, aim) / np.linalg.norm(points[u] - here)),
        )
        chain.append(best)
    return chain if chain[-1] == stop else None


def test_a_family_with_a_triangle_falls_back_as_before(tmp_path):
    """a triangle leaves the family without a multiblock grid.

    ``recover_family`` gives the sheet or tube recovery's own reason; under
    ``method = "grid"`` the refinement is refused naming it. Control: the
    same sheet without the split cell is a multiblock grid.
    """
    sheet = l_sheet(split={(2, 2)})
    why = recover_family(sheet.verts, sheet.faces, set())
    assert isinstance(why, str) and why == recover_grid(sheet.verts, sheet.faces, set())
    assert isinstance(recover_family(l_sheet().verts, l_sheet().faces, set()), Blocks)
    mesh = Composite(sheet.verts, {"L": sheet.faces}, np.zeros((0, 3)), {})
    src = write_source(
        tmp_path / "src", "plate", mesh, refine_toml='[families.L]\nmethod = "grid"\nfactor = 2\n'
    )
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src)
    text = str(caught.value)
    assert "family L" in text and f"no grid was recovered: {why}" in text
    assert not (tmp_path / "plate_R2").exists()


@pytest.mark.parametrize("name", SHEETS)
def test_refine_mesh_reports_a_multiblock_grid_and_its_audit_passes(tmp_path, name):
    """refine.json names the layout and patches.

    At factor 1 the level's faces are the source's and G6 passes; at 2 and
    0.5 the gates G1 to G5 pass. refine.json gives the method grid, the
    layout multiblock, the patch count and each patch's rows and columns.
    Control: chordwise is refused on the multiblock family, naming it.
    """
    make, dims, _ = SHEETS[name]
    sheet = make()
    mesh = Composite(sheet.verts, {"M": sheet.faces}, np.zeros((0, 3)), {})
    src = write_source(tmp_path / "src", "plate", mesh)
    verts, families = read_mesh(src)
    source = face_coordinates(verts, families["M"])
    for factor in (1.0, 2.0, 0.5):
        level = refine_mesh(src, factor, out_dir=tmp_path / f"f{factor}")
        entry = level.report["M"]
        assert entry["method"] == "grid" and entry["layout"] == "multiblock", entry
        assert entry["patches"] == len(dims)
        assert sorted((b["rows"][0], b["columns"][0]) for b in entry["blocks"]) == sorted(dims)
        assert (
            entry["reason"] == f"a multiblock of {len(dims)} four-sided patches of quadrilaterals"
        )
        gates = {g.name: g.verdict for g in level.audit.gates}
        assert all(g.passed for g in level.audit.gates), gates
        written = face_coordinates(*_family(level.obj, "M"))
        assert (written == source) == (factor == 1.0)
        if factor == 1.0:
            assert gates.get("G6") == "pass", gates
    # With no trailing edge, a directional factor is told against the longest side of the
    # family's boundary, warned; the reference is not written as a trailing edge.
    with pytest.warns(GeometryWarning, match="longest side of its boundary"):
        level = refine_mesh(src, normal=2.0, out_dir=tmp_path / "c")
    assert level.report["M"]["method"] == "grid"
    assert not list(level.folder.glob("*.te.txt"))


@pytest.mark.parametrize("factor", [0.01, 0.2, 0.24])
def test_a_patch_side_a_factor_cannot_coarsen_is_kept_and_reported(tmp_path, capsys, factor):
    """a factor below 1/m of a patch side keeps that side.

    The L-shaped sheet's shortest patch side holds 4 source intervals, so a
    factor below 1/4 cannot coarsen it. Decision of 2026-10-22 (it
    replaces the refusal RV-08 of the 0.38.0 release review): the family is
    not refused; such a side keeps max(1, round(f m)) intervals, the rest of
    the family is resampled, the level is written and its audit gates pass,
    the warning names the family, each patch and side and where it lies, and
    refine.json lists the same under ``kept_sides``. The command exits 0.
    Control: exactly 1/4 keeps no side, and the report says so.
    """
    from cfdbender.mesh import cli

    sheet = l_sheet()
    mesh = Composite(sheet.verts, {"M": sheet.faces}, np.zeros((0, 3)), {})
    src = write_source(tmp_path / "src", "plate", mesh)
    with pytest.warns(GeometryWarning) as caught:
        level = refine_mesh(src, factor, out_dir=tmp_path / "a")
    text = " ".join(str(w.message) for w in caught)
    assert f"family M: the factor {factor:g} cannot coarsen" in text, text
    entry = level.report["M"]
    assert entry["layout"] == "multiblock"
    kept = entry["kept_sides"]
    assert kept, entry
    for k in kept:
        m = k["source_intervals"]
        assert factor * m < 1.0
        assert k["intervals"] == max(1, round(m * factor)) >= 1
        assert f"patch {k['patch']} {k['side']}" in text
        assert all(
            lo <= c <= hi
            for lo, c, hi in zip(k["bbox_min"], k["centroid"], k["bbox_max"], strict=True)
        )
    assert any(k["source_intervals"] == 4 for k in kept)
    # The written level, not the report's own formula: each kept side's patch holds
    # that many intervals along the side (a column side spans the patch's rows).
    blocks = entry["blocks"]
    for k in kept:
        along = "rows" if k["side"].endswith("column") else "columns"
        assert blocks[k["patch"] - 1][along] == [k["source_intervals"], k["intervals"]], blocks
    assert all(g.passed for g in level.audit.gates), {g.name: g.verdict for g in level.audit.gates}
    capsys.readouterr()
    assert cli.main(["refine", str(src), str(factor), "--out", str(tmp_path / "b")]) == 0
    err = capsys.readouterr().err
    for stage in ("read plate.obj", "family 1 of 1: M", "grid 1 of 1: M", "writing", "auditing"):
        assert f"] {stage}" in err, err
    assert any(line.startswith("[refine ") for line in err.splitlines())
    level = refine_mesh(src, 0.25, out_dir=tmp_path / "c")
    assert level.report["M"]["kept_sides"] == []
    assert min(min(b["rows"][1], b["columns"][1]) for b in level.report["M"]["blocks"]) == 1


def test_refine_progress_names_each_stage_and_quiet_hides_it(tmp_path, capsys):
    """refine_mesh reports its stages to ``progress``, in order.

    Called with a callback it receives one line per stage and family, read,
    family, grid, writing and auditing in that order; without one it says
    nothing. The command prints the lines as ``[refine HH:MM:SS +N.N s] ...`` on
    stderr, and ``--log-level quiet`` hides them while the level is written.
    """
    from cfdbender.mesh import cli

    sheet = l_sheet()
    mesh = Composite(sheet.verts, {"M": sheet.faces}, np.zeros((0, 3)), {})
    src = write_source(tmp_path / "src", "plate", mesh)
    lines: list[str] = []
    refine_mesh(src, 0.5, out_dir=tmp_path / "a", progress=lines.append)
    stages = ["read plate.obj", "family 1 of 1: M", "grid 1 of 1: M", "writing", "auditing"]
    found = [next(i for i, line in enumerate(lines) if line.startswith(s)) for s in stages]
    assert found == sorted(found), lines
    capsys.readouterr()
    refine_mesh(src, 0.5, out_dir=tmp_path / "b")
    assert "[refine " not in capsys.readouterr().err
    assert cli.main(["refine", str(src), "0.5", "--quiet", "--out", str(tmp_path / "q")]) == 0
    assert "[refine " not in capsys.readouterr().err
    assert (tmp_path / "q" / "plate_R0p5" / "plate_R0p5.obj").is_file()


# ------------------------------------------- grids resampled as one layout (0.44.0)

# (): touching grid families are resampled as one layout.
#
# The source is synthetic, a tail in miniature built on one structured surface
# of 13 chordwise by 18 spanwise cells, its numbering shuffled: ``W``, a sheet
# (the stabilizer) over spanwise cells 10 to 18; ``C``, a sheet (the centre
# strip) over 3 to 7; ``J``, two junction strips over 0 to 3 and 7 to 10, one
# family and so a multiblock grid of two patches; and ``T``, a strip of
# triangles beyond W's tip that no factor selects. The trailing edge is the
# chordwise index 0 line of every family. W and J share a station, C and J
# two. Before 0.44.0 only the first of them in the source's order stayed a grid.

NX = 13  # chordwise cells
BANDS = {"W": [(10, 18)], "C": [(3, 7)], "J": [(0, 3), (7, 10)]}
NY = 18
FACTOR = 0.76923


def _point(i, j):
    """Return the surface at the (fractional) indices: chordwise i from the trailing edge."""
    u, v = np.asarray(i, float) / NX, np.asarray(j, float) / NY
    x = 1.0 - u - 0.05 * u**3 + 0.1 * v
    y = 2.0 * v + 0.05 * v**2
    z = 0.06 * np.sin(np.pi * u) * (1.0 - 0.3 * v)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def tail(*, strip: bool = True, seed: int = 7) -> Composite:
    """Return the tail: W, C, J (and T beyond W), numbering shuffled, faces row by row."""
    rows = NY + (5 if strip else 0)
    used = [(i, j) for j in range(rows + 1) for i in range(NX + 1)]
    perm = np.random.default_rng(seed).permutation(len(used))
    number = {ij: int(perm[n]) for n, ij in enumerate(used)}
    verts = np.empty((len(used), 3))
    for (i, j), n in number.items():
        verts[n] = _point(i, j)
    families: dict[str, list[list[int]]] = {}
    for name, bands in BANDS.items():
        families[name] = [
            [number[(i, j)], number[(i + 1, j)], number[(i + 1, j + 1)], number[(i, j + 1)]]
            for lo, hi in bands
            for j in range(lo, hi)
            for i in range(NX)
        ]
    if strip:
        tri = []
        for j in range(NY, rows):
            for i in range(NX):
                a, b = number[(i, j)], number[(i + 1, j)]
                c, d = number[(i + 1, j + 1)], number[(i, j + 1)]
                tri += [[a, b, c], [a, c, d]] if (i + j) % 2 else [[a, b, d], [b, c, d]]
        families["T"] = tri
    te = 0.5 * (_point(0, np.arange(NY)) + _point(0, np.arange(1, NY + 1)))
    return Composite(verts, families, te, {})


def _source(tmp_path: Path, refine_toml: str | None = None, **kwargs) -> Path:
    return write_source(tmp_path / "src", "tail", tail(**kwargs), refine_toml=refine_toml)


def _nodes(faces) -> set[int]:
    return {v for f in faces for v in f}


def _centroids(verts, faces) -> np.ndarray:
    return np.array([verts[f].mean(axis=0) for f in faces])


def _in_source_order(level_verts, level_faces, source_verts, source_faces) -> bool:
    """Return whether the level's faces follow their nearest source faces in order."""
    old = _centroids(source_verts, source_faces)
    new = _centroids(level_verts, level_faces)
    nearest = np.argmin(np.linalg.norm(new[:, None, :] - old[None, :, :], axis=2), axis=1)
    return bool(np.all(np.diff(nearest) >= 0))


def _gates(level) -> dict[str, str]:
    return {f"{g.name} {g.family}": g.verdict for g in level.audit.gates}


GROUP = ["W", "C", "J"]


def test_three_touching_grids_stay_grids_in_their_own_families_and_conform(tmp_path):
    """W, C and J at 0.76923 are all grids, conforming.

    Each is a grid of its own family that names the other two under
    ``grouped_with``; every curve two of them share holds round(13 f) + 1 = 11
    nodes from both sides; the audit's gates G1 to G5 pass, G5 judged on each
    pair of families that shared nodes in the source; each family's faces follow its own
    source faces in order; T, unchanged, is remeshed only in its band beside W.
    Controls: the order check fails on a family's faces reversed; before
    C and J were remeshed.
    """
    src = _source(tmp_path)
    level = refine_mesh(src, FACTOR, families=GROUP)
    report = level.report
    for name in GROUP:
        assert report[name]["method"] == "grid", report[name]
        assert report[name]["grouped_with"] == [n for n in GROUP if n != name]
    assert report["J"]["layout"] == "multiblock" and report["W"]["layout"] == "sheet"
    assert report["T"]["method"] == "unchanged" and report["T"]["band"] > 0
    gates = _gates(level)
    assert all(g.passed for g in level.audit.gates), gates
    assert {"G5 W and J", "G5 C and J", "G5 W and T"} <= set(gates), gates
    sv, sf = read_mesh(src)
    lv, lf = read_mesh(level.obj)
    for a, b, curves in (("W", "J", 1), ("C", "J", 2)):
        assert len(_nodes(lf[a]) & _nodes(lf[b])) == curves * (round(NX * FACTOR) + 1)
        assert len(_nodes(sf[a]) & _nodes(sf[b])) == curves * (NX + 1)
    assert not _nodes(lf["W"]) & _nodes(lf["C"])
    for name in GROUP:
        assert len(lf[name]) < len(sf[name])
        assert _in_source_order(lv, lf[name], sv, sf[name]), name
    assert not _in_source_order(lv, lf["C"][::-1], sv, sf["C"])


def test_at_factor_one_each_grouped_family_is_its_source_face_for_face(tmp_path):
    """at factor 1 every family is the source, G6 passing.

    Control: the level's faces of J rotated by one vertex differ from the source's.
    """
    src = _source(tmp_path, strip=False)
    level = refine_mesh(src, 1.0)
    sv, sf = read_mesh(src)
    lv, lf = read_mesh(level.obj)
    for name in GROUP:
        assert face_coordinates(lv, lf[name]) == face_coordinates(sv, sf[name]), name
    rotated = [f[1:] + f[:1] for f in lf["J"]]
    assert face_coordinates(lv, rotated) != face_coordinates(sv, sf["J"])
    gates = _gates(level)
    assert all(g.passed for g in level.audit.gates) and "G6 W" in gates, gates


def test_two_grids_giving_a_shared_curve_different_counts_follow_the_largest(tmp_path):
    """J at 0.5 beside W and C at 0.76923 follows the larger family.

    A station of 13 intervals would hold round(13 x 0.76923) = 10 intervals on
    one side and round(13 x 0.5) = 6 on the other. With no governing family the
    declared family of the largest perimeter on the curves leads, J follows,
    warned, and all three stay grids. Before this was refused.
    """
    toml = (
        f"[families.W]\nfactor = {FACTOR}\n\n[families.C]\nfactor = {FACTOR}\n\n"
        "[families.J]\nfactor = 0.5\n"
    )
    src = _source(tmp_path, toml, strip=False)
    with pytest.warns(GeometryWarning, match=r"family W follows J \(declared, of the largest"):
        level = refine_mesh(src, out_dir=tmp_path / "lead")
    assert [level.report[n]["method"] for n in ("W", "C", "J")] == ["grid"] * 3


def test_a_governing_family_gives_its_counts_to_the_curves_it_shares(tmp_path):
    """W marked governing = true leads J, which the largest-perimeter rule would pick."""
    toml = (
        f"[families.W]\nfactor = {FACTOR}\ngoverning = true\n\n"
        f"[families.C]\nfactor = {FACTOR}\n\n[families.J]\nfactor = 0.5\n"
    )
    src = _source(tmp_path, toml, strip=False)
    with pytest.warns(GeometryWarning, match=r"family J follows W \(governing\)"):
        level = refine_mesh(src, out_dir=tmp_path / "gov")
    assert [level.report[n]["method"] for n in ("W", "C", "J")] == ["grid"] * 3


def test_explicit_grid_auto_takes_the_governing_family_factors(tmp_path):
    """with a governing family, W and C explicitly declare grid_auto.

    They are refined as grids with J's factor (0.76923), and written as
    method grid_auto in the level's selection.
    """
    followers = '\n[families.W]\nmethod = "grid_auto"\n[families.C]\nmethod = "grid_auto"\n'
    toml = f"[families.J]\nfactor = {FACTOR}\ngoverning = true\n" + followers
    src = _source(tmp_path, toml, strip=False)
    level = refine_mesh(src, out_dir=tmp_path / "auto")
    assert [level.report[n]["method"] for n in ("W", "C", "J")] == ["grid"] * 3
    assert level.report["W"]["grid_auto"] == {
        "follows": "J",
        "factors": [FACTOR, FACTOR],
        "directions": "from J",
    }
    assert "grid_auto" not in level.report["J"]
    control = refine_mesh(
        _source(tmp_path / "c", f"[families.J]\nfactor = {FACTOR}\n" + followers, strip=False),
        out_dir=tmp_path / "ctl",
    )
    assert control.report.get("W", {}).get("method") != "grid"


def test_grid_auto_without_a_governing_family_is_kept_unchanged(tmp_path):
    """method = "grid_auto" with no governing family is warned and kept."""
    toml = f'[families.J]\nfactor = {FACTOR}\n\n[families.W]\nmethod = "grid_auto"\n'
    src = _source(tmp_path, toml, strip=False)
    with pytest.warns(GeometryWarning, match="marks none"):
        level = refine_mesh(src, out_dir=tmp_path / "none", accept_remesh=True)
    assert level.report.get("W", {}).get("method") != "grid"


# ------------------------------------------- chordwise and spanwise on a multiblock


def _pairs(entry) -> list[list[tuple[int, int]]]:
    """Return each patch's (before, after) interval pairs, its two directions sorted."""
    return sorted(sorted([tuple(b["rows"]), tuple(b["columns"])]) for b in entry["blocks"])


def test_the_junction_coarsened_chordwise_only_keeps_its_spanwise_counts(tmp_path):
    """(): J at chordwise 0.76923, spanwise 1.

    J's arcs are told chordwise or spanwise from its trailing edge: its
    spanwise arcs (3 intervals across each strip, along the trailing edge)
    keep 3, its chordwise arcs (13 intervals) take 10, and so do the curves it
    shares with W and C at 0.76923; the level conforms and its gates pass.
    Control: J at factor 0.76923 in both directions takes 2 across each strip.
    """
    toml = (
        f"[families.W]\nfactor = {FACTOR}\n\n[families.C]\nfactor = {FACTOR}\n\n"
        f"[families.J]\nchordwise = {FACTOR}\nspanwise = 1\n"
    )
    src = _source(tmp_path, toml, strip=False)
    level = refine_mesh(src)
    report = level.report
    lv, lf = read_mesh(level.obj)
    assert len(_nodes(lf["C"]) & _nodes(lf["J"])) == 2 * (round(NX * FACTOR) + 1)
    # Count the written OBJ's span stations and cells, independently of refine.json.
    for lo, hi in BANDS["J"]:
        cells = [f for f in lf["J"] if _point(0, lo)[1] < lv[f, 1].mean() < _point(0, hi)[1]]
        assert len(cells) == 3 * round(NX * FACTOR)
        assert len(np.unique(np.round(lv[sorted(_nodes(cells)), 1], 12))) == 4
    assert _pairs(report["J"]) == [[(3, 3), (13, 10)]] * 2, report["J"]
    assert _pairs(report["C"]) == [[(4, 3), (13, 10)]]
    assert _pairs(report["W"]) == [[(8, 6), (13, 10)]]
    assert report["J"]["kept_sides"] == []
    assert all(g.passed for g in level.audit.gates), _gates(level)
    both = refine_mesh(src, FACTOR, out_dir=tmp_path / "both").report
    assert _pairs(both["J"]) == [[(3, 2), (13, 10)]] * 2


def _l_with_trailing_edge(shear: float = 0.0) -> Composite:
    """Return the L-shaped multiblock sheet M with its j = 0 row as the trailing edge."""
    sheet = l_sheet()
    verts = sheet.verts.copy()
    verts[:, 0] += shear * verts[:, 1]
    row = sorted((v for v, (i, j) in sheet.place.items() if j == 0), key=lambda v: sheet.place[v])
    te = 0.5 * (verts[row[:-1]] + verts[row[1:]])
    return Composite(verts, {"M": sheet.faces}, te, {})


def test_a_lone_multiblock_takes_chordwise_and_spanwise_from_its_trailing_edge(tmp_path):
    """(): the L-shaped multiblock M at chordwise 0.5, spanwise 1.

    Its trailing edge runs along its rows, so every row arc is spanwise and
    keeps its count, every column arc is chordwise and halves. Control: the
    single factor 0.5 halves both, as before.
    """
    src = write_source(tmp_path / "src", "plate", _l_with_trailing_edge())
    level = refine_mesh(src, normal=0.5, parallel=1.0, out_dir=tmp_path / "a")
    assert _pairs(level.report["M"]) == [
        [(4, 2), (5, 5)],
        [(4, 2), (7, 7)],
        [(6, 3), (7, 7)],
    ], level.report["M"]
    assert all(g.passed for g in level.audit.gates), _gates(level)
    half = refine_mesh(src, 0.5, out_dir=tmp_path / "b").report["M"]
    assert _pairs(half) == [[(4, 2), (5, 2)], [(4, 2), (7, 4)], [(6, 3), (7, 4)]]


def test_a_patch_whose_sides_both_run_near_the_edge_is_told_by_the_closer_pair(tmp_path):
    """(): M sheared until its columns run near the edge.

    Sheared by 1.5, its columns run within 35 degrees of the trailing edge,
    like its rows. The pair of opposite sides that runs more along the edge is
    parallel and the other normal, patch by patch, so the level is written and
    nothing is refused. Control: the single factor is accepted on the sheared M.
    """
    src = write_source(tmp_path / "src", "plate", _l_with_trailing_edge(shear=1.5))
    level = refine_mesh(src, normal=0.5, parallel=1.0, out_dir=tmp_path / "a")
    assert level.report["M"]["method"] == "grid"
    assert refine_mesh(src, 0.5, out_dir=tmp_path / "b").report["M"]["method"] == "grid"


@pytest.mark.parametrize("direction", ["chordwise", "spanwise"])
def test_mesh_blocks_multiblock_direction_without_trailing_edge_takes_the_longest_side(
    tmp_path, direction
):
    """A multiblock family with no trailing edge is told against its longest boundary side."""
    mesh = _l_with_trailing_edge()
    mesh.te = np.empty((0, 3))
    src = write_source(tmp_path / "src", "plate", mesh)
    with pytest.warns(GeometryWarning, match="longest side of its boundary"):
        level = refine_mesh(
            src,
            **{"normal" if direction == "chordwise" else "parallel": 0.5},
            out_dir=tmp_path / "side",
        )
    assert level.report["M"]["method"] == "grid"
    assert refine_mesh(src, 0.5, out_dir=tmp_path / "control").report["M"]["method"] == "grid"


@pytest.mark.parametrize("dy,expected", [(1.0, True), (1.0 - 1e-10, True), (1.0 + 1e-10, False)])
def test_mesh_blocks_spanwise_includes_the_45_degree_boundary(dy, expected):
    from cfdbender.mesh._refine._blocks import runs_spanwise

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, dy, 0.0]])
    edge = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    assert runs_spanwise(vertices, [0, 1], edge) is expected


def test_mesh_blocks_union_without_four_sided_layout_joins_none():
    from cfdbender.mesh._refine._blocks import plan
    from cfdbender.mesh._refine._config import FamilySpec

    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, -1, 0], [0, -1, 0], [1, 0, 1], [0, 0, 1]],
        dtype=float,
    )
    families = {"A": [[0, 1, 2, 3]], "B": [[0, 1, 4, 5]], "C": [[0, 1, 6, 7]]}
    edges = {name: {(0, 1)} for name in families}
    grids = {
        n: recover_family(vertices, faces, {v for e in edges[n] for v in e})
        for n, faces in families.items()
    }
    assert all(isinstance(grid, Grid) for grid in grids.values())
    layouts, late = plan(
        vertices,
        families,
        grids,
        {n: FamilySpec(factor=2.0) for n in families},
        edges,
        where="three sheets sharing one edge",
    )
    assert layouts == {}
    assert [row[:2] for row in late] == [("B", "A"), ("C", "A")]
    assert all("form no layout" in row[2] for row in late)


def test_mesh_blocks_planner_retains_each_arc_direction_and_source():
    """orientation survives planning, including shared-curve propagation."""
    from cfdbender.mesh._refine._blocks import plan
    from cfdbender.mesh._refine._config import FamilySpec

    vertices = np.array([[x, y, 0] for x in range(3) for y in range(2)], dtype=float)
    families = {"A": [[0, 2, 3, 1]], "B": [[2, 4, 5, 3]]}
    edges = {"A": {(0, 1)}, "B": set()}
    grids = {
        n: recover_family(vertices, faces, {0, 1} if n == "A" else {2, 3})
        for n, faces in families.items()
    }
    layouts, late = plan(
        vertices,
        families,
        grids,
        {"A": FamilySpec(factor=2, governing=True), "B": FamilySpec(factor=2)},
        edges,
        where="two sheets",
        followers={"B": "A"},
    )
    assert not late
    directions = layouts["A"].directions
    assert set(directions) == set(range(len(layouts["A"].blocks.patches)))
    records = {}
    for name in families:
        arcs = {a for p in layouts[name].patches_of(name) for a, _ in layouts[name].blocks.sides[p]}
        records[name] = {
            a: d for p in layouts[name].patches_of(name) for a, d in directions[p].items()
        }
        assert set(records[name]) == arcs
        assert {d.direction for d in records[name].values()} == {"parallel", "normal"}
        assert {d.source_family for d in records[name].values()} == {"A"}
    assert {d.provenance for d in records["A"].values()} == {"listed_te"}
    assert {d.provenance for d in records["B"].values()} == {"propagated_from:A"}
    assumed, _ = plan(
        vertices,
        families,
        grids,
        {"A": FamilySpec(factor=2, governing=True), "B": FamilySpec(factor=2)},
        edges,
        where="two sheets",
        followers={"B": "A"},
        assumed={"A"},
    )
    assert {
        d.provenance
        for p in assumed["A"].patches_of("A")
        for d in assumed["A"].directions[p].values()
    } == {"longest_side"}


# ------------------------------------------- refine-r2: the longest side and the order


def _cube_sphere(n: int = 4):
    """Six n-by-n patches of quadrilaterals closing a sphere: a family with no boundary."""
    index: dict[tuple[float, ...], int] = {}
    verts: list[np.ndarray] = []

    def vertex(p: np.ndarray) -> int:
        key = tuple(np.round(p, 9))
        if key not in index:
            index[key] = len(verts)
            verts.append(p)
        return index[key]

    g = np.linspace(-1.0, 1.0, n + 1)
    faces = []
    for ax in range(3):
        other = [x for x in range(3) if x != ax]
        for s in (-1.0, 1.0):
            for a in range(n):
                for b in range(n):
                    quad = []
                    for i, j in ((a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1)):
                        p = np.zeros(3)
                        p[ax], p[other[0]], p[other[1]] = s, g[i], g[j]
                        quad.append(vertex(p / np.linalg.norm(p)))
                    faces.append(quad if s > 0 else quad[::-1])
    return np.asarray(verts), faces


def test_refine_r2_qa1_a_closed_multiblock_family_takes_the_parallel_factor(tmp_path):
    """(refine-r2 QA-1): a family with no boundary and no trailing edge is not a crash."""
    verts, faces = _cube_sphere()
    src = write_source(
        tmp_path / "src", "ball", Composite(verts, {"S": faces}, np.empty((0, 3)), {})
    )
    with pytest.warns(GeometryWarning, match="no boundary to orient it"):
        level = refine_mesh(src, normal=0.5, out_dir=tmp_path / "n")
    assert level.report["S"]["method"] == "grid"


def test_refine_r2_qa2_a_hole_is_a_side_of_the_boundary():
    """(refine-r2 QA-2): a corner-free loop beside a cornered one is walked too."""
    from cfdbender.mesh._refine._blocks import boundary_runs

    square = []
    for k in range(10):
        square.append((k - 5.0, -5.0, 0.0))
    for k in range(10):
        square.append((5.0, k - 5.0, 0.0))
    for k in range(10):
        square.append((5.0 - k, 5.0, 0.0))
    for k in range(10):
        square.append((-5.0, 5.0 - k, 0.0))
    n = len(square)
    touching = {k: [(k - 1) % n, (k + 1) % n] for k in range(n)}
    hole = [
        (3.0 * np.cos(2 * np.pi * k / 72), 3.0 * np.sin(2 * np.pi * k / 72), 0.0) for k in range(72)
    ]
    touching.update({n + k: [n + (k - 1) % 72, n + (k + 1) % 72] for k in range(72)})
    pts = np.asarray(square + hole)
    runs = boundary_runs(pts, touching)
    length = [float(np.linalg.norm(np.diff(pts[r], axis=0), axis=1).sum()) for r in runs]
    assert max(length) == pytest.approx(2 * np.pi * 3, rel=1e-3)
    walked = [tuple(sorted(e)) for r in runs for e in zip(r, r[1:], strict=False)]
    assert len(walked) == len(set(walked)) == n + 72


def test_refine_r2_qa3_the_directions_do_not_hang_on_the_patch_numbering():
    """(refine-r2 QA-3): a curved strip of three patches reads the same in any order."""
    from types import SimpleNamespace

    from cfdbender.mesh._refine._blocks import te_directions

    pts = []

    def add(p):
        pts.append(p)
        return len(pts) - 1

    spoke = [
        [add((r * np.cos(np.radians(a)), r * np.sin(np.radians(a)), 0.0)) for r in (1.0, 1.5, 2.0)]
        for a in (0, 60, 120, 180)
    ]

    def ring_arc(k, r):
        i = 0 if r == 1.0 else 2
        mid = add((r * np.cos(np.radians(60 * k + 30)), r * np.sin(np.radians(60 * k + 30)), 0.0))
        return (spoke[k][i], mid, spoke[k + 1][i])

    arcs, rings = [], []
    for k in range(3):
        rings.append((len(arcs), len(arcs) + 1))
        arcs += [ring_arc(k, 1.0), ring_arc(k, 2.0)]
    spokes = [len(arcs) + i for i in range(4)]
    arcs += [tuple(s) for s in spoke]
    sides = tuple(
        ((rings[k][0], False), (rings[k][1], False), (spokes[k], False), (spokes[k + 1], False))
        for k in range(3)
    )
    blocks = SimpleNamespace(
        arcs=tuple(arcs), sides=sides, patches=tuple(np.array([[0]]) for _ in range(3))
    )
    xs = np.linspace(-3.0, 3.0, 13)
    te = np.stack(
        [
            np.c_[xs[:-1], -0.1 + 0 * xs[:-1], 0 * xs[:-1]],
            np.c_[xs[1:], -0.1 + 0 * xs[1:], 0 * xs[1:]],
        ],
        axis=1,
    )
    told = [
        te_directions(np.asarray(pts), blocks, order, te, family="F", what="x")
        for order in ([0, 1, 2], [1, 0, 2], [2, 1, 0])
    ]
    assert told[0] == told[1] == told[2]


def test_mesh_blocks_governing_conflict_keeps_each_patch_orientation():
    """Q4: a conflict must not replace a whole family by parallel arcs."""
    import inspect
    import warnings
    from types import SimpleNamespace

    from cfdbender.mesh._refine._blocks import te_directions

    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0)])
    blocks = SimpleNamespace(
        arcs=((0, 1), (2, 3), (0, 2), (1, 3), (1, 4), (3, 5)),
        sides=(
            ((0, False), (1, False), (2, False), (3, False)),
            ((0, False), (2, False), (4, False), (5, False)),
        ),
        patches=(np.array([[0, 1], [2, 3]]), np.array([[0, 2], [4, 5]])),
    )
    local = {}
    options = (
        {"patch_runs": local} if "patch_runs" in inspect.signature(te_directions).parameters else {}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = te_directions(
            points,
            blocks,
            [0, 1],
            np.array([[[0, 0, 0], [1, 0, 0]]]),
            family="governing",
            what="two patches",
            **options,
        )
    if not local:
        local = {p: {a: runs[a] for a, _ in blocks.sides[p]} for p in (0, 1)}
    assert all(set(told.values()) == {"chordwise", "spanwise"} for told in local.values())
    assert any(
        "patch" in str(w.message) and "own geometric orientation" in str(w.message) for w in caught
    )


# ------------------------------------- a node a neighbour adds on a junction perimeter


def _junction_side(mesh, sign: float, side: str, *, extra: bool, nx: int, splits=()) -> list:
    """Add one side of a wing-body junction to ``mesh`` (verts list, families, te list).

    The wing spans ``nx`` station intervals along x and 19 across y, one
    family W, or W1, W2, ... cut across at the rows in ``splits``. The body
    B below it shares the wing's junction nodes; with ``extra`` B holds one
    more node on the junction in
    each station interval (two edges where the wing has one), as a finer
    neighbour mesh does, so the wing's shared edge meets a corner at every
    station. Each wing family's trailing edge is its far row. ``sign`` mirrors
    the side in x. Returns the wing's family stems.
    """
    verts, families, te = mesh
    ny, by = 19, 4
    at: dict[tuple, int] = {}

    def node(key, x, y):
        if key not in at:
            at[key] = len(verts)
            verts.append((sign * (1.0 + x), y, 0.02 * np.sin(x) * (1.0 + y)))
        return at[key]

    for k in range(nx + 1):
        for j in range(ny + 1):
            node(("w", k, j), 1.5 * k / nx, 0.1 * j)
    cuts = [0, *splits, ny]
    stems = [f"W{n + 1}" for n in range(len(cuts) - 1)] if splits else ["W"]
    for stem, lo, hi in zip(stems, cuts, cuts[1:], strict=False):
        families[stem + side] = [
            [at[("w", k, j)], at[("w", k + 1, j)], at[("w", k + 1, j + 1)], at[("w", k, j + 1)]]
            for k in range(nx)
            for j in range(lo, hi)
        ]
    bx = 2 * nx if extra else nx
    for k in range(bx + 1):
        for j in range(by + 1):
            if j == by and (k * nx) % bx == 0:
                at[("b", k, j)] = at[("w", k * nx // bx, 0)]
            node(("b", k, j), 1.5 * k / bx, -0.1 * (by - j))
    families["B" + side] = [
        [at[("b", k, j)], at[("b", k + 1, j)], at[("b", k + 1, j + 1)], at[("b", k, j + 1)]]
        for k in range(bx)
        for j in range(by)
    ]
    if sign < 0:
        for name in [n for n in families if n.endswith(side)]:
            families[name] = [f[::-1] for f in families[name]]
    for row in cuts[1:]:
        far = np.array([verts[at[("w", k, row)]] for k in range(nx + 1)])
        te.append(0.5 * (far[:-1] + far[1:]))
    return stems


_GOVERNED = "normal_factor = 0.455\nparallel_factor = 0.75\ngoverning_parallel = true\n"


@pytest.mark.parametrize(
    ("nx", "splits", "table", "station_counts"),
    [
        (3, (), _GOVERNED, [3, 2]),
        (3, (9,), _GOVERNED, [3, 2]),
        (8, (5, 10, 15), "factor = 0.35013\n", [8, 3]),
    ],
    ids=["one-family", "two-stacked", "corner-at-every-station"],
)
def test_mesh_blocks_an_extra_junction_node_does_not_stop_a_family_coarsening(
    tmp_path, nx, splits, table, station_counts
):
    """a family beside a finer neighbour coarsens like its mirror.

    On the right the body holds an extra node on the junction in every station
    interval: each node it shares with the wing is where two boundaries only
    touch, and its separatrix cut every wing family into patches of 1 station
    interval, which no factor coarsens (3 stations at normal 0.455 and
    parallel 0.75: 57 to 27 faces, the mirror 57 to 18; 8 stations at 0.35013
    in four stacked families: 8 kept sides of 1 per family). Joined across
    those loose lines, each wing family is one patch counted over its whole
    run of station intervals, round(f m) as the unsplit mirror, and the body,
    declared to nobody, is unchanged.
    """
    mesh: tuple[list, dict, list] = ([], {}, [])
    stems = _junction_side(mesh, 1.0, "R", extra=True, nx=nx, splits=splits)
    _junction_side(mesh, -1.0, "L", extra=False, nx=nx, splits=splits)
    verts, families, te = mesh
    toml = "".join(f"[families.{w}{s}]\n{table}\n" for w in stems for s in "RL")
    source = Composite(np.array(verts, dtype=float), families, np.vstack(te), {})
    src = write_source(tmp_path / "src", "junction", source, refine_toml=toml)
    report = refine_mesh(src, out_dir=tmp_path / "level").report
    for w in stems:
        right, left = report[w + "R"], report[w + "L"]
        assert right["method"] == left["method"] == "grid", (right, left)
        assert right["blocks"] == left["blocks"], (right["blocks"], left["blocks"])
        assert right["faces"] == left["faces"], (right["faces"], left["faces"])
        assert [b["rows"] for b in right["blocks"]] == [station_counts], right["blocks"]
        assert right["kept_sides"] == [], right["kept_sides"]
    if not splits:
        assert report["WR"]["faces"] == [57, 18]
        assert report["WR"]["blocks"] == [{"rows": [3, 2], "columns": [19, 9]}]
    assert report["BR"]["faces"][0] == report["BR"]["faces"][1] == 8 * nx


# ------------------- review R1: the corners the loose-line join must keep


def _wing_and_body(*, strip_families: bool = False, bend_far: bool = False, ridge: bool = False):
    """Return (verts, faces, family of each face, te vertices) of the extra-node junction, nx=3.

    The wing node (k, j) is vertex ``20 k + j``. ``strip_families`` gives each station
    interval its own family; ``bend_far`` moves the far-row nodes at stations 1 and 2 in y so
    the boundary turns there; ``ridge`` lifts odd stations so every station line is a ridge.
    """
    mesh: tuple[list, dict, list] = ([], {}, [])
    _junction_side(mesh, 1.0, "R", extra=True, nx=3)
    verts = [list(v) for v in mesh[0]]
    for k in (1, 2):
        if bend_far:
            verts[20 * k + 19][1] += 1.0
    if ridge:
        for k in range(4):
            for j in range(20):
                verts[20 * k + j][2] += 0.5 * (k % 2)
    wing, body = mesh[1]["WR"], mesh[1]["BR"]
    faces = [*wing, *body]
    family = [f // 19 if strip_families else 0 for f in range(len(wing))] + [3] * len(body)
    return np.array(verts, dtype=float), faces, family, {20 * k + 19 for k in range(4)}


def _wing_patches(case, te=None) -> list[tuple]:
    """Return the wing's patches (vertices below 80) of the joined layout, as vertex tuples."""
    from cfdbender.mesh._refine._layout import edge_faces

    verts, faces, family, te_far = case
    joins = {
        e for e, fs in edge_faces(faces).items() if len(fs) == 2 and family[fs[0]] != family[fs[1]]
    }
    found = recover_joined(verts, faces, te_far if te is None else te, joins)
    assert isinstance(found, Blocks), found
    return sorted(tuple(p.ravel().tolist()) for p in found.patches if int(p.max()) < 80)


def test_mesh_blocks_f2_control_the_loose_lines_join_the_wing_into_one_patch():
    """with no corner on them, the two station lines are joined (one 3 x 19 patch)."""
    assert len(_wing_patches(_wing_and_body())) == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"bend_far": True}, {"ridge": True}, {"strip_families": True}],
    ids=["end-bends-over-45", "ridge-on-the-line", "line-shared-with-another-family"],
)
def test_mesh_blocks_f2_a_real_corner_on_the_cut_line_stays_cut(kwargs):
    """a bend, a ridge or a family change on the cut line keeps its patches apart."""
    assert len(_wing_patches(_wing_and_body(**kwargs))) == 3  # the traced layout: 3 strips


def test_mesh_blocks_f2_a_trailing_edge_on_the_cut_line_stays_cut():
    te = {20 * k + j for k in (1, 2) for j in range(20)}
    assert len(_wing_patches(_wing_and_body(), te=te)) > 3  # nothing joined


# ------- regression of a base strip between two trailing edges cut every station


def _blunt_trailing_edge(tmp_path: Path, families: str, toml: str = "") -> dict:
    """Refine an upper sheet U and a lower sheet L (8 station intervals by 6) and their base B.

    B, one element across and with no [families] table, closes the blunt trailing
    edge: its two long sides are U's and L's trailing edges, both in the
    ``[[perimeter]]`` node list at every station, so every edge across B joins two
    trailing-edge nodes without being a trailing-edge edge.
    """
    verts: list[tuple[float, float, float]] = []
    at: dict[tuple[str, int, int], int] = {}
    for side, z in (("u", 0.05), ("l", -0.05)):
        for k in range(9):
            for i in range(7):
                at[side, k, i] = len(verts)
                x = 1.5 * k / 8
                verts.append((x, 0.1 * i, z * (1.0 + 0.2 * i) + 0.02 * np.sin(2 * x)))
    every = {
        "U": [
            [at["u", k, i], at["u", k + 1, i], at["u", k + 1, i + 1], at["u", k, i + 1]]
            for k in range(8)
            for i in range(6)
        ],
        "L": [
            [at["l", k, i + 1], at["l", k + 1, i + 1], at["l", k + 1, i], at["l", k, i]]
            for k in range(8)
            for i in range(6)
        ],
        "B": [
            [at["l", k, 0], at["l", k + 1, 0], at["u", k + 1, 0], at["u", k, 0]] for k in range(8)
        ],
    }
    rows = [np.array([verts[at[s, k, 0]] for k in range(9)]) for s in "ul"]
    te = np.vstack([0.5 * (row[:-1] + row[1:]) for row in rows])
    source = Composite(np.array(verts), {n: every[n] for n in families}, te, {})
    declared = (
        "[families.U]\nnormal_factor = 0.5\nparallel_factor = 0.5\ngoverning_parallel = true\n\n"
        "[families.L]\nnormal_factor = 0.5\nparallel_factor = 0.5\n\n"
    )
    src = write_source(tmp_path / "src", "blunt", source, refine_toml=declared + toml)
    return refine_mesh(src, out_dir=tmp_path / "level").report


def test_mesh_blocks_a_base_strip_between_two_trailing_edges_leaves_each_family_one_patch(tmp_path):
    """U and L coarsen parallel as one patch each; B follows.

    Since (0.49.0) the undeclared base B joins the layout of U and L, and
    each edge across it, its two ends on the listed trailing edges, was traced as a
    trailing-edge line: every station was cut, U and L held their 8 parallel
    intervals (48 faces to 24) and B was unchanged; before B stayed out of
    the layout and U and L were one patch, 48 faces to 12.
    """
    report = _blunt_trailing_edge(tmp_path, "ULB")
    for name in ("U", "L"):
        assert report[name]["faces"] == [48, 12], report[name]
        assert report[name]["blocks"] == [{"rows": [8, 4], "columns": [6, 3]}], report[name]
    (base,) = report["B"]["blocks"]  # one patch: 8 station intervals to 4, 1 across kept
    assert sorted([base["rows"], base["columns"]]) == [[1, 1], [8, 4]], report["B"]
    assert report["B"]["faces"] == [8, 4]
    assert report["B"]["continuity"]["directions"] == ["parallel"], report["B"]


def test_mesh_blocks_control_without_the_base_strip_each_sheet_is_one_patch(tmp_path):
    """control: U and L alone, the same lists and factors, coarsen as before."""
    report = _blunt_trailing_edge(tmp_path, "UL")
    for name in ("U", "L"):
        assert report[name]["faces"] == [48, 12], report[name]


def test_mesh_blocks_perimeter_list_orients_never_cuts_a_rectangle(tmp_path):
    """front panel F, hinge strip S and rear panel R stay one patch.

    S, one element across with no [families] table, joins F and R at the hinge. As
    prepare-to-refine lists every patch's rearmost line, the node list holds F's
    rear row, S's rear row and R's trailing edge: three listed nodes per station,
    and both long sides of S are reference lines (R, longer at its front, is
    oriented by it). Each family is one patch, its rows along the stations (the
    parallel direction), coarsening 8 station intervals to 4 with the factor 0.5,
    S keeping its one interval across. Before every edge across S was
    traced as a trailing edge: 8 patches per family, the parallel count held and
    S unchanged.
    """
    xs = (0.0, 0.25, 0.5, 0.75, 1.0, 1.05, 1.3, 1.55, 1.8, 2.05)
    verts: list[tuple[float, float, float]] = []
    at: dict[tuple[int, int], int] = {}
    for k in range(9):
        for i, x in enumerate(xs):
            y = 1.5 * k / 8
            at[k, i] = len(verts)
            verts.append((x, y, 0.05 * np.sin(np.pi * x / 2.1) + 0.02 * (3.0 - x) * np.sin(2 * y)))

    def quads(lo: int, hi: int) -> list[list[int]]:
        return [
            [at[k, i], at[k, i + 1], at[k + 1, i + 1], at[k + 1, i]]
            for k in range(8)
            for i in range(lo, hi)
        ]

    families = {"F": quads(0, 4), "S": quads(4, 5), "R": quads(5, 9)}
    folder = tmp_path / "src"
    src = write_source(folder, "trio", Composite(np.array(verts), families, np.zeros((0, 3)), {}))
    listed = [at[k, i] for i in (4, 5, 9) for k in range(9)]
    rows = [f"{v + 1}, , {x:.9f}, {y:.9f}, {z:.9f}" for v in listed for x, y, z in [verts[v]]]
    (folder / "trio.nodes.txt").write_text("#  Id, Name, X , Y, Z\n" + "\n".join(rows) + "\n")
    toml = "".join(
        f"[families.{n}]\nnormal_factor = 0.5\nparallel_factor = 0.5\n{lead}\n"
        for n, lead in (("F", "governing_parallel = true\n"), ("R", ""))
    )
    toml += '[[perimeter]]\nfile = "trio.nodes.txt"\nte_families = ["F", "S", "R"]\n'
    (folder / "trio.refine.toml").write_text(toml)
    report = refine_mesh(src, out_dir=tmp_path / "level").report
    for name, across in (("F", [4, 2]), ("R", [4, 2]), ("S", [1, 1])):
        assert report[name]["blocks"] == [{"rows": [8, 4], "columns": across}], report[name]
    assert report["F"]["faces"] == report["R"]["faces"] == [32, 8]
    assert report["S"]["faces"] == [8, 4]
    assert report["S"]["continuity"]["directions"] == ["parallel"], report["S"]


def _te_file_points(path: Path) -> np.ndarray:
    rows = [r for r in path.read_text().splitlines()[1:] if r.strip()]
    return np.array([[float(w) for w in r.split(",")] for r in rows]).reshape(-1, 3)


def test_mesh_blocks_rear_panel_te_txt_holds_its_own_trailing_edge_whole(tmp_path):
    """A rear panel's te.txt holds its own trailing edge, every coarsened interval present.

    Upper and lower skins meet at a sharp trailing edge, slightly curved in
    planform; a swept hinge, longer than the trailing edge, splits each skin into
    a front and a rear panel; the prepare-style list holds every panel's rearmost
    line and ``te_families`` names all four. The rule of the reference
    ([[perimeter]] te_families): a family's trailing edge is its rearmost listed
    line, the edges its surface lies forward of. The rear panel lies aft of the
    hinge, so its trailing edge, and its file, is the trailing edge alone, with
    the mid-point of every one of its 4 coarsened intervals: a chord across the
    curved line sits off it by more than the grid's surface distance, and its
    ends, source nodes, decide. Before the fix the rear panel kept the hinge as
    the longest listed piece on it, so its te.txt and the level's held the hinge
    and not the trailing edge, and the mid-point off the curved line was dropped.
    """
    verts: list[tuple[float, float, float]] = []
    at: dict[tuple[str, int, int], int] = {}
    for k in range(9):
        y = 1.5 * k / 8
        x_hinge, x_te = 0.8 + 0.4 * y, 2.0 + 0.08 * np.sin(2 * y)
        for i in range(9):
            x = x_hinge * i / 4 if i <= 4 else x_hinge + (x_te - x_hinge) * (i - 4) / 4
            s = x / x_te
            thick = 0.12 * np.sqrt(s) * (1.0 - s)
            for side, sign in (("u", 1.0), ("l", -1.0)):
                if i == 8 and side == "l":
                    at["l", k, 8] = at["u", k, 8]  # the sharp trailing edge: one node
                    continue
                at[side, k, i] = len(verts)
                verts.append((x, y, sign * thick))

    def quads(side: str, lo: int, hi: int) -> list[list[int]]:
        out = []
        for k in range(8):
            for i in range(lo, hi):
                q = [at[side, k, i], at[side, k, i + 1], at[side, k + 1, i + 1], at[side, k + 1, i]]
                out.append(q if side == "u" else q[::-1])
        return out

    families = {
        "FU": quads("u", 0, 4),
        "RU": quads("u", 4, 8),
        "FL": quads("l", 0, 4),
        "RL": quads("l", 4, 8),
    }
    folder = tmp_path / "src"
    src = write_source(folder, "tail", Composite(np.array(verts), families, np.zeros((0, 3)), {}))
    listed = sorted({at[s, k, i] for s in "ul" for i in (4, 8) for k in range(9)})
    rows = [f"{v + 1}, , {x:.9f}, {y:.9f}, {z:.9f}" for v in listed for x, y, z in [verts[v]]]
    (folder / "tail.nodes.txt").write_text("#  Id, Name, X , Y, Z\n" + "\n".join(rows) + "\n")
    toml = "".join(
        f"[families.{n}]\nnormal_factor = 0.5\nparallel_factor = 0.5\n{lead}\n"
        for n, lead in (("FU", "governing_parallel = true\n"), ("RU", ""), ("FL", ""), ("RL", ""))
    )
    toml += '[[perimeter]]\nfile = "tail.nodes.txt"\nte_families = ["FU", "RU", "FL", "RL"]\n'
    (folder / "tail.refine.toml").write_text(toml)
    report = refine_mesh(src, out_dir=tmp_path / "level").report
    (level,) = [p for p in (tmp_path / "level").iterdir() if p.is_dir()]
    for name in ("RU", "RL"):
        assert report[name]["blocks"] == [{"rows": [8, 4], "columns": [4, 2]}], report[name]
        recorded = np.array(report[name]["trailing_edge_points"])
        assert len(recorded) == 8 and (recorded[:, 0] > 1.9).all(), recorded  # its own line
        own = _te_file_points(level / f"{level.name}.{name}.te.txt")
        assert len(own) == 4, own  # every coarsened interval, the curved line's included
        assert (own[:, 0] > 1.9).all(), own  # the trailing edge, never the hinge
    front = _te_file_points(level / f"{level.name}.FU.te.txt")
    assert len(front) == 4 and (front[:, 0] < 1.5).all(), front  # the front panel's rear line
    whole = _te_file_points(level / f"{level.name}.te.txt")
    assert len(whole) == 12 and (whole[:, 0] > 1.9).sum() == 4, whole
