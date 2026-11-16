"""the structured grid family, recovered and resampled.

Every fixture is built in tests/p0380_mesh_fixtures.py from an analytic
surface whose coordinates are cubic polynomials of the grid indices (a
cambered, tapered, swept wing with
clustering toward the leading edge and the tip, and a cambered thin sheet
with clustering at both edges). The not-a-knot spline reproduces a cubic
exactly, so the level's nodes are known in advance: each level node is
matched to the analytic node of its new indices, which proves the counts and
the knots and gives every node its (station, chordwise) place for the face
order checks. The vertex numbering of every source is shuffled, so nothing
is read from the file's vertex order.
"""
# ruff: noqa: E501

from __future__ import annotations

import numpy as np
import pytest
from mesh_fixtures import (
    CELL_DIAG,
    CELL_QUAD,
    FIXTURES,
    MIRRORED,
    NODES,
    Fixture,
    build_grid,
    sheet_quads,
    tube_pole_quads,
    tube_pole_triangles,
    tube_zipper,
)

from cfdbender._errors import InputArtifactError
from cfdbender.mesh._refine._geometry import GRID_SURFACE_DISTANCE
from cfdbender.mesh._refine._grid import (
    Grid,
    check_factors,
    not_a_knot,
    recover_grid,
    refine_grid,
)


def _recovered(fx: Fixture) -> Grid:
    grid = recover_grid(fx.verts, fx.faces, fx.te)
    assert isinstance(grid, Grid), grid
    return grid


# ------------------------------------------------------------------ the oracle


def _params(fx: Fixture, chordwise: float, spanwise: float):
    """Return the new chordwise and station indices of round(f m) per stretch."""
    t = np.linspace(0, fx.stations - 1, round(spanwise * (fx.stations - 1)) + 1)
    if not fx.wrap:
        return np.linspace(0, fx.nodes - 1, round(chordwise * (fx.nodes - 1)) + 1), t
    h, n = fx.le, fx.nodes
    n1 = round(chordwise * h)
    n2 = n1 if "zipper" in fx.ends else round(chordwise * (n - h))
    s = np.concatenate([np.linspace(0, h, n1 + 1)[:-1], h + np.linspace(0, n - h, n2 + 1)[:-1]])
    return s, t


def _places(fx: Fixture, level, chordwise: float, spanwise: float):
    """Match every level node to its analytic node; return node -> (k', i') and the shape."""
    s, t = _params(fx, chordwise, spanwise)
    expected = fx.surface(s, t)
    flat = expected.reshape(-1, 3)
    grid_points = level.points[:-1] if fx.pole is not None else level.points
    d = np.linalg.norm(grid_points[:, None, :] - flat[None, :, :], axis=2)
    best = d.argmin(axis=1)
    assert float(d.min(axis=1).max()) < 1e-9 * fx.size
    assert len(grid_points) == len(flat) and len(set(best.tolist())) == len(flat)
    shape = expected.shape[:2]
    return {p: divmod(int(b), shape[1]) for p, b in enumerate(best)}, shape


def _expected_lateral(fx: Fixture, where, shape, chordwise: float):
    """Return the lateral faces in the source's sweep: k-major, i ascending, source templates."""
    node = {v: p for p, v in where.items()}
    kp, nc = shape
    s, _ = _params(fx, chordwise, 1.0)
    ends = np.append(s, fx.nodes) if fx.wrap else s
    old_i = np.floor(0.5 * (ends[:-1] + ends[1:])).astype(int)
    out = []
    for k in range(kp - 1):
        for i in range(nc if fx.wrap else nc - 1):
            j = (i + 1) % nc
            corner = {(0, 0): (k, i), (0, 1): (k, j), (1, 1): (k + 1, j), (1, 0): (k + 1, i)}
            source_i = int(old_i[i]) % fx.nodes if fx.wrap else min(int(old_i[i]), fx.nodes - 2)
            template = CELL_QUAD if fx.diag is None else CELL_DIAG[fx.diag[source_i]]
            out += [[node[corner[o]] for o in face] for face in template]
    return out


def _sweep_holds(fx, level, where, shape, chordwise, faces=None):
    """Return whether the level's lateral faces are the source's sweep, face for face."""
    faces = level.faces if faces is None else faces
    expected = _expected_lateral(fx, where, shape, chordwise)
    return faces[: len(expected)] == expected


def _cell_runs(faces, where, count):
    """Group the first `count` faces into runs of one cell each, keyed by the cell."""
    runs: list[tuple[tuple[int, int], list[list[int]]]] = []
    for f in faces[:count]:
        places = [where[v] for v in f]
        key = (min(k for k, _ in places), min(i for _, i in places))
        if runs and runs[-1][0] == key:
            runs[-1][1].append(f)
        else:
            runs.append((key, [f]))
    return runs


def _reverse_sweep(faces, where, count):
    """Control writer: the cells in reverse order, each cell's faces unchanged."""
    runs = _cell_runs(faces, where, count)
    return [f for _, run in runs[::-1] for f in run] + faces[count:]


def _transpose_sweep(faces, where, count):
    """Control writer: the cells column by column instead of row by row."""
    runs = sorted(_cell_runs(faces, where, count), key=lambda r: (r[0][1], r[0][0]))
    return [f for _, run in runs for f in run] + faces[count:]


def _rotate_faces(faces):
    """Control writer: every face starts one vertex later."""
    return [f[1:] + f[:1] for f in faces]


def _distance_to_surface(points, verts, faces):
    """Return each point's distance to the source's faces (fanned into triangles)."""
    tri = np.array([[f[0], f[k], f[k + 1]] for f in faces for k in range(1, len(f) - 1)])
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    out = np.empty(len(points))
    for n0 in range(0, len(points), 64):
        p = points[n0 : n0 + 64, None, :]
        best = np.full(p.shape[0], np.inf)
        for x, y in ((a, b), (b, c), (c, a)):
            e = y - x
            t = np.clip(((p - x) * e).sum(-1) / (e * e).sum(-1), 0.0, 1.0)
            best = np.minimum(best, np.linalg.norm(p - (x + t[..., None] * e), axis=2).min(1))
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        height = ((p - a) * normal).sum(-1)
        q = p - height[..., None] * normal
        inside = np.ones(height.shape, bool)
        for x, y in ((a, b), (b, c), (c, a)):
            inside &= (np.cross(y - x, q - x) * normal).sum(-1) >= -1e-15
        plane = np.where(inside, np.abs(height), np.inf).min(axis=1)
        out[n0 : n0 + 64] = np.minimum(best, plane)
    return out


# ------------------------------------------------------------------ the tests


@pytest.mark.parametrize("make", FIXTURES, ids=lambda m: m.__name__)
def test_factor_one_is_the_source_in_coordinates_and_order(make):
    """at factor 1 a grid's faces equal the source's, in order.

    Coordinates face for face and vertex for vertex, so the start vertex of
    every face is checked too. Control: each control writer fails the check.
    """
    fx = make()
    level = refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=1.0, spanwise=1.0)

    def same(faces):
        return len(faces) == len(fx.faces) and all(
            np.array_equal(level.points[f], fx.verts[g])
            for f, g in zip(faces, fx.faces, strict=True)
        )

    assert same(level.faces)
    assert {tuple(p) for p in level.points} == {tuple(p) for p in fx.verts}
    assert not same(_rotate_faces(level.faces))
    assert not same(level.faces[::-1])


@pytest.mark.parametrize("make", FIXTURES, ids=lambda m: m.__name__)
def test_factor_two_doubles_the_intervals_keeps_the_knots_on_the_surface(make):
    """factor 2 gives 2m by 2n intervals, knots exact, nodes on it.

    Every source node is a level node bit for bit (the trailing edge, the
    leading edge and the end stations among them), and every level node lies
    within 1e-3 of the size from the source's faces. Control: a node moved by
    2e-3 of the size off the surface is caught.

    Release review of 0.38.0: proves the fix of RV-04.
    """
    fx = make()
    grid = _recovered(fx)
    level = refine_grid(fx.verts, fx.faces, grid, chordwise=2.0, spanwise=2.0)
    where, shape = _places(fx, level, 2.0, 2.0)
    m_chord, m_span = grid.intervals
    assert level.report["intervals"] == {
        "chordwise": [m_chord, 2 * m_chord],
        "spanwise": [m_span, 2 * m_span],
    }
    assert shape == (2 * m_span + 1, 2 * m_chord + (0 if fx.wrap else 1))
    held = {tuple(p) for p in level.points}
    assert all(tuple(p) in held for p in fx.verts)
    distance = _distance_to_surface(level.points, fx.verts, fx.faces)
    assert float(distance.max()) <= GRID_SURFACE_DISTANCE * fx.size
    off = level.points.copy()
    off[len(off) // 3] += [0.0, 0.0, 2e-3 * fx.size]
    assert _distance_to_surface(off, fx.verts, fx.faces).max() > GRID_SURFACE_DISTANCE * fx.size


@pytest.mark.parametrize("make", FIXTURES, ids=lambda m: m.__name__)
def test_factor_two_is_written_in_the_source_sweep(make):
    """the level keeps the source's start cell, directions, cell order.

    Controls: a writer that sweeps the cells in reverse, one that transposes
    the index directions and one that rotates each face's vertices all fail.
    """
    fx = make()
    level = refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=2.0, spanwise=2.0)
    where, shape = _places(fx, level, 2.0, 2.0)
    count = len(_expected_lateral(fx, where, shape, 2.0))
    assert level.report["order"] == "sweep"
    assert _sweep_holds(fx, level, where, shape, 2.0)
    for control in (
        _reverse_sweep(level.faces, where, count),
        _transpose_sweep(level.faces, where, count),
        _rotate_faces(level.faces),
    ):
        assert not _sweep_holds(fx, level, where, shape, 2.0, control)


def _diagonals(fx, level, where):
    """Return the diagonal each new split cell uses, keyed by the cell."""
    found = {}
    faces = level.faces[: 4 * fx.lateral]
    for a, b in zip(faces[0::2], faces[1::2], strict=True):
        (k0, i0), (_, i1) = sorted(where[v] for v in set(a) & set(b))
        rising = (i1 - i0) % (2 * fx.nodes) == 1
        found[(k0, i0 if rising else i1)] = 0 if rising else 1
    return found


def test_each_new_cell_is_split_along_its_source_cells_diagonal():
    """a mirrored split is kept cell by cell at factor 2.

    The fixture splits the upper half of the section along one diagonal and
    the lower half along the other. Control: the same wing split along one
    diagonal everywhere does not match the mirrored split.
    """
    fx = tube_pole_triangles()
    level = refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=2.0, spanwise=2.0)
    got = _diagonals(fx, level, _places(fx, level, 2.0, 2.0)[0])
    assert len(got) == 2 * fx.lateral  # one entry per new cell, two triangles each
    assert all(d == MIRRORED[i // 2] for (_, i), d in got.items())
    plain = build_grid(
        Fixture(
            "tube-one-diagonal", True, 9, NODES, ("open", "pole"), dict.fromkeys(range(NODES), 0)
        )
    )
    level = refine_grid(plain.verts, plain.faces, _recovered(plain), chordwise=2.0, spanwise=2.0)
    other = _diagonals(plain, level, _places(plain, level, 2.0, 2.0)[0])
    assert set(other.values()) == {0}
    assert not all(d == MIRRORED[i // 2] for (_, i), d in other.items())


@pytest.mark.parametrize(
    ("chordwise", "spanwise"), [(0.5, 0.5), (2.0, 1.0), (1.0, 2.0), (1.5, 0.7)]
)
def test_counts_round_each_direction_and_the_knots_stay(chordwise, spanwise):
    """round(f m) intervals per direction; TE, LE, ends are knots.

    Factor 0.5, each direction separately, and uneven factors. The trailing
    edge, the leading edge and the two end stations keep the source's nodes
    bit for bit. Control: the source counts are not the level's.
    """
    fx = tube_pole_quads()
    grid = _recovered(fx)
    level = refine_grid(fx.verts, fx.faces, grid, chordwise=chordwise, spanwise=spanwise)
    where, shape = _places(fx, level, chordwise, spanwise)
    s, t = _params(fx, chordwise, spanwise)
    assert shape == (len(t), len(s))
    assert level.report["intervals"]["spanwise"] == [8, round(8 * spanwise)]
    assert level.report["intervals"]["chordwise"][1] == len(s)
    node = {v: p for p, v in where.items()}
    source = {place: v for v, place in fx.place.items()}
    le_new = int(np.nonzero(s == fx.le)[0][0])
    for k_old, k_new in ((0, 0), (fx.stations - 1, len(t) - 1)):
        for i_new, i_old in enumerate(s):
            if i_old == int(i_old):
                assert np.array_equal(
                    level.points[node[(k_new, i_new)]], fx.verts[source[(k_old, int(i_old))]]
                )
    for k_new, k_old in enumerate(t):
        if k_old == int(k_old):
            for i_new, i_old in ((0, 0), (le_new, fx.le)):
                assert np.array_equal(
                    level.points[node[(k_new, i_new)]], fx.verts[source[(int(k_old), i_old)]]
                )
    assert (chordwise, spanwise) == (1.0, 1.0) or shape != (fx.stations, fx.nodes)


@pytest.mark.parametrize("direction", ["chordwise", "spanwise"])
def test_a_factor_below_one_over_m_is_refused(direction):
    """a factor below 1/m is refused naming family, direction, value.

    Exactly 1/m is accepted and leaves one interval (the control); zero,
    negative, infinite and not-a-number factors are refused the same way.
    The chordwise case runs on the sheet, since one interval around a tube
    leaves a section its caps cannot close (IND-03, refused on its own).
    """
    fx = tube_pole_quads() if direction == "spanwise" else sheet_quads()
    grid = _recovered(fx)
    m = dict(zip(("chordwise", "spanwise"), grid.intervals, strict=True))[direction]
    factors = {"chordwise": 1.0, "spanwise": 1.0}
    for bad in (0.99 / m, 0.0, -1.0, float("inf"), float("nan")):
        factors[direction] = bad
        with pytest.raises(InputArtifactError) as caught:
            check_factors(grid, "Wing", **factors)
        text = str(caught.value)
        named = {"chordwise": "normal", "spanwise": "parallel"}[direction]
        assert "family Wing" in text and f"{named} factor" in text
        # refine-r1 TW-F2: a factor below 1/m names the least; one not above zero, zero.
        assert (f"1/{m}" if 0 < bad < 1 else "greater than zero") in text
        assert text.endswith("Nothing was written.")
        with pytest.raises(InputArtifactError):
            refine_grid(fx.verts, fx.faces, grid, family="Wing", **factors)
    factors[direction] = 1.0 / m
    check_factors(grid, "Wing", **factors)
    level = refine_grid(fx.verts, fx.faces, grid, **factors)
    assert level.report["intervals"][direction][1] >= 1


@pytest.mark.parametrize("make", [tube_pole_quads, tube_zipper], ids=lambda m: m.__name__)
def test_the_end_faces_start_on_the_source_role(make):
    """pole fans and zipper caps start where the source's start.

    The source's fan triangles start at the pole and its cap faces start one
    vertex after the cap builder's first; the level at factor 2 keeps both.
    Control: the level's end faces rotated by one vertex fail.
    """
    fx = make()
    level = refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=2.0, spanwise=2.0)
    where, shape = _places(fx, level, 2.0, 2.0)
    pole_new = len(level.points) - 1 if fx.pole is not None else None
    h_new = int(np.nonzero(_params(fx, 2.0, 2.0)[0] == fx.le)[0][0])

    def first_roles(faces, place, pole, h, last):
        roles = set()
        for f in faces:
            if pole is not None and pole in f:
                roles.add(("fan", "P" if f[0] == pole else "ring"))
                continue
            spots = [place.get(v) for v in f]
            if None in spots or len({k for k, _ in spots}) != 1 or spots[0][0] not in (0, last):
                continue
            p = [i for _, i in spots]
            upper = sorted(x for x in p if 0 < x < h)
            role = "TE" if p[0] == 0 else "LE" if p[0] == h else "U" if p[0] < h else "L"
            if len(f) == 4 and role == "U":
                role = "U0" if p[0] == upper[0] else "U1"
            roles.add((spots[0][0] == 0, len(f), role))
        return roles

    source = first_roles(fx.faces, fx.place, fx.pole, fx.le, fx.stations - 1)
    got = first_roles(level.faces, where, pole_new, h_new, shape[0] - 1)
    assert source and got == source
    count = len(_expected_lateral(fx, where, shape, 2.0))
    rotated = level.faces[:count] + _rotate_faces(level.faces[count:])
    assert first_roles(rotated, where, pole_new, h_new, shape[0] - 1) != source


def test_a_source_not_in_one_sweep_takes_the_nearest_source_face_order():
    """faces follow their nearest source face, starting like it.

    The source's faces are shuffled, so they form no sweep. At factor 1 the
    level is still the source in coordinates and order; at factor 2 the
    positions of the nearest source faces never go back, and every face starts
    at the vertex nearest its source face's first vertex. Controls: the level
    reversed, and its faces rotated, fail.
    """
    fx = build_grid(
        Fixture("tube-shuffled", True, 9, NODES, ("open", "pole"), MIRRORED), shuffle_faces=True
    )
    grid = _recovered(fx)
    same = refine_grid(fx.verts, fx.faces, grid, chordwise=1.0, spanwise=1.0)
    assert same.report["order"] == "nearest"
    assert all(
        np.array_equal(same.points[f], fx.verts[g])
        for f, g in zip(same.faces, fx.faces, strict=True)
    )
    level = refine_grid(fx.verts, fx.faces, grid, chordwise=2.0, spanwise=2.0)
    centroids = np.array([fx.verts[g].mean(axis=0) for g in fx.faces])

    def nearest_order_holds(faces):
        mine = np.array([level.points[f].mean(axis=0) for f in faces])
        j = np.linalg.norm(mine[:, None] - centroids[None], axis=2).argmin(axis=1)
        if np.any(np.diff(j) < 0):
            return False
        for f, k in zip(faces, j, strict=True):
            first = fx.verts[fx.faces[k][0]]
            if int(np.argmin(np.linalg.norm(level.points[f] - first, axis=1))) != 0:
                return False
        return True

    assert nearest_order_holds(level.faces)
    assert not nearest_order_holds(level.faces[::-1])
    assert not nearest_order_holds(_rotate_faces(level.faces))


def test_recovery_reads_the_connectivity_and_names_why_it_fails():
    """a grid is recovered from connectivity, or the reason is named.

    Each fixture's numbering is shuffled and its layout and ends are still
    found. Planted bad inputs: a face removed, no trailing-edge vertex, and
    trailing-edge vertices that are no grid line each return their reason.
    """
    for make, layout, ends in (
        (tube_pole_triangles, "tube", ("open", "pole")),
        (tube_pole_quads, "tube", ("open", "pole")),
        (tube_zipper, "tube", ("zipper", "zipper")),
        (sheet_quads, "sheet", ("edge", "edge")),
    ):
        fx = make()
        grid = _recovered(fx)
        assert (grid.layout, grid.ends) == (layout, ends)
        assert grid.i_le == fx.le
        assert grid.describe().startswith(f"a {layout} of {fx.stations} stations")
    fx = tube_pole_quads()
    holed = recover_grid(fx.verts, fx.faces[:40] + fx.faces[41:], fx.te)
    assert isinstance(holed, str) and holed
    assert recover_grid(fx.verts, fx.faces, set()) == "no trailing-edge vertex on the end loop"
    sheet = sheet_quads()
    stray = set(list(sheet.place)[:5])
    assert recover_grid(sheet.verts, sheet.faces, stray) == "the trailing edge is not a grid line"


def test_an_incomplete_trailing_edge_on_a_sheet_boundary_still_orients_the_grid():
    """Two trailing-edge vertices missing from the points: the sheet is still recovered.

    Every listed vertex lies on one boundary line and covers most of it, so that line
    is the trailing edge, and the grid is the one the complete list gives.
    """
    sheet = sheet_quads()
    whole = recover_grid(sheet.verts, sheet.faces, sheet.te)
    ordered = sorted(sheet.te)
    partial = set(ordered) - {ordered[len(ordered) // 2], ordered[len(ordered) // 3]}
    found = recover_grid(sheet.verts, sheet.faces, partial)
    assert not isinstance(found, str), found
    assert np.array_equal(found.ids, whole.ids)


def test_a_sheet_with_no_trailing_edge_has_four_boundary_sides_one_of_them_a_grid_line():
    """The sides the level tries when a sheet found no trailing-edge point.

    The boundary of a sheet splits at its four corners into four sides; the sides
    on the trailing edge's index line recover the grid that the listed points give.
    """
    from cfdbender.mesh._refine._lines import (
        boundary_sides as _boundary_sides,
    )
    from cfdbender.mesh._refine._lines import (
        side_length as _side_length,
    )

    sheet = sheet_quads()
    sides = _boundary_sides(sheet.verts, sheet.faces)
    assert len(sides) == 4
    assert any(set(s) == set(sheet.te) for s in sides)
    longest = max(sides, key=lambda s: _side_length(sheet.verts, s))
    assert not isinstance(recover_grid(sheet.verts, sheet.faces, longest), str)


def test_the_spline_reproduces_a_cubic_and_keeps_its_knots():
    """the index-space spline is exact on a cubic and at the knots.

    A not-a-knot spline reproduces any cubic. Control: a linear interpolation
    through the same nodes misses it.
    """
    m = 12
    x = np.arange(m + 1, dtype=float)
    cubic = np.stack([0.3 * x**3 - 2 * x**2 + x - 4, -0.01 * x**3 + 5.0], axis=1)
    p = np.linspace(0, m, 97)
    want = np.stack([0.3 * p**3 - 2 * p**2 + p - 4, -0.01 * p**3 + 5.0], axis=1)
    got = not_a_knot(cubic, p)
    assert float(np.abs(got - want).max()) < 1e-9
    knots = not_a_knot(cubic, x)
    assert np.array_equal(knots, cubic)
    noisy = np.random.default_rng(1).normal(size=(m + 1, 3))
    assert np.array_equal(not_a_knot(noisy, x), noisy)
    natural = np.interp(p, x, cubic[:, 0])
    assert float(np.abs(natural - want[:, 0]).max()) > 1e-3


def test_the_spline_equals_scipys_not_a_knot_where_scipy_is_installed():
    """the numpy spline is scipy's not-a-knot to 1e-12.

    Needs scipy for its control; the module itself
    never imports it. Control: scipy's natural spline differs.
    """
    from scipy import interpolate

    rng = np.random.default_rng(7)
    for m in (3, 4, 5, 8, 33, 200):
        y = rng.normal(size=(m + 1, 2, 3)) * 10
        p = np.sort(np.concatenate([rng.uniform(0, m, 300), np.arange(m + 1)]))
        spline = interpolate.CubicSpline(np.arange(m + 1), y, axis=0, bc_type="not-a-knot")
        assert float(np.abs(not_a_knot(y, p) - spline(p)).max()) < 1e-12
        natural = interpolate.CubicSpline(np.arange(m + 1), y, axis=0, bc_type="natural")
        assert float(np.abs(not_a_knot(y, p) - natural(p)).max()) > 1e-6


def test_f326_a_short_zipper_tube_is_recovered():
    """F-326: a zipper tube of few cells, mostly triangles by count, is still a grid."""
    for stations, nodes in ((2, 6), (3, 6), (2, 8)):
        fx = build_grid(Fixture("short", True, stations, nodes, ("zipper", "zipper"), None))
        grid = _recovered(fx)
        assert grid.ids.shape == (stations, nodes)
        assert grid.ends == ("zipper", "zipper")


def test_f328_a_twisted_grid_keeps_its_trailing_edge_column():
    """F-328: the alignment that keeps the trailing edge a column beats the shortest."""
    stations, nodes = 4, 8
    verts = np.array(
        [
            [
                np.cos(2 * np.pi * (i - 0.9 * k) / nodes),
                np.sin(2 * np.pi * (i - 0.9 * k) / nodes),
                1.0 + 0.5 * k,
            ]
            for k in range(stations)
            for i in range(nodes)
        ]
    )

    def gid(k, i):
        return k * nodes + i % nodes

    faces = []
    for k in range(stations - 1):
        for i in range(nodes):
            faces.append([gid(k, i), gid(k, i + 1), gid(k + 1, i + 1)])
            faces.append([gid(k, i), gid(k + 1, i + 1), gid(k + 1, i)])
    grid = recover_grid(verts, faces, {gid(k, 0) for k in range(stations)})
    assert isinstance(grid, Grid), grid
    assert [int(v) for v in grid.ids[:, 0]] == [gid(k, 0) for k in range(stations)]


def test_f329_a_sweep_starting_inside_the_ring_is_read_across_the_seam():
    """F-329: the sweep's direction does not rest on cells zero and one being adjacent."""
    fx = tube_pole_quads()
    grid = _recovered(fx)
    k, n = grid.ids.shape
    for start, step in ((1, 1), (n - 2, -1), (5, 1)):
        cols = [(start + step * t) % n for t in range(n)]
        faces = []
        for row in range(k - 1):
            for i in cols:
                j = (i + 1) % n
                ids = grid.ids
                faces.append(
                    [int(ids[row, i]), int(ids[row, j]), int(ids[row + 1, j]), int(ids[row + 1, i])]
                )
        level = refine_grid(fx.verts, faces, grid, chordwise=2.0, spanwise=1.0)
        assert level.report["order"] == "sweep", (start, step)
        where, shape = _places(fx, level, 2.0, 1.0)
        nc = shape[1]
        seen = [where[f[0]][1] for f in level.faces[:nc]]
        first = round(start * (nc - 1) / (n - 1))
        assert seen == [(first + step * t) % nc for t in range(nc)], (start, step)
        assert where[level.faces[0][0]][0] == 0, (start, step)
