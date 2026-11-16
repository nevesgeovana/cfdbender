"""The wheel: the arithmetic of a full wheel built from one sector, and the synthetic sector.

Pipeline role: tier-1 oracle, on a sector built here in
numpy: a spinner without a base, a blade whose trailing edge is blunt (a base of one quad row
between two rims) or sharp, and a nacelle whose aft disc is its base, all in a sector that
spans -180/N to +180/N degrees about an axis. The checks do not use the code under test: the
area, the turn and the open edges are computed here from the sector's own faces.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from cfdbender.mesh._wheel import (
    merge_nodes,
    neighbour_pairs,
    parse_axis,
    rotate,
    sector_to_wheel,
)

STEM = "sector"


@dataclass
class Built:
    """A synthetic sector on disk and what the tests compare the wheel with."""

    obj: Path
    copies: int
    axis: str
    hub: tuple[float, float, float]
    verts: np.ndarray
    families: dict[str, list[list[int]]]
    seam: set[int] = field(default_factory=set)
    base_faces: int = 3
    notes: dict[str, int] = field(default_factory=dict)
    rims: dict[str, list[tuple[int, int]]] = field(default_factory=dict)


def _fan_area(verts: np.ndarray, polys: list[list[int]]) -> float:
    total = 0.0
    for poly in polys:
        for k in range(1, len(poly) - 1):
            u, v = verts[poly[k]] - verts[poly[0]], verts[poly[k + 1]] - verts[poly[0]]
            total += float(np.linalg.norm(np.cross(u, v)) / 2.0)
    return total


def _write_points(path: Path, points: list[np.ndarray]) -> None:
    rows = ["METER", *[",".join(repr(float(c)) for c in p) for p in points]]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_sector(
    folder: Path,
    *,
    copies: int = 6,
    blunt: bool = True,
    offset: float = 0.0,
    separate_base: bool = False,
    permuted: bool = False,
    hub: tuple[float, float, float] = (0.0, 0.0, 0.0),
    declare: bool = True,
) -> Built:
    """Write a sector OBJ with its boundaries file and rim points; return it."""
    half = math.pi / copies
    cells = 4
    verts: list[tuple[float, float, float]] = []
    seam: set[int] = set()

    def node(x: float, r: float, theta: float, *, at_seam: bool = False) -> int:
        shift = offset if at_seam and theta > 0 else 0.0
        verts.append((x + shift, r * math.cos(theta), r * math.sin(theta)))
        if at_seam or r == 0.0:
            seam.add(len(verts) - 1)
        return len(verts) - 1

    def grid(xs: list[float], radii: list[float]) -> list[list[int]]:
        thetas = [-half + 2 * half * j / cells for j in range(cells + 1)]
        return [
            [node(x, r, t, at_seam=j in (0, cells)) for j, t in enumerate(thetas)]
            for x, r in zip(xs, radii, strict=True)
        ]

    spinner = grid([0.3 * i for i in range(4)], [0.1 + 0.05 * i for i in range(4)])
    s_faces: list[list[int]] = []
    for i in range(3):
        for j in range(cells):
            a, b, c, d = spinner[i][j], spinner[i + 1][j], spinner[i + 1][j + 1], spinner[i][j + 1]
            s_faces += [[a, b, c], [a, c, d]]
    nac = grid([1.2 + 0.8 * k / 3 for k in range(4)], [0.3] * 4)
    n_faces = [
        [nac[k][j], nac[k + 1][j], nac[k + 1][j + 1], nac[k][j + 1]]
        for k in range(3)
        for j in range(cells)
    ]
    centre = node(2.0, 0.0, 0.0)
    disc = [[centre, nac[3][j + 1], nac[3][j]] for j in range(cells)]
    n_rim = [(nac[3][j], nac[3][j + 1]) for j in range(cells)]

    spans, chords = 3, 3
    radii = [0.35 + 0.1 * j for j in range(spans + 1)]
    le = [node(0.4, r, 0.0) for r in radii]
    up: dict[int, list[int]] = {}
    lo: dict[int, list[int]] = {}
    for i in range(1, chords + 1):
        half_thick = 0.02 if blunt else 0.02 * (chords - i) / (chords - 1)
        if not blunt and i == chords:
            up[i] = lo[i] = [node(0.4 + 0.1 * i, r, 0.0) for r in radii]
            continue
        up[i] = [_blade_node(verts, 0.4 + 0.1 * i, r, +half_thick) for r in radii]
        lo[i] = [_blade_node(verts, 0.4 + 0.1 * i, r, -half_thick) for r in radii]
    top = {0: le, **up}
    bottom = {0: le, **lo}
    blade: list[list[int]] = []
    for i in range(chords):
        for j in range(spans):
            blade.append([top[i][j], top[i + 1][j], top[i + 1][j + 1], top[i][j + 1]])
            blade.append([bottom[i][j], bottom[i][j + 1], bottom[i + 1][j + 1], bottom[i + 1][j]])
    rims: dict[str, list[tuple[int, int]]]
    base = [
        [up[chords][j], up[chords][j + 1], lo[chords][j + 1], lo[chords][j]] for j in range(spans)
    ]
    if blunt:
        rims = {
            "TE_a": [(up[chords][j], up[chords][j + 1]) for j in range(spans)],
            "TE_b": [(lo[chords][j], lo[chords][j + 1]) for j in range(spans)],
        }
    else:
        rims = {"TE": [(up[chords][j], up[chords][j + 1]) for j in range(spans)]}
    rims["N"] = n_rim

    families: dict[str, list[list[int]]] = {}
    if separate_base:
        families = {"Blade1": blade + ([] if not blunt else []), "S": s_faces, "N": n_faces}
        families = {"Blade1": blade, "Base1": base if blunt else [], "S": s_faces, "N": n_faces}
        families["NBase"] = disc
        if not blunt:
            del families["Base1"]
    else:
        families = {"Blade1": blade + (base if blunt else []), "S": s_faces, "N": n_faces + disc}

    array = np.array(verts)
    axis = "x"
    if permuted:
        array = array[:, [1, 2, 0]]
        axis = "z"
    array = array + np.array(hub)
    _emit(folder, array, families, rims, separate_base, blunt, declare)
    return Built(
        folder / f"{STEM}.obj",
        copies,
        axis,
        hub,
        array,
        families,
        seam,
        spans,
        {"disc": len(disc), "spans": spans},
        rims,
    )


def _blade_node(verts: list[tuple[float, float, float]], x: float, r: float, z: float) -> int:
    verts.append((x, r, z))
    return len(verts) - 1


def _emit(
    folder: Path,
    array: np.ndarray,
    families: dict[str, list[list[int]]],
    rims: dict[str, list[tuple[int, int]]],
    separate_base: bool,
    blunt: bool,
    declare: bool,
) -> None:
    lines = ["# synthetic sector"]
    lines += ["v " + " ".join(repr(float(c)) for c in p) for p in array]
    for name, polys in families.items():
        lines.append(f"g {name}")
        lines += ["f " + " ".join(str(i + 1) for i in poly) for poly in polys]
    (folder / f"{STEM}.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    side = [
        "boundaries = [" + ", ".join(f'"{n}"' for n in families) + "]",
        "[import]",
        'units = "METER"',
    ]
    for label, edges in rims.items():
        _write_points(
            folder / f"{STEM}.{label}.te.txt", [(array[a] + array[b]) / 2 for a, b in edges]
        )
        side += [
            f"[trailing_edges.{label}]",
            f'file = "{STEM}.{label}.te.txt"',
            'type = "STANDARD"',
        ]
    if declare and separate_base:
        side += ["[base_regions.NBase]", 'trailing_edges = "auto"', 'merge_into = "N"']
        if blunt:
            side += ["[base_regions.Base1]", 'trailing_edges = "auto"', 'merge_into = "Blade1"']
    elif declare:
        side += ["[base_regions.N]", 'trailing_edges = "auto"']
        if blunt:
            side += ["[base_regions.Blade1]", 'trailing_edges = "auto"']
    if not declare:
        side = side[:2]
    (folder / f"{STEM}.obj.boundaries.toml").write_text("\n".join(side) + "\n", encoding="utf-8")


def read_wheel(path: Path) -> tuple[np.ndarray, dict[str, list[list[int]]]]:
    """Read an OBJ with a parser of its own: vertices and the faces of each group."""
    verts: list[list[float]] = []
    groups: dict[str, list[list[int]]] = {}
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.split()
        if not word or word[0].startswith("#"):
            continue
        if word[0] == "v":
            verts.append([float(w) for w in word[1:4]])
        elif word[0] in ("g", "o"):
            current = word[1]
            groups[current] = []
        elif word[0] == "f":
            groups[current].append([int(w.split("/")[0]) - 1 for w in word[1:]])
    return np.array(verts), groups


def edge_stats(polys: list[list[int]]) -> Counter[tuple[int, int]]:
    count: Counter[tuple[int, int]] = Counter()
    for poly in polys:
        for a, b in zip(poly, poly[1:] + poly[:1], strict=True):
            count[(min(a, b), max(a, b))] += 1
    return count


def all_faces(groups: dict[str, list[list[int]]]) -> list[list[int]]:
    return [f for polys in groups.values() for f in polys]


def turn(points: np.ndarray, axis: np.ndarray, hub: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate right-handed about an axis through a point (Rodrigues, written out here)."""
    t = math.radians(degrees)
    q = points - hub
    k = axis / np.linalg.norm(axis)
    return (
        hub
        + q * math.cos(t)
        + np.cross(k, q) * math.sin(t)
        + np.outer(q @ k, k) * (1 - math.cos(t))
    )


def maps_onto(
    wheel: np.ndarray, polys: list[list[int]], moved: np.ndarray, tol: float
) -> tuple[int, int]:
    """Return how many turned nodes land on a node and how many turned faces on a face."""
    distance = np.linalg.norm(moved[:, None, :] - wheel[None, :, :], axis=2)
    image = distance.argmin(axis=1)
    hit = distance.min(axis=1) <= tol
    known = {tuple(sorted(f)) for f in polys}
    faces = sum(
        1
        for f in polys
        if all(hit[i] for i in f) and tuple(sorted(int(image[i]) for i in f)) in known
    )
    return int(hit.sum()), faces


AXIS = {"x": np.array([1.0, 0, 0]), "z": np.array([0.0, 0, 1])}


@pytest.mark.parametrize(("copies", "blunt"), [(6, True), (6, False), (4, True), (5, False)])
def test_mesh_wheel_the_wheel_area_is_n_times_the_sector_area(tmp_path, copies, blunt):
    sector = build_sector(tmp_path, copies=copies, blunt=blunt)
    result = sector_to_wheel(sector.obj, copies=copies, blades=["Blade1"], axis="x")
    wheel, groups = read_wheel(result.output)
    area = sum(_fan_area(wheel, polys) for polys in groups.values())
    expected = copies * sum(_fan_area(sector.verts, p) for p in sector.families.values())
    assert area == pytest.approx(expected, rel=1e-12)
    assert result.report["area_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert sum(len(p) for p in groups.values()) == copies * sum(
        len(p) for p in sector.families.values()
    )


@pytest.mark.parametrize(
    ("copies", "permuted", "hub"), [(6, False, (0, 0, 0)), (4, True, (0.5, -1.0, 2.0))]
)
def test_mesh_wheel_a_turn_of_one_copy_maps_the_wheel_and_a_derangement_does_not(
    tmp_path, copies, permuted, hub
):
    sector = build_sector(tmp_path, copies=copies, permuted=permuted, hub=hub)
    result = sector_to_wheel(
        sector.obj, copies=copies, blades=["Blade1"], axis=sector.axis, center=hub
    )
    wheel, groups = read_wheel(result.output)
    polys = all_faces(groups)
    axis, centre = AXIS[sector.axis], np.array(hub)
    nodes, faces = maps_onto(wheel, polys, turn(wheel, axis, centre, 360.0 / copies), 1e-9)
    assert (nodes, faces) == (len(wheel), len(polys))
    nodes, faces = maps_onto(wheel, polys, turn(wheel, axis, centre, 360.0 / copies + 1.0), 1e-9)
    assert nodes < len(wheel) and faces < len(polys)
    assert result.report["invariance"]["faces_mapped"] == len(polys)
    assert result.report["control"]["faces_mapped"] < len(polys)


@pytest.mark.parametrize("copies", [6, 4])
def test_mesh_wheel_open_edges_are_the_sector_s_less_its_seams_and_nothing_is_degenerate(
    tmp_path, copies
):
    sector = build_sector(tmp_path, copies=copies)
    result = sector_to_wheel(sector.obj, copies=copies, blades=["Blade1"], axis="x")
    count = edge_stats(all_faces(sector.families))
    opened = [e for e, c in count.items() if c == 1]
    seam = [e for e in opened if e[0] in sector.seam and e[1] in sector.seam]
    wheel, groups = read_wheel(result.output)
    merged = edge_stats(all_faces(groups))
    report = result.report
    assert report["open_edges"]["sector"] == len(opened)
    assert report["open_edges"]["unmerged"] == copies * len(opened) > report["open_edges"]["merged"]
    assert report["open_edges"]["merged"] == copies * (len(opened) - len(seam))
    assert sum(1 for c in merged.values() if c == 1) == copies * (len(opened) - len(seam))
    assert not any(c > 2 for c in merged.values()) and report["non_manifold_edges"]["merged"] == 0
    assert report["degenerate_faces"] == 0
    assert all(len(set(f)) == len(f) for f in all_faces(groups))


def test_mesh_wheel_a_seam_offset_beyond_1e5_and_within_1e4_is_merged_only_by_the_default(tmp_path):
    sector = build_sector(tmp_path, offset=5e-5)
    tight = sector_to_wheel(
        sector.obj, copies=6, blades=["Blade1"], tolerance=1e-5, out=tmp_path / "tight.obj"
    )
    loose = sector_to_wheel(sector.obj, copies=6, blades=["Blade1"], out=tmp_path / "loose.obj")
    assert tight.report["nodes"]["merged"] < loose.report["nodes"]["merged"]
    assert tight.report["open_edges"]["merged"] > loose.report["open_edges"]["merged"]
    count = edge_stats(all_faces(sector.families))
    opened = [e for e, c in count.items() if c == 1]
    seam = [e for e in opened if e[0] in sector.seam and e[1] in sector.seam]
    assert loose.report["open_edges"]["merged"] == 6 * (len(opened) - len(seam))
    assert loose.report["degenerate_faces"] == 0
    assert loose.report["area_ratio"] == pytest.approx(1.0, abs=2e-3)


def test_mesh_wheel_the_grid_search_finds_the_pairs_a_brute_force_finds():
    rng = np.random.default_rng(7)
    points = rng.random((400, 3))
    points[200:] = points[:200] + rng.normal(scale=2e-3, size=(200, 3))
    i, j, d = neighbour_pairs(points, points, 0.01)
    found = {(a, b) for a, b in zip(i.tolist(), j.tolist(), strict=True)}
    gap = np.linalg.norm(points[:, None] - points[None], axis=2)
    expected = {(a, b) for a, b in zip(*np.nonzero(gap <= 0.01), strict=True)}
    assert found == {(int(a), int(b)) for a, b in expected}
    roots = merge_nodes(points, 0.01)
    assert (roots <= np.arange(len(points))).all()
    assert len(set(roots.tolist())) < len(points)


def test_mesh_wheel_axis_and_rotation_follow_the_right_hand_rule():
    assert parse_axis("-z").tolist() == [0.0, 0.0, -1.0]
    assert parse_axis("0, 3, 4").tolist() == pytest.approx([0.0, 0.6, 0.8])
    with pytest.raises(Exception, match="not a direction"):
        parse_axis("0,0,0")
    moved = rotate(np.array([[0.0, 1.0, 0.0]]), parse_axis("x"), np.zeros(3), 90.0)
    assert moved[0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-15)
    about = rotate(np.array([[2.0, 1.0, 0.0]]), parse_axis("z"), np.array([1.0, 0, 0]), 90.0)
    assert about[0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-15)


def test_mesh_wheel_merging_a_hundred_thousand_faces_of_copies_takes_seconds():
    import time

    n = 600_000
    rng = np.random.default_rng(1)
    points = rng.random((n, 3))
    points[n // 2 :] = points[: n // 2]
    started = time.perf_counter()
    roots = merge_nodes(points, 1e-6)
    assert time.perf_counter() - started < 30.0
    assert len(set(roots.tolist())) == n // 2


def test_mesh_wheel_a_tolerance_that_reaches_too_many_pairs_is_refused_before_it_allocates():
    from cfdbender._errors import InputValueError

    points = np.random.default_rng(3).random((3000, 3)) * 1e-3
    with pytest.raises(InputValueError, match="--tolerance"):
        neighbour_pairs(points, points, 1.0)
    with pytest.raises(InputValueError, match="--tolerance"):
        merge_nodes(points, 1.0)


def test_mesh_wheel_generated_paths_are_role_qualified_and_case_safe():
    """Equal or case-only equal names never replace one another's point file."""
    from cfdbender.mesh._wheel import reserve_name

    taken: set[str] = set()
    stem = "wheel"
    first = reserve_name(taken, stem, "Rim.te.txt", "trailing_edge")
    second = reserve_name(taken, stem, "Rim.te.txt", "base")
    third = reserve_name(taken, stem, "RIM.te.txt", "base")
    assert first == "wheel.Rim.te.txt"
    assert second == "wheel.base.Rim.te.txt"
    assert len({first.casefold(), second.casefold(), third.casefold()}) == 3
