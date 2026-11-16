"""Block topology, counts, carry rows and poles of the full-grid strategy (strategy 2).

Pipeline role: the pure planning layer of strategy 2. The workspace mesher reads the faces of a
configuration from the kernel and hands this module only numbers: the signed edge walk of every
face with the turn at each vertex, the edge lengths, and the edges that meet on a junction. This
module then chooses the four corners of every face (a face with more than four edges becomes a
block with composite sides, which is how a junction curve replaces one side of a block, delta
F.2), solves one integer cell count per edge so that opposite sides of every block are equal
(E09), the junction ring holds its count on every side (E08) and the cell width changes by at
most ``max_growth`` from one edge to the next along a carry row (E10), splits a half-body
azimuth count over the body's sectors, and checks that every irregular node is a declared
valence 3 pole (E11) and that no cell is a triangle (E07).

The error codes and their wording are those of :data:`cfdbender.geometry._mesh_schema.ERRORS`,
so an input refused before meshing and a realized defect read the same.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cfdbender._errors import InputValueError

from ._mesh_schema import ERRORS

__all__ = [
    "TOPOLOGIES",
    "Block",
    "azimuth_split",
    "block_from_walk",
    "carry_pairs_growth",
    "carry_plan",
    "check_poles",
    "check_triangles",
    "component_topology",
    "irregular_nodes",
    "node_valence",
    "ring_count",
    "solve_counts",
]

#: The block topologies of MESH_STRATEGY section 2.5: O around a body axis, C around a leading
#: edge, H for flat pieces, a collar ring around a root, a cap closing a tip.
TOPOLOGIES = ("O", "C", "H", "collar", "cap")

#: A vertex where the walk turns back by more than this is a cusp (degrees).
_CUSP_DEG = 170.0

#: The status ``scipy.optimize.milp`` returns for a problem it calls infeasible.
_MILP_INFEASIBLE = 2

#: The topology each component type takes when its ``blocks`` table does not name one.
_DEFAULT_TOPOLOGY = {
    "body": {"topology": "O", "i": "axial", "j": "azimuth"},
    "lifting_surface": {"topology": "C", "i": "chord", "j": "span"},
    "wing_fuselage_fairing": {"topology": "collar", "i": "around", "j": "across"},
    "fillet": {"topology": "collar", "i": "around", "j": "across"},
    "tip_cap": {"topology": "cap", "i": "chord", "j": "across"},
    "pylon": {"topology": "H", "i": "chord", "j": "span"},
}


def _refuse(code: str, where: str) -> InputValueError:
    return InputValueError(f"{code}: {ERRORS[code]}: {where}")


def component_topology(name: str, kind: str, table: Mapping[str, Any] | None = None) -> dict:
    """Return the block topology of one component: its kind, index directions and poles.

    Parameters
    ----------
    name : str
        The component, for the messages.
    kind : str
        Its type (``body``, ``lifting_surface``, ``wing_fuselage_fairing``, ...).
    table : mapping, optional
        Its ``blocks`` table of the mesh controls; keys it gives override the defaults.

    Returns
    -------
    dict
        ``topology``, ``i``, ``j`` and ``poles`` (a body closes both ends with valence 3 poles
        unless its table says ``open``).

    Raises
    ------
    InputValueError
        For a topology outside :data:`TOPOLOGIES`, or a pole that is not ``valence3`` or
        ``open`` (E11).
    """
    given = dict(table or {})
    result: dict[str, Any] = dict(
        _DEFAULT_TOPOLOGY.get(kind, {"topology": "H", "i": "u", "j": "v"})
    )
    result.update({k: given[k] for k in ("topology", "i", "j") if k in given})
    if result["topology"] not in TOPOLOGIES:
        raise InputValueError(
            f"{name}.blocks.topology: {result['topology']!r} is not one of {list(TOPOLOGIES)}"
        )
    poles = {"nose": "valence3", "base": "valence3"} if result["topology"] == "O" else {}
    poles.update(dict(given.get("poles", {})))
    for end, value in poles.items():
        if value not in ("valence3", "open"):
            raise _refuse("E11", f"{name} {end}: {value} pole refused")
    result["poles"] = poles
    return result


@dataclass(frozen=True)
class Block:
    """One face as a block: its name, component and four sides of edge tags in walk order.

    A side is a tuple of edge tags (unsigned); a side of more than one edge is a composite side,
    which is how a junction curve cut into pieces stays one side of the block.
    """

    name: str
    component: str
    sides: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    corners: tuple[int, int, int, int]


def block_from_walk(
    name: str,
    component: str,
    walk: Sequence[int],
    turns: Sequence[float],
    *,
    cusps: bool = True,
) -> Block | None:
    """Choose the four corners of a face from the turns of its walk; ``None`` if it has none.

    ``walk`` is the closed signed edge walk and ``turns[k]`` the turn in degrees at the vertex
    where edge ``k`` starts. The four sharpest turns that are not cusps are the corners, so a
    face of five or more edges becomes a block with composite sides. A face of fewer than four
    edges, or with fewer than four corners, is not a block: strategy 2 reports it by name as
    not full grid (delta F.2) instead of refusing. A cusp (the walk turning back) is a corner
    only when the face has too few other corners, and never with ``cusps=False``.

    Parameters
    ----------
    name : str
        The block name.
    component : str
        The component the face belongs to.
    walk : sequence of int
        The closed signed edge walk of the face.
    turns : sequence of float
        The turn in degrees at the start vertex of each edge of ``walk``.
    cusps : bool, optional
        Whether a cusp may serve as a corner when the face has too few other corners.

    Returns
    -------
    Block or None
        The block with its four sides and corner indices, or ``None`` when the face is not one.
    """
    sharp = sum(t < _CUSP_DEG for t in turns)
    if len(walk) < 4 or (not cusps and sharp < 4):
        return None
    rank = sorted(range(len(walk)), key=lambda k: (turns[k] >= _CUSP_DEG, -turns[k], k))
    chosen = sorted(rank[:4])
    sides = []
    for q in range(4):
        start, stop = chosen[q], chosen[(q + 1) % 4]
        span = range(start, stop) if stop > start else [*range(start, len(walk)), *range(stop)]
        sides.append(tuple(abs(int(walk[k])) for k in span))
    return Block(name, component, (sides[0], sides[1], sides[2], sides[3]), tuple(chosen))  # type: ignore[arg-type]


def ring_count(chord_cells: int, te_cells: int) -> int:
    """Return the node count around a lifting root ring: both sides plus the base (2.5).

    Parameters
    ----------
    chord_cells : int
        The cells along the chord on each side.
    te_cells : int
        The cells across the trailing edge base.

    Returns
    -------
    int
        Twice ``chord_cells`` plus ``te_cells``.
    """
    return 2 * int(chord_cells) + int(te_cells)


def _index(blocks: Sequence[Block], lengths: Mapping[int, float]) -> dict[int, int]:
    edges = sorted({e for b in blocks for side in b.sides for e in side} | set(lengths))
    return {e: i for i, e in enumerate(edges)}


def _row(index: Mapping[int, int], plus: Iterable[int], minus: Iterable[int] = ()) -> np.ndarray:
    row = np.zeros(len(index))
    for e in plus:
        row[index[e]] += 1
    for e in minus:
        row[index[e]] -= 1
    return row


def _direct_conflicts(blocks: Sequence[Block], fixed: Mapping[int, int]) -> None:
    """Name the first block whose opposite single-edge sides carry two different fixed counts."""
    for block in blocks:
        for a, b in ((0, 2), (1, 3)):
            sa, sb = block.sides[a], block.sides[b]
            if all(e in fixed for e in sa + sb):
                na, nb = sum(fixed[e] for e in sa), sum(fixed[e] for e in sb)
                if na != nb:
                    raise _refuse("E09", f"{block.name} sides {a} and {b}: {na} and {nb}")


@dataclass(frozen=True)
class _Problem:
    """The linear rows of a count problem: equalities, inequalities and the targets."""

    index: dict[int, int]
    equal: list[np.ndarray]
    equal_rhs: list[float]
    upper: list[np.ndarray]
    target: np.ndarray
    even: list[np.ndarray]


def _carry_rows(index, lengths, carry, max_growth) -> list[np.ndarray]:
    """``L_a n_b - g L_b n_a <= 0`` both ways: the width ratio of a carry pair stays under g."""
    rows = []
    for a, b in carry:
        for first, second in ((a, b), (b, a)):
            row = np.zeros(len(index))
            row[index[second]] += lengths[first]
            row[index[first]] -= max_growth * lengths[second]
            rows.append(row)
    return rows


def _problem(blocks, lengths, target_m, *, fixed, totals, carry, max_growth, even) -> _Problem:
    index = _index(blocks, lengths)
    equal, rhs = [], []
    for block in blocks:
        for a, b in ((0, 2), (1, 3)):
            equal.append(_row(index, block.sides[a], block.sides[b]))
            rhs.append(0.0)
    for e, n in fixed.items():
        equal.append(_row(index, [e]))
        rhs.append(float(n))
    for edges, n in totals:
        equal.append(_row(index, edges))
        rhs.append(float(n))
    upper = _carry_rows(index, lengths, carry, max_growth) if carry else []
    size = np.array([target_m.get(e, 0.0) for e in index])
    length = np.array([lengths[e] for e in index])
    target = np.maximum(1.0, length / np.where(size > 0, size, np.inf))
    target[size <= 0] = 1.0
    loops = [_row(index, edges) for edges in even]
    return _Problem(index, equal, rhs, upper, target, loops)


def _padded(rows: list[np.ndarray], width: int) -> np.ndarray:
    return np.hstack([np.array(rows), np.zeros((len(rows), width - len(rows[0])))])


def _solve(problem: _Problem) -> np.ndarray | None:
    """Minimize the relative deviation from the targets; variables are n, |n - t| and halves.

    Each loop of ``even`` (the boundary of a face meshed unstructured) holds an even node count,
    so the recombination can close it with quads only: its sum equals twice an integer.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    n, m = len(problem.index), len(problem.even)
    width = 2 * n + m
    eye = np.eye(n)
    constraints = [
        LinearConstraint(_padded([*np.hstack([eye, -eye])], width), -np.inf, problem.target),
        LinearConstraint(_padded([*np.hstack([-eye, -eye])], width), -np.inf, -problem.target),
    ]
    if problem.equal:
        rows = _padded(problem.equal, width)
        constraints.append(LinearConstraint(rows, problem.equal_rhs, problem.equal_rhs))
    if problem.upper:
        constraints.append(LinearConstraint(_padded(problem.upper, width), -np.inf, 0.0))
    if m:
        rows = np.hstack([np.array(problem.even), np.zeros((m, n)), -2.0 * np.eye(m)])
        constraints.append(LinearConstraint(rows, 0.0, 0.0))
    cost = np.r_[np.zeros(n), 1.0 / problem.target, np.zeros(m)]
    low = np.r_[np.ones(n), np.zeros(n), np.full(m, 2.0)]  # a loop of four nodes is one quad
    high = np.r_[np.full(n, 1e5), np.full(n, 1e7), np.full(m, 1e5)]
    integer = np.r_[np.ones(n), np.zeros(n), np.ones(m)]
    posed = {"constraints": constraints, "integrality": integer, "bounds": Bounds(low, high)}
    result = milp(cost, **posed)
    if result.status == _MILP_INFEASIBLE:
        # The presolve of the HiGHS in scipy 1.15, the declared floor, calls the WB ring rows
        # infeasible; the same rows reach the same optimum with the presolve off (1.15 and 1.18)
        # and with it on (1.18). An infeasible verdict is taken only from the solve without it.
        result = milp(cost, **posed, options={"presolve": False})
    return None if result.x is None else np.round(result.x[:n]).astype(int)


def solve_counts(
    blocks: Sequence[Block],
    lengths: Mapping[int, float],
    target_m: Mapping[int, float],
    *,
    fixed: Mapping[int, int] | None = None,
    totals: Sequence[tuple[Sequence[int], int]] = (),
    carry: Sequence[tuple[int, int]] = (),
    max_growth: float = 1.2,
    even: Sequence[Sequence[int]] = (),
) -> dict[int, int]:
    """Solve one cell count per edge for the whole block graph.

    Opposite sides of every block hold equal counts (a composite side counts the sum of its
    edges); ``fixed`` pins edges; each ``totals`` row pins the sum over a set of edges (a junction
    ring, a body azimuth ring, a lifting span line); each ``carry`` pair ``(a, b)`` of edges that
    continue one grid line across a junction keeps the ratio of their cell widths at most
    ``max_growth``. Among the counts that satisfy all of that, the one closest to the target
    cell size ``target_m`` per edge (relative deviation summed) is returned. Each loop of
    ``even`` holds an even total, so a face that is not a block closes with quads only.

    Parameters
    ----------
    blocks : sequence of Block
        The blocks of the graph.
    lengths : mapping of int to float
        The length of each edge.
    target_m : mapping of int to float
        The target cell size of each edge.
    fixed : mapping of int to int, optional
        Edges pinned to a count.
    totals : sequence of tuple, optional
        Rows of ``(edges, total)`` pinning the sum of counts over a set of edges.
    carry : sequence of tuple of int, optional
        Pairs of edges that continue one grid line across a junction.
    max_growth : float, optional
        The largest width ratio kept over a carry pair.
    even : sequence of sequence of int, optional
        Loops of edges whose total count is even.

    Returns
    -------
    dict of int to int
        The cell count of each edge.

    Raises
    ------
    InputValueError
        E09 when two fixed counts contradict on opposite sides of a block, E08 when a ring total
        cannot be held on every side, E10 when the carry pairs cannot keep the growth limit.
    """
    fixed = dict(fixed or {})
    _direct_conflicts(blocks, fixed)

    def attempt(rows, pairs):
        problem = _problem(
            blocks,
            lengths,
            target_m,
            fixed=fixed,
            totals=rows,
            carry=pairs,
            max_growth=max_growth,
            even=even,
        )
        counts = _solve(problem)
        return None if counts is None else {e: int(counts[i]) for e, i in problem.index.items()}

    plain = attempt((), ())
    if plain is None:
        raise _refuse("E09", "no counts hold the opposite sides of every block with these pins")
    ringed = attempt(totals, ()) if totals else plain
    if ringed is None:
        alone = [k for k, row in enumerate(totals) if attempt([row], ()) is None]
        held = [(len(totals[k][0]), totals[k][1]) for k in alone] or "together"
        raise _refuse("E08", f"ring totals (edges, count) {held} cannot be held on every side")
    if not carry:
        return ringed
    graded = attempt(totals, carry)
    if graded is None:
        growth = carry_pairs_growth(ringed, lengths, carry)
        raise _refuse("E10", f"carry rows: growth {growth:.3g} > {max_growth:.3g}")
    return graded


def carry_pairs_growth(
    counts: Mapping[int, int], lengths: Mapping[int, float], carry: Sequence[tuple[int, int]]
) -> float:
    """Return the largest width ratio over the carry pairs (1.0 when there is none).

    Parameters
    ----------
    counts : mapping of int to int
        The cell count of each edge.
    lengths : mapping of int to float
        The length of each edge.
    carry : sequence of tuple of int
        The pairs of edges that continue one grid line across a junction.

    Returns
    -------
    float
        The largest ratio of the wider to the narrower cell width over the pairs.
    """
    ratios = [1.0]
    for a, b in carry:
        wa, wb = lengths[a] / counts[a], lengths[b] / counts[b]
        ratios.append(max(wa, wb) / min(wa, wb))
    return float(max(ratios))


def carry_plan(
    junction_m: float, far_m: float, max_growth: float, rows: int | None = None
) -> dict[str, Any]:
    """Plan the carry rows that relax a junction cell width to the component's own width.

    Parameters
    ----------
    junction_m, far_m : float
        The cell width at the junction ring and the component's own width.
    max_growth : float
        The largest ratio of successive widths along a carry row.
    rows : int, optional
        The rows the controls give; derived (the fewest that keep the limit) when absent.

    Returns
    -------
    dict
        ``rows``, ``growth`` (the realized ratio per row) and ``widths`` (one per row, from the
        junction outward, ending at ``far_m``).

    Raises
    ------
    InputValueError
        E10 when the given rows need a ratio above ``max_growth``.
    """
    for name, value in (("junction_m", junction_m), ("far_m", far_m)):
        if not (math.isfinite(value) and value > 0):
            raise InputValueError(f"carry rows: {name} must be a finite positive width")
    if not (math.isfinite(max_growth) and max_growth > 1):
        raise InputValueError("carry rows: max_growth must be above 1")
    jump = max(junction_m, far_m) / min(junction_m, far_m)
    needed = 0 if jump == 1 else math.ceil(math.log(jump) / math.log(max_growth) - 1e-12)
    rows = needed if rows is None else int(rows)
    growth = 1.0 if jump == 1 else jump ** (1.0 / max(rows, 1))
    if growth > max_growth * (1 + 1e-12):
        raise _refuse("E10", f"growth {growth:.3g} > {max_growth:.3g} over {rows} rows; add rows")
    step = growth if far_m >= junction_m else 1.0 / growth
    widths = [junction_m * step**k for k in range(1, rows + 1)]
    return {"rows": rows, "growth": float(growth), "widths": widths}


def azimuth_split(total: int, arcs: Sequence[float], *, floor: int = 1) -> list[int]:
    """Split a half-body azimuth count over its sectors in proportion to their arc lengths.

    Largest remainders take the leftover cells; every sector holds at least ``floor`` cells.

    Parameters
    ----------
    total : int
        The half-body azimuth cell count to distribute.
    arcs : sequence of float
        The finite positive arc length of each sector.
    floor : int, optional
        The fewest cells any sector holds.

    Returns
    -------
    list of int
        One cell count per sector; they sum to ``total``.

    Raises
    ------
    InputValueError
        When the total cannot give every sector its floor.
    """
    lengths = np.asarray(arcs, dtype=float)
    if not len(lengths) or np.any(lengths <= 0) or not np.all(np.isfinite(lengths)):
        raise InputValueError("azimuth: finite positive sector arcs are required")
    if total < floor * len(lengths):
        raise InputValueError(
            f"azimuth: {total} cells cannot give {len(lengths)} sectors {floor} cells each"
        )
    share = floor + (total - floor * len(lengths)) * lengths / lengths.sum()
    counts = np.floor(share).astype(int)
    for k in np.argsort(-(share - counts), kind="stable")[: total - int(counts.sum())]:
        counts[k] += 1
    return [int(c) for c in counts]


def node_valence(cells: Sequence[Sequence[int]]) -> dict[int, int]:
    """Return the number of cells around each node."""
    valence: dict[int, int] = {}
    for cell in cells:
        for node in cell:
            valence[node] = valence.get(node, 0) + 1
    return valence


def _boundary_nodes(cells: Sequence[Sequence[int]]) -> set[int]:
    uses: dict[tuple[int, int], int] = {}
    for cell in cells:
        for a, b in zip(cell, [*cell[1:], cell[0]], strict=True):
            key = (min(a, b), max(a, b))
            uses[key] = uses.get(key, 0) + 1
    return {n for edge, k in uses.items() if k == 1 for n in edge}


def irregular_nodes(
    xyz: np.ndarray,
    cells: Sequence[Sequence[int]],
    poles: Sequence[Sequence[float]],
    *,
    tolerance_m: float,
) -> tuple[int, list[tuple[int, int]]]:
    """Return the valence 3 poles found and the other irregular interior nodes.

    ``poles`` are the declared pole points (the nose and base closures whose poles are
    ``valence3``); a node within ``tolerance_m`` of one of them with valence 3 is a pole. Every
    other interior node of valence other than four, and any declared pole with a different
    valence, is returned as ``(node, valence)``. Open
    boundary nodes (the symmetry plane) are not counted.

    Parameters
    ----------
    xyz : numpy.ndarray
        The node coordinates, one row per node.
    cells : sequence of sequence of int
        The node indices of each cell.
    poles : sequence of sequence of float
        The declared pole points.
    tolerance_m : float
        The distance within which a node is taken to be a declared pole.

    Returns
    -------
    tuple of int and list of tuple of int
        The count of valence 3 poles, and the other irregular interior nodes as
        ``(node, valence)``.
    """
    points = np.asarray(xyz, dtype=float)
    declared = np.asarray(poles, dtype=float).reshape(-1, 3)
    boundary = _boundary_nodes(cells)
    found, other = 0, []
    for node, valence in sorted(node_valence(cells).items()):
        if node in boundary:
            continue
        near = bool(len(declared)) and bool(
            np.min(np.linalg.norm(declared - points[node], axis=1)) <= tolerance_m
        )
        if valence == 3 and near:
            found += 1
        elif valence != 4 or near:
            other.append((node, valence))
    return found, other


def check_poles(
    xyz: np.ndarray,
    cells: Sequence[Sequence[int]],
    poles: Sequence[Sequence[float]],
    *,
    tolerance_m: float,
) -> dict[str, Any]:
    """Refuse irregular interior nodes or a declared pole whose valence is not three (E11).

    Returns
    -------
    dict
        ``valence3_poles`` (the count found at the declared points) and ``irregular`` (0).
    """
    found, other = irregular_nodes(xyz, cells, poles, tolerance_m=tolerance_m)
    if other:
        node, valence = other[0]
        where = np.round(np.asarray(xyz, dtype=float)[node], 6).tolist()
        raise _refuse(
            "E11",
            f"node at {where}: valence {valence}, not a declared valence 3 pole "
            f"({len(other)} such nodes)",
        )
    return {"valence3_poles": found, "irregular": 0}


def check_triangles(cells: Sequence[Sequence[int]], components: Sequence[str]) -> None:
    """Refuse any triangle (E07): strategy 2 is a full quad grid everywhere.

    Parameters
    ----------
    cells : sequence of sequence of int
        The node indices of each cell.
    components : sequence of str
        The component name of each cell, in the order of ``cells``.

    Raises
    ------
    InputValueError
        E07 when any cell has three nodes, naming the count per component.
    """
    by_component: dict[str, int] = {}
    for cell, name in zip(cells, components, strict=True):
        if len(cell) == 3:
            by_component[name] = by_component.get(name, 0) + 1
    if by_component:
        raise _refuse("E07", f"realized triangles {by_component}")
