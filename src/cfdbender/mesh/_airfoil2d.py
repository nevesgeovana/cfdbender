"""The 2D airfoil volume mesh of ``cfd-mesh airfoil-2d``: a structured C-mesh made with gmsh.

A section (a NACA 4 or 5 digit designation, or a coordinates file in Selig order) is normalized
to a unit chord and meshed as four transfinite gmsh blocks (lower wake, lower surface, upper
surface, upper wake) that share the wake cut, recombined to quadrilaterals. A blunt trailing
edge keeps its base, which becomes a fifth block between the two wake blocks with its own cells,
its edge in the separate airfoil_base wall marker. The far field is a semicircle of radius D
chords about the trailing edge, closed by straight lines to the outflow at D chords downstream
of it.

Wall-normal spacing: the first cell comes from a target y+ and the Reynolds number through the
turbulent flat-plate skin friction of Schlichting, ``Cf = (2 log10 Re - 0.65)^-2.3``, so
``y1 / c = y+ / (Re sqrt(Cf / 2))``; the normal cells then grow geometrically to the far field,
the ratio solved for the level's count. The levels carry the counts of the NASA Turbulence
Modeling Resource NACA 0012 family (225, 449 and 897 nodes around the C).

Two files are written: ``<name>.su2`` (two dimensions, quadrilaterals, markers ``airfoil`` and
``farfield`` plus ``airfoil_base`` for a blunt edge) written here from the gmsh nodes and
elements, and ``<name>.msh`` (gmsh 2.2 ASCII,
the same plane mesh extruded one cell along z, with ``airfoil``, ``farfield``, ``front``,
``back``, optional ``airfoil_base``, and the volume ``fluid``), the form an OpenFOAM workspace
imports with gmshToFoam.
The kernel runs in a disposable worker (:mod:`cfdbender.mesh._gmsh_worker`), so a fault leaves
nothing behind.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender._textio import write_text
from cfdbender.geometry._airfoils import naca_digits, normalized_naca
from cfdbender.mesh._gmsh_worker import run_isolated

__all__ = [
    "BASE_CELLS",
    "DEFAULT_FARFIELD",
    "DEFAULT_LEVEL",
    "DEFAULT_TRAILING_EDGE",
    "DEFAULT_YPLUS",
    "GROWTH_CAP",
    "LEVELS",
    "SPAN",
    "TRAILING_EDGES",
    "Level",
    "Section",
    "build_mesh",
    "first_cell",
    "naca_section",
    "plan_spacing",
    "read_coordinates",
    "shape_trailing_edge",
    "skin_friction",
    "two_sided",
    "write_mesh_files",
]

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Level:
    """The cell counts and wall spacings of one mesh level, in chords.

    Attributes
    ----------
    wall_cells : int
        Cells along each side of the airfoil, leading edge to trailing edge.
    normal_cells : int
        Cells from the wall (or the wake cut) to the far field.
    wake_cells : int
        Cells along the wake cut on each side, trailing edge to outflow.
    leading_edge_spacing, trailing_edge_spacing : float
        The tangential wall spacing at the leading and trailing edges; the wake starts at the
        trailing-edge spacing.
    """

    wall_cells: int
    normal_cells: int
    wake_cells: int
    leading_edge_spacing: float
    trailing_edge_spacing: float


#: The levels, with the counts and wall spacings of the NASA TMR NACA 0012 grids N-225, N-449
#: and N-897 (wall nodes 129, 257, 513; wake nodes 48, 96, 192 per side; normal nodes 65, 129,
#: 257).
LEVELS: dict[str, Level] = {
    "coarse": Level(64, 64, 48, 4.2740e-04, 4.2599e-03),
    "medium": Level(128, 128, 96, 2.0573e-04, 2.0545e-03),
    "fine": Level(256, 256, 192, 1.0094e-04, 1.0090e-03),
}
DEFAULT_LEVEL = "coarse"
#: The target y+ of the first cell.
DEFAULT_YPLUS = 1.0
#: The far-field distance in chords, as the TMR family (outflow 500 chords behind the edge).
DEFAULT_FARFIELD = 500.0
#: The nearest far field the verb accepts, in chords.
MIN_FARFIELD = 5.0
TRAILING_EDGES = ("sharp", "blunt")
DEFAULT_TRAILING_EDGE = "sharp"
#: The largest geometric growth ratio of the cells normal to the wall and along the wake.
GROWTH_CAP = 1.35
#: Cells across a blunt trailing-edge base, evenly spaced.
BASE_CELLS = 8
#: The extrusion of the OpenFOAM form along z, in chords: one chord, so a reference area of 1
#: is right for both solvers.
SPAN = 1.0
#: A gap between the two ends of a coordinates file above this many chords is an open contour.
MAX_GAP = 0.05
#: A trailing-edge gap at or below this many chords is closed.
CLOSED_GAP = 1e-9
#: The fewest points a coordinates file may hold.
MIN_POINTS = 5
#: The kernel's geometric tolerance, relative to the model size. Its default (1e-8 of a model
#: a thousand chords across) would merge the nodes of the first cells, a few millionths of a
#: chord apart, and turn their extruded faces into triangles.
GEOMETRY_TOLERANCE = 1e-14
#: Samples per side of the arc-length table of the section spline.
_SAMPLES = 20001


def skin_friction(reynolds: float) -> float:
    """Return the turbulent flat-plate skin friction of Schlichting at ``reynolds``.

    ``Cf = (2 log10 Re - 0.65)^-2.3``.
    """
    return float((2.0 * math.log10(reynolds) - 0.65) ** -2.3)


def first_cell(yplus: float, reynolds: float) -> float:
    """Return the first-cell height in chords that gives ``yplus`` at ``reynolds``.

    ``y1 / c = y+ / (Re sqrt(Cf / 2))`` with :func:`skin_friction`.
    """
    return yplus / (reynolds * math.sqrt(skin_friction(reynolds) / 2.0))


def _series(first: float, ratio: float, cells: int) -> float:
    """Return ``first * (ratio**cells - 1) / (ratio - 1)``, the length of a geometric series."""
    if ratio == 1.0:
        return first * cells
    return first * math.expm1(cells * math.log(ratio)) / (ratio - 1.0)


def _growth(first: float, length: float, cells: int) -> float:
    """Return the ratio whose geometric series of ``cells`` from ``first`` spans ``length``."""
    if first * cells >= length:
        raise InputValueError(
            f"a first cell of {first:.4g} chords is too large for {cells} cells over "
            f"{length:g} chords; lower y+ or raise the Reynolds number. Nothing was written."
        )
    low, high = 1.0, 2.0
    while _series(first, high, cells) < length:
        high *= 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _series(first, middle, cells) < length:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _cells_for(first: float, length: float) -> int:
    """Return the fewest cells growing at :data:`GROWTH_CAP` from ``first`` that span ``length``."""
    count = math.ceil(math.log1p(length * (GROWTH_CAP - 1.0) / first) / math.log(GROWTH_CAP))
    while _series(first, GROWTH_CAP, count) < length:
        count += 1
    return count


def _positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise InputValueError(
            f"{name} must be a positive number, not {value!r}. Nothing was written."
        )


def plan_spacing(*, reynolds: float, yplus: float, level: str, farfield: float) -> dict[str, Any]:
    """Return the spacing of the mesh: first cell, counts and the solved growth ratios.

    Parameters
    ----------
    reynolds : float
        The chord Reynolds number the first cell is made for.
    yplus : float
        The target y+ of the first cell.
    level : str
        A key of :data:`LEVELS`.
    farfield : float
        The far-field distance D in chords, at least :data:`MIN_FARFIELD`.

    Returns
    -------
    dict
        ``reynolds``, ``yplus``, ``first_cell``, ``level``, the counts and wall spacings of the
        level, ``normal_growth`` (from the first cell over D) and ``wake_growth`` (from the
        trailing-edge spacing over D), and ``farfield``.

    Raises
    ------
    InputValueError
        A value out of range, or a growth ratio above :data:`GROWTH_CAP`; the message names the
        count that would satisfy the cap.
    """
    _positive("reynolds", reynolds)
    if reynolds <= 10**0.325:
        raise InputValueError(
            "Reynolds number is outside the flat-plate correlation; increase it above "
            f"{10**0.325:.4g}. Nothing was written."
        )
    _positive("y+", yplus)
    if level not in LEVELS:
        raise InputValueError(
            f"level {level!r} is not one of {', '.join(LEVELS)}. Nothing was written."
        )
    if not (math.isfinite(farfield) and farfield >= MIN_FARFIELD):
        raise InputValueError(
            f"the far field must be at least {MIN_FARFIELD:g} chords, not {farfield!r}. "
            "Nothing was written."
        )
    counts = LEVELS[level]
    height = first_cell(yplus, reynolds)
    normal = _growth(height, farfield, counts.normal_cells)
    if normal > GROWTH_CAP:
        raise InputValueError(
            f"the wall-normal growth would be {normal:.4f}, above the cap {GROWTH_CAP}: a first "
            f"cell of {height:.4g} chords (y+ {yplus:g} at Re {reynolds:g}) to a far field of "
            f"{farfield:g} chords needs at least {_cells_for(height, farfield)} normal cells, "
            f"and level {level} has {counts.normal_cells}. Raise y+, choose a finer level or a "
            "nearer far field. Nothing was written."
        )
    wake = _growth(counts.trailing_edge_spacing, farfield, counts.wake_cells)
    if wake > GROWTH_CAP:
        raise InputValueError(
            f"the wake growth would be {wake:.4f}, above the cap {GROWTH_CAP}; level {level} "
            f"needs at least {_cells_for(counts.trailing_edge_spacing, farfield)} wake cells for "
            f"a far field of {farfield:g} chords. Choose a nearer far field. Nothing was written."
        )
    return {
        "reynolds": reynolds,
        "yplus": yplus,
        "first_cell": height,
        "level": level,
        "wall_cells": counts.wall_cells,
        "normal_cells": counts.normal_cells,
        "wake_cells": counts.wake_cells,
        "leading_edge_spacing": counts.leading_edge_spacing,
        "trailing_edge_spacing": counts.trailing_edge_spacing,
        "normal_growth": normal,
        "wake_growth": wake,
        "farfield": farfield,
    }


def _stretch(cells: int, start: float, end: float) -> Array:
    """Return Vinokur's two-sided stretching on [0, 1] for end spacings near ``start``, ``end``."""
    xi = np.arange(cells + 1, dtype=float) / cells
    product = cells * math.sqrt(start * end)
    asymmetry = math.sqrt(end / start)
    target = 1.0 / product
    if abs(target - 1.0) < 1e-9:
        u = xi
    elif target > 1.0:
        low, high = 1e-12, 1.0
        while math.sinh(high) / high < target:
            high *= 2.0
        for _ in range(200):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if math.sinh(middle) / middle < target else (low, middle)
        delta = 0.5 * (low + high)
        u = 0.5 * (1.0 + np.tanh(delta * (xi - 0.5)) / math.tanh(delta / 2.0))
    else:
        low, high = 1e-12, math.pi - 1e-9
        for _ in range(200):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if math.sin(middle) / middle > target else (low, middle)
        delta = 0.5 * (low + high)
        u = 0.5 * (1.0 + np.tan(delta * (xi - 0.5)) / math.tan(delta / 2.0))
    points: Array = u / (asymmetry + (1.0 - asymmetry) * u)
    points[0], points[-1] = 0.0, 1.0
    return points


def two_sided(length: float, cells: int, start: float, end: float) -> list[float]:
    """Return ``cells + 1`` positions on [0, length] whose end steps are ``start`` and ``end``.

    Vinokur's two-sided stretching, its input spacings corrected until the end steps meet the
    request.
    """
    want = np.array([start, end]) / length
    ask = want.copy()
    points = _stretch(cells, float(ask[0]), float(ask[1]))
    for _ in range(50):
        got = np.array([points[1] - points[0], points[-1] - points[-2]])
        if np.all(np.abs(got / want - 1.0) < 1e-10):
            break
        ask = ask * want / got
        points = _stretch(cells, float(ask[0]), float(ask[1]))
    return [float(p) * length for p in points]


@dataclass(frozen=True)
class Section:
    """A section normalized to a unit chord along +x, leading edge at the origin.

    Attributes
    ----------
    upper, lower : ndarray
        The two sides as (n, 2) points from the leading edge to the trailing edge.
    source : dict
        ``{"naca": digits}`` or ``{"coordinates": path}``.
    transform : dict
        How the input was normalized (``nose``, ``chord``, ``rotation``).
    gap_removed : float
        The trailing-edge gap a sharp trailing edge collapsed, in chords.
    """

    upper: Array
    lower: Array
    source: dict[str, str]
    transform: dict[str, Any] = field(default_factory=dict)
    gap_removed: float = 0.0

    @property
    def gap(self) -> float:
        """The trailing-edge gap in chords: the distance between the ends of the two sides."""
        return float(np.linalg.norm(self.upper[-1] - self.lower[-1]))


def naca_section(designation: str) -> Section:
    """Return the NACA 4 or 5 digit section ``designation``, normalized, its gap as published.

    The published thickness formula keeps the coefficient -0.1015, so the trailing edge is blunt
    (0.00252 chord for a NACA 0012); :func:`shape_trailing_edge` collapses it for a sharp edge.
    """
    try:
        digits = naca_digits(designation)  # type: ignore[no-untyped-call]
    except InputValueError as error:
        raise InputValueError(
            f"{designation!r} is not a NACA 4 or 5 digit designation with nonzero thickness "
            "(for example 0012 or 23012). Nothing was written."
        ) from error
    upper, lower, transform = normalized_naca(digits)  # type: ignore[no-untyped-call]
    return Section(
        np.asarray(upper, dtype=float),
        np.asarray(lower, dtype=float),
        {"naca": digits},
        dict(transform),
    )


def _number_rows(path: Path) -> list[tuple[float, float]]:
    """Return the point rows of a coordinates file, refusing a line that is not two numbers."""
    if not path.is_file():
        raise InputArtifactError(
            f"coordinates file not found: {path}. Nothing was written.", kind="coordinates"
        )
    rows: list[tuple[float, float]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for number, line in enumerate(lines, start=1):
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        try:
            if len(parts) != 2:
                raise ValueError
            row = (float(parts[0]), float(parts[1]))
        except ValueError:
            if not rows and number == next(i for i, t in enumerate(lines, 1) if t.strip()):
                continue
            raise InputArtifactError(
                f"line {number} of {path} is not two numbers: {line.strip()!r}. A coordinates "
                "file is an optional title line, then one 'x y' pair per line. Nothing was "
                "written.",
                kind="coordinates",
            ) from None
        if not all(math.isfinite(v) for v in row):
            raise InputArtifactError(
                f"line {number} of {path} holds a value that is not finite. Nothing was written.",
                kind="coordinates",
            )
        if not rows and min(abs(row[0]), abs(row[1])) > 1.5 and row[0] == int(row[0]):
            raise InputArtifactError(
                f"{path} starts with the point counts of the Lednicer layout ({line.strip()}); "
                "write it in Selig order (trailing edge, upper surface, leading edge, lower "
                "surface, trailing edge). Nothing was written.",
                kind="coordinates",
            )
        if rows and rows[-1] == row:
            continue
        rows.append(row)
    if len(rows) < MIN_POINTS:
        raise InputArtifactError(
            f"{path} holds {len(rows)} points; a section needs at least {MIN_POINTS}. Nothing "
            "was written.",
            kind="coordinates",
        )
    return rows


def _crossing(contour: Array) -> Array | None:
    """Return a point where the closed polygon ``contour`` crosses itself, or None."""
    closed = np.vstack([contour, contour[:1]])
    if np.linalg.norm(contour[0] - contour[-1]) <= CLOSED_GAP:
        closed = contour.copy()
    a, b = closed[:-1], closed[1:]
    count = len(a)
    d = b - a
    for i in range(count - 2):
        j = np.arange(i + 2, count)
        if i == 0 and np.allclose(closed[0], closed[-1]):
            j = j[j != count - 1]
        if j.size == 0:
            continue
        r, s = d[i], d[j]
        denominator = r[0] * s[:, 1] - r[1] * s[:, 0]
        q = a[j] - a[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / denominator
            u = (q[:, 0] * r[1] - q[:, 1] * r[0]) / denominator
        hit = (denominator != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if hit.any():
            k = int(np.argmax(hit))
            found: Array = a[i] + t[k] * r
            return found
        # Parallel segments can overlap, including a repeated vertex away from the nose.
        scale = np.linalg.norm(r) * np.linalg.norm(s, axis=1)
        parallel = np.abs(denominator) <= 1e-12 * scale
        collinear = np.abs(q[:, 0] * r[1] - q[:, 1] * r[0]) <= (
            1e-12 * np.linalg.norm(r) * np.linalg.norm(q, axis=1)
        )
        length2 = float(np.dot(r, r))
        if length2:
            start = q @ r / length2
            end = (q + s) @ r / length2
            overlap = (
                parallel & collinear & (np.maximum(start, end) >= 0) & (np.minimum(start, end) <= 1)
            )
            if overlap.any():
                k = int(np.argmax(overlap))
                found = a[i] + max(0.0, min(float(start[k]), float(end[k]))) * r
                return found
    return None


def read_coordinates(path: str | Path) -> Section:
    """Return the section of a coordinates file in Selig order, normalized to a unit chord.

    The file holds an optional title line, then one ``x y`` pair per line, from the trailing
    edge over the upper surface to the leading edge and back under the lower surface (the
    opposite direction is accepted). The leading edge is the point farthest from the
    trailing-edge midpoint; the section is moved, turned and scaled so that point is the origin
    and the trailing-edge midpoint is (1, 0).

    Raises
    ------
    InputArtifactError
        The file is missing, a line is not two numbers, it is in the Lednicer layout, or it
        holds fewer than :data:`MIN_POINTS` points.
    InputValueError
        The contour is open (its ends more than :data:`MAX_GAP` chords apart) or crosses itself.
    """
    source = Path(path)
    contour = np.array(_number_rows(source), dtype=float)
    x, y = contour[:, 0], contour[:, 1]
    if 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0:
        contour = contour[::-1].copy()
    middle = 0.5 * (contour[0] + contour[-1])
    nose_index = int(np.argmax(np.linalg.norm(contour - middle, axis=1)))
    if nose_index in (0, len(contour) - 1):
        raise InputValueError(
            f"{source} does not run from the trailing edge round the leading edge and back "
            "(Selig order). Nothing was written."
        )
    nose = contour[nose_index].copy()
    axis = middle - nose
    chord = float(np.linalg.norm(axis))
    cos, sin = axis / chord
    rotation = np.array([[cos, -sin], [sin, cos]])
    normalised: Array = (contour - nose) @ rotation / chord
    gap = float(np.linalg.norm(normalised[0] - normalised[-1]))
    if gap > MAX_GAP:
        raise InputValueError(
            f"the contour of {source} is open: its two ends are {gap:.4g} chords apart, and a "
            f"trailing-edge gap is at most {MAX_GAP:g} chords. Give both surfaces, trailing "
            "edge to trailing edge. Nothing was written."
        )
    crossing = _crossing(normalised)
    if crossing is not None:
        raise InputValueError(
            f"the contour of {source} crosses itself near x = {crossing[0]:.4g}, y = "
            f"{crossing[1]:.4g} chords. Remove overlapping or crossing segments. "
            "Nothing was written."
        )
    upper = normalised[: nose_index + 1][::-1].copy()
    lower = normalised[nose_index:].copy()
    if gap <= CLOSED_GAP:
        upper[-1] = lower[-1] = (1.0, 0.0)
    return Section(
        upper,
        lower,
        {"coordinates": str(source)},
        {"nose": nose.tolist(), "chord": chord, "rotation": rotation.tolist()},
    )


def _fraction(side: Array) -> Array:
    """Return the cumulative arc-length fraction of the points of ``side``."""
    steps = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(side, axis=0), axis=1))]
    fraction: Array = steps / steps[-1]
    return fraction


def shape_trailing_edge(section: Section, trailing_edge: str) -> Section:
    """Return ``section`` with a sharp or a blunt trailing edge.

    ``sharp`` collapses the gap: each side is moved by the vector from its end to the
    trailing-edge midpoint (1, 0), weighted by its arc-length fraction from the leading edge, and
    the gap removed is recorded. ``blunt`` keeps the gap, which becomes the base of its own block.

    Raises
    ------
    InputValueError
        An unknown treatment, or ``blunt`` on a closed trailing edge.
    """
    if trailing_edge not in TRAILING_EDGES:
        raise InputValueError(
            f"the trailing edge must be sharp or blunt, not {trailing_edge!r}. Nothing was written."
        )
    gap = section.gap
    if trailing_edge == "blunt":
        if gap <= CLOSED_GAP:
            raise InputValueError(
                "a blunt trailing edge needs a gap, and this section has a closed trailing "
                "edge; mesh it sharp. Nothing was written."
            )
        return replace(section, gap_removed=0.0)
    if gap <= CLOSED_GAP:
        return replace(section, gap_removed=0.0)
    edge = np.array([1.0, 0.0])
    sides = []
    for side in (section.upper, section.lower):
        moved = side + np.outer(_fraction(side), edge - side[-1])
        moved[0], moved[-1] = side[0], edge
        sides.append(moved)
    return replace(section, upper=sides[0], lower=sides[1], gap_removed=gap)


# ----------------------------------------------------------------------------------------------
# The mesh


def _wall_nodes(
    upper: Array, lower: Array, cells: int, start: float, end: float
) -> tuple[Array, Array]:
    """Return the wall nodes of both sides, from the leading edge to the trailing edge.

    One cubic spline runs through the whole contour, parametrised by its chord length, so the
    two sides meet smoothly at the leading edge; the nodes follow :func:`two_sided` in arc
    length from ``start`` at the leading edge to ``end`` at the trailing edge.
    """
    contour = np.vstack([upper[::-1], lower[1:]])
    t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))]
    spline = CubicSpline(t, contour)
    nose = t[len(upper) - 1]
    sides = []
    for side, finish in ((upper, t[0]), (lower, t[-1])):
        dense = np.linspace(nose, finish, _SAMPLES)
        samples = spline(dense)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
        wanted = two_sided(float(arc[-1]), cells, start, end)
        nodes: Array = np.asarray(spline(np.interp(wanted, arc, dense)), dtype=float)
        nodes[0], nodes[-1] = side[0], side[-1]
        sides.append(nodes)
    return sides[0], sides[1]


def _arc_fractions(nodes: Array, upper: bool) -> Array:
    """Return where each wall node's grid line meets the far-field arc, as a fraction of 90 deg.

    The line leaves the wall along its outward normal where it can: the normal's angle from
    upstream, clipped to 0 to 90 degrees and made monotone, blended with a tenth of the
    arc-length fraction so the fractions strictly increase.
    """
    tangent = np.gradient(nodes, axis=0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    if not upper:
        normal = -normal
    angle = np.degrees(np.arctan2(np.abs(normal[:, 1]), -normal[:, 0]))
    monotone = np.maximum.accumulate(np.clip(angle, 0.0, 90.0) / 90.0)
    fractions: Array = 0.9 * monotone + 0.1 * _fraction(nodes)
    fractions[0], fractions[-1] = 0.0, 1.0
    return fractions


class _Builder:
    """The gmsh geo model of the C-mesh, built curve by curve."""

    def __init__(self, gmsh: Any) -> None:
        self.gmsh = gmsh
        self.geo = gmsh.model.geo

    def point(self, xy: Sequence[float]) -> int:
        return int(self.geo.addPoint(float(xy[0]), float(xy[1]), 0.0))

    def polyline(self, ends: tuple[int, int], inner: Sequence[Sequence[float]]) -> list[int]:
        """Return one straight curve per cell through the given inner points."""
        tags = [ends[0], *(self.point(p) for p in inner), ends[1]]
        curves = [int(self.geo.addLine(a, b)) for a, b in zip(tags, tags[1:], strict=False)]
        for curve in curves:
            self.geo.mesh.setTransfiniteCurve(curve, 2)
        return curves

    def arc(
        self, ends: tuple[int, int], centre: int, inner: Sequence[Sequence[float]]
    ) -> list[int]:
        """Return one circle arc per cell through the given inner points about ``center``."""
        tags = [ends[0], *(self.point(p) for p in inner), ends[1]]
        curves = [
            int(self.geo.addCircleArc(a, centre, b)) for a, b in zip(tags, tags[1:], strict=False)
        ]
        for curve in curves:
            self.geo.mesh.setTransfiniteCurve(curve, 2)
        return curves

    def line(
        self,
        ends: tuple[int, int],
        start: Sequence[float],
        end: Sequence[float],
        cells: int,
        first: float,
    ) -> list[int]:
        """Return a straight polyline of ``cells`` growing geometrically from ``first`` at start.

        The nodes are placed here, not by the kernel's progression law, which does not place
        a first cell a hundred-millionth of the line's length exactly.
        """
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        length = float(np.linalg.norm(b - a))
        ratio = _growth(first, length, cells)
        inner = [
            (a + (b - a) * (_series(first, ratio, k) / length)).tolist() for k in range(1, cells)
        ]
        return self.polyline(ends, inner)

    def block(self, loop: Sequence[int], corners: Sequence[int]) -> int:
        surface = int(self.geo.addPlaneSurface([self.geo.addCurveLoop(list(loop))]))
        self.geo.mesh.setTransfiniteSurface(surface, "Left", list(corners))
        self.geo.mesh.setRecombine(2, surface)
        return surface


def _reverse(curves: Sequence[int]) -> list[int]:
    return [-c for c in reversed(curves)]


def _build_model(
    builder: _Builder, upper: Array, lower: Array, plan: dict[str, Any], blunt: bool
) -> dict[str, Any]:
    """Build the blocks and return their surfaces and the curves of each marker."""
    far = float(plan["farfield"])
    normal_cells, wall_cells = int(plan["normal_cells"]), int(plan["wall_cells"])
    wake_cells, y1 = int(plan["wake_cells"]), float(plan["first_cell"])
    spacing = float(plan["trailing_edge_spacing"])
    upper_wall, lower_wall = _wall_nodes(
        upper, lower, wall_cells, float(plan["leading_edge_spacing"]), spacing
    )
    nose_xy, edge_up_xy, edge_low_xy = upper_wall[0], upper_wall[-1], lower_wall[-1]
    nose = builder.point(nose_xy)
    edge_upper = builder.point(edge_up_xy)
    edge_lower = builder.point(edge_low_xy) if blunt else edge_upper
    centre = builder.point((1.0, 0.0))
    front_xy, top_xy, bottom_xy = (1.0 - far, 0.0), (1.0, far), (1.0, -far)
    front, top, bottom = builder.point(front_xy), builder.point(top_xy), builder.point(bottom_xy)
    out_top_xy, out_bottom_xy = (1.0 + far, far), (1.0 + far, -far)
    out_top, out_bottom = builder.point(out_top_xy), builder.point(out_bottom_xy)
    out_upper_xy = (1.0 + far, float(edge_up_xy[1]))
    out_lower_xy = (1.0 + far, float(edge_low_xy[1]))
    out_upper = builder.point(out_upper_xy)
    out_lower = builder.point(out_lower_xy) if blunt else out_upper

    upper_curves = builder.polyline((nose, edge_upper), upper_wall[1:-1].tolist())
    lower_curves = builder.polyline((nose, edge_lower), lower_wall[1:-1].tolist())

    def arc_points(fractions: Array, sign: float) -> list[list[float]]:
        angle = fractions[1:-1] * math.pi / 2.0
        return [[1.0 - far * math.cos(a), sign * far * math.sin(a)] for a in angle.tolist()]

    upper_arc = builder.arc((front, top), centre, arc_points(_arc_fractions(upper_wall, True), 1.0))
    lower_arc = builder.arc(
        (front, bottom), centre, arc_points(_arc_fractions(lower_wall, False), -1.0)
    )
    nose_line = builder.line((nose, front), nose_xy, front_xy, normal_cells, y1)
    edge_up_line = builder.line((edge_upper, top), edge_up_xy, top_xy, normal_cells, y1)
    edge_low_line = builder.line((edge_lower, bottom), edge_low_xy, bottom_xy, normal_cells, y1)
    out_up_line = builder.line((out_upper, out_top), out_upper_xy, out_top_xy, normal_cells, y1)
    out_low_line = builder.line(
        (out_lower, out_bottom), out_lower_xy, out_bottom_xy, normal_cells, y1
    )
    cut_upper = builder.line((edge_upper, out_upper), edge_up_xy, out_upper_xy, wake_cells, spacing)
    cut_lower = cut_upper
    if blunt:
        cut_lower = builder.line(
            (edge_lower, out_lower), edge_low_xy, out_lower_xy, wake_cells, spacing
        )
    top_line = builder.line((top, out_top), top_xy, out_top_xy, wake_cells, spacing)
    bottom_line = builder.line((bottom, out_bottom), bottom_xy, out_bottom_xy, wake_cells, spacing)

    blocks = [
        builder.block(
            [*edge_low_line, *bottom_line, *_reverse(out_low_line), *_reverse(cut_lower)],
            [edge_lower, bottom, out_bottom, out_lower],
        ),
        builder.block(
            [*nose_line, *lower_arc, *_reverse(edge_low_line), *_reverse(lower_curves)],
            [nose, front, bottom, edge_lower],
        ),
        builder.block(
            [*upper_curves, *edge_up_line, *_reverse(upper_arc), *_reverse(nose_line)],
            [nose, edge_upper, top, front],
        ),
        builder.block(
            [*cut_upper, *out_up_line, *_reverse(top_line), *_reverse(edge_up_line)],
            [edge_upper, out_upper, out_top, top],
        ),
    ]
    airfoil = [*lower_curves, *upper_curves]
    farfield = [
        *out_low_line,
        *bottom_line,
        *lower_arc,
        *upper_arc,
        *top_line,
        *out_up_line,
    ]
    base: list[int] = []
    if blunt:
        across = [i / BASE_CELLS for i in range(1, BASE_CELLS)]
        step = edge_up_xy - edge_low_xy
        base_inner = [(edge_low_xy + f * step).tolist() for f in across]
        rise = out_upper_xy[1] - out_lower_xy[1]
        out_inner = [[1.0 + far, out_lower_xy[1] + f * rise] for f in across]
        base = builder.polyline((edge_lower, edge_upper), base_inner)
        base_out = builder.polyline((out_lower, out_upper), out_inner)
        blocks.append(
            builder.block(
                [*cut_lower, *base_out, *_reverse(cut_upper), *_reverse(base)],
                [edge_lower, out_lower, out_upper, edge_upper],
            )
        )
        farfield += base_out
    return {
        "blocks": blocks,
        "airfoil": airfoil,
        "base": base,
        "farfield": farfield,
        "radial": [
            (nose_line, nose_xy.tolist()),
            (edge_up_line, edge_up_xy.tolist()),
            (edge_low_line, edge_low_xy.tolist()),
            (out_up_line, list(out_upper_xy)),
            (out_low_line, list(out_lower_xy)),
        ],
    }


def _classify(gmsh: Any, extruded: Sequence[tuple[int, int]], model: dict[str, Any]) -> None:
    """Add the physical groups of the extruded mesh: the four patches and the volume."""
    airfoil, base, farfield = set(model["airfoil"]), set(model["base"]), set(model["farfield"])
    groups: dict[str, list[int]] = {"airfoil": [], "farfield": [], "back": [], "fluid": []}
    for position, (dim, tag) in enumerate(extruded):
        if dim == 3:
            groups["fluid"].append(tag)
        elif position + 1 < len(extruded) and extruded[position + 1][0] == 3:
            groups["back"].append(tag)
        else:
            curves = {abs(c) for _, c in gmsh.model.getBoundary([(2, tag)], oriented=False)}
            if curves & airfoil:
                groups["airfoil"].append(tag)
            elif curves & base:
                groups.setdefault("airfoil_base", []).append(tag)
            elif curves & farfield:
                groups["farfield"].append(tag)
    groups["front"] = list(model["blocks"])
    for name in ("airfoil", "airfoil_base", "farfield", "front", "back", "fluid"):
        if name not in groups:
            continue
        dim = 3 if name == "fluid" else 2
        group = gmsh.model.addPhysicalGroup(dim, sorted(groups[name]))
        gmsh.model.setPhysicalName(dim, group, name)


def _plane_mesh(gmsh: Any, model: dict[str, Any]) -> dict[str, Any]:
    """Return the z = 0 plane mesh: points, counter-clockwise quadrilaterals and marker edges."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    tags, xyz = np.asarray(tags)[order], np.asarray(coords).reshape(-1, 3)[order]
    blocks = []
    for surface in model["blocks"]:
        kinds, _, nodes = gmsh.model.mesh.getElements(2, surface)
        for kind, members in zip(kinds, nodes, strict=True):
            if int(kind) != 3:
                raise InputArtifactError(
                    "the 2D airfoil mesh holds a cell that is not a quadrilateral; the section "
                    "cannot be meshed as a structured C-mesh. Nothing was written.",
                    kind="geometry",
                )
            blocks.append(np.asarray(members, dtype=np.int64).reshape(-1, 4))
    raw = np.vstack(blocks)
    used = np.unique(raw)
    points = xyz[np.searchsorted(tags, used)][:, :2]
    cells = np.searchsorted(used, raw)
    xy = points[cells]
    area = 0.5 * np.sum(
        xy[..., 0] * np.roll(xy[..., 1], -1, axis=1) - np.roll(xy[..., 0], -1, axis=1) * xy[..., 1],
        axis=1,
    )
    flip = area < 0
    cells[flip] = cells[flip][:, ::-1]

    def edges(curves: Sequence[int]) -> NDArray[np.int64]:
        found = []
        for curve in curves:
            kinds, _, nodes = gmsh.model.mesh.getElements(1, curve)
            for kind, members in zip(kinds, nodes, strict=True):
                if int(kind) == 1:
                    found.append(np.asarray(members, dtype=np.int64).reshape(-1, 2))
        joined = np.vstack(found) if found else np.zeros((0, 2), dtype=np.int64)
        result: NDArray[np.int64] = np.searchsorted(used, joined)
        return result

    markers = {"airfoil": edges(model["airfoil"]), "farfield": edges(model["farfield"])}
    if model["base"]:
        markers["airfoil_base"] = edges(model["base"])
    return {
        "points": points,
        "cells": cells,
        "area": np.abs(area),
        "markers": markers,
        "base": edges(model["base"]),
    }


def _leaving(cells: NDArray[np.int64], points: Array, wall: NDArray[np.int64]) -> Array:
    """Return the lengths of the cell edges that leave the wall edges ``wall`` off the wall."""
    if len(wall) == 0:
        return np.zeros(0)
    marked = {(int(min(a, b)), int(max(a, b))) for a, b in wall}
    lengths: dict[tuple[int, int], float] = {}
    for cell in cells[np.isin(cells, np.unique(wall)).any(axis=1)]:
        ring = [int(v) for v in cell]
        for k in range(4):
            a, b = ring[k], ring[(k + 1) % 4]
            if (min(a, b), max(a, b)) not in marked:
                continue
            for p, q in ((a, ring[(k - 1) % 4]), (b, ring[(k + 2) % 4])):
                lengths[(min(p, q), max(p, q))] = float(np.linalg.norm(points[p] - points[q]))
    return np.array(sorted(lengths.values()))


def _growth_along(gmsh: Any, curves: Sequence[int], start: Sequence[float]) -> float:
    """Return the largest ratio of successive cell lengths along a straight polyline."""
    found = [
        np.asarray(gmsh.model.mesh.getNodes(1, c, includeBoundary=True)[1]).reshape(-1, 3)
        for c in curves
    ]
    xy = np.unique(np.vstack(found)[:, :2], axis=0)
    xy = xy[np.argsort(np.linalg.norm(xy - np.asarray(start), axis=1))]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return float(np.max(steps[1:] / steps[:-1]))


def _quality(gmsh: Any, plane: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Return the measured quality of the plane mesh, the checks a hand-made 2D mesh gets."""
    points, cells = plane["points"], plane["cells"]
    ring = np.stack([cells, np.roll(cells, -1, axis=1)], axis=-1).reshape(-1, 2)
    ring.sort(axis=1)
    unique, counts = np.unique(ring, axis=0, return_counts=True)
    boundary = {(int(a), int(b)) for (a, b), n in zip(unique, counts, strict=True) if n == 1}
    marked: set[tuple[int, int]] = set()
    for found in plane["markers"].values():
        marked |= {(int(min(a, b)), int(max(a, b))) for a, b in found}
    base = {(int(min(a, b)), int(max(a, b))) for a, b in plane["base"]}
    surface = np.array(
        [e for e in plane["markers"]["airfoil"] if (int(min(e)), int(max(e))) not in base],
        dtype=np.int64,
    ).reshape(-1, 2)
    first = _leaving(cells, points, surface)
    far_nodes = np.unique(plane["markers"]["farfield"])
    distance = np.linalg.norm(points[far_nodes] - np.array([1.0, 0.0]), axis=1)
    sides = np.linalg.norm(points[cells] - points[np.roll(cells, -1, axis=1)], axis=2)
    record: dict[str, Any] = {
        "cells": int(len(cells)),
        "nodes": int(len(points)),
        "markers": {name: int(len(found)) for name, found in plane["markers"].items()},
        "first_cell": {
            "min": float(first.min()),
            "mean": float(first.mean()),
            "max": float(first.max()),
        },
        "largest_growth": max(_growth_along(gmsh, c, s) for c, s in model["radial"]),
        "farfield": {"min": float(distance.min()), "max": float(distance.max())},
        "outflow_x": float(points[far_nodes][:, 0].max()),
        "smallest_area": float(plane["area"].min()),
        "largest_aspect_ratio": float((sides.max(axis=1) / sides.min(axis=1)).max()),
        "duplicate_nodes": int(len(points) - len(np.unique(np.round(points, 12), axis=0))),
        "unmarked_boundary_edges": len(boundary - marked),
        "marked_interior_edges": len(marked - boundary),
    }
    if len(plane["base"]):
        record["base_first_cell"] = float(_leaving(cells, points, plane["base"]).mean())
    return record


def _su2_text(plane: dict[str, Any]) -> str:
    """Return the native SU2 text of the plane mesh: quadrilaterals (9) and marker lines (3)."""
    points, cells = plane["points"], plane["cells"]
    lines = ["NDIME= 2", f"NELEM= {len(cells)}"]
    lines += [f"9 {a} {b} {c} {d} {i}" for i, (a, b, c, d) in enumerate(cells.tolist())]
    lines.append(f"NPOIN= {len(points)}")
    lines += [f"{x:.17g} {y:.17g} {i}" for i, (x, y) in enumerate(points.tolist())]
    lines.append(f"NMARK= {len(plane['markers'])}")
    for name, found in plane["markers"].items():
        lines += [f"MARKER_TAG= {name}", f"MARKER_ELEMS= {len(found)}"]
        lines += [f"3 {a} {b}" for a, b in found.tolist()]
    return "\n".join(lines) + "\n"


def write_mesh_files(
    *,
    out: str,
    upper: list[list[float]],
    lower: list[list[float]],
    plan: dict[str, Any],
    trailing_edge: str,
) -> dict[str, Any]:
    """Mesh the shaped section with gmsh and write ``<out>.su2`` and ``<out>.msh``.

    The worker side of :func:`build_mesh`; its arguments are JSON values.

    Returns
    -------
    dict
        ``files``, the gmsh version, the parameters and the measured quality.
    """
    from cfdbender.mesh._gmsh import session

    blunt = trailing_edge == "blunt"
    su2, msh = Path(out + ".su2"), Path(out + ".msh")
    su2.parent.mkdir(parents=True, exist_ok=True)
    with session("the 2D airfoil mesh of cfd-mesh airfoil-2d") as gmsh:
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Geometry.Tolerance", GEOMETRY_TOLERANCE)
        gmsh.model.add("airfoil2d")
        builder = _Builder(gmsh)
        model = _build_model(
            builder, np.asarray(upper, dtype=float), np.asarray(lower, dtype=float), plan, blunt
        )
        extruded = builder.geo.extrude(
            [(2, s) for s in model["blocks"]], 0.0, 0.0, SPAN, numElements=[1], recombine=True
        )
        builder.geo.synchronize()
        _classify(gmsh, extruded, model)
        gmsh.model.mesh.generate(3)
        plane = _plane_mesh(gmsh, model)
        record = _quality(gmsh, plane, model)
        write_text(su2, _su2_text(plane), encoding="ascii")
        gmsh.write(str(msh))
        version = str(gmsh.option.getString("General.Version"))
    parameters = {
        **plan,
        "trailing_edge": trailing_edge,
        "base_cells": BASE_CELLS if blunt else 0,
        "span": SPAN,
    }
    return {
        "files": {"su2": str(su2), "msh": str(msh)},
        "gmsh": version,
        "parameters": parameters,
        **record,
    }


def build_mesh(
    out: str | Path,
    section: Section,
    *,
    reynolds: float,
    yplus: float = DEFAULT_YPLUS,
    level: str = DEFAULT_LEVEL,
    farfield: float = DEFAULT_FARFIELD,
    trailing_edge: str = DEFAULT_TRAILING_EDGE,
) -> dict[str, Any]:
    """Write the 2D C-mesh of ``section`` as ``<out>.su2`` and ``<out>.msh``; return its record.

    Parameters
    ----------
    out : str or Path
        The output path without a suffix; its folder must exist.
    section : Section
        From :func:`naca_section` or :func:`read_coordinates`.
    reynolds, yplus, level, farfield
        As :func:`plan_spacing`.
    trailing_edge : str
        ``sharp`` (default) or ``blunt``, as :func:`shape_trailing_edge`.

    Returns
    -------
    dict
        ``files`` (``su2``, ``msh``), ``gmsh``, ``parameters`` (every value after defaults, the
        section's source and transform and the gap removed) and the measured quality: ``cells``,
        ``nodes``, ``markers``, ``first_cell`` (min, mean, max on the airfoil surface),
        ``largest_growth``, ``farfield`` (distance from the trailing edge, min and max),
        ``outflow_x``, ``smallest_area``, ``largest_aspect_ratio``, ``duplicate_nodes``,
        ``unmarked_boundary_edges``, ``marked_interior_edges`` and, for a blunt edge,
        ``base_first_cell``.

    Raises
    ------
    InputValueError
        A value out of range, as :func:`plan_spacing` and :func:`shape_trailing_edge`.
    InputArtifactError
        The kernel failed; nothing was written.
    """
    shaped = shape_trailing_edge(section, trailing_edge)
    plan = plan_spacing(reynolds=reynolds, yplus=yplus, level=level, farfield=farfield)
    arguments = {
        "out": str(out),
        "upper": shaped.upper.tolist(),
        "lower": shaped.lower.tolist(),
        "plan": plan,
        "trailing_edge": trailing_edge,
    }
    record = run_isolated("cfdbender.mesh._airfoil2d:write_mesh_files", arguments, write_mesh_files)
    record["parameters"].update(
        {
            "gap_removed": shaped.gap_removed,
            "source": dict(shaped.source),
            "transform": dict(shaped.transform),
        }
    )
    return record
