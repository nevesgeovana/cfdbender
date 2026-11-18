"""Cut a surface by planes and write the section Cp and sectional-load tables.

The reduction is solver agnostic: it takes a surface (triangles and
quadrilaterals, point data, optional family labels), the reference quantities and
one ``[[sections.distributions]]`` entry, and returns per station the Cp
distribution and the sectional loads. The tables have the file names, the column
names, the column order, the units and the signs of the sectional-loads and Cp
exports of the FlightStream post-processing, so a table from one solver reads
like a table from another. Columns the caller does not list as fillable are
absent from the files, never empty.

Definitions used
----------------
* A plane of a frame is named by the two axes it holds: ``XY`` holds x and y and
  its offset runs along z, ``XZ`` holds x and z with the offset along y, ``YZ``
  holds y and z with the offset along x. Axes and offsets are those of the
  distribution's frame.
* ``count`` planes lie at the centres of ``count`` equal bins across the extent
  of the selected cells along the offset axis (the stations are strip centres).
* Forces are per unit offset (N/m) in the frame's axes, the first and the second
  axis of the plane as ``Fx`` and ``Fz``. ``Moment`` is per unit offset (N m/m),
  about the moment reference point, along the offset axis.
* The pressure force of a station is ``-(p - p_inf) n dA`` integrated over the
  strip of surface between two neighbouring planes, per unit offset. The
  normal ``n`` of a cell is that of its winding, times ``Surface.normal_sign``.
* The integrated columns follow the strip rule: strip edges at midpoints between
  stations and half a spacing beyond each end station.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CP_COLUMNS",
    "INTEGRATED_COLUMNS",
    "SLOADS_COLUMNS",
    "Distribution",
    "Frame",
    "PlaneBlock",
    "Reference",
    "SectionError",
    "SectionsResult",
    "Station",
    "Surface",
    "compute_sections",
    "distribution_names",
    "strip_lengths",
    "write_section_files",
]

#: The columns of the Cp table after ``SECTION``, in order.
CP_COLUMNS: tuple[str, ...] = (
    "Section_direction_value",
    "X",
    "Y",
    "Z",
    "nx",
    "ny",
    "nz",
    "L",
    "Cp",
    "Mach",
    "vx",
    "vy",
    "vz",
    "vtot",
    "Cp_ref",
    "Theta",
    "CF",
    "Delta*",
    "Delta",
    "H",
)
#: The columns of the sectional-loads table, in order.
SLOADS_COLUMNS: tuple[str, ...] = ("Offset", "Chord", "X_QC", "Z_QC", "Fx", "Fz", "Moment")
#: The columns an ``integrate = true`` entry adds after ``Moment``.
INTEGRATED_COLUMNS: tuple[str, ...] = ("Strip_length", "Fx_int", "Fz_int", "My_int")

#: Plane name to (first in-plane axis, second in-plane axis, offset axis).
_PLANES: dict[str, tuple[int, int, int]] = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}
#: The solver fields a Cp row takes from the surface by column name.
_FIELD_COLUMNS: tuple[str, ...] = ("Mach", "vx", "vy", "vz", "Theta", "CF", "Delta*", "Delta", "H")
#: A facet whose normal is this close to the offset axis is parallel to the cut.
_PARALLEL_FACET = 1e-12
#: The quarter-chord station of the chord, from the leading edge.
QUARTER_CHORD = 0.25


class SectionError(ValueError):
    """A declaration or a surface the sectioning cannot work with."""


@dataclass(frozen=True, eq=False)
class Frame:
    """A frame as an origin and three axes, in reference coordinates.

    Attributes
    ----------
    origin : tuple of float
        The frame origin.
    axes : tuple of tuple of float
        The unit vectors of the frame's x, y and z axes (one row each).
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axes: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def matrix(self) -> NDArray[np.float64]:
        """Return the axes as a 3 x 3 array, one axis per row."""
        return np.asarray(self.axes, dtype=np.float64)

    def to_frame(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return reference-coordinate points in this frame's coordinates."""
        out: NDArray[np.float64] = (points - np.asarray(self.origin)) @ self.matrix().T
        return out


@dataclass(frozen=True, eq=False)
class Surface:
    """A surface with named cell families and point data.

    Attributes
    ----------
    points : numpy.ndarray
        Point coordinates, ``(n, 3)``, reference coordinates, metres.
    triangles, quads : numpy.ndarray
        Point indices of the cells, ``(m, 3)`` and ``(k, 4)``.
    pressure : numpy.ndarray
        Static pressure at the points, Pa.
    fields : mapping of str to numpy.ndarray
        Point data named by the Cp column it fills (``Mach``, ``vx``, ``vy``, ``vz``,
        ``vtot``, ``Cp_ref``, ``CF``, ``Theta``, ``Delta*``, ``Delta``, ``H``).
    families : tuple of str
        The family names.
    triangle_family, quad_family : numpy.ndarray
        Index into ``families`` for each triangle and each quad; unstated, all
        cells belong to the one family.
    normal_sign : int
        ``+1`` when the winding normal of a cell points out of the body into the
        fluid, ``-1`` when it points into the body.
    """

    points: NDArray[np.float64]
    triangles: NDArray[np.int64]
    quads: NDArray[np.int64]
    pressure: NDArray[np.float64]
    fields: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    families: tuple[str, ...] = ("surface",)
    triangle_family: NDArray[np.int64] | None = None
    quad_family: NDArray[np.int64] | None = None
    normal_sign: int = 1


@dataclass(frozen=True)
class Reference:
    """The reference pressure and dynamic pressure of the Cp columns."""

    p_inf: float
    q_ref: float


@dataclass(frozen=True)
class Distribution:
    """One ``[[sections.distributions]]`` entry, with ``count`` and ``plot_direction`` resolved.

    Attributes
    ----------
    families : tuple of str
        Family names to cut; empty cuts every cell.
    frame : str
        The frame name; the caller maps it to a :class:`Frame`.
    planes : tuple of str
        ``XY``, ``XZ`` or ``YZ``; each plane is one block of the files.
    count : int
        Number of sections of each plane.
    plot_direction : int
        1 or 2, the in-plane axis ``Section_direction_value`` runs along.
    integrate : bool
        Adds the strip columns and the integrated totals.
    """

    families: tuple[str, ...] = ()
    frame: str = "MRP"
    planes: tuple[str, ...] = ("XZ",)
    count: int = 50
    plot_direction: int = 1
    integrate: bool = False


@dataclass
class Station:
    """The result of one plane: the loads and the Cp rows."""

    section: int
    offset: float
    chord: float
    x_qc: float
    z_qc: float
    fx: float
    fz: float
    moment: float
    cp: dict[str, NDArray[np.float64]]
    strip_length: float | None = None


@dataclass
class PlaneBlock:
    """The stations of one plane of one distribution."""

    plane: str
    stations: list[Station]
    integrated: bool = False


@dataclass
class SectionsResult:
    """Every plane of one distribution, the columns to write and what was left out."""

    family_label: str
    blocks: list[PlaneBlock]
    cp_columns: tuple[str, ...]
    sloads_columns: tuple[str, ...]
    skips: list[str] = field(default_factory=list)


def strip_lengths(offsets: Sequence[float]) -> list[float]:
    """Return the strip length of each station by the strip rule.

    Interior edges lie at midpoints, the outer edges half an adjacent spacing
    beyond the end stations. The offsets must be strictly monotonic and at least
    two.
    """
    s = np.asarray(offsets, dtype=np.float64)
    if s.size < 2:
        raise SectionError("integration needs at least two stations in each block")
    if not np.all(np.isfinite(s)):
        raise SectionError("integration needs finite offsets")
    gap = np.diff(s)
    if not (np.all(gap > 0) or np.all(gap < 0)):
        raise SectionError("integration needs strictly monotonic Offset in each block")
    half = np.abs(gap) / 2.0
    return [float(v) for v in np.concatenate(([2 * half[0]], half[:-1] + half[1:], [2 * half[-1]]))]


def distribution_names(selections: Sequence[str | Sequence[str]]) -> list[str]:
    """Return the file-name word of each distribution entry.

    A word is the entry's ``families`` word, a list joined by ``-``; characters a
    file name cannot hold become ``_`` and trailing spaces and dots are removed.
    Names that collide, without regard to case, take ``_<k>`` (the entry's 1-based
    position) until they are unique.
    """
    names = [
        re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s if isinstance(s, str) else "-".join(s)).rstrip(" .")
        or "distribution"
        for s in selections
    ]
    while True:
        counts = Counter(n.casefold() for n in names)
        clashes = {i for i, n in enumerate(names) if counts[n.casefold()] > 1}
        if not clashes:
            return names
        names = [f"{n}_{i + 1}" if i in clashes else n for i, n in enumerate(names)]


def _chains(
    keys_a: NDArray[np.int64], keys_b: NDArray[np.int64]
) -> list[tuple[list[int], list[int]]]:
    """Order segments into chains; return each as (segment indices, point keys)."""
    adjacent: dict[int, list[int]] = {}
    for i, (a, b) in enumerate(zip(keys_a.tolist(), keys_b.tolist(), strict=True)):
        adjacent.setdefault(a, []).append(i)
        adjacent.setdefault(b, []).append(i)
    used = [False] * len(keys_a)
    out: list[tuple[list[int], list[int]]] = []

    def walk(start_key: int, closed: bool) -> None:
        segs: list[int] = []
        keys = [start_key]
        key = start_key
        while True:
            nxt = next((s for s in adjacent[key] if not used[s]), None)
            if nxt is None:
                break
            used[nxt] = True
            segs.append(nxt)
            key = int(keys_b[nxt] if keys_a[nxt] == key else keys_a[nxt])
            if closed and key == start_key:
                break
            keys.append(key)
        out.append((segs, keys))

    for key in sorted(k for k, v in adjacent.items() if len(v) == 1):
        if any(not used[s] for s in adjacent[key]):
            walk(key, False)
    for i in range(len(keys_a)):
        if not used[i]:
            walk(int(keys_a[i]), True)
    return out


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def compute_sections(
    surface: Surface,
    reference: Reference,
    distribution: Distribution,
    *,
    frames: Mapping[str, Frame] | None = None,
    fillable: Collection[str] | None = None,
    moment_point: Sequence[float] = (0.0, 0.0, 0.0),
    include_symmetry: bool = False,
) -> SectionsResult:
    """Cut ``surface`` by the planes of ``distribution`` and compute the station tables.

    Parameters
    ----------
    surface, reference, distribution
        The surface, the Cp reference quantities and the declared entry.
    frames : mapping of str to Frame, optional
        Frames by name; ``MRP`` is the reference frame when not given.
    fillable : collection of str, optional
        The columns of the Cp and sectional-loads tables the solver can fill.
        Columns outside it are absent from the files. Every column when None.
    moment_point : sequence of float
        The moment reference point, in reference coordinates.
    include_symmetry : bool
        Refused when true: no symmetry copy is made here.

    Returns
    -------
    SectionsResult

    Raises
    ------
    SectionError
        An unknown frame, plane or family, an unsupported option, or a fillable
        column the surface carries no data for.
    """
    if include_symmetry:
        raise SectionError("include_symmetry = true is not supported: no symmetry copy is made")
    if distribution.plot_direction not in (1, 2):
        raise SectionError(f"plot_direction must be 1 or 2, not {distribution.plot_direction!r}")
    if distribution.count < 1:
        raise SectionError("count must be at least 1")
    frame_map = {"MRP": Frame(), **(dict(frames) if frames else {})}
    if distribution.frame not in frame_map:
        raise SectionError(f"unknown frame {distribution.frame!r}")
    frame = frame_map[distribution.frame]

    all_cp = set(CP_COLUMNS)
    all_loads = set(SLOADS_COLUMNS) | set(INTEGRATED_COLUMNS)
    if fillable is None:
        have_v = "vtot" in surface.fields or all(c in surface.fields for c in ("vx", "vy", "vz"))
        wanted = {c for c in all_cp | all_loads if c not in _FIELD_COLUMNS or c in surface.fields}
        if not have_v:
            wanted.discard("vtot")
    else:
        wanted = set(fillable)
    cp_columns = tuple(c for c in CP_COLUMNS if c in wanted)
    sloads_columns = tuple(
        c
        for c in (*SLOADS_COLUMNS, *(INTEGRATED_COLUMNS if distribution.integrate else ()))
        if c in wanted
    )
    for column in cp_columns:
        if column in _FIELD_COLUMNS and column not in surface.fields:
            raise SectionError(
                f"column {column!r} is declared fillable but the surface has no field {column!r}"
            )
    if "vtot" in cp_columns and "vtot" not in surface.fields:
        if not all(c in surface.fields for c in ("vx", "vy", "vz")):
            raise SectionError("column 'vtot' needs the field 'vtot' or the fields vx, vy and vz")

    # cells of the selected families as triangles (a quad is split along 0-2)
    index = {name: i for i, name in enumerate(surface.families)}
    for name in distribution.families:
        if name not in index:
            raise SectionError(f"family {name!r} is not in the surface (it has {sorted(index)})")
    chosen = {index[n] for n in distribution.families}
    tri_family = (
        np.zeros(len(surface.triangles), dtype=np.int64)
        if surface.triangle_family is None
        else np.asarray(surface.triangle_family)
    )
    quad_family = (
        np.zeros(len(surface.quads), dtype=np.int64)
        if surface.quad_family is None
        else np.asarray(surface.quad_family)
    )
    quads = np.asarray(surface.quads, dtype=np.int64).reshape(-1, 4)
    tris = np.concatenate(
        [
            np.asarray(surface.triangles, dtype=np.int64).reshape(-1, 3),
            quads[:, [0, 1, 2]],
            quads[:, [0, 2, 3]],
        ]
    )
    fam = np.concatenate([tri_family, quad_family, quad_family])
    if chosen:
        keep = np.isin(fam, list(chosen))
        tris = tris[keep]
    if len(tris) == 0:
        raise SectionError("the selected families have no cells")

    pts = np.asarray(surface.points, dtype=np.float64)
    fpts = frame.to_frame(pts)
    cross = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
    normals = _unit(cross) * float(surface.normal_sign)
    degenerate = np.linalg.norm(cross, axis=1) == 0.0

    # per-point quantities that are interpolated along cut edges
    names: list[str] = ["pressure", *(f for f in _FIELD_COLUMNS if f in surface.fields)]
    for extra in ("vtot", "Cp_ref"):
        if extra in surface.fields:
            names.append(extra)
    quantity = np.column_stack(
        [np.asarray(surface.pressure, dtype=np.float64)]
        + [np.asarray(surface.fields[n], dtype=np.float64) for n in names[1:]]
    )

    mp = np.asarray(moment_point, dtype=np.float64)
    result = SectionsResult(
        family_label="+".join(distribution.families) or "all",
        blocks=[],
        cp_columns=cp_columns,
        sloads_columns=sloads_columns,
    )
    for plane in distribution.planes:
        if plane not in _PLANES:
            raise SectionError(f"unknown plane {plane!r} (XY, XZ or YZ)")
        a1, a2, na = _PLANES[plane]
        used = np.unique(tris[~degenerate])
        lo, hi = float(fpts[used, na].min()), float(fpts[used, na].max())
        width = (hi - lo) / distribution.count
        block = PlaneBlock(plane=plane, stations=[])
        for i in range(distribution.count):
            offset = lo + (i + 0.5) * width
            station = _station(
                i + 1,
                offset,
                (a1, a2, na),
                distribution.plot_direction,
                tris[~degenerate],
                normals[~degenerate],
                pts,
                fpts,
                quantity,
                names,
                frame,
                reference,
                mp,
            )
            if station is None:
                result.skips.append(f"{plane} section {i + 1} at offset {offset!r}: no cut")
                continue
            block.stations.append(station)
        if distribution.integrate:
            try:
                lengths = strip_lengths([s.offset for s in block.stations])
                for s, length in zip(block.stations, lengths, strict=True):
                    s.strip_length = length
                block.integrated = True
            except SectionError as exc:
                result.skips.append(f"{plane} #integration: {exc}")
        result.blocks.append(block)
    if distribution.integrate and not all(b.integrated for b in result.blocks):
        result.sloads_columns = tuple(c for c in sloads_columns if c not in INTEGRATED_COLUMNS)
    if not any(b.stations for b in result.blocks):
        result.skips.append("no stations: the planes of the distribution miss the surface")
    return result


def _station(
    section: int,
    offset: float,
    axes: tuple[int, int, int],
    plot_direction: int,
    tris: NDArray[np.int64],
    normals: NDArray[np.float64],
    pts: NDArray[np.float64],
    fpts: NDArray[np.float64],
    quantity: NDArray[np.float64],
    names: list[str],
    frame: Frame,
    reference: Reference,
    moment_point: NDArray[np.float64],
) -> Station | None:
    a1, a2, na = axes
    d = fpts[tris, na] - offset
    above = d >= 0.0
    count_above = above.sum(axis=1)
    cut = np.flatnonzero((count_above > 0) & (count_above < 3))
    if cut.size == 0:
        return None
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    crossing = above[cut][:, edges[:, 0]] != above[cut][:, edges[:, 1]]
    pick = np.argsort(~crossing, axis=1, kind="stable")[:, :2]
    n_pts = len(pts)
    end_keys: list[NDArray[np.int64]] = []
    end_pts: list[NDArray[np.float64]] = []
    end_q: list[NDArray[np.float64]] = []
    for col in range(2):
        e = edges[pick[:, col]]
        va = tris[cut, e[:, 0]]
        vb = tris[cut, e[:, 1]]
        da = d[cut, e[:, 0]]
        db = d[cut, e[:, 1]]
        t = (da / (da - db))[:, None]
        on_a, on_b = (da == 0.0)[:, None], (db == 0.0)[:, None]
        end_pts.append(
            np.where(on_a, pts[va], np.where(on_b, pts[vb], pts[va] + t * (pts[vb] - pts[va])))
        )
        end_q.append(
            np.where(
                on_a,
                quantity[va],
                np.where(on_b, quantity[vb], quantity[va] + t * (quantity[vb] - quantity[va])),
            )
        )
        # a point on a vertex is that vertex whatever edge reached it; else the edge names it
        edge_key = n_pts + np.minimum(va, vb) * n_pts + np.maximum(va, vb)
        end_keys.append(np.where(da == 0.0, va, np.where(db == 0.0, vb, edge_key)))
    p0, p1 = end_pts
    q0, q1 = end_q
    seg_n = normals[cut]
    seg_len = np.linalg.norm(p1 - p0, axis=1)
    live = (seg_len > 0.0) & (end_keys[0] != end_keys[1])
    if not live.any():
        return None
    p0, p1, q0, q1, seg_n, seg_len = (x[live] for x in (p0, p1, q0, q1, seg_n, seg_len))
    k0, k1 = end_keys[0][live], end_keys[1][live]

    axis_n = frame.matrix()[na]
    ne = seg_n @ axis_n
    sin = np.sqrt(np.clip(1.0 - ne * ne, 0.0, None))
    ok = sin > math.sqrt(_PARALLEL_FACET)
    if not ok.any():
        return None
    weight = np.where(ok, seg_len / np.where(ok, sin, 1.0), 0.0)
    g0 = q0[:, 0] - reference.p_inf
    g1 = q1[:, 0] - reference.p_inf
    force = -((g0 + g1) / 2.0 * weight)[:, None] * seg_n
    total = force.sum(axis=0)
    r0 = p0 - moment_point
    r1 = p1 - moment_point
    integral = (
        2 * r0 * g0[:, None] + r0 * g1[:, None] + r1 * g0[:, None] + 2 * r1 * g1[:, None]
    ) / 6.0
    moment_vec = -(np.cross(integral, seg_n) * weight[:, None]).sum(axis=0)
    moment = float(moment_vec @ axis_n)
    mat = frame.matrix()
    fx = float(total @ mat[a1])
    fz = float(total @ mat[a2])

    # chord from the cut: the farthest pair of cut points
    ends_f = frame.to_frame(np.concatenate([p0, p1]))[:, [a1, a2]]
    uniq = np.unique(ends_f, axis=0)
    chord, lead, trail = _diameter(uniq, plot_direction - 1)
    qc = lead + QUARTER_CHORD * (trail - lead)

    chains = _chains(k0, k1)
    rows: list[dict[str, NDArray[np.float64]]] = []
    for segs, keys in chains:
        rows.append(
            _chain_rows(
                segs,
                keys,
                k0,
                k1,
                p0,
                p1,
                q0,
                q1,
                seg_n,
                names,
                frame,
                (a1, a2),
                plot_direction,
                reference,
            )
        )
    cp = {c: np.concatenate([r[c] for r in rows]) for c in rows[0]}
    return Station(
        section=section,
        offset=offset,
        chord=chord,
        x_qc=float(qc[0]),
        z_qc=float(qc[1]),
        fx=fx,
        fz=fz,
        moment=moment,
        cp=cp,
    )


def _diameter(
    pts2: NDArray[np.float64], direction: int
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return the largest distance between two points, with its leading and trailing point."""
    best = -1.0
    pair = (0, 0)
    for start in range(0, len(pts2), 512):
        block = pts2[start : start + 512]
        dist = np.linalg.norm(block[:, None, :] - pts2[None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmax(dist)), dist.shape)
        if dist[i, j] > best:
            best, pair = float(dist[i, j]), (start + int(i), int(j))
    a, b = pts2[pair[0]], pts2[pair[1]]
    return (best, a, b) if a[direction] <= b[direction] else (best, b, a)


def _chain_rows(
    segs: list[int],
    keys: list[int],
    k0: NDArray[np.int64],
    k1: NDArray[np.int64],
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    q0: NDArray[np.float64],
    q1: NDArray[np.float64],
    seg_n: NDArray[np.float64],
    names: list[str],
    frame: Frame,
    axes: tuple[int, int],
    plot_direction: int,
    reference: Reference,
) -> dict[str, NDArray[np.float64]]:
    """Order one chain into Cp rows: closed loops counter-clockwise, from the lowest plot value."""
    point: dict[int, NDArray[np.float64]] = {}
    value: dict[int, NDArray[np.float64]] = {}
    normal: dict[int, NDArray[np.float64]] = {}
    for s in segs:
        for key, pp, qq in ((int(k0[s]), p0[s], q0[s]), (int(k1[s]), p1[s], q1[s])):
            point.setdefault(key, pp)
            value.setdefault(key, qq)
            normal[key] = normal[key] + seg_n[s] if key in normal else seg_n[s].copy()
    order = list(keys)
    closed = len(segs) == len(order)
    fp = frame.to_frame(np.array([point[k] for k in order]))[:, list(axes)]
    direction = plot_direction - 1
    if closed:
        area = 0.5 * float(
            np.sum(fp[:, 0] * np.roll(fp[:, 1], -1) - np.roll(fp[:, 0], -1) * fp[:, 1])
        )
        if area < 0:
            order = order[::-1]
            fp = fp[::-1]
        first = int(np.argmin(fp[:, direction]))
        order = order[first:] + order[:first]
    elif fp[0, direction] > fp[-1, direction]:
        order = order[::-1]
    xyz = np.array([point[k] for k in order])
    fpo = frame.to_frame(xyz)[:, list(axes)]
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(fpo, axis=0), axis=1))))
    qv = np.array([value[k] for k in order])
    nrm = _unit(np.array([normal[k] for k in order]))
    col = {n: qv[:, i] for i, n in enumerate(names)}
    cp = (col["pressure"] - reference.p_inf) / reference.q_ref
    out: dict[str, NDArray[np.float64]] = {
        "Section_direction_value": fpo[:, direction],
        "X": xyz[:, 0],
        "Y": xyz[:, 1],
        "Z": xyz[:, 2],
        "nx": nrm[:, 0],
        "ny": nrm[:, 1],
        "nz": nrm[:, 2],
        "L": arc,
        "Cp": cp,
    }
    for name in _FIELD_COLUMNS:
        if name in col:
            out[name] = col[name]
    if "vtot" in col:
        out["vtot"] = col["vtot"]
    elif all(c in col for c in ("vx", "vy", "vz")):
        out["vtot"] = np.sqrt(col["vx"] ** 2 + col["vy"] ** 2 + col["vz"] ** 2)
    out["Cp_ref"] = col["Cp_ref"] if "Cp_ref" in col else cp
    return out


def _fmt(value: float) -> str:
    return repr(float(value))


def write_section_files(
    result: SectionsResult,
    directory: str | Path,
    point: str,
    name: str,
    *,
    leading: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write ``sections/<point>_cp_<name>.csv`` and ``sections/<point>_sloads_<name>.csv``.

    Parameters
    ----------
    result : SectionsResult
        From :func:`compute_sections`.
    directory : path
        The folder that holds ``sections/``.
    point, name : str
        The point label and the distribution's file-name word
        (see :func:`distribution_names`).
    leading : mapping of str to str, optional
        Identity and condition columns written first, in order, with one value
        for every row. ``FAMILY`` and ``PLANE``, when present, take the value of
        the block.

    Returns
    -------
    tuple of Path
        The Cp file and the sectional-loads file. The text is UTF-8 with line feed
        ends, the same input giving the same bytes.
    """
    folder = Path(directory) / "sections"
    folder.mkdir(parents=True, exist_ok=True)
    lead = dict(leading or {})

    def prefix(plane: str) -> list[str]:
        cells = dict(lead)
        if "FAMILY" in cells:
            cells["FAMILY"] = result.family_label
        if "PLANE" in cells:
            cells["PLANE"] = plane
        return list(cells.values())

    cp_lines = [",".join([*lead, "SECTION", *result.cp_columns])]
    sl_lines = [",".join([*lead, *result.sloads_columns])]
    loads_of = {
        "Offset": lambda s: s.offset,
        "Chord": lambda s: s.chord,
        "X_QC": lambda s: s.x_qc,
        "Z_QC": lambda s: s.z_qc,
        "Fx": lambda s: s.fx,
        "Fz": lambda s: s.fz,
        "Moment": lambda s: s.moment,
        "Strip_length": lambda s: s.strip_length,
        "Fx_int": lambda s: s.fx * s.strip_length,
        "Fz_int": lambda s: s.fz * s.strip_length,
        "My_int": lambda s: s.moment * s.strip_length,
    }
    for block in result.blocks:
        pre = prefix(block.plane)
        for st in block.stations:
            n_rows = len(st.cp["L"])
            for r in range(n_rows):
                cells = [_fmt(st.cp[c][r]) for c in result.cp_columns]
                cp_lines.append(",".join([*pre, str(st.section), *cells]))
            sl_lines.append(
                ",".join([*pre, *(_fmt(loads_of[c](st)) for c in result.sloads_columns)])
            )
    cp_path = folder / f"{point}_cp_{name}.csv"
    sl_path = folder / f"{point}_sloads_{name}.csv"
    cp_path.write_bytes(("\n".join(cp_lines) + "\n").encode("utf-8"))
    sl_path.write_bytes(("\n".join(sl_lines) + "\n").encode("utf-8"))
    return cp_path, sl_path
