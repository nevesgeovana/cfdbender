"""A full wheel from one sector of an OBJ.

Pipeline role: the mesh operation behind ``cfd-mesh sector-to-wheel``. A sector holds one blade, the
hub surfaces between the two seams of its angular span and the declarations
of its boundaries; the wheel is the sector copied ``copies`` times about an
axis, with the nodes of the seams merged, the blade surfaces numbered one per
copy and the other surfaces kept as one group each.

THE METHOD, for an OBJ (:func:`sector_to_wheel`)

* The sector's surfaces are its OBJ groups, in file order. A trailing-edge
  entry of its boundaries file names mid-points of mesh edges; the points are
  matched to the edges of the mesh within the entry's tolerance, and the
  surface the matched edges lie in is the surface the entry rims.
* A surface with two or more rim entries that stays one piece when the faces
  are joined only across edges that are not rim edges has a blunt trailing
  edge: its base is the faces whose every node lies on a rim line (the
  three-node rule for a triangle). A surface the rim edges cut into two pieces
  (the aft base of a body) has the smaller piece as its base. One rim entry
  and one piece is a sharp trailing edge: no base is made.
* Copy ``k`` is the sector turned right-handed about the axis through the hub
  by ``360 k / copies`` degrees. Nodes of all copies closer than the tolerance
  are one node (a grid search and a union-find, linear in the nodes). Faces
  keep the sector's order inside each copy.
* The wheel's boundaries file is written in the named form: one
  ``[trailing_edges.<label>_<k>]`` and one ``[base_regions.Base<k>]`` per copy
  of a blade surface, one entry for each other surface.

Nothing in the method fixes a name: the surfaces that are blades are the ones
``blades`` names, and every other surface is shared by the copies.

Tolerance is in meters. The mesh unit comes from the boundaries file's
``[import] units``; one that is not stated, or that names no scale, is warned
and read as meters.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._digest import file_sha256
from cfdbender._errors import (
    GeometryWarning,
    InputArtifactError,
    InputValueError,
    NotApplicableWarning,
    warn,
)
from cfdbender._lengths import METRES_PER_UNIT
from cfdbender.mesh._refine._obj import (
    ObjMesh,
    obj_text,
    points_from_file,
    read_obj,
    read_sidecar,
    write_te,
)
from cfdbender.mesh._sidecars import (
    face_count_lines,
    inventory_sidecar,
    read_boundary_declarations,
    toml_value,
)

__all__ = [
    "DEFAULT_TOLERANCE_M",
    "WheelResult",
    "sector_to_wheel",
]

#: The default merge tolerance, in meters: a seam that sits 1e-5 m apart is merged by it and
#: so is one that sits 1e-4 m apart, which a seam of a mesh saved in single precision can.
DEFAULT_TOLERANCE_M = 1e-4

_AXES = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}
_PRIME = numpy.array([73856093, 19349663, 83492791], dtype=numpy.int64)
_CELLS = numpy.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=numpy.int64)
_NOTHING = "No file was written."
_ANGLE_SLACK_DEG = 0.05  # a sector this close to 360/N closes within the rounding of saved nodes
#: The keys of a sector's boundaries file the wheel's states again, from the wheel or the sector.
_CARRIED = frozenset(
    {
        "base_regions",
        "boundaries",
        "boundary_faces",
        "file",
        "import",
        "mesh_faces",
        "mesh_sha256",
        "trailing_edges",
    }
)
#: Pairs a neighbor search may compare: a node of a seam has one partner per copy and a cell a
#: few nodes, so a real merge stays far under this, and the floor keeps a small mesh out of it.
_PAIRS_FLOOR, _PAIRS_PER_POINT = 1_000_000, 64


@dataclass
class WheelResult:
    """What a sector-to-wheel run did: whether it wrote, the files and the checks.

    Attributes
    ----------
    written : bool
        False when nothing was written.
    output : Path or None
        The wheel OBJ; None when nothing was written.
    files : list of Path
        Every file written: the wheel, its boundaries file and rim points files for an OBJ,
        and ``<out stem>.wheel.json``.
    report : dict
        What ``<out stem>.wheel.json`` holds: the source sha256, the copies, axis, hub and
        tolerance, the nodes merged, the open edges, the area ratio and the turn check.
    """

    written: bool
    output: Path | None = None
    files: list[Path] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def lines(self) -> list[str]:
        """Return the printed summary: the files written, then the checks."""
        out = [f"wrote {path}" for path in self.files]
        report = self.report
        if not report:
            return out
        merged = report["nodes"]
        out.append(
            f"nodes: {merged['unmerged']} unmerged, {merged['wheel']} in the wheel, "
            f"{merged['merged']} merged at {report['tolerance_m']:g} m"
        )
        out.append(f"degenerate faces removed: {report['degenerate_faces']}")
        for key in ("open_edges", "non_manifold_edges"):
            e = report[key]
            out.append(
                f"{key.replace('_', ' ')}: sector {e['sector']}, unmerged wheel {e['unmerged']}, "
                f"merged wheel {e['merged']}"
            )
        out.append(f"area ratio, wheel over copies x sector: {report['area_ratio']:.15f}")
        for tag in ("invariance", "control"):
            t = report[tag]
            out.append(
                f"{tag} {t['angle_deg']:g} deg: nodes mapped {t['nodes_mapped']} of {t['nodes']}, "
                f"faces mapped {t['faces_mapped']} of {t['faces']}"
            )
        return out


# --- the arithmetic every route shares ------------------------------------------------------


def parse_axis(axis: str | Sequence[float]) -> NDArray[numpy.float64]:
    """Return the unit vector of an axis: ``x``, ``y``, ``z``, ``-x``... or three numbers."""
    if isinstance(axis, str):
        key = axis.strip().lower()
        sign = -1.0 if key.startswith("-") else 1.0
        if key.lstrip("+-") in _AXES:
            vector = [sign * c for c in _AXES[key.lstrip("+-")]]
        else:
            try:
                vector = [float(w) for w in key.replace(" ", "").split(",")]
            except ValueError:
                vector = []
    else:
        vector = [float(c) for c in axis]
    array = numpy.asarray(vector, dtype=float)
    if array.shape != (3,) or not numpy.isfinite(array).all() or not numpy.any(array):
        raise InputValueError(
            f"axis {axis!r} is not a direction; write x, y, z, -x, -y or -z, or three comma-"
            f"separated numbers such as 1,0,0. {_NOTHING}"
        )
    array /= numpy.max(numpy.abs(array))
    return array / numpy.linalg.norm(array)


def rotate(
    points: NDArray[numpy.float64],
    axis: NDArray[numpy.float64],
    hub: NDArray[numpy.float64],
    degrees: float,
) -> NDArray[numpy.float64]:
    """Turn points right-handed about the axis through the hub by an angle (Rodrigues)."""
    angle = math.radians(degrees)
    q = points - hub
    cross = numpy.cross(axis, q)
    turned = (
        q * math.cos(angle)
        + cross * math.sin(angle)
        + numpy.outer(q @ axis, axis) * (1.0 - math.cos(angle))
    )
    return turned + hub


def azimuth_range(psi: NDArray[numpy.float64]) -> tuple[float, float]:
    """Return the middle azimuth and the angle of a set of azimuths, in radians.

    The angle is the turn less the largest gap between neighboring azimuths, so it does not
    depend on which azimuth comes first: a difference measured from one node of a seam reads
    half a turn as plus or minus half a turn according to the rounding of its last digit.
    """
    turn = 2.0 * math.pi
    ordered = numpy.sort(numpy.mod(psi, turn))
    gaps = numpy.diff(numpy.r_[ordered, ordered[0] + turn])
    widest = int(numpy.argmax(gaps))
    span = turn - float(gaps[widest])
    start = float(ordered[(widest + 1) % len(ordered)])
    return math.remainder(start + 0.5 * span, turn), span


def _keys(cells: NDArray[numpy.int64]) -> NDArray[numpy.int64]:
    """Hash grid cells to one integer each (a collision only adds a candidate pair)."""
    scaled = cells * _PRIME
    return scaled[:, 0] ^ scaled[:, 1] ^ scaled[:, 2]


def neighbour_pairs(
    a: NDArray[numpy.float64], b: NDArray[numpy.float64], tol: float
) -> tuple[NDArray[numpy.int64], NDArray[numpy.int64], NDArray[numpy.float64]]:
    """Return every pair (i of a, j of b) within ``tol`` and its distance, by a grid.

    The cost is linear in the points and the pairs: a point is compared only with the
    points of its own cell and the 26 around it.

    Raises
    ------
    InputValueError
        The distance puts so many pairs in reach that the search would not fit in memory.
    """
    empty = numpy.zeros(0, dtype=numpy.int64)
    if not len(a) or not len(b) or not tol > 0.0:
        return empty, empty, numpy.zeros(0)
    cell_a = numpy.floor(a / tol).astype(numpy.int64)
    key_b = _keys(numpy.floor(b / tol).astype(numpy.int64))
    order = numpy.argsort(key_b, kind="stable")
    sorted_b = key_b[order]
    found: list[NDArray[numpy.int64]] = []
    reach, limit = 0, _PAIRS_FLOOR + _PAIRS_PER_POINT * (len(a) + len(b))
    for offset in _CELLS:
        key = _keys(cell_a + offset)
        low = numpy.searchsorted(sorted_b, key, "left")
        count = numpy.searchsorted(sorted_b, key, "right") - low
        total = int(count.sum())
        if not total:
            continue
        reach += total
        if reach > limit:
            raise InputValueError(
                f"a distance of {tol:g}, in the mesh's unit, puts over {limit} pairs of its "
                f"{len(a)} points in reach of each other, so the mesh would collapse and the "
                "search does not fit in memory. Lower the distance; for the merge of a wheel "
                f"it is tolerance (CLI: --tolerance), in meters. {_NOTHING}"
            )
        i = numpy.repeat(numpy.arange(len(a)), count)
        start = numpy.repeat(low - (numpy.cumsum(count) - count), count)
        j = order[start + numpy.arange(total)]
        near = numpy.linalg.norm(a[i] - b[j], axis=1) <= tol
        found.append(i[near] * len(b) + j[near])
    if not found:
        return empty, empty, numpy.zeros(0)
    both = numpy.unique(numpy.concatenate(found))
    i, j = both // len(b), both % len(b)
    return i, j, numpy.linalg.norm(a[i] - b[j], axis=1)


def merge_nodes(points: NDArray[numpy.float64], tol: float) -> NDArray[numpy.int64]:
    """Return, for every point, the first point of the chain of points within ``tol`` of it."""
    parent = list(range(len(points)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    i, j, _ = neighbour_pairs(points, points, tol)
    for a, b in zip(i[i < j].tolist(), j[i < j].tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return numpy.array([find(k) for k in range(len(points))], dtype=numpy.int64)


def nearest(
    points: NDArray[numpy.float64], targets: NDArray[numpy.float64], tol: float
) -> NDArray[numpy.int64]:
    """Return the index of the nearest target within ``tol`` of each point, or -1."""
    i, j, d = neighbour_pairs(points, targets, tol)
    out = numpy.full(len(points), -1, dtype=numpy.int64)
    order = numpy.lexsort((d, i))
    i, j = i[order], j[order]
    first = numpy.concatenate(([True], i[1:] != i[:-1])) if len(i) else numpy.zeros(0, bool)
    out[i[first]] = j[first]
    return out


@dataclass
class Faces:
    """Polygons as one flat index array and the size of each, so no loop runs over a face."""

    flat: NDArray[numpy.int64]
    sizes: NDArray[numpy.int64]

    @classmethod
    def of(cls, faces: Iterable[Sequence[int]]) -> Faces:
        """Return the polygons of a list of index lists."""
        rows = list(faces)
        flat = numpy.fromiter(itertools.chain.from_iterable(rows), dtype=numpy.int64)
        return cls(flat, numpy.fromiter(map(len, rows), dtype=numpy.int64, count=len(rows)))

    def __len__(self) -> int:
        return len(self.sizes)

    @property
    def starts(self) -> NDArray[numpy.int64]:
        """The position of each face's first index in ``flat``."""
        return numpy.cumsum(self.sizes) - self.sizes

    def pick(self, mask: NDArray[numpy.bool_]) -> Faces:
        """Return the faces a mask keeps, in order."""
        return Faces(self.flat[numpy.repeat(mask, self.sizes)], self.sizes[mask])

    def mapped(self, node: NDArray[numpy.int64], offset: int = 0) -> Faces:
        """Return the faces with every index sent through ``node`` after adding ``offset``."""
        return Faces(node[self.flat + offset], self.sizes)

    def lists(self) -> list[list[int]]:
        """Return the faces as index lists."""
        pieces = numpy.split(self.flat, numpy.cumsum(self.sizes)[:-1]) if len(self) else []
        return [p.tolist() for p in pieces]

    def edges(self) -> tuple[NDArray[numpy.int64], NDArray[numpy.int64], NDArray[numpy.int64]]:
        """Return each directed edge as (from, to, owning face)."""
        ends = numpy.cumsum(self.sizes)
        following = numpy.arange(len(self.flat)) + 1
        following[ends - 1] = ends - self.sizes
        return self.flat, self.flat[following], numpy.repeat(numpy.arange(len(self)), self.sizes)

    def repeated(self) -> NDArray[numpy.bool_]:
        """Return which faces name one node twice (a face the merge collapsed)."""
        out = numpy.zeros(len(self), dtype=bool)
        for size in numpy.unique(self.sizes):
            rows = numpy.flatnonzero(self.sizes == size)
            block = numpy.sort(self.flat[self.starts[rows][:, None] + numpy.arange(size)], axis=1)
            out[rows] = (numpy.diff(block, axis=1) == 0).any(axis=1)
        return out

    def area(self, verts: NDArray[numpy.float64]) -> float:
        """Return the summed area, each polygon a fan about its first node."""
        total = 0.0
        for size in numpy.unique(self.sizes):
            rows = numpy.flatnonzero(self.sizes == size)
            block = self.flat[self.starts[rows][:, None] + numpy.arange(size)]
            for k in range(1, int(size) - 1):
                u = verts[block[:, k]] - verts[block[:, 0]]
                v = verts[block[:, k + 1]] - verts[block[:, 0]]
                total += float(numpy.linalg.norm(numpy.cross(u, v), axis=1).sum() / 2.0)
        return total


def concat(parts: Sequence[Faces]) -> Faces:
    """Return the faces of several sets, one after the other."""
    if not parts:
        return Faces(numpy.zeros(0, dtype=numpy.int64), numpy.zeros(0, dtype=numpy.int64))
    return Faces(
        numpy.concatenate([p.flat for p in parts]), numpy.concatenate([p.sizes for p in parts])
    )


def edge_keys(faces: Faces, count: int) -> NDArray[numpy.int64]:
    """Return one integer per undirected edge occurrence, equal for the two directions."""
    a, b, _ = faces.edges()
    return numpy.minimum(a, b) * count + numpy.maximum(a, b)


def edge_counts(faces: Faces, count: int) -> tuple[int, int]:
    """Return the number of open edges (one face) and non-manifold edges (more than two)."""
    _, times = numpy.unique(edge_keys(faces, count), return_counts=True)
    return int((times == 1).sum()), int((times > 2).sum())


def components(count: int, left: NDArray[numpy.int64], right: NDArray[numpy.int64]) -> list[int]:
    """Return the component label of each of ``count`` items joined by the pairs."""
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(count)]


def pieces(faces: Faces, rim: NDArray[numpy.int64], count: int) -> list[NDArray[numpy.bool_]]:
    """Return the face masks of the pieces joined across every edge that is not a rim edge."""
    a, b, face_of = faces.edges()
    keys = numpy.minimum(a, b) * count + numpy.maximum(a, b)
    keep = ~numpy.isin(keys, rim)
    keys, face_of = keys[keep], face_of[keep]
    order = numpy.argsort(keys, kind="stable")
    keys, face_of = keys[order], face_of[order]
    same = keys[1:] == keys[:-1]
    label = numpy.array(components(len(faces), face_of[:-1][same], face_of[1:][same]))
    return [label == k for k in numpy.unique(label)] if len(faces) else []


def _invariance(
    nodes: NDArray[numpy.float64],
    faces: Faces,
    axis: NDArray[numpy.float64],
    hub: NDArray[numpy.float64],
    *,
    degrees: float,
    tol: float,
) -> dict[str, Any]:
    """Count the nodes and faces a turn by an angle maps onto the wheel's own."""
    image = nearest(rotate(nodes, axis, hub, degrees), nodes, tol)
    flat = image[faces.flat]
    ok = numpy.add.reduceat((flat < 0).astype(int), faces.starts) == 0 if len(faces) else []
    known = {tuple(sorted(f)) for f in faces.lists()}
    mapped = sum(
        1
        for f, fine in zip(Faces(flat, faces.sizes).lists(), ok, strict=True)
        if fine and tuple(sorted(f)) in known
    )
    return {
        "angle_deg": degrees,
        "nodes": len(nodes),
        "nodes_mapped": int((image >= 0).sum()),
        "faces": len(faces),
        "faces_mapped": mapped,
    }


def wheel_symmetry(
    nodes: NDArray[numpy.float64],
    faces: Faces,
    axis: NDArray[numpy.float64],
    hub: NDArray[numpy.float64],
    *,
    copies: int,
    tol: float,
) -> dict[str, dict[str, Any]]:
    """Return the N-fold invariance of the wheel and the control turn that must fail it."""
    step = 360.0 / copies
    return {
        "invariance": _invariance(nodes, faces, axis, hub, degrees=step, tol=tol),
        "control": _invariance(nodes, faces, axis, hub, degrees=step + 1.0, tol=tol),
    }


def refuse_existing(paths: Iterable[Path], overwrite: bool) -> None:
    """Refuse, before any file is written, an output that exists and is not to be replaced."""
    clash = [p for p in paths if p.exists()]
    if clash and not overwrite:
        raise InputArtifactError(
            f"{', '.join(p.name for p in clash)} already exist(s); pass --overwrite (overwrite="
            f"True) to replace them, or name another out (CLI: --out). {_NOTHING}"
        )


def refuse_the_source(target: Path, source: Path) -> None:
    """Refuse an output that is the sector read: it would be lost and its record would be wrong."""
    if target.exists() and target.samefile(source):
        raise InputArtifactError(
            f"{target.name} is the sector itself: it would be replaced, and the record would "
            f"state the output's digest as the source's. Name another out (CLI: --out). {_NOTHING}"
        )


def refuse_a_collapsed_wheel(faces_left: int, tolerance_m: float) -> None:
    """Refuse a wheel whose every face the tolerance collapsed: that result is no wheel."""
    if not faces_left:
        raise InputValueError(
            f"tolerance (CLI: --tolerance) {tolerance_m:g} m joins the nodes of every face of "
            f"the wheel, so no face is left. Lower it. {_NOTHING}"
        )


def unit_of(declared: str | None, where: str) -> float:
    """Return the meters per mesh unit, warning when the unit is not stated or is unknown."""
    if declared is not None and declared in METRES_PER_UNIT:
        return float(METRES_PER_UNIT[declared])
    cause = "states no [import] units" if declared is None else f"states the unit {declared!r}"
    warn(
        f"{where} {cause}, which names no length scale the package knows; the tolerance, the "
        "mesh and its points are read as meters.",
        GeometryWarning,
        stacklevel=3,
    )
    return 1.0


def stem_of(family: str) -> str:
    """Return a family's name without its trailing digits, the stem a copy's name is built on."""
    return family.rstrip("0123456789")


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a wheel record as JSON through the package's text route."""
    _textio.write_text(path, json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path


# --- the OBJ route --------------------------------------------------------------------------


@dataclass
class _Rim:
    """One trailing-edge entry: its points files in the mesh's unit and what it rims."""

    label: str
    sets: list[tuple[str, NDArray[numpy.float64]]]
    unit: str
    to_file: float
    tolerance_m: float
    family: str = ""
    keys: NDArray[numpy.int64] = field(default_factory=lambda: numpy.zeros(0, dtype=numpy.int64))


@dataclass
class _Surface:
    """One surface of the wheel before it is copied: faces of the sector and its group names."""

    family: str
    faces: Faces
    replicated: bool
    names: list[str]
    parent: str | None = None
    generated: bool = False


def _rims(obj: Path, declarations: Any) -> list[_Rim]:
    """Read the trailing-edge entries of the sector's boundaries file, skipping what cannot copy."""
    rims: list[_Rim] = []
    for entry in declarations.trailing_edges:
        marking = entry.marking
        if marking.route != "file":
            warn(
                f"the trailing-edge entry {entry.label!r} marks by detection and names no points, "
                "so it cannot be copied about the axis; it is skipped. Declare it with a points "
                "file to carry it.",
                NotApplicableWarning,
                stacklevel=4,
            )
            continue
        sets: list[tuple[str, NDArray[numpy.float64]]] = []
        unit, to_file = "METER", 1.0
        for one in marking.sets:
            loaded = points_from_file(obj, Path(str(one.points_file)))
            assert loaded is not None
            sets.append((one.edge_type, loaded.points))
            unit, to_file = loaded.unit, loaded.to_file
        rims.append(
            _Rim(
                entry.label,
                sets,
                unit,
                to_file,
                marking.tolerance,
            )
        )
    return rims


def _match_rims(
    rims: list[_Rim],
    verts: NDArray[numpy.float64],
    families: dict[str, Faces],
    declared: set[str],
    metres: float,
) -> None:
    """Match each entry's points to mesh edges and find the surface each entry rims."""
    count = len(verts)
    whole = concat(list(families.values()))
    a, b, _ = whole.edges()
    keys = numpy.minimum(a, b) * count + numpy.maximum(a, b)
    unique, first = numpy.unique(keys, return_index=True)
    mids = 0.5 * (verts[a[first]] + verts[b[first]])
    per_family = {name: numpy.unique(edge_keys(f, count)) for name, f in families.items()}
    for rim in rims:
        points = numpy.vstack([p for _, p in rim.sets]) if rim.sets else numpy.zeros((0, 3))
        found = nearest(points, mids, rim.tolerance_m / metres)
        if (found < 0).any():
            warn(
                f"trailing-edge entry {rim.label!r}: {int((found < 0).sum())} of {len(points)} "
                f"points match no mesh edge within {rim.tolerance_m:g} m; those points are "
                "copied into the wheel's file and not used to split a base.",
                GeometryWarning,
                stacklevel=4,
            )
        rim.keys = numpy.unique(unique[found[found >= 0]])
        score = {n: int(numpy.isin(rim.keys, k).sum()) for n, k in per_family.items()}
        pool = {n: s for n, s in score.items() if n not in declared} or score
        rim.family = max(pool, key=lambda n: pool[n]) if pool and max(pool.values()) else ""


def _split(
    faces: Faces, rim: NDArray[numpy.int64], count: int, entries: int
) -> NDArray[numpy.bool_] | None:
    """Return the mask of a surface's base faces, or None when it has a sharp edge or none."""
    if not rim.size:
        return None
    parts = pieces(faces, rim, count)
    if len(parts) == 2:
        return min(parts, key=lambda m: int(m.sum()))
    if len(parts) != 1:
        warn(
            f"a surface falls into {len(parts)} pieces across its rim edges, and only one or two "
            "are read; it is copied whole and no base is split from it.",
            GeometryWarning,
            stacklevel=4,
        )
        return None
    if entries < 2:
        return None
    a, b, _ = faces.edges()
    hit = numpy.isin(numpy.minimum(a, b) * count + numpy.maximum(a, b), rim)
    on_rim = numpy.zeros(count, dtype=bool)
    on_rim[a[hit]] = True
    on_rim[b[hit]] = True
    inside = numpy.add.reduceat((~on_rim[faces.flat]).astype(int), faces.starts) == 0
    return inside if inside.any() and not inside.all() else None


def _base_name(blade: str, blades: Sequence[str], k: int) -> str:
    """Return a blade's base name: ``Base<k>`` for one blade family, else led by the stem."""
    lead = "" if len(blades) == 1 else stem_of(blade)
    return f"{lead}Base{k + 1}"


def _surfaces(
    families: dict[str, Faces],
    rims: list[_Rim],
    declared: dict[str, Any],
    *,
    blades: Sequence[str],
    copies: int,
    count: int,
) -> list[_Surface]:
    """Cut the sector's families into surfaces: each blade, its base, and the shared others."""
    out: list[_Surface] = []
    for name, faces in families.items():
        parent = declared[name].merge_into if name in declared else None
        parent = parent if parent in families else None
        mine = [r for r in rims if r.family == name]
        keys = numpy.unique(numpy.concatenate([r.keys for r in mine])) if mine else numpy.zeros(0)
        base = None if parent else _split(faces, keys.astype(numpy.int64), count, len(mine))
        if parent in blades:
            names = [_base_name(str(parent), blades, k) for k in range(copies)]
        elif name in blades:
            names = [f"{stem_of(name)}{k + 1}" for k in range(copies)]
        else:
            names = [name]
        replicated = name in blades or parent in blades
        if base is None:
            out.append(_Surface(name, faces, replicated, names, parent))
            continue
        out.append(_Surface(name, faces.pick(~base), replicated, names))
        added = (
            [_base_name(name, blades, k) for k in range(copies)]
            if name in blades
            else [f"{name}Base"]
        )
        out.append(_Surface(name, faces.pick(base), name in blades, added, name, True))
    return out


@dataclass
class Options:
    """What every route is asked: copies, blade surfaces, axis, hub, tolerance, output."""

    copies: int
    blades: list[str]
    axis: NDArray[numpy.float64]
    hub: NDArray[numpy.float64]
    tolerance_m: float
    out: Path
    overwrite: bool


def _assemble(
    verts: NDArray[numpy.float64], surfaces: list[_Surface], opts: Options, tol: float
) -> tuple[NDArray[numpy.float64], list[tuple[str, Faces]], dict[str, Any]]:
    """Copy and merge: the wheel's nodes, its groups in file order, and the counts of the merge."""
    count, n = len(verts), opts.copies
    stack = numpy.vstack([rotate(verts, opts.axis, opts.hub, 360.0 * k / n) for k in range(n)])
    root = merge_nodes(stack, tol)
    used = numpy.unique(root)
    index = numpy.full(len(stack), -1, dtype=numpy.int64)
    index[used] = numpy.arange(len(used))
    node = index[root]
    groups: list[tuple[str, Faces]] = []
    for s in [s for s in surfaces if s.replicated]:
        groups += [(s.names[k], s.faces.mapped(node, k * count)) for k in range(n)]
    for s in [s for s in surfaces if not s.replicated]:
        groups.append((s.names[0], concat([s.faces.mapped(node, k * count) for k in range(n)])))
    identity = numpy.arange(count * n)
    raw = concat(
        [concat([s.faces.mapped(identity, k * count) for s in surfaces]) for k in range(n)]
    )
    degenerate = 0
    kept: list[tuple[str, Faces]] = []
    for name, faces in groups:
        bad = faces.repeated()
        degenerate += int(bad.sum())
        kept.append((name, faces.pick(~bad)))
    live = numpy.unique(concat([f for _, f in kept]).flat)
    compact = numpy.full(len(used), -1, dtype=numpy.int64)
    compact[live] = numpy.arange(len(live))
    facts = {
        "raw": raw,
        "stack": len(stack),
        "wheel": len(live),
        "degenerate": degenerate,
        "merged": len(stack) - len(used),
    }
    return stack[used][live], [(nm, f.mapped(compact)) for nm, f in kept], facts


def warn_sector_angle(bodies: NDArray[numpy.float64], opts: Options, tol: float) -> None:
    """Warn when the surfaces that are not blades do not span one copy's turn about the axis.

    The turn check cannot say this, since copies of anything about any axis map onto each other.
    A sector with no such surface, or with all of it on the axis, is not measured. A seam the
    tolerance ``tol`` (in the mesh's unit) closes can read off the angle by ``tol / radius``, so
    a node nearer the axis than that allows says nothing of the angle; a surface small against
    the tolerance is read over its outer half.
    """
    q = bodies - opts.hub
    radial = q - numpy.outer(q @ opts.axis, opts.axis)
    radius = numpy.linalg.norm(radial, axis=1)
    if not len(radius):
        return
    largest = float(radius.max())
    keep = radius > max(1e-6 * largest, min(tol / math.radians(_ANGLE_SLACK_DEG), 0.5 * largest))
    if not keep.any():
        return
    first = radial[keep][0] / radius[keep][0]
    second = numpy.cross(opts.axis, first)
    psi = numpy.arctan2(radial[keep] @ second, radial[keep] @ first)
    angle = math.degrees(azimuth_range(psi)[1])
    expected = 360.0 / opts.copies
    if abs(angle - expected) > _ANGLE_SLACK_DEG:
        warn(
            f"the sector's surfaces that are not blades: the sector spans {angle:.4f} deg about "
            f"the axis through the hub, and one of {opts.copies} copies is {expected:g} deg, so "
            "the copies do not close. Check --axis, --center and --copies.",
            GeometryWarning,
            stacklevel=4,
        )


def report_of(
    sector: tuple[NDArray[numpy.float64], Faces],
    wheel: tuple[NDArray[numpy.float64], Faces],
    facts: dict[str, Any],
    opts: Options,
    tol: float,
) -> dict[str, Any]:
    """Return the checks of a merge: counts, open edges, area and the turn against its control."""
    verts, faces = sector
    nodes, joined = wheel
    sector_open = edge_counts(faces, len(verts))
    raw_open = edge_counts(facts["raw"], len(verts) * opts.copies)
    wheel_open = edge_counts(joined, len(nodes))
    return {
        "copies": opts.copies,
        "axis": opts.axis.tolist(),
        "hub": opts.hub.tolist(),
        "tolerance_m": opts.tolerance_m,
        "nodes": {
            "sector": len(verts),
            "unmerged": facts["stack"],
            "wheel": facts["wheel"],
            "merged": facts["merged"],
        },
        "faces": {"sector": len(faces), "wheel": len(joined)},
        "degenerate_faces": facts["degenerate"],
        "open_edges": {"sector": sector_open[0], "unmerged": raw_open[0], "merged": wheel_open[0]},
        "non_manifold_edges": {
            "sector": sector_open[1],
            "unmerged": raw_open[1],
            "merged": wheel_open[1],
        },
        "area_ratio": joined.area(nodes) / (opts.copies * faces.area(verts)),
        **wheel_symmetry(nodes, joined, opts.axis, opts.hub, copies=opts.copies, tol=tol),
    }


def _compact(mesh: ObjMesh) -> tuple[NDArray[numpy.float64], dict[str, Faces]]:
    """Return the vertices the faces use and the families re-indexed to them."""
    families = {name: Faces.of(faces) for name, faces in mesh.families.items()}
    used = numpy.unique(concat(list(families.values())).flat)
    remap = numpy.full(len(mesh.verts), -1, dtype=numpy.int64)
    remap[used] = numpy.arange(len(used))
    return mesh.verts[used], {n: f.mapped(remap) for n, f in families.items()}


def _copy_points(
    points: NDArray[numpy.float64], opts: Options, copy: int | None, tol: float
) -> NDArray[numpy.float64]:
    """Return a rim's points turned into copy ``copy``, or into every copy, repeats merged."""
    step = 360.0 / opts.copies
    if copy is not None:
        return rotate(points, opts.axis, opts.hub, step * copy)
    every = numpy.vstack(
        [rotate(points, opts.axis, opts.hub, step * k) for k in range(opts.copies)]
    )
    return every[numpy.unique(merge_nodes(every, tol))]


def _table(prefix: str, name: str) -> str:
    """Return a TOML table header ``[prefix.name]``, the name quoted when it is not bare."""
    bare = name.isascii() and name.replace("_", "").replace("-", "").isalnum()
    return f"[{prefix}.{name if bare else toml_value(name)}]"


def reserve_name(taken: set[str], stem: str, body: str, role: str) -> str:
    """Return a file name ``<stem>.<body>`` that ``taken`` does not hold, ignoring case, and add it.

    A trailing-edge entry and a base region live in separate namespaces and may carry one name;
    the first keeps the plain name and a later one takes its role in the name
    (``<stem>.<role>.<body>``), then a number, so no point file replaces another.
    """
    candidates = (f"{stem}.{body}", f"{stem}.{role}.{body}")
    for name in candidates:
        if name.casefold() not in taken:
            taken.add(name.casefold())
            return name
    number = 2
    while f"{stem}.{role}{number}.{body}".casefold() in taken:
        number += 1
    name = f"{stem}.{role}{number}.{body}"
    taken.add(name.casefold())
    return name


@dataclass
class _Written:
    """The rim point files a wheel writes (path to unit and points) and its sidecar tables."""

    points: dict[Path, tuple[str, NDArray[numpy.float64]]] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)
    #: The file names handed out, case-folded: a file system may fold case, so two names that
    #: differ only in case would otherwise write one file.
    taken: set[str] = field(default_factory=set)

    def reserve(self, stem: str, body: str, role: str) -> str:
        """Return a file name no earlier reservation shares (see :func:`reserve_name`)."""
        return reserve_name(self.taken, stem, body, role)


def _rim_tables(
    rims: list[_Rim],
    surfaces: list[_Surface],
    opts: Options,
    *,
    obj: Path,
    tol: float,
    out: _Written,
) -> None:
    """Write each trailing-edge entry once per copy of a blade surface, once for any other."""
    copied = {s.family for s in surfaces if s.replicated and not s.generated}
    for rim in rims:
        for k in range(opts.copies) if rim.family in copied else [None]:
            label = rim.label if k is None else f"{rim.label}_{k + 1}"
            files = []
            for j, (kind, points) in enumerate(rim.sets):
                tag = f".{j + 1}" if len(rim.sets) > 1 else ""
                name = out.reserve(opts.out.stem, f"{label}{tag}.te.txt", "trailing_edge")
                out.points[opts.out.with_name(name)] = (
                    rim.unit,
                    _copy_points(points, opts, k, tol) * rim.to_file,
                )
                files.append((name, kind))
            out.lines += ["", _table("trailing_edges", label)]
            if len(files) == 1:
                out.lines += [
                    f"file = {toml_value(files[0][0])}",
                    f"type = {toml_value(files[0][1])}",
                ]
            else:
                items = ", ".join(
                    f"{{ file = {toml_value(f)}, type = {toml_value(t)} }}" for f, t in files
                )
                out.lines.append(f"files = [{items}]")
            if rim.tolerance_m != DEFAULT_TOLERANCE_M:
                out.lines.append(f"tolerance = {rim.tolerance_m!r}")


def _base_tables(
    surfaces: list[_Surface],
    declared: dict[str, Any],
    opts: Options,
    *,
    obj: Path,
    tol: float,
    out: _Written,
) -> None:
    """Write a base region for each base surface and carry a declaration on any other."""
    main = {s.family: s.names for s in surfaces if not s.generated}
    split = {s.family for s in surfaces if s.generated}
    for s in surfaces:
        if s.generated:
            region, parents = declared.get(s.family), main[s.family]
        elif s.parent:
            region, parents = declared[s.family], main[s.parent]
        elif s.family in declared and s.family not in split:
            region, parents = declared[s.family], None
        else:
            continue
        for k in range(len(s.names)):
            parent = None if parents is None else parents[k if len(parents) > 1 else 0]
            _one_base(s, k, parent, region, where=(opts, obj, tol), out=out)


def _one_base(
    s: _Surface,
    k: int,
    parent: str | None,
    region: Any,
    *,
    where: tuple[Options, Path, float],
    out: _Written,
) -> None:
    """Write one ``[base_regions.<name>]`` table, its rim file turned into copy ``k``."""
    opts, obj, tol = where
    name = s.names[k]
    out.lines += ["", _table("base_regions", name)]
    rim: Any = "auto" if region is None else region.trailing_edges
    if isinstance(rim, str) or not rim.sets:
        out.lines.append('trailing_edges = "auto"')
    else:
        one = rim.sets[0]
        loaded = points_from_file(obj, Path(str(one.points_file)))
        assert loaded is not None
        copy = k if s.replicated else None
        file = out.reserve(opts.out.stem, f"{name}.te.txt", "base")
        out.points[opts.out.with_name(file)] = (
            loaded.unit,
            _copy_points(loaded.points, opts, copy, tol) * loaded.to_file,
        )
        out.lines.append(
            f"trailing_edges = {{ file = {toml_value(file)}, type = {toml_value(one.edge_type)} }}"
        )
    if region is not None and region.model != "EMPIRICAL":
        out.lines.append(f"model = {toml_value(region.model)}")
    if region is not None and region.cp != -0.2:
        out.lines.append(f"cp = {region.cp!r}")
    if parent is not None:
        out.lines.append(f"merge_into = {toml_value(parent)}")


def validate_blades(
    blades: Sequence[str], known: Iterable[str], copies: int, what: str
) -> list[str]:
    """Return the blade surfaces, refusing a missing name, a repeat or a name clash."""
    names = list(known)
    chosen = list(dict.fromkeys(blades))
    if copies < 2:
        raise InputValueError(
            f"copies (CLI: --copies) is {copies}; a wheel needs 2 or more. {_NOTHING}"
        )
    if not chosen:
        raise InputValueError(
            f"blades (CLI: --blades) names no {what}; name the {what}(s) that are the blade, "
            "one of "
            f"{', '.join(names)}. {_NOTHING}"
        )
    missing = [b for b in chosen if b not in names]
    if missing:
        raise InputValueError(
            f"blades (CLI: --blades) names {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} "
            f"not a {what} of the sector; the sector has {', '.join(names)}. {_NOTHING}"
        )
    stems = [stem_of(b) for b in chosen]
    if "" in stems or len(set(stems)) != len(stems):
        raise InputValueError(
            f"blades (CLI: --blades) {', '.join(chosen)} would give copies the same name or none: "
            "the stem of "
            "a name is the name without its trailing digits, and stems must be non-empty and "
            f"differ. Rename the {what}s. {_NOTHING}"
        )
    return chosen


def _center_point(center: Sequence[float] | None) -> Any:
    """Resolve the rotor-axis point: three finite numbers, the origin by default."""
    if center is None:
        return numpy.zeros(3)
    try:
        point = numpy.asarray(center, dtype=float)
    except (TypeError, ValueError) as error:
        raise InputValueError(
            f"center (CLI: --center) must be three finite numbers. {_NOTHING}"
        ) from error
    if point.shape != (3,) or not numpy.isfinite(point).all():
        raise InputValueError(f"center (CLI: --center) must be three finite numbers. {_NOTHING}")
    return point


def sector_to_wheel(
    sector: str | Path,
    *,
    copies: int,
    blades: Sequence[str],
    axis: str | Sequence[float] = "x",
    center: Sequence[float] | None = None,
    tolerance: float = DEFAULT_TOLERANCE_M,
    out: str | Path | None = None,
    overwrite: bool = False,
) -> WheelResult:
    """Copy one sector about an axis into a full wheel and write it.

    An OBJ sector gives an OBJ wheel, with its boundaries file and the rim points files. The
    nodes of the seams are merged within ``tolerance`` and the faces keep the sector's order
    inside each copy.

    Parameters
    ----------
    sector : str or Path
        The sector: an OBJ with its ``<name>.obj.boundaries.toml`` and trailing-edge points
        files beside it.
    copies : int
        The number of copies, 2 or more; copy ``k`` is turned by ``360 k / copies`` degrees.
    blades : sequence of str
        The surfaces that are blades (OBJ groups);
        each is numbered one per copy, ``<stem><k>`` with the stem the name without its
        trailing digits, and every other surface is one group shared by the copies.
    axis : str or sequence of float
        ``x``, ``y``, ``z``, a signed one of them, or a vector; the turn is right-handed.
    center : sequence of float, optional
        A point of the axis, in the mesh's unit (default the origin).
    tolerance : float
        The distance, in meters, within which two nodes are one.
    out : str or Path, optional
        The wheel OBJ; default ``<sector stem>_wheel.obj`` beside the sector.
    overwrite : bool
        Replace the files of a wheel that exists; without it an existing file is refused.

    Returns
    -------
    WheelResult
        ``written`` is True and the report holds the checks.

    Raises
    ------
    InputArtifactError
        A sector that is not an OBJ, a file that cannot be read, or an output that exists
        without ``overwrite``.
    InputValueError
        Fewer than two copies, a blade surface the sector lacks, a bad axis or tolerance.
    """
    path = Path(sector)
    suffix = path.suffix.lower()
    if suffix != ".obj":
        raise InputArtifactError(
            f"{path}: a sector is an OBJ (a saved simulation is not accepted here), and this "
            f"is {suffix or 'no file type'}. {_NOTHING}"
        )
    if not path.is_file():
        raise InputArtifactError(f"{path}: no such sector file; name an existing one. {_NOTHING}")
    target = Path(out) if out is not None else path.with_name(f"{path.stem}_wheel.obj")
    if target.suffix.lower() != ".obj":
        warn(
            f"out (CLI: --out) {target.name} is not an OBJ; the wheel is an OBJ.",
            NotApplicableWarning,
            stacklevel=2,
        )
        target = target.with_suffix(".obj")
    refuse_the_source(target, path)
    if not (math.isfinite(tolerance) and tolerance > 0.0):
        raise InputValueError(
            f"tolerance (CLI: --tolerance) {tolerance!r} must be a positive length in "
            f"meters. {_NOTHING}"
        )
    point = _center_point(center)
    opts = Options(copies, list(blades), parse_axis(axis), point, tolerance, target, overwrite)
    return _obj_route(path, opts)


def _load(path: Path) -> tuple[ObjMesh, Any, str | None]:
    """Read the sector OBJ, its declarations (None without a boundaries file) and its unit."""
    mesh = read_obj(path)
    found = read_sidecar(path)
    if found is None:
        warn(
            f"{path.name} has no {inventory_sidecar(path, 'obj').name}, so no trailing edge and "
            "no base region is declared: the wheel is written with none, and no base is split.",
            GeometryWarning,
            stacklevel=3,
        )
        return mesh, None, None
    declarations = read_boundary_declarations(found[0])
    table = found[1].get("import")
    unit = table.get("units") if isinstance(table, dict) else None
    skipped = sorted(set(found[1]) - _CARRIED)
    if isinstance(table, dict) and table.get("operations"):
        skipped.insert(0, "import.operations")
    if skipped:
        warn(
            f"{found[0].name} states {', '.join(skipped)}, which a wheel's boundaries file does "
            "not carry; they are skipped. State them again in the wheel's boundaries file.",
            NotApplicableWarning,
            stacklevel=3,
        )
    return mesh, declarations, unit.strip().upper() if isinstance(unit, str) else None


def provenance(
    path: Path, kind: str, unit: str | None, groups: Sequence[tuple[str, int]]
) -> dict[str, Any]:
    """Return the record's identity of the source and the groups of the wheel."""
    return {
        "source": path.name,
        "source_sha256": file_sha256(path),
        "kind": kind,
        "mesh_unit": unit,
        "groups": [{"name": n, "faces": count} for n, count in groups],
    }


def _obj_route(path: Path, opts: Options) -> WheelResult:
    """Run the OBJ route: read, split, copy, merge, check and write."""
    mesh, declarations, unit = _load(path)
    opts.blades = validate_blades(opts.blades, mesh.families, opts.copies, "group")
    metres = unit_of(unit, inventory_sidecar(path, "obj").name)
    tol = opts.tolerance_m / metres
    verts, families = _compact(mesh)
    declared = {r.family: r for r in declarations.base_regions} if declarations else {}
    rims = _rims(path, declarations) if declarations else []
    bases = {f for f, r in declared.items() if r.merge_into}
    _match_rims(rims, verts, families, bases, metres)
    surfaces = _surfaces(
        families, rims, declared, blades=opts.blades, copies=opts.copies, count=len(verts)
    )
    shared = concat([s.faces for s in surfaces if not s.replicated])
    warn_sector_angle(verts[numpy.unique(shared.flat)], opts, tol)
    nodes, groups, facts = _assemble(verts, surfaces, opts, tol)
    names = [n for n, _ in groups]
    if len(set(names)) != len(names):
        raise InputValueError(
            "the wheel would hold two groups named alike "
            f"({sorted({n for n in names if names.count(n) > 1})}); rename a group of the "
            f"sector. {_NOTHING}"
        )
    joined = concat([f for _, f in groups])
    refuse_a_collapsed_wheel(len(joined), opts.tolerance_m)
    if facts["degenerate"]:
        warn(
            f"{facts['degenerate']} faces of the wheel name one node twice: the tolerance joins "
            "nodes of one face. They are removed, and each leaves a hole. Lower --tolerance.",
            GeometryWarning,
            stacklevel=3,
        )
    report = report_of((verts, concat(list(families.values()))), (nodes, joined), facts, opts, tol)
    report.update(provenance(path, "obj", unit, [(n, len(f)) for n, f in groups]))
    written = _Written()
    if declarations is not None:
        _rim_tables(rims, surfaces, opts, obj=path, tol=tol, out=written)
        _base_tables(surfaces, declared, opts, obj=path, tol=tol, out=written)
    return _write_obj(path, opts, (nodes, groups), report, (written, declarations, unit))


def _write_obj(
    path: Path,
    opts: Options,
    wheel: tuple[NDArray[numpy.float64], list[tuple[str, Faces]]],
    report: dict[str, Any],
    declared: tuple[_Written, Any, str | None],
) -> WheelResult:
    """Write the wheel OBJ, its boundaries file, its points files and its record."""
    nodes, groups = wheel
    written, declarations, unit = declared
    out = opts.out
    sidecar = inventory_sidecar(out, "obj")
    record = out.with_name(f"{out.stem}.wheel.json")
    refuse_existing([out, sidecar, record, *written.points], opts.overwrite)
    earlier = _earlier_points(record, out) if opts.overwrite else []
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# Full wheel of {path.name}: {opts.copies} copies about axis "
        f"{opts.axis.round(12).tolist()} through {opts.hub.tolist()}, nodes merged within "
        f"{opts.tolerance_m:g} m."
    ]
    _textio.write_text(
        out, obj_text(ObjMesh(nodes, {n: f.lists() for n, f in groups}, header, "g"))
    )
    top = [
        f"# Boundaries of {out.name}, a full wheel of {path.name}; its declarations are the",
        "# sector's, copied about the axis (one entry per copy of a blade surface).",
    ]
    top += ["boundaries = [", *[f"    {toml_value(n)}," for n, _ in groups], "]"]
    top += face_count_lines(out, len(groups))
    if declarations is not None and unit:
        top += ["", "[import]", f"units = {toml_value(unit)}"]
    if declarations is not None and declarations.no_trailing_edges:
        top += ["", "[trailing_edges]", "none = true"]
    _textio.write_text(sidecar, "\n".join(top + written.lines) + "\n")
    files = [out, sidecar]
    for file, (file_unit, points) in written.points.items():
        write_te(file, file_unit, points)
        files.append(file)
    for stale in earlier:
        if stale not in written.points:
            stale.unlink(missing_ok=True)
    report["points_files"] = sorted(file.name for file in written.points)
    files.append(write_json(record, report))
    return WheelResult(True, out, files, report)


def _earlier_points(record: Path, out: Path) -> list[Path]:
    """Return the points files the earlier wheel of this name wrote, as its record lists them.

    The record, not the boundaries file, says what the verb wrote: a boundaries file also
    declares files the user made. A target with no such list has nothing of a wheel's to remove.
    """
    try:
        names = json.loads(record.read_text(encoding="utf-8")).get("points_files", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [out.with_name(Path(name).name) for name in names if isinstance(name, str)]
