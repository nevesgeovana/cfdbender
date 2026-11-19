"""The reader profile gate: measure a written IGES file against what the solver reader accepts.

Pure text work on the file, with no kernel: :func:`check_profile` reads the entities back and
reports one finding per defect. A solid entity is refused with an error, because the package
builds surfaces only. Every other defect is a ``fail`` finding that names its check and entity,
so a loop of parameter sets keeps its report. A degree beyond the observed ones, which no source
limits yet, a surface of revolution (120), and an inner loop while the profile does not record
inner loops as read, are a ``warn``. The profile itself is
:mod:`cfdbender.geometry.iges_profile`.

A trimmed face (144) is read from its parameter list, not from where this package writes things:
its surface, its outer boundary, its inner boundaries and its property pointers are parsed, so a
file from another system is measured by the same rules. The outer boundary (142) must close in the
parameter space of its surface, stay inside the surface's domain, and have a model-space curve
that lies on the surface within the resolution the file states.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import InputArtifactError
from cfdbender._lengths import METRES_PER_UNIT
from cfdbender.geometry._iges_names import read_iges_names
from cfdbender.geometry._iges_write import read_iges
from cfdbender.geometry.iges_profile import (
    OBSERVED,
    SOLID_ENTITIES,
    GateFinding,
    ReaderProfile,
    surface_findings,
)

__all__ = ["GateFinding", "GateReport", "base_displacement", "check_profile", "read_bases"]


def base_displacement(path, sources) -> float:
    """Compare each written face's base coefficients with its other written copies.

    Control poles are placed in meters, including an IGES transformation when
    present. Knots and degrees must also match. No surface sampling or fitting
    can hide a changed control net behind a small geometric residual.
    """
    document = read_iges(path)
    names = read_iges_names(path)
    unit = METRES_PER_UNIT.get(names.unit or "")
    if unit is None:
        return 1.0
    factor = float(unit)
    entities = {e["id"]: e for e in document["entities"]}
    faces = [e for e in document["entities"] if e["type"] == 144]
    worst = 0.0
    for face, name in zip(faces, names.faces, strict=True):
        entity = entities[int(face["parameters"][1])]
        if entity["type"] != 128:
            return 1.0
        surface = _Surface(entity["parameters"])
        points = surface.points.swapaxes(0, 1)
        transform = int(entity["d1"][6])
        if transform:
            matrix = np.asarray(
                [_number(v) for v in entities[transform]["parameters"][1:13]]
            ).reshape(3, 4)
            points = points @ matrix[:, :3].T + matrix[:, 3]
        error = min(
            (_base_error(surface, points * factor, p) for p in sources.get(name, [])), default=1.0
        )
        worst = max(worst, error)
    return worst


def read_bases(path):
    """Read the serialized base coefficients per face name, without a kernel or a fit.

    These are the comparison values for other copies of the same source base.
    Serialization rounding is shared by the copies; it does not change the
    independently retained numerical source identity.
    """
    document = read_iges(path)
    names = read_iges_names(path)
    entities = {e["id"]: e for e in document["entities"]}
    faces = [e for e in document["entities"] if e["type"] == 144]
    result = {}
    for face, name in zip(faces, names.faces, strict=True):
        entity = entities[int(face["parameters"][1])]
        surface = _Surface(entity["parameters"])
        points = surface.points.swapaxes(0, 1)
        transform = int(entity["d1"][6])
        if transform:
            matrix = np.asarray(
                [_number(v) for v in entities[transform]["parameters"][1:13]]
            ).reshape(3, 4)
            points = points @ matrix[:, :3].T + matrix[:, 3]
        result[name] = {
            "controls": points * float(METRES_PER_UNIT[names.unit or "m"]),
            "knots_u": surface.knots[0],
            "knots_v": surface.knots[1],
            "mult_u": np.ones(len(surface.knots[0]), int),
            "mult_v": np.ones(len(surface.knots[1]), int),
            "degree_u": surface.degrees[0],
            "degree_v": surface.degrees[1],
        }
    return result


def _base_error(surface, points, expected):
    controls = np.asarray(expected["controls"])
    knots = [np.repeat(expected[f"knots_{d}"], expected[f"mult_{d}"]) for d in ("u", "v")]
    if (
        points.shape != controls.shape
        or surface.degrees != (expected["degree_u"], expected["degree_v"])
        or any(a.shape != b.shape for a, b in zip(surface.knots, knots, strict=True))
    ):
        return 1.0
    values = [float(np.max(np.abs(points - controls))), float(np.max(np.abs(surface.weights - 1)))]
    values += [float(np.max(np.abs(a - b))) for a, b in zip(surface.knots, knots, strict=True)]
    return max(values) if np.isfinite(values).all() else 1.0


#: Points taken along each curve of a boundary when it is measured.
_BOUNDARY_SAMPLES = 9
#: Starting grid, per direction, of the search for the closest point of a surface.
_GRID = 21
#: Newton steps of that search; the surface is smooth, so a handful reach the resolution.
_REFINE = 12
#: Position of the minimum user-intended resolution (global parameter 19), zero based.
_RESOLUTION_AT = 18
#: Used when a file states no resolution; one micro unit of the file's own length unit.
_DEFAULT_RESOLUTION = 1e-6


@dataclass(frozen=True)
class GateReport:
    """The findings of one file and how many surfaces it holds."""

    path: Path
    surfaces: int
    findings: tuple[GateFinding, ...]

    @property
    def failed(self) -> tuple[GateFinding, ...]:
        """Return the findings that fail the gate."""
        return tuple(f for f in self.findings if f.severity == "fail")

    @property
    def passed(self) -> bool:
        """Return whether no finding fails the gate; warnings do not."""
        return not self.failed


@dataclass(frozen=True)
class _Face:
    """The pointers of one trimmed surface (144), parsed from its parameter list."""

    surface: int
    outer: int
    inner: tuple[int, ...]
    properties: tuple[int, ...]


def _number(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _parse_face(values: list[str]) -> _Face:
    """Parse a 144: its surface, outer boundary, N2 inner boundaries, then its properties."""
    inner_count = int(values[3])
    inner = tuple(int(v) for v in values[5 : 5 + inner_count])
    rest = values[5 + inner_count :]
    associations = int(rest[0]) if rest else 0
    tail = rest[1 + associations :]
    count = int(tail[0]) if tail else 0
    return _Face(
        surface=int(values[1]),
        outer=int(values[4]),
        inner=inner,
        properties=tuple(int(p) for p in tail[1 : 1 + count]),
    )


def _boundary_curves(boundary: dict[str, Any]) -> tuple[int, int]:
    """Return the parameter-space and the model-space curve pointers of a 142."""
    values = boundary["parameters"]
    return int(values[3]), int(values[4])


def _outer_sides(face: _Face, by_id: dict[int, dict[str, Any]]) -> int | None:
    outer = by_id.get(face.outer) if face.outer else None
    if outer is None or outer["type"] != 142:
        return None
    for pointer in reversed(_boundary_curves(outer)):
        curve = by_id.get(pointer) if pointer else None
        if curve is not None and curve["type"] == 102:
            return int(curve["parameters"][1])
    return None


# --- evaluation of the entities a trimmed face stands on ---------------------------------------


def _basis(knots: np.ndarray, degree: int, x: float) -> np.ndarray:
    """Return every B-spline basis function of the knot vector at ``x`` (zero outside its span)."""
    count = len(knots) - degree - 1
    span = int(np.clip(np.searchsorted(knots, x, side="right") - 1, degree, count - 1))
    out = [1.0]
    left = [0.0] * (degree + 1)
    right = [0.0] * (degree + 1)
    for j in range(1, degree + 1):
        left[j] = x - knots[span + 1 - j]
        right[j] = knots[span + j] - x
        saved = 0.0
        for r in range(j):
            denominator = right[r + 1] + left[j - r]
            share = out[r] / denominator if denominator else 0.0
            out[r] = saved + right[r + 1] * share
            saved = left[j - r] * share
        out.append(saved)
    row = np.zeros(count)
    row[span - degree : span + 1] = out
    return row


class _Spline:
    """A rational B-spline curve (126), or a straight line (110) as its degree 1 form."""

    def __init__(self, knots: np.ndarray, degree: int, weights: np.ndarray, points: np.ndarray):
        self.knots, self.degree, self.weights, self.points = knots, degree, weights, points
        self.low, self.high = float(knots[degree]), float(knots[-degree - 1])

    def at(self, t: float) -> np.ndarray:
        weighted = _basis(self.knots, self.degree, t) * self.weights
        return np.asarray(weighted @ self.points / weighted.sum())


def _spline_126(values: list[str]) -> _Spline:
    k, m = int(values[1]), int(values[2])
    numbers = [_number(v) for v in values[7:]]
    knots = np.array(numbers[: k + m + 2])
    weights = np.array(numbers[k + m + 2 : k + m + 3 + k])
    points = np.array(numbers[2 * k + m + 3 : 2 * k + m + 3 + 3 * (k + 1)]).reshape(-1, 3)
    curve = _Spline(knots, m, weights, points)
    ends = numbers[5 * k + m + 6 : 5 * k + m + 8]  # V(0) and V(1), the range the curve uses
    if len(ends) == 2 and curve.low <= ends[0] < ends[1] <= curve.high:
        curve.low, curve.high = float(ends[0]), float(ends[1])
    return curve


def _spline_110(values: list[str]) -> _Spline:
    ends = np.array([_number(v) for v in values[1:7]]).reshape(2, 3)
    return _Spline(np.array([0.0, 0.0, 1.0, 1.0]), 1, np.ones(2), ends)


def _pieces(entity: dict[str, Any], by_id: dict[int, dict[str, Any]]) -> list[_Spline] | None:
    """Return the curves of a boundary entity in order.

    ``None`` means it holds one this gate cannot evaluate: anything but 126, 110 and a composite
    102 of those.
    """
    kind = entity["type"]
    if kind == 126:
        return [_spline_126(entity["parameters"])]
    if kind == 110:
        return [_spline_110(entity["parameters"])]
    if kind != 102:
        return None
    out: list[_Spline] = []
    for pointer in entity["parameters"][2 : 2 + int(entity["parameters"][1])]:
        member = by_id.get(int(pointer))
        if member is None:
            continue
        found = None if member["type"] == 102 else _pieces(member, by_id)
        if found is None:
            return None
        out += found
    return out


def _samples(pieces: list[_Spline]) -> np.ndarray:
    """Return ``_BOUNDARY_SAMPLES`` points along each piece, the two ends included."""
    return np.array(
        [p.at(t) for p in pieces for t in np.linspace(p.low, p.high, _BOUNDARY_SAMPLES)]
    )


class _Surface:
    """A rational B-spline surface (128) with its domain."""

    def __init__(self, values: list[str]) -> None:
        k1, k2, m1, m2 = (int(values[i]) for i in range(1, 5))
        numbers = [_number(v) for v in values[10:]]
        n1, n2 = k1 + m1 + 2, k2 + m2 + 2
        self.knots = (np.array(numbers[:n1]), np.array(numbers[n1 : n1 + n2]))
        self.degrees = (m1, m2)
        count = (k1 + 1) * (k2 + 1)
        at = n1 + n2
        self.weights = np.array(numbers[at : at + count]).reshape(k2 + 1, k1 + 1)
        at += count
        self.points = np.array(numbers[at : at + 3 * count]).reshape(k2 + 1, k1 + 1, 3)
        self.domain = tuple(numbers[at + 3 * count : at + 3 * count + 4])

    def at(self, u: float, v: float) -> np.ndarray:
        nu = _basis(self.knots[0], self.degrees[0], u)
        nv = _basis(self.knots[1], self.degrees[1], v)
        weighted = nv[:, None] * nu[None, :] * self.weights
        return np.asarray(np.tensordot(weighted, self.points, axes=2) / weighted.sum())

    def distance(self, point: np.ndarray, *, seeds: np.ndarray | None = None) -> float:
        """Return the distance from ``point`` to the surface: grid start, then Newton steps."""
        u0, u1, v0, v1 = self.domain
        grid = [(u, v) for u in np.linspace(u0, u1, _GRID) for v in np.linspace(v0, v1, _GRID)]
        if seeds is not None:
            grid.extend(tuple(uv) for uv in np.clip(seeds, [u0, v0], [u1, v1]))
        best = min(grid, key=lambda uv: float(np.linalg.norm(self.at(*uv) - point)))
        u, v = float(best[0]), float(best[1])
        closest = float(np.linalg.norm(self.at(u, v) - point))
        ranges = np.array([u1 - u0, v1 - v0])
        steps = 1e-6 * ranges
        for _ in range(_REFINE):
            here = self.at(u, v)
            ua, ub = max(u - steps[0], u0), min(u + steps[0], u1)
            va, vb = max(v - steps[1], v0), min(v + steps[1], v1)
            du = (self.at(ub, v) - self.at(ua, v)) / (ub - ua)
            dv = (self.at(u, vb) - self.at(u, va)) / (vb - va)
            jacobian = np.stack([du, dv], axis=1)
            previous = np.array([u, v])
            move = _bounded_move(jacobian, point - here, previous, [u0, v0], [u1, v1])
            u, v = float(np.clip(u + move[0], u0, u1)), float(np.clip(v + move[1], v0, v1))
            closest = min(closest, float(np.linalg.norm(self.at(u, v) - point)))
            if np.max(np.abs((np.array([u, v]) - previous) / ranges)) <= 1e-12:
                break
        return closest


def _bounded_move(jacobian, residual, position, low, high):
    """Solve again in the free direction when a bound removes a Newton direction."""
    move = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
    fixed = ((position <= low) & (move < 0)) | ((position >= high) & (move > 0))
    if fixed.any():
        move[fixed] = 0.0
        free = ~fixed
        if free.any():
            move[free] = np.linalg.lstsq(jacobian[:, free], residual, rcond=None)[0]
    return move


# --- the findings of one trimmed face --------------------------------------------------------


def _closed(
    ident: int,
    pieces: list[_Spline],
    profile: ReaderProfile,
    *,
    surface: _Surface,
    resolution: float,
) -> list[GateFinding]:
    """Find the joints of a loop that are open in (u, v) and also apart on the surface.

    A writer's pcurves meet only to its own tolerance (6e-7 in (u, v) after a kernel fragment), so
    a joint is open when its parameter gap exceeds ``loop_gap`` and its two ends, mapped on the
    surface, are farther apart than the resolution the file states.
    """
    found = []
    for k, piece in enumerate(pieces):
        following = pieces[(k + 1) % len(pieces)]
        end, start = piece.at(piece.high)[:2], following.at(following.low)[:2]
        gap = float(np.abs(end - start).max())
        apart = float(np.linalg.norm(surface.at(*end) - surface.at(*start)))
        if gap > profile.loop_gap and apart > resolution:
            found.append(
                GateFinding(
                    "profile.loop_open",
                    "fail",
                    ident,
                    f"the outer loop does not close in (u, v): piece {k} ends {gap:.3g} from "
                    "the next",
                )
            )
    return found


def _in_domain(
    ident: int, pieces: list[_Spline], domain: tuple[float, ...], slack: float
) -> list[GateFinding]:
    u0, u1, v0, v1 = domain
    uv = _samples(pieces)[:, :2]
    outside = (uv < [u0 - slack, v0 - slack]) | (uv > [u1 + slack, v1 + slack])
    if not outside.any():
        return []
    where = tuple(float(x) for x in uv[outside.any(axis=1)][0])
    return [
        GateFinding(
            "profile.loop_domain",
            "fail",
            ident,
            f"the outer loop leaves the surface's domain at {where}",
        )
    ]


def _on_surface(
    ident: int,
    pieces: list[_Spline],
    surface: _Surface,
    resolution: float,
    *,
    seeds: np.ndarray | None = None,
) -> list[GateFinding]:
    worst = max(surface.distance(p, seeds=seeds) for p in _samples(pieces))
    if worst <= resolution:
        return []
    return [
        GateFinding(
            "profile.loop_surface",
            "fail",
            ident,
            f"the model-space boundary lies {worst:.3g} off the surface; the file states a "
            f"resolution of {resolution:.3g}",
        )
    ]


def _loop_findings(
    ident: int,
    boundary: dict[str, Any],
    surface: _Surface,
    by_id: dict[int, dict[str, Any]],
    profile: ReaderProfile,
    *,
    resolution: float,
) -> list[GateFinding]:
    """Measure one outer boundary: closed in (u, v), inside the domain, and on the surface."""
    parametric, spatial = _boundary_curves(boundary)
    pieces = _pieces(by_id[parametric], by_id) if parametric in by_id else []
    if pieces is None:
        return [
            GateFinding(
                "profile.loop_unmeasured",
                "warn",
                ident,
                "its parameter-space boundary is not a 126, 110 or composite of them",
            )
        ]
    found = (
        _closed(ident, pieces, profile, surface=surface, resolution=resolution) if pieces else []
    )
    found += _in_domain(ident, pieces, surface.domain, profile.loop_gap) if pieces else []
    model = _pieces(by_id[spatial], by_id) if spatial in by_id else None
    seeds = _samples(pieces)[:, :2] if pieces else None
    return found + (_on_surface(ident, model, surface, resolution, seeds=seeds) if model else [])


#: Entity use flag (status digits 5 and 6) of a curve in a surface's parameter space.
_PARAMETER_SPACE_USE = "05"


def _direct_read_findings(
    ident: int, face: _Face, by_id: dict[int, dict[str, Any]]
) -> list[GateFinding]:
    """Warn on an outer boundary the solver's direct CAD import reads as no face.

    Measured on a direct CAD import: a face is read only when its outer
    boundary (142) has a parameter-space curve (BPTR) flagged 2D parametric (use 05) and a PREF
    other than 2. A file written by this package meets all three; a file from elsewhere may not,
    and ``iges-convert`` rewrites it into the form that is read.
    """
    boundary = by_id.get(face.outer) if face.outer else None
    if boundary is None or boundary["type"] != 142:
        reason = "no parameter-space outer boundary (142)"
    else:
        values = boundary["parameters"]
        bptr = int(values[3] or 0) if len(values) > 3 else 0
        curve = by_id.get(bptr) if bptr else None
        if curve is None:
            reason = "its outer boundary has no parameter-space curve (BPTR 0)"
        elif curve["d1"][8][4:6] != _PARAMETER_SPACE_USE:
            reason = f"its parameter-space curve has entity use {curve['d1'][8][4:6]}, not 05"
        elif len(values) > 5 and values[5].strip() == "2":
            reason = "its outer boundary states PREF 2 (model space)"
        else:
            return []
    return [
        GateFinding(
            "profile.trim",
            "warn",
            ident,
            f"{reason}; the solver's direct CAD import reads such a face as no face, run "
            "iges-convert on the file",
        )
    ]


def face_findings(
    entity: dict[str, Any],
    by_id: dict[int, dict[str, Any]],
    profile: ReaderProfile,
    resolution: float = _DEFAULT_RESOLUTION,
) -> list[GateFinding]:
    """Return the findings of one trimmed surface (144) measured against ``profile``."""
    ident = entity["id"]
    face = _parse_face(entity["parameters"])
    found: list[GateFinding] = []
    surface = by_id.get(face.surface)
    if surface is not None and surface["type"] not in (120, 128):
        found.append(
            GateFinding(
                "profile.entity",
                "warn",
                ident,
                f"the surface is entity {surface['type']}; only 128 was observed read and no "
                "source states the rest",
            )
        )
    if face.inner and not profile.inner_loops:
        found.append(
            GateFinding(
                "profile.inner_loop",
                "warn",
                ident,
                f"the face has {len(face.inner)} inner boundaries; whether the reader honors "
                "them is not observed",
            )
        )
    named = any(
        (prop := by_id.get(p)) is not None and prop["type"] == 406 and int(prop["d2"][4]) == 15
        for p in face.properties
    )
    if not named:
        found.append(GateFinding("profile.name", "fail", ident, "no 406 form 15 NAME on the face"))
    found += _direct_read_findings(ident, face, by_id)
    sides = _outer_sides(face, by_id)
    if sides in profile.bad_side_counts:
        found.append(
            GateFinding("profile.sides", "fail", ident, f"outer boundary has {sides} curves")
        )
    boundary = by_id.get(face.outer) if face.outer else None
    if (
        surface is not None
        and surface["type"] == 128
        and boundary is not None
        and boundary["type"] == 142
    ):
        found += _loop_findings(
            ident, boundary, _Surface(surface["parameters"]), by_id, profile, resolution=resolution
        )
    return found


def _resolution(document: dict[str, Any]) -> float:
    """Return the minimum resolution the file's global section states, in the file's unit."""
    try:
        value = _number(document["global"][_RESOLUTION_AT])
    except (IndexError, ValueError):
        return _DEFAULT_RESOLUTION
    return value if value > 0 else _DEFAULT_RESOLUTION


def check_profile(path: str | Path, profile: ReaderProfile = OBSERVED) -> GateReport:
    """Measure the IGES file at ``path`` against ``profile``.

    Raises :class:`InputArtifactError` for a solid entity (surfaces only) and for a file that
    cannot be read as IGES. Every other defect is a finding of the returned report, and a file with
    no trimmed surface fails: it holds nothing the gate could have measured.
    """
    try:
        document = read_iges(path)
    except (UnicodeDecodeError, ValueError, IndexError) as error:
        raise InputArtifactError(
            f"{path}: this cannot be read as an IGES file ({error}). Write the file again and "
            "run again. Nothing was written.",
            kind="geometry",
        ) from error
    entities = document["entities"]
    solids = sorted({e["type"] for e in entities if e["type"] in SOLID_ENTITIES})
    if solids:
        raise InputArtifactError(
            f"{path}: the file holds solid entities {solids}; this package builds surfaces only. "
            "Export surfaces from the CAD system and run again. Nothing was written.",
            kind="geometry",
        )
    by_id = {e["id"]: e for e in entities}
    findings: list[GateFinding] = []
    faces = [e for e in entities if e["type"] == 144]
    if not faces:
        findings.append(GateFinding("profile.surfaces", "fail", None, "no trimmed surface (144)"))
    trimmed = {_parse_face(e["parameters"]).surface for e in faces}
    for entity in entities:
        if entity["type"] == 120:
            findings.append(
                GateFinding(
                    "profile.entity",
                    "warn",
                    entity["id"],
                    "a surface of revolution (120) was not observed read; none is written",
                )
            )
        if entity["type"] == 128:
            findings += surface_findings(entity, profile)
            if entity["id"] not in trimmed:
                findings.append(
                    GateFinding("profile.name", "fail", entity["id"], "a surface no face names")
                )
    resolution = _resolution(document)
    for entity in faces:
        findings += face_findings(entity, by_id, profile, resolution)
    return GateReport(path=Path(path), surfaces=len(faces), findings=tuple(findings))
