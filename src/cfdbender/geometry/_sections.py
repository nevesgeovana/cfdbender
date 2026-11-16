"""The one section interface of the parametric geometry, and the CST form behind it.

Pipeline role: pure numerical stage of the geometry package. A lifting-surface section is
read from a NACA designation, from CST coefficients or from a file of ordinates, and held as
a ``SectionForm``. CST is the only stored form; a second form implements the same protocol and
changes nothing that calls it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from cfdbender._errors import InputValueError

from ._airfoils import (
    cst_airfoil,
    naca_camber,
    naca_digits,
    naca_to_cst,
    normalized_naca,
    ordinates_to_cst,
)
from ._bspline import clamped_knots, curve_basis, elevate_curve, fit_curve, union_knots
from ._distributions import elevate_coefficients

__all__ = [
    "BSplineSection",
    "CstSection",
    "SectionForm",
    "bspline_sections",
    "fit_or_read",
    "identity_report",
    "read_ordinates",
    "section_checks",
    "section_forms",
    "section_from_bspline",
    "trailing_edge_notes",
    "section_from_physical",
    "shape_unchanged",
    "thickness_profile",
    "section_from_item",
    "section_from_ordinates",
    "section_from_stored",
]

_COMPARE_X = (1 - np.cos(np.linspace(0, np.pi, 401))) / 2


@runtime_checkable
class SectionForm(Protocol):
    """What the rest of the geometry asks of a section, whatever its stored form."""

    def evaluate(self, x):
        """Return the upper and lower (x, z) points at chord fractions ``x``."""

    def nose_radius(self) -> float:
        """Return the leading-edge radius in chord lengths."""

    def te_thickness(self) -> float:
        """Return the trailing-edge thickness in chord lengths."""

    def to_stored(self) -> dict:
        """Return the stored form as plain data."""

    def compat(self, others) -> float:
        """Return the largest ordinate difference to ``others``, in chord lengths."""


@dataclass(frozen=True)
class CstSection:
    """A class-shape-transformation section: Bernstein coefficients and a trailing-edge base."""

    upper: tuple[float, ...]
    lower: tuple[float, ...]
    trailing_edge_thickness: float
    n1: float = 0.5
    n2: float = 1.0
    max_residual: float = 0.0
    chord_transform: dict | None = None

    @property
    def degree(self) -> int:
        """Return the Bernstein degree."""
        return len(self.upper) - 1

    def evaluate(self, x):
        """Return the upper and lower (x, z) points at chord fractions ``x``."""
        return cst_airfoil(
            x, self.upper, self.lower, self.trailing_edge_thickness, self.n1, self.n2
        )

    def nose_radius(self) -> float:
        """Return the leading-edge radius: z^2 = 2 r x at the nose gives r = a0^2 / 2 per side."""
        return float((self.upper[0] ** 2 + self.lower[0] ** 2) / 4)

    def te_thickness(self) -> float:
        """Return the trailing-edge thickness in chord lengths."""
        return float(self.trailing_edge_thickness)

    def to_stored(self) -> dict:
        """Return the stored form as plain data."""
        return {
            "upper": list(self.upper),
            "lower": list(self.lower),
            "degree": self.degree,
            "n1": self.n1,
            "n2": self.n2,
            "trailing_edge_thickness": self.trailing_edge_thickness,
            "max_residual": self.max_residual,
            "chord_transform": self.chord_transform,
        }

    def compat(self, others) -> float:
        """Return the largest ordinate difference to each form of ``others``, as one number."""
        mine = self.evaluate(_COMPARE_X)
        worst = 0.0
        for other in others:
            theirs = other.evaluate(_COMPARE_X)
            for a, b in zip(mine, theirs, strict=True):
                worst = max(worst, float(np.max(np.abs(a[:, 1] - b[:, 1]))))
        return worst


def section_from_stored(stored) -> CstSection | BSplineSection:
    """Return the section held by the stored form ``to_stored`` wrote."""
    if "poles" in stored:
        return _bspline_from_stored(stored)
    return _cst_from_stored(stored)


def _cst_from_stored(stored) -> CstSection:
    return CstSection(
        upper=tuple(float(v) for v in stored["upper"]),
        lower=tuple(float(v) for v in stored["lower"]),
        trailing_edge_thickness=float(stored["trailing_edge_thickness"]),
        n1=float(stored.get("n1", 0.5)),
        n2=float(stored.get("n2", 1.0)),
        max_residual=float(stored.get("max_residual", 0.0)),
        chord_transform=stored.get("chord_transform"),
    )


def read_ordinates(path):
    """Read x, z pairs from a text file; one leading-edge point, trailing edge at both ends.

    Rows are two numbers separated by blanks, commas or semicolons; a row that is not two
    numbers (a name or a count line) is skipped. The points run from the upper trailing edge
    round the nose to the lower trailing edge, or the other way round.
    """
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise InputValueError(
            f"sections: ordinates file {str(path)!r}: {error.strerror}"
        ) from error
    rows = []
    for line in text.splitlines():
        parts = line.replace(",", " ").replace(";", " ").split()
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue
        if len(values) == 2 and all(math.isfinite(v) for v in values):
            rows.append(values)
    if len(rows) < 8:
        raise InputValueError(f"sections: ordinates file {str(path)!r}: fewer than 8 x, z rows")
    return np.asarray(rows, float)


def _chord_frame(points, label):
    """Return upper and lower sides, leading edge to trailing edge, and the chord transform.

    The chord is the line from the leading edge (the point farthest from the trailing-edge
    midpoint) to that midpoint; the points are moved to the unit chord frame.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 8 or not np.all(np.isfinite(p)):
        raise InputValueError(f"{label}: at least 8 finite (x, z) points required")
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    te = (p[0] + p[-1]) / 2
    nose = int(np.argmax(np.linalg.norm(p - te, axis=1)))
    if nose in (0, len(p) - 1):
        raise InputValueError(f"{label}: no leading edge between the two trailing-edge ends")
    chord_vec = te - p[nose]
    chord = float(np.linalg.norm(chord_vec))
    axis = chord_vec / chord
    rotation = np.array([[axis[0], -axis[1]], [axis[1], axis[0]]])
    q = (p - p[nose]) @ rotation / chord
    first, second = q[: nose + 1][::-1].copy(), q[nose:].copy()
    upper, lower = (first, second) if first[-1, 1] >= second[-1, 1] else (second, first)
    for name, side in (("upper", upper), ("lower", lower)):
        side[0] = [0.0, 0.0]
        side[-1, 0] = 1.0
        if np.any(np.diff(side[:, 0]) <= 0):
            raise InputValueError(f"{label}: the {name} surface is not a function of x")
    return upper, lower, {"nose": p[nose].tolist(), "chord": chord, "rotation": rotation.tolist()}


def section_from_ordinates(
    points, label, *, degree=5, tolerance=5e-4, max_degree=20, trailing_edge_thickness=None
) -> CstSection:
    """Fit a CST section to a closed contour of (x, z) points.

    The chord is the line from the leading edge (the point farthest from the trailing-edge
    midpoint) to that midpoint; the points are moved to the unit chord frame first.
    """
    upper, lower, transform = _chord_frame(points, label)
    fit = ordinates_to_cst(
        upper, lower, transform, f"{label}",
        degree=degree, tolerance=tolerance, max_degree=max_degree,
        trailing_edge_thickness=trailing_edge_thickness,
    )  # fmt: skip
    return _cst_from_stored(fit)


def _station_label(eta) -> str:
    """Return how a message names the section of the station at ``eta``, as the user wrote it."""
    return "stations.section" if eta is None else f"stations[eta={float(eta):g}].section"


def section_from_item(item, *, degree=5, trailing_edge_thickness=None) -> CstSection:
    """Return the section a ``sections.items`` row names: NACA digits, CST or ordinates.

    A message names the station's ``section`` as the user wrote it, never the resolver's row.
    """
    label = _station_label(item.get("eta"))
    if "bspline" in item:
        raise InputValueError(
            f'{label}.kind: "bspline" resolves, but this version\'s generator does not build '
            'it at this step yet; state the section with kind "naca", "cst" or "coords"'
        )
    if "naca" in item:
        fit = naca_to_cst(
            f"NACA{item['naca']}", degree=degree, trailing_edge_thickness=trailing_edge_thickness
        )
        return _cst_from_stored(fit)
    if "cst_upper" in item:
        return CstSection(
            upper=tuple(float(v) for v in item["cst_upper"]),
            lower=tuple(float(v) for v in item["cst_lower"]),
            trailing_edge_thickness=float(item.get("te", trailing_edge_thickness or 0.0)),
        )
    return section_from_ordinates(
        read_ordinates(item["file"]),
        f"{label}.file {item['file']!r}",
        degree=degree,
        trailing_edge_thickness=trailing_edge_thickness,
    )


def section_from_physical(physical, *, degree=5, trailing_edge_thickness=0.0) -> CstSection:
    """Return the symmetric section whose nose radius, boat-tail angle and thickness are given.

    ``nose_radius`` is the leading-edge radius in chord lengths, ``boat_tail_deg`` the angle
    between each surface and the chord at the trailing edge, ``thickness_at`` the maximum
    thickness in chord lengths and ``te_thickness`` the blunt base (the section default when
    absent). The nose and trailing-edge coefficients follow from the first two, the others
    are one smooth ramp between them scaled until the maximum thickness is the given one.
    """
    missing = [k for k in ("nose_radius", "boat_tail_deg", "thickness_at") if k not in physical]
    if missing:
        raise InputValueError(f"sections.physical: {', '.join(missing)} required")
    te = float(physical.get("te_thickness", trailing_edge_thickness))
    first = math.sqrt(2.0 * float(physical["nose_radius"]))
    last = math.tan(math.radians(float(physical["boat_tail_deg"]))) + te / 2
    wanted = float(physical["thickness_at"])
    grid = np.linspace(0.0, 1.0, 801)
    ramp = np.linspace(first, last, degree + 1)

    def coefficients(scale):
        values = ramp.copy()
        values[1:-1] *= scale
        return values

    def excess(scale):
        values = coefficients(scale)
        upper, lower = cst_airfoil(grid, values, -values, te)
        return float(np.max(upper[:, 1] - lower[:, 1])) - wanted

    if excess(1e-3) > 0 or excess(1e3) < 0:
        raise InputValueError(
            "sections.physical: no section has this nose radius, boat-tail angle and thickness"
        )
    values = coefficients(brentq(excess, 1e-3, 1e3, xtol=1e-13))
    return CstSection(
        upper=tuple(float(v) for v in values),
        lower=tuple(float(-v) for v in values),
        trailing_edge_thickness=te,
    )


def fit_or_read(sections) -> list[CstSection]:
    """Return the section of every station of a resolved ``sections`` table (stage S1).

    A ``physical`` entry gives one section for all stations, ``items`` rows are read from a
    designation, coefficients or a file of ordinates, and a plain airfoil list is fitted.
    A file of ordinates keeps its own trailing-edge thickness, so its S1 residual is measured
    against the file as given; ``trailing_edge_thickness`` sets the base of the other forms.
    """
    degree, te = sections["degree"], sections["trailing_edge_thickness"]
    if sections.get("physical"):
        one = section_from_physical(sections["physical"], degree=degree, trailing_edge_thickness=te)
        return [one for _ in sections["eta"]]
    if sections.get("items"):
        return [
            section_from_item(
                item, degree=degree, trailing_edge_thickness=None if "file" in item else te
            )
            for item in sections["items"]
        ]
    return [
        _cst_from_stored(naca_to_cst(name, degree=degree, trailing_edge_thickness=te))
        for name in sections["airfoils"]
    ]


def _inflections(z):
    """Count the sign changes of the second difference of one surface."""
    curvature = np.diff(z, 2)
    curvature = curvature[np.abs(curvature) > 1e-9]
    return int(np.count_nonzero(np.diff(np.sign(curvature))))


def section_checks(section, *, residual_limit=1e-4) -> list[dict]:
    """Return the S1 check records of one section: residual, nose radius, base, inflections.

    Each record has ``id``, ``status`` (``pass``, ``fail``, ``report`` or ``skipped``),
    ``value`` and ``note``. The residual is skipped when the section was given as coefficients,
    which have no ordinates to compare with. One nose radius is reported, never failed, until
    the stored form is decided.
    """
    if isinstance(section, BSplineSection):
        return _bspline_checks(section)
    x = (1 - np.cos(np.linspace(0, np.pi, 801))) / 2
    upper, lower = section.evaluate(x)
    residual = section.max_residual
    if residual == 0.0:
        status, note = "skipped", "no ordinates to compare"
    else:
        status = "pass" if residual <= residual_limit else "fail"
        note = f"limit {residual_limit} chord lengths"
    return [
        {"id": "section.residual", "status": status, "value": residual, "note": note},
        {
            "id": "section.one_nose_radius",
            "status": "report",
            "value": [section.upper[0] ** 2 / 2, section.lower[0] ** 2 / 2],
            "note": "upper and lower nose radius, chord lengths",
        },
        {
            "id": "section.te_thickness",
            "status": "pass" if section.te_thickness() >= 0 else "fail",
            "value": section.te_thickness(),
            "note": "blunt base, chord lengths",
        },
        {
            "id": "section.inflection",
            "status": "report",
            "value": [_inflections(upper[:, 1]), _inflections(lower[:, 1])],
            "note": "curvature sign changes, upper and lower",
        },
    ]


def shape_unchanged(sections, limit=1e-9) -> dict:
    """Return the S3 record: degree elevation to the common degree must not move any section."""
    if sections.get("form") == "bspline":
        return _bspline_unchanged(sections, limit)
    forms = fit_or_read(sections)
    degree = max(form.degree for form in forms)
    worst = 0.0
    for form in forms:
        stored = form.to_stored()
        raised = _cst_from_stored(
            {
                **stored,
                "upper": elevate_coefficients(stored["upper"], degree).tolist(),
                "lower": elevate_coefficients(stored["lower"], degree).tolist(),
            }
        )
        worst = max(worst, form.compat([raised]))
    return {
        "id": "compat.shape_unchanged",
        "status": "pass" if worst <= limit else "fail",
        "value": worst,
        "note": f"limit {limit} chord lengths, common degree {degree}",
    }


#: Differences below these are the same section for the report: thickness in chord lengths,
#: position in chord fractions.
THICKNESS_TOLERANCE = 2e-3
POSITION_TOLERANCE = 2e-2


def thickness_profile(section: SectionForm) -> dict:
    """Return the maximum thickness and its chord fraction of a section, on a dense grid."""
    x = (1 - np.cos(np.linspace(0, np.pi, 4001))) / 2
    upper, lower = section.evaluate(x)
    thickness = upper[:, 1] - lower[:, 1]
    index = int(np.argmax(thickness))
    return {"thickness": float(thickness[index]), "position": float(x[index])}


def identity_report(seed, measured=None) -> list[dict]:
    """Return one record per measured section: the seed's numbers against the measured ones.

    ``seed`` is a four or five digit designation, as written in the input (``"0012"`` or
    ``"NACA0012"``); ``measured`` is a mapping from a label to a ``SectionForm``. Without
    measured sections the report holds the seed alone. Each record has ``id``
    (``identity.listed``), ``status`` (``match``, ``mismatch`` or ``seed only``; never a
    failure), the two sets of numbers and the differences.
    """
    name = seed if str(seed).upper().startswith("NACA") else f"NACA{seed}"
    own = thickness_profile(_cst_from_stored(naca_to_cst(name.upper())))
    if not measured:
        return [{"id": "identity.listed", "section": str(name), "status": "seed only", "seed": own}]
    records = []
    for label, section in measured.items():
        found = thickness_profile(section)
        d_thickness = found["thickness"] - own["thickness"]
        d_position = found["position"] - own["position"]
        same = abs(d_thickness) <= THICKNESS_TOLERANCE and abs(d_position) <= POSITION_TOLERANCE
        records.append(
            {
                "id": "identity.listed",
                "section": str(label),
                "status": "match" if same else "mismatch",
                "seed": own,
                "measured": found,
                "difference": {"thickness": d_thickness, "position": d_position},
            }
        )
    return records


# ---------------------------------------------------------------------------------------------
# The stored form: one direct B-spline per section
# ---------------------------------------------------------------------------------------------

#: Every stored section is degree 5 with interior knots of multiplicity 3 (C2), the form the
#: surveyed lifting skins use.
SECTION_DEGREE = 5
SECTION_MULTIPLICITY = 3
#: The conversion starts at this many spans and rises by one until every section meets its
#: limit; poles = 3 spans + 3. The cap only stops a runaway: a fit above its limit is reported
#: by ``section.residual``, never refused.
MIN_SPANS, MAX_SPANS = 8, 60
#: Maximum normal distance of a converted analytic section (NACA or CST), in chord lengths.
ANALYTIC_LIMIT = 2e-5
#: The default limit for a file of ordinates; the ``tolerance`` of the item states another.
FILE_LIMIT = 1e-4
#: A base at or below this is a sharp edge (chord lengths): a closed section as tabulated still
#: carries a base of the order of 1e-5 from its rounded ordinates.
SHARP_EDGE = 1e-4
#: The nose radius of a symmetric NACA section is this times t squared; a cambered one depends
#: on the camber slope at the nose and is reported, not checked.
NOSE_FACTOR = 1.1019
#: A stored section with more poles than this draws a warning (the original's skin has 93).
POLE_WARNING = 150
#: Curvature below this (per chord length, a radius of 100 chords) is straight for the inflection
#: count: the data and a fit differ by noise there, not by a shape.
CURVATURE_FLOOR = 1e-2
_SAMPLES = 2001
_CHECK_X = (1 - np.cos(np.linspace(0, np.pi, 801))) / 2
_DENSE_X = (1 - np.cos(np.linspace(0, np.pi, _SAMPLES))) / 2


def _curve(poles, knots, degree) -> BSpline:
    return BSpline(np.asarray(knots, float), np.asarray(poles, float), int(degree))


@dataclass(frozen=True)
class BSplineSection:
    """A section as one open clamped non-rational B-spline in chord units.

    The curve runs from the lower trailing edge through the leading edge to the upper trailing
    edge; the parameter is 0.5 at the leading edge. ``limit`` is the largest normal distance the
    conversion was held to (0 for a curve read exactly) and ``max_residual`` the one measured.
    """

    poles: tuple[tuple[float, float], ...]
    knots: tuple[float, ...]
    degree: int = SECTION_DEGREE
    max_residual: float = 0.0
    limit: float = 0.0
    source_inflections: tuple[int, int] | None = None
    nose_reference: float | None = None
    stated_te: float | None = None
    chord_transform: dict | None = None

    def _spline(self) -> BSpline:
        return _curve(self.poles, self.knots, self.degree)

    def evaluate(self, x):
        """Return the upper and lower (x, z) points at chord fractions ``x``."""
        x = np.asarray(x, float)
        if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise InputValueError("x: unit chord range required")
        spline = self._spline()
        sides = []
        for end in (1.0, 0.0):  # upper, then lower
            low, high = np.full(x.shape, 0.5), np.full(x.shape, end)
            for _ in range(60):
                middle = (low + high) / 2
                behind = spline(middle)[..., 0] < x
                low, high = np.where(behind, middle, low), np.where(behind, high, middle)
            sides.append(np.stack((x, spline((low + high) / 2)[..., 1]), axis=-1))
        return sides[0], sides[1]

    def nose_radius(self) -> float:
        """Return the radius of curvature at the leading edge, in chord lengths."""
        spline = self._spline()
        d1, d2 = spline(0.5, 1), spline(0.5, 2)
        cross = abs(float(d1[0] * d2[1] - d1[1] * d2[0]))
        return float(np.linalg.norm(d1) ** 3 / cross) if cross > 0 else math.inf

    def te_thickness(self) -> float:
        """Return the distance between the two trailing-edge ends, in chord lengths."""
        return float(np.linalg.norm(np.asarray(self.poles[-1]) - np.asarray(self.poles[0])))

    def to_stored(self) -> dict:
        """Return the stored form as plain data."""
        return {
            "form": "bspline",
            "degree": self.degree,
            "poles": [list(p) for p in self.poles],
            "knots": list(self.knots),
            "max_residual": self.max_residual,
            "limit": self.limit,
            "source_inflections": None
            if self.source_inflections is None
            else list(self.source_inflections),
            "nose_reference": self.nose_reference,
            "stated_te": self.stated_te,
            "chord_transform": self.chord_transform,
        }

    def compat(self, others) -> float:
        """Return the largest ordinate difference to each form of ``others``, as one number."""
        return _compat(self, others)


def _compat(first, others) -> float:
    mine = first.evaluate(_COMPARE_X)
    worst = 0.0
    for other in others:
        for a, b in zip(mine, other.evaluate(_COMPARE_X), strict=True):
            worst = max(worst, float(np.max(np.abs(a[:, 1] - b[:, 1]))))
    return worst


def _bspline_from_stored(stored) -> BSplineSection:
    inflections = stored.get("source_inflections")
    return BSplineSection(
        poles=tuple((float(p[0]), float(p[1])) for p in stored["poles"]),
        knots=tuple(float(k) for k in stored["knots"]),
        degree=int(stored.get("degree", SECTION_DEGREE)),
        max_residual=float(stored.get("max_residual", 0.0)),
        limit=float(stored.get("limit", 0.0)),
        source_inflections=None
        if inflections is None
        else (int(inflections[0]), int(inflections[1])),
        nose_reference=stored.get("nose_reference"),
        stated_te=stored.get("stated_te"),
        chord_transform=stored.get("chord_transform"),
    )


def section_from_bspline(source, label="sections.items.bspline") -> BSplineSection:
    """Return the section a ``bspline`` input names, read exactly: poles, knots and degree.

    ``source`` is a mapping with ``poles`` (rows of x, z), ``knots`` (the full clamped vector on
    [0, 1]) and ``degree``, or the path of a JSON file holding one. A degree below 5 is raised
    to 5 without changing the curve.
    """
    if not isinstance(source, dict):
        try:
            source = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise InputValueError(f"{label}: cannot read {str(source)!r}: {error}") from error
    if not isinstance(source, dict) or {"poles", "knots", "degree"} - set(source):
        raise InputValueError(f"{label}: expected poles, knots and degree")
    poles, knots = np.asarray(source["poles"], float), np.asarray(source["knots"], float)
    degree = source["degree"]
    if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= SECTION_DEGREE:
        raise InputValueError(f"{label}.degree: expected an integer from 1 to {SECTION_DEGREE}")
    if poles.ndim != 2 or poles.shape[1] != 2 or not np.all(np.isfinite(poles)):
        raise InputValueError(f"{label}.poles: expected finite rows of x, z")
    clamped = (
        knots.ndim == 1
        and len(knots) == len(poles) + degree + 1
        and np.all(np.isfinite(knots))
        and np.all(np.diff(knots) >= 0)
        and np.all(knots[: degree + 1] == 0.0)
        and np.all(knots[-degree - 1 :] == 1.0)
    )
    if not clamped:
        raise InputValueError(
            f"{label}.knots: expected a clamped vector on [0, 1], poles + degree + 1"
        )
    if degree < SECTION_DEGREE:
        poles, knots, _ = elevate_curve(poles, knots, degree, SECTION_DEGREE - degree)
    return BSplineSection(
        poles=tuple((float(p[0]), float(p[1])) for p in poles),
        knots=tuple(float(k) for k in knots),
        degree=SECTION_DEGREE,
    )


@dataclass(frozen=True)
class _Prepared:
    """One section ready for the joint fit: dense sides in the chord frame and its limits."""

    upper: np.ndarray
    lower: np.ndarray
    limit: float
    inflections: tuple[int, int] | None
    transform: dict | None = None
    nose_reference: float | None = None
    stated_te: float | None = None
    exact: BSplineSection | None = None


def _wedge(upper, lower, thickness):
    """Return the sides with the trailing-edge base changed to ``thickness`` by a linear wedge."""
    up, low = upper.copy(), lower.copy()
    up[:, 1] += up[:, 0] * (thickness / 2 - upper[-1, 1])
    low[:, 1] += low[:, 0] * (-thickness / 2 - lower[-1, 1])
    return up, low


def _flips(x, z) -> int:
    """Count the curvature sign changes of z(x), ignoring curvature below ``CURVATURE_FLOOR``."""
    h = np.diff(x)
    slope = np.diff(z) / h
    curvature = 2 * np.diff(slope) / (x[2:] - x[:-2])
    curvature = curvature[np.abs(curvature) > CURVATURE_FLOOR]
    return int(np.count_nonzero(np.diff(np.sign(curvature))))


def _source_inflections(upper, lower):
    out = []
    for side in (upper, lower):
        z = PchipInterpolator(side[:, 0], side[:, 1])(_CHECK_X)
        out.append(_flips(_CHECK_X, z))
    return out[0], out[1]


def _prepared(item, te, label, warnings) -> _Prepared:
    """Return the dense sides and limits of one items row; ``te`` is the default base."""
    if "bspline" in item:
        return _Prepared(
            np.empty((0, 2)), np.empty((0, 2)), 0.0, None,
            exact=section_from_bspline(item["bspline"], label),
        )  # fmt: skip
    nose = stated = transform = None
    limit = ANALYTIC_LIMIT
    if "naca" in item:
        digits = naca_digits(f"NACA{item['naca']}")
        upper, lower, transform = normalized_naca(f"NACA{digits}", _SAMPLES)
        if not np.any(naca_camber(digits, [0.3, 0.6])[0]):
            nose = NOSE_FACTOR * (int(digits[-2:]) / 100) ** 2
        if te is not None:
            upper, lower = _wedge(upper, lower, te)
    elif "cst_upper" in item:
        section = CstSection(
            tuple(float(v) for v in item["cst_upper"]),
            tuple(float(v) for v in item["cst_lower"]),
            float(item.get("te", te or 0.0)),
        )
        upper, lower = section.evaluate(_DENSE_X)
    else:
        upper, lower, transform = _chord_frame(read_ordinates(item["file"]), label)
        limit = float(item.get("tolerance", FILE_LIMIT))
        upper, lower = _resampled(upper, lower)
        if upper[-1, 1] - lower[-1, 1] <= SHARP_EDGE and te is not None and te > SHARP_EDGE:
            warnings.append(
                f"{label}.file: sharp trailing edge, the base {te} is used instead (blunt in the "
                "IGES, collapsed at mesh generation)"
            )
            upper, lower = _wedge(upper, lower, te)
    if "naca" in item or "cst_upper" in item:
        stated = float(upper[-1, 1] - lower[-1, 1])
    return _Prepared(
        upper, lower, limit, _source_inflections(upper, lower), transform, nose, stated
    )


def _resampled(upper, lower):
    """Return the sides of a file on the dense grid, through shape-preserving interpolation."""
    out = []
    for side in (upper, lower):
        z = PchipInterpolator(side[:, 0], side[:, 1])(_DENSE_X)
        z[-1] = side[-1, 1]
        out.append(np.stack((_DENSE_X, z), axis=-1))
    return out[0], out[1]


def _side_parameter(side, pins):
    """Return the chord-length parameter t in [0, 1] of a side, with each pin at t = sqrt(x)."""
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(side, axis=0), axis=1))]
    at = np.interp(pins, side[:, 0], arc)
    return np.interp(arc, np.r_[0.0, at, arc[-1]], np.r_[0.0, np.sqrt(pins), 1.0])


def _check_pins(pins):
    values = np.asarray(pins if pins is not None else [], float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
        raise InputValueError("sections.pin: expected chord fractions strictly inside (0, 1)")
    if np.any(np.diff(values) <= 0):
        raise InputValueError("sections.pin: expected strictly increasing chord fractions")
    return values


def _required_knots(pins):
    half = 0.5 * np.sqrt(pins)
    return sorted({0.5, *(0.5 - half).tolist(), *(0.5 + half).tolist()})


def _warp(u):
    """Return the nose-clustered coordinate of a parameter: sqrt of the distance to 0.5."""
    d = 2 * np.asarray(u, float) - 1
    return 0.5 + 0.5 * np.sign(d) * np.sqrt(np.abs(d))


def _unwarp(q):
    d = 2 * np.asarray(q, float) - 1
    return float(0.5 + 0.5 * np.sign(d) * d * d)


def _interior_knots(spans, pins):
    """Return the ``spans - 1`` distinct interior knots: the pinned, then halved gaps.

    Gaps are measured in a coordinate that clusters at the nose, where a section bends most.
    """
    knots = _required_knots(pins)
    if len(knots) > spans - 1:
        raise InputValueError("sections.pin: more pinned parameters than interior knots")
    while len(knots) < spans - 1:
        edges = np.r_[0.0, _warp(knots), 1.0]
        at = int(np.argmax(np.diff(edges)))
        middle = _unwarp((edges[at] + edges[at + 1]) / 2)
        knots.insert(at, middle)
        mirror = 1.0 - middle
        if len(knots) < spans - 1 and all(abs(k - mirror) > 1e-12 for k in knots):
            knots = sorted([*knots, mirror])
    return knots


def full_knot_vector(interior) -> np.ndarray:
    """Return the degree 5 clamped vector with every interior knot of multiplicity 3."""
    return clamped_knots(interior, SECTION_DEGREE, [SECTION_MULTIPLICITY] * len(interior))


def _normal_distance(points, poles, knots) -> float:
    """Return the largest distance from ``points`` to the curve, measured to its segments."""
    dense = _curve(poles, knots, SECTION_DEGREE)(np.linspace(0.0, 1.0, 20001))
    near = cKDTree(dense).query(points)[1]
    best = np.full(len(points), np.inf)
    for shift in (-1, 0):
        a = dense[np.clip(near + shift, 0, len(dense) - 2)]
        b = dense[np.clip(near + shift + 1, 1, len(dense) - 1)]
        edge = b - a
        t = np.clip(np.sum((points - a) * edge, axis=1) / np.sum(edge * edge, axis=1), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * edge), axis=1))
    return float(best.max())


#: Weight of a pinned sample in the least squares fit: the curve passes through it at its own
#: parameter to well below the limit, which a plain fit does not do (it fixes the distance to
#: the curve, not the position along it).
PIN_WEIGHT = 1e4


def _fit_one(prep, knots, pins):
    tu, tl = _side_parameter(prep.upper, pins), _side_parameter(prep.lower, pins)
    points = np.vstack([prep.lower[::-1], prep.upper[1:]])
    parameters = np.r_[0.5 - 0.5 * tl[::-1], 0.5 + 0.5 * tu[1:]]
    weights = np.ones(len(points))
    if len(pins):
        spots = [(side, sign) for side, sign in ((prep.lower, -1.0), (prep.upper, 1.0))]
        extra = [
            (np.array([x, np.interp(x, side[:, 0], side[:, 1])]), 0.5 + sign * 0.5 * np.sqrt(x))
            for side, sign in spots
            for x in pins
        ]
        points = np.vstack([points, [e[0] for e in extra]])
        parameters = np.r_[parameters, [e[1] for e in extra]]
        weights = np.r_[weights, np.full(len(extra), PIN_WEIGHT)]
        order = np.argsort(parameters, kind="stable")
        points, parameters, weights = points[order], parameters[order], weights[order]
    basis = curve_basis(knots, SECTION_DEGREE, parameters)
    rhs = points - np.outer(basis[:, 0], points[0]) - np.outer(basis[:, -1], points[-1])
    inner = basis[:, 1:-1] * weights[:, None]
    middle = np.linalg.lstsq(inner, rhs * weights[:, None], rcond=None)[0]
    poles = np.vstack([points[:1], middle, points[-1:]])
    return poles, _normal_distance(points, poles, knots)


def _joint_fit(preps, pins):
    """Fit every analytic or file section on one knot vector with the fewest spans."""
    spans = max(MIN_SPANS, len(_required_knots(pins)) + 1)
    while True:
        knots = full_knot_vector(_interior_knots(spans, pins))
        fits = [_fit_one(p, knots, pins) for p in preps]
        sound = all(_sound(p, knots, fit) for p, fit in zip(preps, fits, strict=True))
        if spans >= MAX_SPANS or sound:
            return knots, fits
        spans += 1


def _sound(prep, knots, fit) -> bool:
    """Return whether a fit meets its limit and the S1 checks that a finer fit can repair."""
    section = _assembled(prep, fit[0], knots, fit[1])
    return all(check["status"] != "fail" for check in _bspline_checks(section))


def _refined(poles, knots, target):
    """Return the same curve on the finer knot vector ``target`` (exact to rounding)."""
    if len(knots) == len(target) and np.array_equal(knots, target):
        return np.asarray(poles, float)
    params = np.unique(np.r_[np.linspace(0, 1, max(801, 4 * len(target))), np.unique(target)])
    refined, _ = fit_curve(
        _curve(poles, knots, SECTION_DEGREE)(params), params, target, SECTION_DEGREE
    )
    return refined


def _assembled(prep, poles, knots, residual) -> BSplineSection:
    return BSplineSection(
        poles=tuple((float(p[0]), float(p[1])) for p in poles),
        knots=tuple(float(k) for k in knots),
        max_residual=float(residual),
        limit=prep.limit,
        source_inflections=prep.inflections,
        nose_reference=prep.nose_reference,
        stated_te=prep.stated_te,
        chord_transform=prep.transform,
    )


def _row_of(item):
    """Return an items row in the internal keys, reading a ``section = {kind = ...}`` table too.

    The proposed input names a section ``kind`` (``bspline``, ``naca``, ``cst`` or ``coords``)
    with ``degree``, ``knots``, ``poles`` or ``file`` beside it; the other keys of the row stay.
    """
    table = item.get("section")
    if not isinstance(table, dict):
        return dict(item)
    row = {k: v for k, v in item.items() if k != "section"}
    kind = table.get("kind")
    body = {k: v for k, v in table.items() if k != "kind"}
    if kind == "bspline":
        row["bspline"] = body.pop("file") if "file" in body else body
    elif kind == "coords":
        row["file"] = body.pop("file", "")
        row.update(body)
    elif kind in ("naca", "cst"):
        row.update(body)
    else:
        raise InputValueError("sections.items.section.kind: expected bspline, naca, cst or coords")
    return row


def _items_of(sections, te, warnings):
    """Return the ``sections.items``-style rows of a resolved table, whatever its form."""
    if sections.get("physical"):
        one = section_from_physical(sections["physical"], degree=sections["degree"],
                                    trailing_edge_thickness=te)  # fmt: skip
        row = {"cst_upper": list(one.upper), "cst_lower": list(one.lower), "te": one.te_thickness()}
        return [dict(row) for _ in sections["eta"]]
    if sections.get("items"):
        return [_row_of(item) for item in sections["items"]]
    return [{"naca": name.upper().replace("NACA", "")} for name in sections["airfoils"]]


def bspline_sections(sections, warnings=None) -> list[BSplineSection]:
    """Return the section of every station of a resolved ``sections`` table as a B-spline.

    All sections share one knot vector: the analytic and file sections are fitted jointly with
    the fewest spans that meet each limit, a ``bspline`` input is read exactly, and the vectors
    are united by exact knot insertion. A file whose trailing edge is sharp takes the base
    ``sections.trailing_edge_thickness`` and adds a warning to ``warnings``.
    """
    warnings = [] if warnings is None else warnings
    te, pins = sections["trailing_edge_thickness"], _check_pins(sections.get("pin"))
    rows = _items_of(sections, te, warnings)
    etas = list(sections.get("eta") or [])
    preps = [
        _prepared(row, te, _station_label(etas[i] if i < len(etas) else None), warnings)
        for i, row in enumerate(rows)
    ]
    fitted = [p for p in preps if p.exact is None]
    fit_knots, fits = _joint_fit(fitted, pins) if fitted else (None, [])
    vectors = ([fit_knots] if fitted else []) + [
        np.asarray(p.exact.knots) for p in preps if p.exact is not None
    ]
    common = full_knot_vector([])
    for i, vector in enumerate(vectors):
        common = vector if i == 0 else union_knots(common, vector)
    out, fit_iter = [], iter(fits)
    for prep in preps:
        if prep.exact is not None:
            poles = _refined(prep.exact.poles, np.asarray(prep.exact.knots), common)
            out.append(_assembled(prep, poles, common, 0.0))
        else:
            poles, residual = next(fit_iter)
            out.append(_assembled(prep, _refined(poles, fit_knots, common), common, residual))
    if out and len(out[0].poles) > POLE_WARNING:
        warnings.append(f"sections: {len(out[0].poles)} poles per section, above {POLE_WARNING}")
    return out


def trailing_edge_notes(name, sections, warnings) -> None:
    """Record the base each file section will use and warn about a sharp edge (resolve time).

    Writes ``trailing_edge_used`` into every ``sections.items`` row that names a ``file``: the
    base the file carries, or ``sections.trailing_edge_thickness`` when the file's edge is sharp.
    A file that cannot be read is left alone; the generator refuses it with the reason.
    """
    default = sections["trailing_edge_thickness"]
    for i, item in enumerate(sections.get("items") or []):
        if not item.get("file"):
            continue
        try:
            upper, lower, _ = _chord_frame(read_ordinates(item["file"]), name)
        except InputValueError:
            continue
        own = float(upper[-1, 1] - lower[-1, 1])
        sharp = own <= SHARP_EDGE and default > SHARP_EDGE
        item["trailing_edge_used"] = float(default if sharp else own)
        if sharp:
            warnings.append(
                f"{name}.sections.items[{i}].file: sharp trailing edge, the base {default} is "
                "used instead (blunt in the IGES, collapsed at mesh generation)"
            )


def section_forms(sections):
    """Return the sections of a resolved table in the form it names (``bspline`` or ``cst``)."""
    return (
        bspline_sections(sections) if sections.get("form") == "bspline" else fit_or_read(sections)
    )


def _bspline_checks(section: BSplineSection) -> list[dict]:
    upper, lower = section.evaluate(_CHECK_X)
    exact = section.limit == 0.0
    residual = {
        "id": "section.residual",
        "status": "pass" if section.max_residual <= section.limit else "fail",
        "value": section.max_residual,
        "note": "read exactly" if exact else f"limit {section.limit} chord lengths",
    }
    radius, reference = section.nose_radius(), section.nose_reference
    nose = {"id": "section.one_nose_radius", "status": "report", "value": radius,
            "note": "one nose radius, chord lengths"}  # fmt: skip
    if reference:
        nose["status"] = "pass" if abs(radius - reference) <= 0.02 * reference else "fail"
        nose["note"] = f"within 2 percent of {NOSE_FACTOR} t^2 = {reference}"
    base = {"id": "section.te_thickness", "status": "report", "value": section.te_thickness(),
            "note": "blunt base, chord lengths"}  # fmt: skip
    if section.stated_te is not None:
        near = abs(section.te_thickness() - section.stated_te) <= 1e-6
        base["status"] = "pass" if near else "fail"
        base["note"] = f"stated {section.stated_te}, within 1e-6 chord lengths"
    found = (_flips(_CHECK_X, upper[:, 1]), _flips(_CHECK_X, lower[:, 1]))
    own = section.source_inflections
    flips = {"id": "section.inflection", "status": "report", "value": list(found),
             "note": "curvature sign changes, upper and lower"}  # fmt: skip
    if own is not None:
        flips["status"] = "pass" if found[0] <= own[0] and found[1] <= own[1] else "fail"
        flips["note"] = f"not more than the data's own {list(own)}"
    return [residual, nose, base, flips]


def _bspline_unchanged(sections, limit) -> dict:
    forms = bspline_sections(sections)
    worst = 0.0
    for form in forms:
        poles, knots, degree = elevate_curve(form.poles, np.asarray(form.knots), form.degree, 1)
        raised = BSplineSection(
            tuple((float(p[0]), float(p[1])) for p in poles),
            tuple(float(k) for k in knots),
            degree,
        )
        worst = max(worst, form.compat([raised]))
    return {
        "id": "compat.shape_unchanged",
        "status": "pass" if worst <= limit else "fail",
        "value": worst,
        "note": f"limit {limit} chord lengths, common degree {SECTION_DEGREE + 1}",
    }
