"""Identifying a lifting surface from an original IGES skin (uses the gmsh kernel).

Pipeline role: the workspace row of the reverse step. A component of an IGES file (named on
its level, the 406 NAME index) is cut by planes normal to its span axis on the trimmed faces
with OpenCASCADE (Newton-corrected polylines, never bridging a missing surface); the cuts feed
the numerical stage of :mod:`cfdbender.geometry._reverse`, which finds the clean band, the
planform between a root and the outermost clean cut and the CST sections. The root is the
plane crossing of the extended leading edge for a surface on a symmetry plane, the meeting of
the extended leading edge with a host surface for a surface on a body, or the span extent of
the original's own trimmed boundary curves for a surface open inside its host.

Nothing is repaired silently: a station that is not one closed airfoil contour is recorded
with its reason and left out of the band. The entry point and its result are
:func:`cfdbender.geometry._identify.identify_lifting_surface` and
:class:`~cfdbender.geometry._identify.LiftingIdentification`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from cfdbender._errors import InputValueError
from cfdbender.geometry._geometry_check import (
    edge_samples,
    edges_of,
    resolve_faces,
)
from cfdbender.geometry.stages import (
    clean_band,
    closed_contour,
    contour_features,
    cosine,
    leading_extension,
    reference_planform,
    refine_tip,
)

__all__ = [
    "COARSE",
    "CUTS",
    "PURPOSE",
    "ROOTS",
    "boundary_extent",
    "convergence",
    "identify_pass",
    "refuse_tolerances",
    "resolve_hosts",
    "span_direction",
    "surface_grid",
]

PURPOSE = "identifying a lifting surface from an original IGES skin"
#: Convergence tolerances of the coarse pass against the fine pass, by planform quantity.
CONVERGENCE_TOLERANCES = {
    "semi_span_m": 0.002,
    "root_chord_m": 0.0005,
    "tip_chord_m": 0.0005,
    "area_m2": 0.01,
    "mac_m": 0.0005,
    "incidence_deg": 0.01,
    "dihedral_deg": 0.01,
    "tip_twist_deg": 0.01,
}
#: The coarse pass: stations and cut-grid points per side.
COARSE = (81, 129)
ROOTS = ("auto", "plane", "host", "boundary")
CUTS = ("plain", "bridge_te", "mirror")


# --- cuts -------------------------------------------------------------------------------


def surface_grid(kernel, face, count=257):
    """Return the face, its parameter bounds and a cosine-spaced grid of surface points."""
    low, high = map(np.asarray, kernel.uv_bounds(face))
    u = low[0] + np.ptp([low[0], high[0]]) * cosine(count)
    v = low[1] + np.ptp([low[1], high[1]]) * cosine(count)
    uv = np.stack(np.meshgrid(u, v), axis=-1)
    xyz = np.asarray(kernel.values(face, uv.reshape(-1, 2))).reshape(count, count, 3)
    return face, low, high, u, v, xyz


def boundary_intersections(kernel, face, normal, station):
    """Return the points where the trim edges of ``face`` cross the plane at ``station``."""
    points = []
    for edge in sorted({abs(t) for t in kernel.boundary(face)}):
        low, high = kernel.edge_bounds(edge)
        t = low + (high - low) * cosine(257)
        p = np.asarray(kernel.edge_values(edge, t))
        f = p @ normal - station

        def value(at, edge=edge):
            return np.asarray(kernel.edge_values(edge, [at]))[0] @ normal - station

        for i in np.flatnonzero(f[:-1] * f[1:] < 0):
            root = brentq(value, t[i], t[i + 1], xtol=1e-13)
            points.append(np.asarray(kernel.edge_values(edge, [root]))[0])
        points.extend(p[np.abs(f) < 1e-10])
    return points


def cut_grid(kernel, grid, normal, station, tolerance=1e-8):
    """Return the contiguous trimmed polylines of one face on a plane, never bridging a gap."""
    import contourpy

    face, low, high, u, v, xyz = grid
    normal = np.asarray(normal, float)
    heights = xyz @ normal
    if station < heights.min() or station > heights.max():
        return []
    generator = contourpy.contour_generator(x=u, y=v, z=heights)
    segments = []
    for q in generator.lines(station):
        for _ in range(12):
            p = np.asarray(kernel.values(face, q))
            error = p @ normal - station
            if np.max(np.abs(error)) < tolerance:
                break
            der = np.asarray(kernel.derivatives(face, q))
            grad = der @ normal
            step = error[:, None] * grad / np.maximum(np.sum(grad**2, axis=1)[:, None], 1e-30)
            q = np.clip(q - step, low, high)
        p = np.asarray(kernel.values(face, q))
        good = np.asarray(kernel.inside(face, q)).astype(bool)
        good &= np.abs(p @ normal - station) <= tolerance
        runs = np.split(np.arange(len(p)), np.flatnonzero(np.diff(good.astype(int))) + 1)
        segments.extend(p[r] for r in runs if len(r) >= 2 and good[r[0]])
    boundary = boundary_intersections(kernel, face, normal, station)
    if segments:
        for point in boundary:
            candidates = [
                (np.linalg.norm(line[k] - point), i, k)
                for i, line in enumerate(segments)
                for k in (0, -1)
            ]
            _, i, k = min(candidates)
            segments[i] = (
                np.vstack([point, segments[i]]) if k == 0 else np.vstack([segments[i], point])
            )
    elif boundary:
        segments = [np.asarray(boundary)]
    return segments


def cut_surfaces(kernel, grids, normal, station):
    """Return every polyline the faces give on a plane; a plane that meets none is refused."""
    segments = [line for grid in grids for line in cut_grid(kernel, grid, normal, station)]
    if not segments:
        raise InputValueError(f"section at {station}: no trimmed intersection")
    return segments


def bridged_cut(kernel, grids, span, station, limit=0.01):
    """Cut an original whose blunt trailing edge is an open slit, closing it with a segment."""
    segments = [np.asarray(s) for s in cut_surfaces(kernel, grids, span, station)]
    ends = [(i, k, s[k]) for i, s in enumerate(segments) for k in (0, -1)]
    loose = []
    for i, k, p in ends:
        partners = [np.linalg.norm(p - q) for j, m, q in ends if (j, m) != (i, k)]
        if min(partners) > 5e-5:
            loose.append(p)
    if len(loose) == 2 and np.linalg.norm(loose[0] - loose[1]) < limit:
        segments.append(np.array([loose[0], loose[1]]))
    return segments


def mirrored_cut(kernel, grids, span, station):
    """Cut a half-body original (one side of its symmetry plane) and add the reflection."""
    segments = cut_surfaces(kernel, grids, span, station)
    return segments + [np.asarray(line) * [1.0, -1.0, 1.0] for line in segments]


def host_intersection(kernel, curve, faces, bracket, tolerance=5e-5):
    """Find where a leading-edge curve meets a trimmed host face; the outermost meeting wins."""

    def closest(station, trimmed=False):
        p = curve(station)
        candidates = []
        for face in faces:
            xyz, uv = kernel.project(face, p.reshape(1, 3))
            if trimmed and not kernel.inside(face, uv[:1])[0]:
                continue
            gap = float(np.linalg.norm(xyz[0] - p))
            candidates.append((gap, face, xyz[0]))
        if not candidates:
            return float("inf"), None, None
        return min(candidates, key=lambda row: row[0])

    stations = np.linspace(*bracket, 81)
    distances = np.array([closest(s)[0] for s in stations])
    minima = (
        np.flatnonzero((distances[1:-1] <= distances[:-2]) & (distances[1:-1] <= distances[2:])) + 1
    )
    candidates = []
    for i in minima:
        fit = minimize_scalar(
            lambda s: closest(s)[0] ** 2,
            bounds=(stations[i - 1], stations[i + 1]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        gap, face, _ = closest(fit.x, trimmed=True)
        if gap <= tolerance:
            candidates.append(
                {
                    "station_m": float(fit.x),
                    "gap_m": gap,
                    "host_face": face,
                    "point_m": curve(fit.x).tolist(),
                }
            )
    if not candidates:
        raise InputValueError(
            f"the leading-edge extension does not meet a trimmed host within {tolerance} m; "
            f"sampled minimum {float(distances.min())} m"
        )
    return max(candidates, key=lambda row: row["station_m"])


def boundary_extent(kernel, faces, span, count=400):
    """Return the span-station extent of the original's own trimmed boundary curves."""
    s = np.concatenate([edge_samples(kernel, e, count) @ span for e in edges_of(kernel, faces)])
    return float(s.min()), float(s.max())


# --- the driver -------------------------------------------------------------------------


def scan(kernel, grids, samples, span, count, *, cutter):
    """Cut the component at ``count`` cosine-spaced stations; return stations, features, reasons."""
    bounds = samples @ span
    lo, hi = float(bounds.min()), float(bounds.max())
    inset = (hi - lo) * 1e-7
    stations = lo + inset + (hi - lo - 2 * inset) * cosine(count)
    features, failures = [], []
    for s in stations:
        try:
            contour = closed_contour(cutter(kernel, grids, span, s))
            features.append(contour_features(contour, span))
            failures.append(None)
        except ValueError as error:
            features.append(None)
            failures.append(str(error))
    return stations, features, failures


def span_direction(samples, span_axis):
    """Return the unit span axis: the one given, or y or z (the larger extent), toward the mass."""
    if span_axis is not None:
        axis = np.asarray(span_axis, float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0:
            raise InputValueError("span_axis: three finite numbers, not all zero, required")
        return axis / np.linalg.norm(axis)
    extent = np.ptp(samples, axis=0)
    axis = np.zeros(3)
    index = 1 if extent[1] >= extent[2] else 2
    axis[index] = 1.0 if samples[:, index].mean() >= 0 else -1.0
    return axis


def _end_plane_station(stations, features, band, span, *, samples, axis):
    """Return the span station where the leading edge reaches the tip end plane, or None.

    The cuts are normal to the span axis, and the end plane of the original is normal to the
    coordinate axis ``axis`` at the extreme sample along it. The published span includes the
    tip cap (the skin ends one cap reach inboard), so the extreme sample lies on the end plane
    whatever the cap. Two reasons put the outermost clean cut short of that plane, and the
    leading edge is extended to it in both:

    - a tilted span axis (dihedral): a cut stops being clean when the far side of the section
      crosses the end plane, at most the section's height across the chord times the tangent
      of the tilt before the leading edge does;
    - a rounded, ridged or lofted tip cap: the skin, and so the clean band, ends one cap reach
      inboard of the end plane. That reach is added back on any tilt (R14-001 measured the
      case of a dihedral wing).

    Returns None when the extended leading edge does not reach the plane within one band
    length of the outermost clean cut; the outermost clean station is then kept.
    """
    side = 1.0 if span[axis] > 0 else -1.0
    end = float(samples[:, axis].max() if side > 0 else samples[:, axis].min())
    curve = leading_extension(stations, features, band)
    last = float(stations[band[-1]])
    reach = float(np.ptp(np.asarray(stations)))
    try:
        station = float(
            brentq(lambda s: (curve(s)[axis] - end) * side, last, last + reach, xtol=1e-10)
        )
    except ValueError:
        return None
    return station


def identify_pass(kernel, grids, samples, hosts, spec, *, count):
    """One pass: scan, band, planform. Returns the planform, band and scan diagnostics."""
    cutter = {"plain": cut_surfaces, "bridge_te": bridged_cut, "mirror": mirrored_cut}[spec["cut"]]
    span = spec["span"].copy()
    for iteration in range(2):
        stations, features, failures = scan(
            kernel, grids, samples, span, min(count, 81) if iteration == 0 else count, cutter=cutter
        )
        band, _, _ = clean_band(stations, features)
        if iteration == 0:
            le = np.array([features[i]["le"] for i in band])[:, 1:]
            slope = np.polyfit(stations[band], le, 1)[0]
            candidate = np.r_[0.0, slope] / np.linalg.norm(slope)
            settled = np.linalg.norm(candidate - span) < 1e-6
            if settled and count <= 81:
                break  # this scan already holds every station asked for
            if not settled:
                span = candidate

    def evaluate(s):
        return contour_features(closed_contour(cutter(kernel, grids, span, s)), span)

    # Labelled before the refinement, which inserts a station the failures do not hold.
    rejected = {f"{s:.6g}": reason for s, reason in zip(stations, failures, strict=True) if reason}
    stations, features, band, bracket = refine_tip(stations, features, band, evaluate)
    mirrored = spec["mirrored"]
    root = spec["root"]
    if root == "boundary":
        lo, hi = boundary_extent(
            kernel, spec["faces"], span
        )  # along the settled axis, as the stations
        result = reference_planform(
            stations, features, band, span, root_station=lo, tip_station=hi,
            hold_incidence=True, mirrored=mirrored,
        )  # fmt: skip
        result["root_kind"] = "minimum span station of the original face boundary curves"
    elif root == "plane":
        axis = int(np.argmax(np.abs(span)))
        result = reference_planform(
            stations, features, band, span,
            boundary=lambda p: p[axis], root_bracket=(stations[0] - 1.0, stations[band[0]] + 1e-4),
            tip_station=_end_plane_station(
                stations, features, band, span, samples=samples, axis=axis
            ),
            mirrored=mirrored,
        )  # fmt: skip
        result["root_kind"] = f"symmetry plane {'xyz'[axis]} = 0"
    else:
        curve = leading_extension(stations, features, band)
        try:
            hit = host_intersection(
                kernel, curve, hosts, (float(stations[0]) - 1.5, float(stations[band[0]])),
                tolerance=spec["host_tolerance"],
            )  # fmt: skip
            result = reference_planform(
                stations, features, band, span, root_station=hit["station_m"], mirrored=mirrored
            )
            result["root_kind"] = "extended leading edge meets host"
            result["root_host_gap_m"] = hit["gap_m"]
        except InputValueError as error:
            lowest = float((samples @ span).min())
            result = reference_planform(
                stations, features, band, span, root_station=lowest, mirrored=mirrored
            )
            result["root_kind"] = "first original sample along the span axis (no host meeting)"
            result["root_host_note"] = str(error)
    result["clean_cut_count"] = len(band)
    diagnostics = {
        "station_m": stations.tolist(),
        "clean_indices": band.tolist(),
        "tip_bracket_m": bracket,
        "rejected": rejected,
    }
    return result, band, features, stations, span, diagnostics


def convergence(coarse, fine):
    checks = {
        key: {
            "coarse": coarse[key],
            "fine": fine[key],
            "absolute_difference": abs(fine[key] - coarse[key]),
            "tolerance": tolerance,
            "passed": abs(fine[key] - coarse[key]) <= tolerance,
        }
        for key, tolerance in CONVERGENCE_TOLERANCES.items()
    }
    return {"accepted": all(c["passed"] for c in checks.values()), "checks": checks}


def refuse_tolerances(section_tolerance_mm, host_tolerance):
    """Refuse a section tolerance no fit can meet and a host gap no meeting can have."""
    if section_tolerance_mm is not None and not section_tolerance_mm > 0:
        raise InputValueError(
            f"section_tolerance_mm: positive required, got {section_tolerance_mm}"
        )
    if not host_tolerance >= 0:
        raise InputValueError(f"host_tolerance: zero or positive required, got {host_tolerance}")


def resolve_hosts(kernel, hosts, source):
    """Return the host faces: names of components of ``source`` or ``path#name`` entries."""
    tags: list = []
    for host in hosts:
        entry = host if "#" in str(host) else f"{source}#{host}"
        tags += resolve_faces(kernel, entry)
    return tags
