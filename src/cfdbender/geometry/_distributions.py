"""Spanwise spline distributions for the parametric geometry.

Pipeline role: pure numerical helper of the geometry package. One cubic
spline interpolates a quantity (chord, twist, a leading-edge offset, or the
coefficients of a section) through its control stations.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from cfdbender._errors import InputValueError

__all__ = [
    "blended_section",
    "chord_spacing",
    "elevate_coefficients",
    "fewest_stations",
    "insert_knot",
    "planform_distributions",
    "planform_laws",
    "planform_values",
    "spline_distribution",
]


def spline_distribution(stations, values, at, breaks=()):
    """Interpolate ``values`` given at ``stations`` and return value and slope at ``at``.

    A not-a-knot cubic is used for three or more stations, a straight line for
    two and a constant for one. Axis zero of ``values`` indexes the stations;
    the remaining axes may hold section coefficients. Extrapolation beyond the
    control stations is refused when there is more than one station. Each station
    in ``breaks`` is a slope break: the spans on its two sides are interpolated
    separately, so the value is continuous there and the slope is not, and a kink
    does not ring into its neighbors.

    Returns
    -------
    tuple of numpy.ndarray
        The interpolated values and their first derivative with respect to the
        station coordinate, both of shape ``at.shape + values.shape[1:]``.
    """
    x, y, q = np.asarray(stations, float), np.asarray(values, float), np.asarray(at, float)
    if x.ndim != 1 or not len(x) or y.ndim < 1 or y.shape[0] != len(x):
        raise InputValueError("stations/values: inconsistent dimensions")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(q)):
        raise InputValueError("stations/values/at: finite values required")
    if len(x) == 1:
        shape = q.shape + y.shape[1:]
        return np.broadcast_to(y[0], shape).copy(), np.zeros(shape)
    if np.any(np.diff(x) <= 0):
        raise InputValueError("stations: strictly increasing values required")
    if np.any(q < x[0]) or np.any(q > x[-1]):
        raise InputValueError("at: outside control stations")
    if len(breaks):
        return _piecewise(x, y, q, breaks)
    spline = CubicSpline(x, y, axis=0, bc_type="not-a-knot", extrapolate=False)
    return spline(q), spline(q, 1)


def _piecewise(x, y, q, breaks):
    """Interpolate each span between slope breaks on its own; the value is shared at a break."""
    cuts = sorted({float(b) for b in breaks})
    for cut in cuts:
        if not np.any(np.isclose(x[1:-1], cut, rtol=0, atol=1e-12)):
            raise InputValueError(f"slope break {cut}: must be an interior station")
    edges = [x[0], *cuts, x[-1]]
    value, slope = np.empty(q.shape + y.shape[1:]), np.empty(q.shape + y.shape[1:])
    for index, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = index == len(edges) - 2
        inside = (q >= a) & ((q <= b) if last else (q < b))
        keep = (x >= a - 1e-12) & (x <= b + 1e-12)
        value[inside], slope[inside] = spline_distribution(x[keep], y[keep], q[inside])
    return value, slope


def elevate_coefficients(coefficients, degree):
    """Degree-elevate Bernstein coefficients to ``degree`` without changing the curve."""
    a = np.asarray(coefficients, float)
    while len(a) <= degree:
        n = len(a)
        q = np.arange(1, n) / n
        a = np.r_[a[0], q * a[:-1] + (1 - q) * a[1:], a[-1]]
    return a


def insert_knot(knots, degree, points, t):
    """Insert the knot ``t`` once into a B-spline curve (Boehm); the curve does not change.

    ``knots`` is the full clamped knot vector, ``points`` the control points of shape (n, d).
    Returns the new knot vector and control points.
    """
    k, p = np.asarray(knots, float), np.asarray(points, float)
    if len(k) != len(p) + degree + 1:
        raise InputValueError("knots/points: knot count must be points + degree + 1")
    if not k[degree] < t < k[len(p)]:
        raise InputValueError("t: must lie strictly inside the parameter range")
    span = int(np.searchsorted(k, t, side="right")) - 1
    new = [p[0]]
    for i in range(1, len(p)):
        if i <= span - degree:
            new.append(p[i])
        elif i > span:
            new.append(p[i - 1])
        else:
            a = (t - k[i]) / (k[i + degree] - k[i])
            new.append(a * p[i] + (1 - a) * p[i - 1])
    new.append(p[-1])
    return np.insert(k, span + 1, t), np.array(new)


def planform_distributions(component):
    """Return ``{key: (stations, values)}`` of the four planform laws, in span-normalized units.

    A law the component supplies is used as given; the others follow from the planform table,
    sampled at its control stations and at its slope breaks.
    """
    p = component["planform"]
    stations = np.unique(np.r_[p["control_stations"], p.get("slope_breaks", [])])
    span = p["span"]
    chord = p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * stations
    le_x = span / 2 * stations * np.tan(np.deg2rad(p["sweep"]))
    if p["sweep_reference"] == "quarter_chord":
        le_x += (p["root_chord"] - chord) / 4
    defaults = {
        "chord": chord / span,
        "leading_edge_x": le_x / span,
        "leading_edge_z": stations / 2 * np.tan(np.deg2rad(p["dihedral"])),
        "twist": p["incidence"] + p["tip_twist"] * stations,
    }
    supplied = component.get("distributions", {})
    return {
        key: (
            (supplied[key]["stations"], supplied[key]["values"])
            if key in supplied
            else (stations, values)
        )
        for key, values in defaults.items()
    }


def planform_values(component, eta):
    """Return chord, leading-edge offsets and twist at normalized span ``eta``, in model units."""
    span = component["planform"]["span"]
    breaks = component["planform"].get("slope_breaks", [])
    result = {}
    for key, (stations, values) in planform_distributions(component).items():
        result[key] = spline_distribution(stations, values, eta, breaks)[0]
        if key != "twist":
            result[key] *= span
    return result


def fewest_stations(stations, values, tolerance, breaks=()):
    """Return the indices of the fewest stations whose spline reproduces all ``values``.

    The ends are always kept. A station is dropped while the spline through the remaining ones
    stays within ``tolerance`` of every given value; the cheapest drop is taken first. A slope
    break is never dropped.
    """
    x, y = np.asarray(stations, float), np.asarray(values, float)
    kept = list(range(len(x)))
    while len(kept) > 2:
        best = None
        for drop in kept[1:-1]:
            if np.any(np.isclose(breaks, x[drop], rtol=0, atol=1e-12)):
                continue
            rest = [i for i in kept if i != drop]
            error = float(np.max(np.abs(spline_distribution(x[rest], y[rest], x, breaks)[0] - y)))
            if error <= tolerance and (best is None or error < best[0]):
                best = (error, drop)
        if best is None:
            break
        kept.remove(best[1])
    return kept


def planform_laws(component, tolerance=1e-9) -> list[dict]:
    """Return the S4 records: each planform law against the fewest stations that describe it.

    The record never fails: it reports how many stations the law uses and how few would do, so
    a planform given by more stations than its shape needs is visible.
    """
    span = component["planform"]["span"]
    records = []
    for key, (stations, values) in planform_distributions(component).items():
        scale = 1.0 if key == "twist" else span
        fewest = fewest_stations(
            stations,
            np.asarray(values, float) * scale,
            tolerance * scale,
            component["planform"].get("slope_breaks", []),
        )
        records.append(
            {
                "id": "law.fewest_stations",
                "law": key,
                "status": "pass" if len(fewest) == len(stations) else "report",
                "value": {"used": len(stations), "fewest": len(fewest)},
                "note": f"tolerance {tolerance}",
            }
        )
    return records


def blended_section(sections, models, eta):
    """Blend the CST coefficients and trailing-edge thickness of the sections at ``eta``."""
    return {
        key: spline_distribution(sections["eta"], [m[key] for m in models], eta)[0]
        for key in ("upper", "lower", "trailing_edge_thickness")
    }


def chord_spacing(start, end, u):
    """Map ``u`` in [0, 1] to chord fractions from ``start`` to ``end``, clustered at the nose.

    The spacing is uniform in the square root of the chord fraction, which resolves the
    leading-edge curvature with the same number of samples as the rest of the chord.
    """
    root_start, root_end = np.sqrt(start), np.sqrt(end)
    u = np.asarray(u, float)
    if np.any(u < 0) or np.any(u > 1):
        raise InputValueError("u: parameter range [0, 1] required")
    return (root_start + (root_end - root_start) * u) ** 2
