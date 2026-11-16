"""Section shapes of the parametric geometry: NACA 4 and 5 digit and CST.

Pipeline role: pure numerical helper of the geometry package. The analytic
section equations follow NASA TM 4741, printed pp. 3, 7, 8 and 9
(https://ntrs.nasa.gov/api/citations/19970008124/downloads/19970008124.pdf).
Raw NACA ordinates retain the published mean-line-normal thickness
convention. The class-shape-transformation (CST) conversion uses the
geometric leading edge and the trailing-edge midpoint as the chord.
"""

from __future__ import annotations

import re
from math import comb

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

from cfdbender._errors import InputValueError

__all__ = [
    "bernstein_matrix",
    "cst_airfoil",
    "fit_cst",
    "naca_camber",
    "naca_coordinates",
    "naca_digits",
    "naca_to_cst",
    "normalized_naca",
    "ordinates_to_cst",
]


def naca_digits(designation):
    """Return the digit string of a NACA 4 or 5 digit designation, refusing any other."""
    digits = designation.upper().replace("NACA", "").strip()
    if not re.fullmatch(r"\d{4}|\d{5}", digits) or int(digits[-2:]) == 0:
        raise InputValueError("designation: expected a nonzero-thickness NACA 4 or 5 digit section")
    return digits


def naca_camber(designation, x):
    """Return the mean-line ordinate and slope of a NACA 4 or 5 digit section at ``x``."""
    digits, x = naca_digits(designation), np.asarray(x, float)
    if len(digits) == 4:
        m, p = int(digits[0]) / 100, int(digits[1]) / 10
        if m == 0:
            return np.zeros_like(x), np.zeros_like(x)
        if p == 0:
            raise InputValueError(
                "designation: cambered four digit section requires nonzero camber position"
            )
        y = np.where(
            x < p,
            m / p**2 * (2 * p * x - x * x),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x * x),
        )
        d = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
        return y, d
    position, reflex = int(digits[1]), int(digits[2])
    normal = {
        1: (0.0580, 361.4),
        2: (0.1260, 51.64),
        3: (0.2025, 15.957),
        4: (0.2900, 6.643),
        5: (0.3910, 3.230),
    }
    reflexed = {
        2: (0.1300, 51.990, 0.000764),
        3: (0.2170, 15.793, 0.00677),
        4: (0.3180, 6.520, 0.0303),
        5: (0.4410, 3.191, 0.1355),
    }
    if reflex == 0 and position in normal:
        r, k = normal[position]
        k *= int(digits[0]) / 2
        y = k / 6 * np.where(x < r, x**3 - 3 * r * x * x + r * r * (3 - r) * x, r**3 * (1 - x))
        d = k / 6 * np.where(x < r, 3 * x * x - 6 * r * x + r * r * (3 - r), -(r**3))
        return y, d
    if reflex == 1 and position in reflexed:
        r, k, ratio = reflexed[position]
        k *= int(digits[0]) / 2
        factor = np.where(x < r, 1.0, ratio)
        y = k / 6 * (factor * (x - r) ** 3 - ratio * (1 - r) ** 3 * x - r**3 * x + r**3)
        d = k / 6 * (3 * factor * (x - r) ** 2 - ratio * (1 - r) ** 3 - r**3)
        return y, d
    raise InputValueError("designation: unsupported five digit mean line")


def naca_coordinates(designation, x):
    """Return upper and lower (x,z), with thickness normal to the mean line."""
    digits, x = naca_digits(designation), np.asarray(x, float)
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise InputValueError("x: unit chord range required")
    t = int(digits[-2:]) / 100
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x**3 - 0.1015 * x**4)
    yc, slope = naca_camber(digits, x)
    theta = np.arctan(slope)
    upper = np.stack((x - yt * np.sin(theta), yc + yt * np.cos(theta)), axis=-1)
    lower = np.stack((x + yt * np.sin(theta), yc - yt * np.cos(theta)), axis=-1)
    return upper, lower


def bernstein_matrix(x, degree):
    """Return the Bernstein basis of the given degree evaluated at ``x``, one column per term."""
    x = np.asarray(x, float)
    return np.stack(
        [comb(degree, i) * x**i * (1 - x) ** (degree - i) for i in range(degree + 1)], axis=-1
    )


def cst_airfoil(x, upper, lower, trailing_edge_thickness, n1=0.5, n2=1.0):
    """Return upper and lower (x, z) points of a class-shape-transformation section."""
    x = np.asarray(x, float)
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise InputValueError("x: unit chord range required")
    if len(upper) != len(lower) or len(upper) < 2:
        raise InputValueError("upper/lower: matching coefficient counts required")
    if trailing_edge_thickness < 0 or n1 <= 0 or n2 <= 0:
        raise InputValueError("trailing_edge_thickness/n1/n2: invalid value")
    matrix = bernstein_matrix(x, len(upper) - 1)
    cls = x**n1 * (1 - x) ** n2
    zu = cls * (matrix @ np.asarray(upper)) + x * trailing_edge_thickness / 2
    zl = cls * (matrix @ np.asarray(lower)) - x * trailing_edge_thickness / 2
    return np.stack((x, zu), axis=-1), np.stack((x, zl), axis=-1)


def normalized_naca(designation, count=4001):
    """Normalize the closed contour's geometric chord, retaining its transform."""
    ends = naca_coordinates(designation, np.array([1.0]))
    te = (ends[0][0] + ends[1][0]) / 2
    candidates = []
    for side in range(2):

        def objective(s, side=side):
            return -float(np.sum((naca_coordinates(designation, np.array([s]))[side][0] - te) ** 2))

        opt = minimize_scalar(
            objective, bounds=(0, 0.05), method="bounded", options={"xatol": 1e-16}
        )
        candidates.extend([(objective(0), side, 0.0), (opt.fun, side, opt.x)])
    _, side, s = min(candidates)
    nose = naca_coordinates(designation, np.array([s]))[side][0]
    axis = te - nose
    chord = float(np.linalg.norm(axis))
    axis /= chord
    rotation = np.array([[axis[0], -axis[1]], [axis[1], axis[0]]])
    x = (1 - np.cos(np.linspace(0, np.pi, count))) / 2
    x = np.unique(np.r_[x, s])
    upper, lower = naca_coordinates(designation, x)
    contour = np.concatenate((upper[::-1], lower[1:]))
    normalized = (contour - nose) @ rotation / chord
    split = int(np.argmin(normalized[:, 0]))
    up, low = normalized[: split + 1][::-1].copy(), normalized[split:].copy()
    projection = max(abs(float(up[-1, 0]) - 1), abs(float(low[-1, 0]) - 1))
    adjusted = []
    for points in (up, low):
        points = np.concatenate((points[:-1][points[:-1, 0] < 1.0], points[-1:]))
        points[0] = [0, 0]
        # The published finite TE is normal to the camber line, not the chord.
        # CST uses a chord-normal base, so endpoint abscissae are projected.
        points[-1, 0] = 1.0
        if np.any(np.diff(points[:, 0]) <= 0):
            raise InputValueError("normalized_naca: contour not a graph in geometric chord frame")
        adjusted.append(points)
    return *adjusted, {
        "nose": nose.tolist(),
        "chord": chord,
        "rotation": rotation.tolist(),
        "te_projection_max": projection,
    }


def fit_cst(x, zu, zl, thickness, degree=5):
    """Least-squares fit of upper and lower CST coefficients to ordinates at ``x``."""
    matrix = bernstein_matrix(x, degree) * np.sqrt(x)[:, None] * (1 - x)[:, None]
    upper = np.linalg.lstsq(matrix, zu - x * thickness / 2, rcond=None)[0]
    lower = np.linalg.lstsq(matrix, zl + x * thickness / 2, rcond=None)[0]
    return upper, lower


def naca_to_cst(designation, degree=5, tolerance=5e-4, max_degree=20, trailing_edge_thickness=None):
    """Raise degree only when a dense independent ordinate check requires it.

    Residual is maximum absolute ordinate error in normalized geometric chord.
    Specifying TE thickness deliberately changes the target by a linear wedge.
    """
    upper, lower, transform = normalized_naca(designation)
    return ordinates_to_cst(
        upper, lower, transform, f"naca_to_cst.{designation}",
        degree=degree, tolerance=tolerance, max_degree=max_degree,
        trailing_edge_thickness=trailing_edge_thickness,
    )  # fmt: skip


def ordinates_to_cst(
    upper,
    lower,
    transform,
    label,
    *,
    degree=5,
    tolerance=5e-4,
    max_degree=20,
    trailing_edge_thickness=None,
):
    """Fit CST coefficients to unit-chord ordinates, raising the degree only as the residual needs.

    ``upper`` and ``lower`` are (x, z) arrays running from the leading edge (0, 0) to the
    trailing edge at x = 1, strictly increasing in x. ``label`` names the section in a refusal.
    """
    fu, fl = (
        PchipInterpolator(upper[:, 0], upper[:, 1]),
        PchipInterpolator(lower[:, 0], lower[:, 1]),
    )
    thickness = (
        float(upper[-1, 1] - lower[-1, 1])
        if trailing_edge_thickness is None
        else trailing_edge_thickness
    )
    te_u, te_l = float(upper[-1, 1]), float(lower[-1, 1])
    x = (1 - np.cos(np.linspace(0, np.pi, 601))) / 2
    check = (1 - np.cos(np.linspace(0, np.pi, 2402))) / 2

    def target(q):
        return fu(q) + q * (thickness / 2 - te_u), fl(q) + q * (-thickness / 2 - te_l)

    zu, zl = target(x)
    for current in range(degree, max_degree + 1):
        a, b = fit_cst(x, zu, zl, thickness, current)
        actual = cst_airfoil(check, a, b, thickness)
        expected = target(check)
        residual = max(float(np.max(abs(actual[i][:, 1] - expected[i]))) for i in (0, 1))
        if residual <= tolerance:
            return {
                "upper": a.tolist(),
                "lower": b.tolist(),
                "degree": current,
                "n1": 0.5,
                "n2": 1.0,
                "trailing_edge_thickness": thickness,
                "max_residual": residual,
                "chord_transform": transform,
            }
    raise InputValueError(
        f"{label}: residual {residual} exceeds {tolerance} at degree {max_degree}"
    )
