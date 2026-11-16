"""Polynomial section measurements for Fowler motion and clearance checks.

Pipeline role: minimize over each spline polynomial interval, including all
stationary roots and end points. Gap measurements are not nearest samples.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline, PPoly

from ._bspline import full_knots
from ._control_curves import chord_parameter, interval, point


def frame(skin, v):
    le = point(skin, 0.5, v)
    chord = (point(skin, 0, v) + point(skin, 1, v)) / 2 - le
    normal = np.array([-chord[2], 0, chord[0]])
    return le, np.stack((chord, normal), axis=1) / (chord @ chord)


def exterior_intervals(skin, entry):
    nose = 1 - entry["chord_ratio"] + entry.get("nose", {}).get("radius", 0.02)
    lip = entry["cove"].get("lip_x", 0.96)
    aft = lip + min(entry.get("nose", {}).get("radius", 0.02), (1 - lip) / 2)
    return (0, chord_parameter(skin, nose, False)), (chord_parameter(skin, aft, True), 1)


def curves(skin, nested, entry):
    return [
        *(interval(nested["skin"], "u", a, b) for a, b in exterior_intervals(skin, entry)),
        *(nested[n] for n in ("nose", "upper_nested", "upper_transition", "trailing_edge")),
    ]


def section(patch, v, le, axes, rotation, *, translation):
    poles = BSpline(
        full_knots(patch.knots_v, patch.multiplicities_v), patch.controls, patch.degree_v, axis=1
    )(v)
    return BSpline(
        full_knots(patch.knots_u, patch.multiplicities_u),
        (poles @ rotation.T + translation - le) @ axes,
        patch.degree_u,
    )


def _roots(poly, width):
    return [r.real for r in poly.roots() if abs(r.imag) < 1e-9 and 0 < r.real < width]


def polynomials(curve):
    pp = [PPoly.from_spline((curve.t, curve.c[:, i], curve.k)) for i in (0, 1)]
    for i, (a, b) in enumerate(zip(pp[0].x, pp[0].x[1:], strict=False)):
        if a < b and 0 <= a < b <= 1:
            yield b - a, [Polynomial(p.c[::-1, i]) for p in pp]


def minimum_x(profiles):
    values = []
    for curve in profiles:
        for width, (x, _) in polynomials(curve):
            values.extend(float(x(t)) for t in (0, width, *_roots(x.deriv(), width)))
    return min(values)


def distance(profiles, start, end=None):
    """Minimum distance from the section curves to a point or closed segment."""
    start = np.asarray(start)
    end = start if end is None else np.asarray(end)
    direction = end - start
    length2 = float(direction @ direction)
    values = []
    for curve in profiles:
        for width, xy in polynomials(curve):
            delta = [xy[i] - start[i] for i in (0, 1)]
            squared = sum(p * p for p in delta)
            candidates = [0, width, *_roots(squared.deriv(), width)]
            if length2:
                projected = sum(delta[i] * direction[i] for i in (0, 1)) / length2
                perpendicular = squared - length2 * projected * projected
                candidates.extend(_roots(perpendicular.deriv(), width))
                candidates.extend(_roots(projected, width))
                candidates.extend(_roots(projected - 1, width))
                far = sum((xy[i] - end[i]) ** 2 for i in (0, 1))
                candidates.extend(_roots(far.deriv(), width))
            points = np.array([[p(t) for p in xy] for t in candidates])
            fraction = np.clip((points - start) @ direction / length2, 0, 1) if length2 else 0
            nearest = start + np.asarray(fraction)[..., None] * direction
            values.append(float(np.linalg.norm(points - nearest, axis=1).min()))
    return min(values)


def cove_points(patches, v, le, axes):
    return [
        (point(patches[n], 0, v) - le) @ axes for n in ("cove_wall", "cove_roof", "lip_base")
    ] + [(point(patches["lip_base"], 1, v) - le) @ axes]


def inside_count(skin, patches, profiles, v, le, *, axes):
    from matplotlib.path import Path

    # The skin rim is monotone on either side of its nose. Inversion uses
    # the actual section, rather than assuming that u is a chord fraction.
    from scipy.optimize import brentq

    def at(u):
        return (point(skin, u, v) - le) @ axes

    xwall, xlip = cove_points(patches, v, le, axes)[0][0], cove_points(patches, v, le, axes)[-1][0]
    low = brentq(lambda u: at(u)[0] - xwall, 0, 0.5)
    high = brentq(lambda u: at(u)[0] - xlip, 0.5, 1)
    polygon = np.vstack(
        ([at(u) for u in np.linspace(low, high, 513)], cove_points(patches, v, le, axes)[::-1])
    )
    samples = np.vstack([curve(np.linspace(0, 1, 129)) for curve in profiles])
    return int(Path(polygon).contains_points(samples, radius=-1e-10).sum())
