"""The wing-fuselage fairing: footprint on the body, junction on the wing and the blend between.

Pipeline role: pure numerical stage of the geometry package. For a wing and the
body it meets, the fairing is twelve patches (six wing contour segments, each
split in a body half and a wing half, listed segment by segment) whose boundary lies on the
body at the
footprint and on the wing at the junction, with matching tangent planes and
prescribed normal curvature at both.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from cfdbender._errors import InputValueError

from ._bodies import body_points, body_profile, body_station_table
from ._caps import contour_segment, junction_normal, wing_contour_point
from ._distributions import blended_section, planform_values, spline_distribution
from ._lifting import section_models, thickness_station
from ._patch import Patch, interpolate_polynomial_strip


def host_y(body, table, x, z):
    """Return the body half-width at axial position ``x`` and height ``z``."""
    axial = x / body["length"]
    if not 0 < axial < 1:
        raise InputValueError("fairing.footprint: outside body axial range")
    w, h, centre, top, bottom = body_profile(body, table, np.asarray(axial))
    fraction = (z - centre) / h
    exponent = top if fraction >= 0 else bottom
    if abs(fraction) >= 1:
        raise InputValueError("fairing.junction: wing lies beyond body height")
    return float(w * (1 - abs(fraction) ** exponent) ** (1 / exponent))


def host_point_normal(body, table, x, theta):
    """Return the body point and unit normal at axial position ``x`` and azimuth ``theta``."""
    axial = x / body["length"]
    point = body_points(body, table, axial, theta)
    tangent_x = body_points(body, table, axial + 1e-6, theta) - body_points(
        body, table, axial - 1e-6, theta
    )
    tangent_theta = body_points(body, table, axial, theta + 1e-6) - body_points(
        body, table, axial, theta - 1e-6
    )
    normal = np.cross(tangent_x, tangent_theta)
    return point, normal / np.linalg.norm(normal)


def footprint(fairing, body, table, root_planform, section, *, maximum, index, u):
    """Return the body point and normal where contour parameter ``u`` of a segment lands."""
    q = contour_segment(section, maximum, index, np.asarray(u))
    upper, lower = (
        contour_segment(section, maximum, 0, np.asarray(1.0)),
        contour_segment(section, maximum, 5, np.asarray(0.0)),
    )
    half = (upper[1] - lower[1]) / 2
    phi = np.arctan2(q[1] / half, 2 * q[0] - 1)
    axial_fraction = (1 + np.cos(phi)) / 2
    n1, n2 = fairing["start_exponent"], fairing["end_exponent"]
    peak = n1 / (n1 + n2)
    width_factor = axial_fraction**n1 * (1 - axial_fraction) ** n2 / (peak**n1 * (1 - peak) ** n2)
    limit = fairing["footprint_upper"] if phi >= 0 else fairing["footprint_lower"]
    theta = np.pi / 2 + width_factor * (np.deg2rad(limit) - np.pi / 2)
    x = root_planform["leading_edge_x"] + root_planform["chord"] * (
        fairing["start"] + (fairing["end"] - fairing["start"]) * axial_fraction
    )
    return host_point_normal(body, table, float(x), float(theta))


def wing_junction(fairing, body, table, wing, models, *, maximum, index, u):
    """Return the wing point, normal and fairing width where the fairing leaves the wing."""
    section = blended_section(wing["sections"], models, 0.0)
    x = contour_segment(section, maximum, index, np.asarray(u))[0]
    root_chord = float(planform_values(wing, 0.0)["chord"])
    width = float(spline_distribution(fairing["stations"], fairing["width"], x)[0]) * root_chord

    def equation(eta):
        p = wing_contour_point(wing, models, maximum, index, u, eta)
        return p[1] - host_y(body, table, p[0], p[2]) - 1.5 * width

    eta = brentq(equation, 0, 1, xtol=1e-14)
    point = wing_contour_point(wing, models, maximum, index, u, eta)
    normal = junction_normal(wing, models, maximum, index, u, eta)
    return point, normal, width


def tangent_direction(vector, normal):
    """Return ``vector`` projected into the plane normal to ``normal``, as a unit vector."""
    projected = vector - np.dot(vector, normal) * normal
    length = np.linalg.norm(projected)
    if length < 1e-12:
        raise InputValueError("fairing.blend: tangent direction is undefined")
    return projected / length


def blend_curve(start, end, n_start, n_end, ridge, *, r_start, r_end, at, end_tangent=None):
    """Quintic endpoint jets plus a sixth-degree ridge correction.

    The correction and its first two derivatives vanish at both hosts, so
    endpoint tangent planes and prescribed normal curvature are retained.
    """
    a, b = 2 * np.linalg.norm(ridge - start), 2 * np.linalg.norm(end - ridge)
    d_start = a * tangent_direction(ridge - start, n_start)
    direction = (
        tangent_direction(end - ridge, n_end)
        if end_tangent is None
        else end_tangent / np.linalg.norm(end_tangent)
    )
    d_end = b * direction
    dd_start, dd_end = a * a / r_start * n_start, b * b / r_end * n_end
    c0, c1, c2 = start, d_start, dd_start / 2
    rhs = np.stack((end - c0 - c1 - c2, d_end - c1 - 2 * c2, dd_end - 2 * c2))
    c3, c4, c5 = np.linalg.solve(
        np.array([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [6.0, 12.0, 20.0]]), rhs
    )
    coefficients = np.stack((c0, c1, c2, c3, c4, c5))

    def evaluate(t):
        return np.polynomial.polynomial.polyval(t, coefficients).T

    t = np.asarray(at)
    return evaluate(t) + 64 * (t**3 * (1 - t) ** 3)[..., None] * (ridge - evaluate(np.asarray(0.5)))


def wing_fuselage_fairing(fairing, body, wing) -> list[Patch]:
    """Return the twelve fairing patches between ``body`` and the root of ``wing`` (body frame)."""
    table, models = body_station_table(body), section_models(wing["sections"])
    root_planform = planform_values(wing, 0.0)
    section = blended_section(wing["sections"], models, 0.0)
    maximum = thickness_station(wing, models, 0.0)
    patches = []
    for index in range(6):
        for low, high in [(0.0, 0.5), (0.5, 1.0)]:
            grid = []
            for u in np.linspace(0, 1, 49):
                p, np_ = footprint(
                    fairing, body, table, root_planform, section, maximum=maximum, index=index, u=u
                )
                q, nq, width = wing_junction(
                    fairing, body, table, wing, models, maximum=maximum, index=index, u=u
                )
                eta = 2 * q[1] / wing["planform"]["span"]
                tangent = wing_contour_point(
                    wing, models, maximum, index, u, min(1.0, eta + 1e-5)
                ) - wing_contour_point(wing, models, maximum, index, u, max(0.0, eta - 1e-5))
                ridge = (p + q) / 2
                ridge[1] = host_y(body, table, ridge[0], ridge[2]) + width
                grid.append(
                    blend_curve(
                        p,
                        q,
                        np_,
                        nq,
                        ridge,
                        r_start=fairing["body_blend_radius"],
                        r_end=fairing["wing_blend_radius"],
                        at=np.linspace(low, high, 7),
                        end_tangent=tangent,
                    )
                )
            family = "fairing_body" if low == 0 else "fairing_wing"
            patches.append(interpolate_polynomial_strip(grid, family=family))
    return patches
