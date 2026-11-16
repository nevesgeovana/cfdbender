"""Geometry: the numerical oracles of the section and curve machinery.

Each test states a closed form or an invariance and also scores a deliberately wrong formula
against the same tolerance, so the tolerance is shown to discriminate.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline

from cfdbender.geometry._airfoils import cst_airfoil, naca_to_cst, ordinates_to_cst
from cfdbender.geometry._distributions import elevate_coefficients, insert_knot


def _naca0012_half_thickness(x):
    """The NACA 4-digit thickness equation, open trailing edge, t = 0.12."""
    return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)


def test_degree_eight_cst_of_naca0012_matches_the_thickness_equation():
    fit = naca_to_cst("NACA0012", degree=8, tolerance=5e-4)
    assert fit["degree"] >= 8
    x = np.linspace(0.02, 0.98, 97)
    upper, lower = cst_airfoil(x, fit["upper"], fit["lower"], fit["trailing_edge_thickness"])
    half = (upper[:, 1] - lower[:, 1]) / 2
    right = float(np.max(np.abs(half - _naca0012_half_thickness(x))))
    # A wrong constant of the equation (closed-trailing-edge 0.1036 in place of 0.1015) and a
    # wrong class exponent both move the section by more than the fit tolerance.
    wrong_constant = 0.6 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    wrong_class = cst_airfoil(x, fit["upper"], fit["lower"], fit["trailing_edge_thickness"], n1=0.4)
    wrong = (wrong_class[0][:, 1] - wrong_class[1][:, 1]) / 2
    # the bound is the fit tolerance, so the closed-trailing-edge constant (up to
    # 1.2e-3 away near the trailing edge) cannot pass where the published one does.
    assert right < 5e-4
    assert float(np.max(np.abs(wrong - _naca0012_half_thickness(x)))) > 10 * right
    assert float(np.max(np.abs(wrong_constant - _naca0012_half_thickness(x)))) > 5e-4


def _curve(knots, degree, points, t):
    return BSpline(knots, points, degree)(t)


def test_knot_insertion_does_not_move_the_curve():
    degree = 3
    knots = np.array([0, 0, 0, 0, 0.3, 0.7, 1, 1, 1, 1], float)
    points = np.array([[0, 0], [1, 2], [2, -1], [3, 3], [4, 0], [5, 2]], float)
    new_knots, new_points = insert_knot(knots, degree, points, 0.5)
    t = np.linspace(0, 1, 201)
    assert len(new_points) == len(points) + 1
    assert (
        np.max(np.abs(_curve(new_knots, degree, new_points, t) - _curve(knots, degree, points, t)))
        < 1e-12
    )
    # The wrong blending weight (1 - alpha) does move it.
    k, p = np.asarray(knots), points
    span = 4
    bad = [p[0], *[p[i] for i in range(1, span - degree + 1)]]
    for i in range(span - degree + 1, span + 1):
        a = (0.5 - k[i]) / (k[i + degree] - k[i])
        bad.append((1 - a) * p[i] + a * p[i - 1])
    bad += [p[i - 1] for i in range(span + 1, len(p))] + [p[-1]]
    wrong = np.array(bad)
    assert (
        np.max(np.abs(_curve(new_knots, degree, wrong, t) - _curve(knots, degree, points, t)))
        > 1e-3
    )


def test_knot_insertion_refuses_a_knot_outside_the_range():
    from cfdbender._errors import InputValueError

    knots = np.array([0, 0, 0, 1, 1, 1], float)
    with pytest.raises(InputValueError, match="strictly inside"):
        insert_knot(knots, 2, np.zeros((3, 2)), 1.0)


def test_degree_elevation_keeps_the_bernstein_polynomial():
    coefficients = [0.3, -0.1, 0.5, 0.2]
    elevated = elevate_coefficients(coefficients, 7)
    x = np.linspace(0, 1, 101)
    from cfdbender.geometry._airfoils import bernstein_matrix

    before = bernstein_matrix(x, 3) @ coefficients
    after = bernstein_matrix(x, 7) @ elevated
    assert len(elevated) == 8
    assert np.max(np.abs(before - after)) < 1e-13
    padded = np.r_[coefficients, np.zeros(4)]
    assert np.max(np.abs(bernstein_matrix(x, 7) @ padded - before)) > 1e-2


def test_cst_fit_recovers_the_exact_degree_of_a_cst_section():
    upper, lower = (0.30, 0.15, 0.25, 0.12), (-0.28, -0.05, -0.2, -0.10)
    te = 0.004
    x = (1 - np.cos(np.linspace(0, np.pi, 1501))) / 2
    u, low = cst_airfoil(x, upper, lower, te)
    fit = ordinates_to_cst(u, low, {}, "oracle", degree=2, tolerance=1e-4, max_degree=12)
    assert fit["degree"] == 3
    assert np.max(np.abs(np.array(fit["upper"]) - upper)) < 1e-3
    assert np.max(np.abs(np.array(fit["lower"]) - lower)) < 1e-3
    # A start above the true degree never lowers it, and a degree cap below it is refused.
    from cfdbender._errors import InputValueError

    with pytest.raises(InputValueError, match="exceeds"):
        ordinates_to_cst(u, low, {}, "oracle", degree=1, tolerance=1e-9, max_degree=2)
