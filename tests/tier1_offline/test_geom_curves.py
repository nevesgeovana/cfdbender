"""Geometry: curve fit on given knots, multiplicity, union, elevation.

Oracles are closed forms: a polynomial a degree-3 space holds is reproduced exactly, a cubic has a
known quartic form, and a knot of multiplicity m in degree p has continuity p - m.
"""

import numpy as np
import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry._bspline import (
    clamped_knots,
    curve_basis,
    elevate_curve,
    fit_curve,
    union_knots,
)


def poly(t):
    return np.stack([t, 2 * t**3 - t**2, 1 - 3 * t**2], axis=1)


def test_clamped_knots_carry_multiplicity():
    assert (
        clamped_knots([0.25, 0.5], 3, [1, 3]).tolist() == [0] * 4 + [0.25, 0.5, 0.5, 0.5] + [1] * 4
    )
    with pytest.raises(InputValueError):
        clamped_knots([0.5], 3, [4])
    with pytest.raises(InputValueError):
        clamped_knots([0.6, 0.5], 3)


def test_basis_is_a_partition_of_unity_through_the_last_parameter():
    t = np.linspace(0, 1, 23)
    basis = curve_basis(clamped_knots([0.3, 0.7], 3), 3, t)
    assert np.allclose(basis.sum(axis=1), 1.0)
    assert basis[-1, -1] == pytest.approx(1.0)


def test_fit_reproduces_a_cubic_on_any_knots_and_pins_the_ends():
    t = np.linspace(0, 1, 40)
    controls, residual = fit_curve(poly(t), t, clamped_knots([0.4], 3), 3)
    assert residual < 1e-12
    assert np.allclose(controls[0], poly(t)[0]) and np.allclose(controls[-1], poly(t)[-1])


def test_fit_reports_the_residual_a_too_coarse_space_leaves():
    t = np.linspace(0, 1, 60)
    circle = np.stack([np.cos(np.pi * t), np.sin(np.pi * t)], axis=1)
    _, coarse = fit_curve(circle, t, clamped_knots([], 3), 3)
    _, fine = fit_curve(circle, t, clamped_knots([0.25, 0.5, 0.75], 3), 3)
    assert coarse > 1e-3 and 0 < fine < coarse / 10


def test_fit_refuses_bad_input():
    t = np.linspace(0, 1, 5)
    with pytest.raises(InputValueError):
        fit_curve(poly(t), t, clamped_knots([0.2, 0.4, 0.6, 0.8], 3), 3)
    with pytest.raises(InputValueError):
        fit_curve(poly(t), t[::-1], clamped_knots([], 3), 3)


def test_a_corner_knot_lets_a_fit_hold_a_kink():
    t = np.linspace(0, 1, 41)
    kink = np.stack([t, np.abs(t - 0.5)], axis=1)
    _, smooth = fit_curve(kink, t, clamped_knots([0.5], 3), 3)
    _, corner = fit_curve(kink, t, clamped_knots([0.5], 3, [3]), 3)
    assert corner < 1e-12 < smooth


def test_union_takes_the_larger_multiplicity_of_each_knot():
    a = clamped_knots([0.25, 0.5], 3, [1, 2])
    b = clamped_knots([0.5, 0.75], 3, [1, 3])
    assert union_knots(a, b).tolist() == [0] * 4 + [0.25, 0.5, 0.5, 0.75, 0.75, 0.75] + [1] * 4
    with pytest.raises(InputValueError):
        union_knots(a, clamped_knots([0.5], 2))


def test_elevation_keeps_the_curve_and_raises_each_knot():
    t = np.linspace(0, 1, 50)
    knots = clamped_knots([0.3, 0.6], 3, [1, 2])
    controls, _ = fit_curve(np.sin(3 * t)[:, None] * np.ones((1, 3)) * [1, 2, 3], t, knots, 3)
    high, new, degree = elevate_curve(controls, knots, 3, 2)
    assert degree == 5
    assert new.tolist() == [0] * 6 + [0.3] * 3 + [0.6] * 4 + [1] * 6
    before = curve_basis(knots, 3, t) @ controls
    assert np.max(np.abs(curve_basis(new, 5, t) @ high - before)) < 1e-11


def test_elevation_of_a_cubic_matches_the_bezier_formula():
    # A cubic Bezier raised to degree 4: Q_i = (i/4) P_{i-1} + (1 - i/4) P_i.
    p = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]])
    high, _, _ = elevate_curve(p, clamped_knots([], 3), 3)
    expect = [
        p[0],
        0.25 * p[0] + 0.75 * p[1],
        0.5 * p[1] + 0.5 * p[2],
        0.75 * p[2] + 0.25 * p[3],
        p[3],
    ]
    assert np.allclose(high, expect)


def test_end_slopes_are_held_exactly_and_a_wrong_slope_shows_in_the_curve():
    t = np.linspace(0, 1, 40)
    knots = clamped_knots([0.5], 3)
    q = poly(t)
    slope = (
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 4.0, -6.0]),
    )  # the derivative of poly at 0, 1
    controls, _ = fit_curve(q, t, knots, 3, slope)
    assert np.allclose(controls[1] - controls[0], knots[4] / 3 * slope[0])
    assert np.allclose(controls[-1] - controls[-2], (1 - knots[-5]) / 3 * slope[1])
    _, wrong = fit_curve(q, t, knots, 3, (slope[0] * 3, slope[1]))
    assert wrong > 1e-3
