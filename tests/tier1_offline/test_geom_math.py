"""Geometry: spanwise distributions and section shapes (NACA 4/5 digit, CST)."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from cfdbender.geometry._airfoils import (
    cst_airfoil,
    naca_camber,
    naca_coordinates,
    naca_to_cst,
)
from cfdbender.geometry._distributions import spline_distribution


def test_two_controls_are_linear():
    q = np.linspace(0, 1, 41)
    y, d = spline_distribution([0, 1], [2, 5], q)
    np.testing.assert_allclose(y, 2 + 3 * q, atol=1e-14)
    np.testing.assert_allclose(d, 3, atol=1e-14)


def test_cubic_and_derivative():
    x = np.array([0, 0.2, 0.6, 1])
    q = np.linspace(0, 1, 101)
    y, d = spline_distribution(x, x**3 - 2 * x * x + 3 * x - 1, q)
    np.testing.assert_allclose(y, q**3 - 2 * q * q + 3 * q - 1, atol=2e-14)
    np.testing.assert_allclose(d, 3 * q * q - 4 * q + 3, atol=2e-14)


def test_end_values():
    y, _ = spline_distribution([0, 0.2, 0.8, 1], [4, 2, 5, 9], [0, 1])
    np.testing.assert_allclose(y, [4, 9], atol=1e-14)


def test_one_section_and_vector_coefficients():
    y, d = spline_distribution([0], [[1, 2, 3]], [0, 0.5, 1])
    np.testing.assert_array_equal(y, [[1, 2, 3]] * 3)
    np.testing.assert_array_equal(d, np.zeros((3, 3)))


@pytest.mark.parametrize("x,y,q", [([0, 0], [1, 2], 0), ([0, 1], [1], 0), ([0, 1], [1, 2], 2)])
def test_bad_distribution(x, y, q):
    with pytest.raises(ValueError):
        spline_distribution(x, y, q)


@pytest.mark.parametrize("name,thickness", [("0012", 0.12), ("23012", 0.12), ("23018", 0.18)])
def test_published_coordinates(name, thickness):
    # Independent direct evaluation of the published thickness and mean-line equations.
    x = np.linspace(0, 1, 1001)
    half = (
        thickness
        / 0.2
        * (0.2969 * x**0.5 - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    )
    if name == "0012":
        mean, slope = np.zeros_like(x), np.zeros_like(x)
    else:
        r, k = 0.2025, 15.957
        mean = np.array(
            [
                k / 6 * (s**3 - 3 * r * s * s + r * r * (3 - r) * s)
                if s < r
                else k * r**3 / 6 * (1 - s)
                for s in x
            ]
        )
        slope = np.array(
            [
                k / 6 * (3 * s * s - 6 * r * s + r * r * (3 - r)) if s < r else -k * r**3 / 6
                for s in x
            ]
        )
    for sign, actual in zip((1, -1), naca_coordinates(name, x), strict=True):
        root = np.sqrt(1 + slope * slope)
        expected = np.column_stack((x - sign * half * slope / root, mean + sign * half / root))
        np.testing.assert_allclose(actual, expected, atol=2e-16)
    opt = minimize_scalar(
        lambda s: -np.linalg.norm(np.subtract(*naca_coordinates(name, np.array([s])))),
        bounds=(0.1, 0.5),
        method="bounded",
    )
    assert opt.x == pytest.approx(0.29983, abs=1e-5)
    assert -opt.fun == pytest.approx(thickness, abs=6e-5)


def test_230_camber_line():
    for name in ("23012", "23018"):
        y, d = naca_camber(name, np.array([0, 0.15, 0.2025, 1]))
        assert y[0] == y[-1] == 0
        assert y[1] == pytest.approx(0.018386447016796887, abs=1e-15)
        assert abs(d[1]) < 1e-4
        assert d[2] == pytest.approx(d[3], abs=1e-15)
    np.testing.assert_array_equal(
        naca_camber("23012", np.linspace(0, 1, 101))[0],
        naca_camber("23018", np.linspace(0, 1, 101))[0],
    )


def test_four_digit_camber_and_reflex():
    assert naca_camber("2412", 0.4)[0] == pytest.approx(0.02)
    for name in ("22112", "23112", "24112", "25112"):
        y, _ = naca_camber(name, np.array([0, 1]))
        np.testing.assert_allclose(y, 0, atol=1e-15)


@pytest.mark.parametrize("name", ["NACA00", "NACA1234567", "NACA0000", "ABCD"])
def test_designation_refusals(name):
    with pytest.raises(ValueError, match="designation"):
        naca_coordinates(name, np.array([0.5]))


@pytest.mark.parametrize("name", ["0012", "23012", "23018"])
def test_cst_geometric_residual(name):
    model = naca_to_cst(name)
    assert model["max_residual"] <= 5e-4
    q = (1 - np.cos(np.linspace(0, np.pi, 20001))) / 2
    fitted = cst_airfoil(q, model["upper"], model["lower"], model["trailing_edge_thickness"])
    original = np.concatenate(naca_coordinates(name, q))
    transform = model["chord_transform"]
    normalized = (
        (original - np.array(transform["nose"]))
        @ np.array(transform["rotation"])
        / transform["chord"]
    )
    candidate = np.concatenate(fitted)
    # Both directions, including the original finite TE and the actual nose.
    forward = np.max(cKDTree(candidate).query(normalized)[0])
    reverse = np.max(cKDTree(normalized).query(candidate)[0])
    assert max(forward, reverse) < 5e-4
    assert fitted[0][-1, 1] - fitted[1][-1, 1] == pytest.approx(model["trailing_edge_thickness"])


def test_coefficient_blend_is_straight():
    a, b = naca_to_cst("23018"), naca_to_cst("23012")
    q = np.linspace(0, 1, 31)
    for side in ("upper", "lower"):
        value, derivative = spline_distribution([0, 1], [a[side], b[side]], q)
        expected = (1 - q[:, None]) * np.array(a[side]) + q[:, None] * np.array(b[side])
        np.testing.assert_allclose(value, expected, atol=1e-15)
        np.testing.assert_allclose(
            derivative, np.broadcast_to(np.subtract(b[side], a[side]), value.shape), atol=1e-15
        )


def test_cst_refuses_x_outside_the_unit_chord():
    with pytest.raises(ValueError, match="unit chord"):
        cst_airfoil(np.array([1.2]), [0.1, 0.1], [-0.1, -0.1], 0.0)
