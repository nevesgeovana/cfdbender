"""Geometry: knot insertion and splitting of a B-spline patch.

Oracles come from the definition of a knot insertion (it changes the net and not the surface) and
from a surface with a closed form (a plane and a cylinder), never from the output under test.
"""

import numpy as np
import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry._bspline import full_knots, insert_knot, split_patch
from cfdbender.geometry._patch import evaluate_patch, interpolate_patch


def rough_patch():
    return interpolate_patch(np.random.default_rng(7).normal(size=(13, 9, 3)))


def test_insertion_leaves_the_surface_unchanged():
    patch = rough_patch()
    for direction in "uv":
        refined = insert_knot(patch, direction, 0.37, times=2)
        assert np.max(np.abs(evaluate_patch(patch, 33) - evaluate_patch(refined, 33))) < 1e-12
    assert refined.controls.shape[1] == patch.controls.shape[1] + 2


def test_insertion_adds_one_control_row_per_knot():
    patch = rough_patch()
    refined = insert_knot(patch, "u", 0.37, times=3)
    assert refined.controls.shape[0] == patch.controls.shape[0] + 3
    assert full_knots(refined.knots_u, refined.multiplicities_u).size == (
        full_knots(patch.knots_u, patch.multiplicities_u).size + 3
    )


@pytest.mark.parametrize("direction", ["u", "v"])
@pytest.mark.parametrize("fraction", [0.5, 0.25])
def test_split_pieces_are_the_original_surface(direction, fraction):
    patch = rough_patch()
    low, high = split_patch(patch, direction, fraction)
    whole = evaluate_patch(patch, 161)
    j = np.arange(41)
    # piece parameter j/40 is original parameter fraction*j/40, or fraction + (1-fraction)*j/40
    index_low = np.round(fraction * j * 4).astype(int)
    index_high = np.round((fraction + (1 - fraction) * j / 40) * 160).astype(int)
    if direction == "u":
        expected_low, expected_high = whole[index_low][:, ::4], whole[index_high][:, ::4]
    else:
        expected_low, expected_high = whole[::4][:, index_low], whole[::4][:, index_high]
    assert np.max(np.abs(evaluate_patch(low, 41) - expected_low)) < 1e-12
    assert np.max(np.abs(evaluate_patch(high, 41) - expected_high)) < 1e-12


def sample(patch, direction, parameters):
    """Evaluate one direction at ``parameters`` along the edge v=0 or u=0 by the lattice."""
    count = 401
    lattice = evaluate_patch(patch, count)
    index = np.round(np.asarray(parameters) * (count - 1)).astype(int)
    return lattice[index, 0] if direction == "u" else lattice[0, index]


@pytest.mark.parametrize("direction", ["u", "v"])
def test_split_edge_curves_follow_the_original_edge(direction):
    patch = rough_patch()
    x = 0.5
    low, high = split_patch(patch, direction, x)
    s = np.linspace(0, 1, 41)
    assert np.max(np.abs(sample(low, direction, s) - sample(patch, direction, s * x))) < 1e-12
    assert (
        np.max(np.abs(sample(high, direction, s) - sample(patch, direction, x + s * (1 - x))))
        < 1e-12
    )


def test_split_pieces_meet_on_the_cut():
    patch = rough_patch()
    low, high = split_patch(patch, "u", 0.43)
    assert np.max(np.abs(low.controls[-1] - high.controls[0])) < 1e-13
    assert np.max(np.abs(evaluate_patch(low, 25)[-1] - evaluate_patch(high, 25)[0])) < 1e-12


def test_split_at_an_existing_knot_equals_split_at_a_new_one():
    patch = rough_patch()
    knot = patch.knots_u[3]
    low, high = split_patch(patch, "u", knot)
    assert low.controls.shape[0] + high.controls.shape[0] == patch.controls.shape[0] + 3


def test_cylinder_split_stays_on_the_cylinder():
    angle = np.linspace(0, np.pi / 2, 9)
    grid = np.stack(
        [
            np.stack([np.full_like(angle, x), np.cos(angle), np.sin(angle)], axis=-1)
            for x in np.linspace(0, 3, 12)
        ]
    )
    low, high = split_patch(interpolate_patch(grid), "u", 0.5)
    for piece in (low, high):
        points = evaluate_patch(piece, 31).reshape(-1, 3)
        assert np.max(np.abs(np.hypot(points[:, 1], points[:, 2]) - 1)) < 5e-4


@pytest.mark.parametrize(
    ("direction", "x", "times"),
    [("w", 0.5, 1), ("u", 0.0, 1), ("u", 1.0, 1), ("u", 1.5, 1), ("u", 0.5, 4), ("u", 0.5, -1)],
)
def test_insertion_refuses_bad_input(direction, x, times):
    with pytest.raises(InputValueError):
        insert_knot(rough_patch(), direction, x, times)


def test_insertion_past_the_degree_counts_existing_multiplicity():
    patch = insert_knot(rough_patch(), "u", 0.4, times=2)
    with pytest.raises(InputValueError, match="multiplicity"):
        insert_knot(patch, "u", 0.4, times=2)


def test_split_refuses_a_cut_outside_the_domain():
    with pytest.raises(InputValueError):
        split_patch(rough_patch(), "u", 1.2)
