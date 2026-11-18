"""Frames that place declared points, and a rotating frame that follows the rotor azimuth."""

from __future__ import annotations

import math

import numpy as np
import pytest
from post_volume_support import box_grid, with_fields

from cfdbender.post.volume import VolumeSampler
from cfdbender.post.volume_frames import REFERENCE, Frame, FrameError, RotorSpin


def test_the_reference_frame_places_points_unchanged() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.0]])
    np.testing.assert_array_equal(REFERENCE.place(points), points)
    np.testing.assert_array_equal(REFERENCE.place(points, step=17), points)


def test_a_fixed_frame_places_by_origin_and_axes() -> None:
    frame = Frame("body", origin=(1.0, 2.0, 3.0), axes=((0, 1, 0), (-1, 0, 0), (0, 0, 1)))
    placed = frame.place(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 5.0]]))
    np.testing.assert_allclose(placed, [[1.0, 3.0, 3.0], [0.0, 2.0, 8.0]], atol=1e-15)


def test_a_rotating_frame_needs_a_step() -> None:
    frame = Frame("rotor", spin=RotorSpin((0, 0, 1), 0.0, 1.0, 72.0))
    with pytest.raises(FrameError, match="at a step"):
        frame.place(np.zeros((1, 3)))


def test_a_blade_fixed_point_sits_at_the_first_blade_azimuth_for_each_step() -> None:
    origin = (1.0, 2.0, 3.0)
    datum, spr = 10.0, 72.0
    for sense in (1.0, -1.0):
        frame = Frame("rotor", origin, spin=RotorSpin((0, 0, 1), datum, sense, spr))
        local = np.array([[0.8, 0.0, 0.25], [0.0, 0.3, -1.0]])
        for step in (0, 1, 18, 36, 71, 72, 100):
            azimuth = math.radians((datum + sense * step * 360.0 / spr) % 360.0)
            c, s = math.cos(azimuth), math.sin(azimuth)
            want = np.array(
                [
                    [origin[0] + 0.8 * c, origin[1] + 0.8 * s, origin[2] + 0.25],
                    [origin[0] - 0.3 * s, origin[1] + 0.3 * c, origin[2] - 1.0],
                ]
            )
            np.testing.assert_allclose(frame.place(local, step), want, atol=1e-13)


def test_a_tilted_axis_keeps_axis_points_fixed_and_distance_to_the_axis() -> None:
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    frame = Frame("rotor", (0.5, 0.0, -1.0), spin=RotorSpin((2.0, 4.0, 4.0), 0.0, 1.0, 8.0))
    on_axis = axis * 1.7
    other = np.array([0.3, -0.4, 0.9])
    placed_axis = frame.place(np.array([on_axis]), step=3)[0]
    np.testing.assert_allclose(placed_axis, np.array([0.5, 0.0, -1.0]) + on_axis, atol=1e-14)
    start = frame.place(np.array([other]), step=0)[0] - np.array([0.5, 0.0, -1.0])
    moved = frame.place(np.array([other]), step=1)[0] - np.array([0.5, 0.0, -1.0])
    assert np.dot(moved, axis) == pytest.approx(np.dot(start, axis), abs=1e-14)
    assert np.linalg.norm(moved) == pytest.approx(np.linalg.norm(start), abs=1e-14)
    # one step of 8 is 45 degrees: the component off the axis turns by that angle
    off_a = start - np.dot(start, axis) * axis
    off_b = moved - np.dot(moved, axis) * axis
    cosine = np.dot(off_a, off_b) / np.linalg.norm(off_a) / np.linalg.norm(off_b)
    assert math.degrees(math.acos(cosine)) == pytest.approx(45.0, abs=1e-9)
    # a full revolution returns every point
    np.testing.assert_allclose(
        frame.place(np.array([other]), step=8), frame.place(np.array([other]), step=0), atol=1e-13
    )
    # right-hand sense: the turn is positive about the axis
    assert np.dot(np.cross(off_a, off_b), axis) > 0.0


def test_a_probe_in_a_rotating_frame_follows_a_blade_fixed_point_of_a_rotated_field() -> None:
    """The analytic field is steady in the rotor frame, so it is read as the rotor turns."""
    origin = np.array([0.5, 0.5, 0.5])
    datum, spr, sense = 25.0, 36.0, -1.0
    frame = Frame("rotor", tuple(origin), spin=RotorSpin((0, 0, 1), datum, sense, spr))

    def blade_fixed(q: np.ndarray) -> np.ndarray:
        return 3.0 * q[:, 0] - q[:, 1] + 0.5 * q[:, 2] + 2.0

    local = np.array([[0.2, 0.0, 0.1], [0.1, 0.15, -0.2], [0.0, 0.3, 0.05]])
    base = with_fields(box_grid("hex", (4, 4, 4)), lambda p: np.zeros(len(p)))
    for step in (0, 5, 9, 18, 30):
        angle = math.radians((datum + sense * step * 360.0 / spr) % 360.0)
        c, s = math.cos(angle), math.sin(angle)

        def lab_field(x: np.ndarray, c: float = c, s: float = s) -> np.ndarray:
            d = x - origin
            # the position in the rotor frame: rotate by minus the azimuth about z
            q = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]])
            return blade_fixed(q)

        grid = with_fields(base, lab_field)
        placed = frame.place(local, step)
        result = VolumeSampler(grid).sample(placed)
        assert result.n_missing == 0
        np.testing.assert_allclose(result.values["f"], blade_fixed(local), atol=1e-12)
        # and the unrotated frame reads something else once the rotor has turned
        if step:
            fixed = VolumeSampler(grid).sample(REFERENCE.place(local)).values["f"]
            assert not np.allclose(fixed, blade_fixed(local), atol=1e-6)
