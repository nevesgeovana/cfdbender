"""The sample layouts of volume cuts and probes, by arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cfdbender.post.volume_layouts import (
    LayoutError,
    circle_points,
    line_points,
    rectangle_points,
    volume_section_points,
)


def test_a_rectangle_cut_has_its_corners_and_default_counts() -> None:
    laid = volume_section_points("rectangle", "XY", corners_m=(-1.0, 2.0, 3.0, 4.0), offset_m=0.5)
    assert laid.shape == (625, 3)
    assert laid[0].tolist() == [-1.0, 2.0, 0.5]
    assert laid[-1].tolist() == [3.0, 4.0, 0.5]
    assert laid[1].tolist() == pytest.approx([-1.0, 2.0 + 2.0 / 24.0, 0.5])


def test_each_refinement_layer_bisects_every_interval() -> None:
    laid = volume_section_points(
        "rectangle", "XZ", corners_m=(0.0, 0.0, 1.0, 1.0), points=(3, 5), refinement_layers=3
    )
    assert laid.shape == (9 * 17, 3)


def test_a_circle_cut_samples_its_centre_once_and_rings_by_arithmetic() -> None:
    laid = volume_section_points("circle", "YZ", radii_m=(0.0, 2.0), points=(3, 4), offset_m=1.0)
    assert laid.shape == (1 + 2 * 4, 3)
    assert laid[0].tolist() == [1.0, 0.0, 0.0]
    ring1 = laid[1:5]
    np.testing.assert_allclose(np.hypot(ring1[:, 1], ring1[:, 2]), 1.0, atol=1e-15)
    np.testing.assert_allclose(ring1[:, 0], 1.0)
    angles = np.degrees(np.arctan2(ring1[:, 2], ring1[:, 1]))
    np.testing.assert_allclose(angles, [0.0, 90.0, 180.0, -90.0], atol=1e-12)
    np.testing.assert_allclose(np.hypot(laid[5:, 1], laid[5:, 2]), 2.0, atol=1e-15)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": "rectangle", "plane": "XY"}, "corners_m"),
        ({"shape": "rectangle", "plane": "XY", "corners_m": (0, 0, 0, 1)}, "no area"),
        ({"shape": "rectangle", "plane": "XY", "corners_m": (0, 0, 1, 1), "points": (1, 5)}, "2"),
        (
            {"shape": "rectangle", "plane": "XY", "corners_m": (0, 0, 1, 1), "radii_m": (0, 1)},
            "circle",
        ),
        ({"shape": "circle", "plane": "XY", "radii_m": (0, 1)}, "points"),
        ({"shape": "circle", "plane": "XY", "radii_m": (1, 1), "points": (3, 4)}, "r1 < r2"),
        ({"shape": "circle", "plane": "XY", "radii_m": (0, 1), "points": (3, 2)}, "azimuthal"),
        (
            {
                "shape": "circle",
                "plane": "XY",
                "corners_m": (0, 0, 1, 1),
                "radii_m": (0, 1),
                "points": (3, 4),
            },
            "rectangle",
        ),
        ({"shape": "disc", "plane": "XY"}, "rectangle"),
        ({"shape": "rectangle", "plane": "AB", "corners_m": (0, 0, 1, 1)}, "plane"),
    ],
)
def test_a_declaration_that_lays_nothing_is_refused_by_name(kwargs: dict, message: str) -> None:
    with pytest.raises(LayoutError, match=message):
        volume_section_points(**kwargs)


def test_probe_shapes_refuse_what_pyflightstream_refuses() -> None:
    with pytest.raises(LayoutError):
        line_points((0, 0, 0), (1, 0, 0), 1)
    with pytest.raises(LayoutError, match="no area"):
        rectangle_points((0, 0, 0), (1, 0, 0), (2, 0, 0), 2, 2)
    with pytest.raises(LayoutError):
        circle_points((0, 0, 0), (0, 0, 0), 1.0, 3, 4)
    with pytest.raises(LayoutError):
        circle_points((0, 0, 0), (0, 0, 1), math.nan, 3, 4)
