"""The sample layouts of volume cuts and probes against the recorded outputs of another library."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cfdbender.post import volume_layouts
from cfdbender.post.volume_layouts import (
    circle_points,
    line_points,
    rectangle_points,
    volume_section_points,
)

REFERENCE = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "volume_layouts_reference.json").read_text(
        encoding="utf-8"
    )
)


#: A circle is laid with a sine and a cosine, which are not bit reproducible across platforms:
#: its points agree to this relative tolerance, with a floor of the same size times the
#: largest coordinate of the cut so that a coordinate that is zero up to rounding passes.
RTOL = 1e-12


@pytest.mark.parametrize("case", REFERENCE["sections"])
def test_a_volume_cut_lays_the_recorded_points(case: dict) -> None:
    declaration = case["declaration"]
    laid = volume_section_points(
        declaration["shape"],
        declaration["plane"],
        corners_m=declaration.get("corners_m"),
        radii_m=declaration.get("radii_m"),
        points=declaration.get("points"),
        refinement_layers=declaration.get("refinement_layers", 1),
        offset_m=declaration.get("offset_m", 0.0),
    )
    recorded = np.array(case["points"])
    if declaration["shape"] == "circle":
        np.testing.assert_allclose(laid, recorded, rtol=RTOL, atol=RTOL * np.abs(recorded).max())
    else:
        np.testing.assert_array_equal(laid, recorded)


def test_the_recorded_cuts_cover_both_shapes_and_every_plane() -> None:
    kinds = {(c["declaration"]["shape"], c["declaration"]["plane"]) for c in REFERENCE["sections"]}
    assert {"rectangle", "circle"} == {k[0] for k in kinds}
    assert {"XY", "XZ", "YZ"} == {k[1] for k in kinds}


@pytest.mark.parametrize("case", REFERENCE["lines"])
def test_a_probe_line_lays_the_recorded_points(case: dict) -> None:
    d = case["declaration"]
    laid = line_points(d["start"], d["end"], d["points"], d["scale"])
    np.testing.assert_array_equal(laid, np.array(case["points"]))


@pytest.mark.parametrize("case", REFERENCE["rectangles"])
def test_a_probe_rectangle_lays_the_recorded_points(case: dict) -> None:
    d = case["declaration"]
    laid = rectangle_points(
        d["origin"], d["along_u"], d["along_v"], d["points_u"], d["points_v"], d["scale"]
    )
    np.testing.assert_array_equal(laid, np.array(case["points"]))


@pytest.mark.parametrize("case", REFERENCE["circles"])
def test_a_probe_circle_lays_the_recorded_points(case: dict) -> None:
    d = case["declaration"]
    laid = circle_points(
        d["center"],
        d["normal"],
        d["radius"],
        d["points_radial"],
        d["points_azimuth"],
        d["scale"],
    )
    recorded = np.array(case["points"])
    # Only permit a one-unit discrepancy when the unrounded coordinate is at
    # the boundary between the recorded and adjacent five-decimal values.
    with patch.object(volume_layouts, "round", lambda value, digits: value, create=True):
        raw = volume_layouts.circle_points(
            d["center"],
            d["normal"],
            d["radius"],
            d["points_radial"],
            d["points_azimuth"],
            d["scale"],
        )
    assert laid.shape == recorded.shape == raw.shape
    difference = np.abs(laid - recorded)
    boundary = np.abs(np.abs(raw - recorded) - 0.5e-5)
    allowance = RTOL * max(1.0, float(np.max(np.abs(raw))))
    assert np.all((difference == 0) | ((difference <= 1.001e-5) & (boundary <= allowance)))
