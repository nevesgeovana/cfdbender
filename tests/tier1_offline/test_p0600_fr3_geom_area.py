"""A rounded cap must survive IGES export within the fixed area limit.

Reproduction: run the WING of geom_geometry_config.toml through the write
and read-back stages. The complete documented command is checked separately.
"""

import json
import tomllib
from pathlib import Path

import pytest

from cfdbender.geometry import pipeline
from cfdbender.geometry.pipeline import Stage, run_geometry
from cfdbender.geometry.stages import stage_row


@pytest.mark.parametrize("component", ["WING"], ids=["P0600-rounded-cap-readback"])
def test_p0600_fr3_fixture_cap_keeps_its_area_on_reimport(tmp_path, component):
    fixture = Path(__file__).parent / "fixtures" / "geom_geometry_config.toml"
    raw = tomllib.loads(fixture.read_text(encoding="utf-8"))
    raw["configurations"]["cap_check"] = [component]
    stages = [Stage(stage_row(s), pipeline.STAGE_FUNCTIONS[s]) for s in ("S0", "S11", "S11.R")]
    report = run_geometry(
        raw, configuration="cap_check", out=tmp_path, until="S11.R", stages=stages
    )
    assert report["status"] != "failed", report
    record = json.loads((tmp_path / "S11.R.json").read_text(encoding="utf-8"))
    check = next(c for c in record["checks"] if c["id"] == "area.read_back")
    assert check["limit"] == 0.001
    assert check["status"] == "passed", check


@pytest.mark.parametrize(
    "points, expected",
    [
        ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], 0.5),
        ([(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0, 1, 0)], 0.75),
    ],
    ids=["P0600-triangle-trim", "P0600-trapezoid-trim"],
)
def test_p0600_fr3_area_does_not_integrate_a_trimmed_bounding_rectangle(points, expected):
    from cfdbender.geometry._geom_area import export_area
    from cfdbender.mesh._gmsh import open_kernel

    kernel = open_kernel("trimmed area control")
    try:
        face = kernel.add_planar_face(points)
        assert export_area(kernel, face) == pytest.approx(expected, rel=1e-9)
    finally:
        kernel.close()
