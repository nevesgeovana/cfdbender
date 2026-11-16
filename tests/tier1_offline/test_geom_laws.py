"""Geometry: section law checks (S1), compatibility (S3) and the physical section.

Expected values are closed forms written here: the leading-edge parabola z^2 = 2 r x, the
trailing-edge slope tan(boat tail), and the maximum thickness read on a dense grid.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from geom_inputs import naca_stations, stations

from cfdbender._errors import InputValueError
from cfdbender.geometry import resolve_config
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._lifting import lifting_surface, section_models
from cfdbender.geometry._sections import (
    CstSection,
    fit_or_read,
    section_checks,
    section_from_physical,
    shape_unchanged,
)


def lifting(sections, stations=None):
    out = {
        "type": "lifting_surface",
        "planform": {"semispan_mm": 600.0, "taper": 0.5, "le_sweep_deg": 0.0},
        "sections": sections,
    }
    if stations is not None:
        out["stations"] = stations
    return out


def resolved(sections, tmp_path, stations=None):
    data = {"components": {"W": lifting(sections, stations)}}
    return unpack(resolve_config(data, base=tmp_path))["components"]["W"]


PHYSICAL = {"nose_radius": 0.0158, "boat_tail_deg": 7.0, "thickness_at": 0.12}


def test_physical_section_meets_its_three_numbers():
    section = section_from_physical(PHYSICAL, degree=5, trailing_edge_thickness=0.002)
    x = np.linspace(0, 1, 20001)
    upper, lower = section.evaluate(x)
    assert float(np.max(upper[:, 1] - lower[:, 1])) == pytest.approx(0.12, abs=1e-6)
    near = 1e-7
    zu = section.evaluate(np.array([near]))[0][0, 1]
    assert zu**2 / (2 * near) == pytest.approx(0.0158, rel=1e-3)
    h = 1e-6
    slope = (upper[-1, 1] - section.evaluate(np.array([1 - h]))[0][0, 1]) / h
    assert slope == pytest.approx(-math.tan(math.radians(7.0)), abs=1e-4)
    assert section.te_thickness() == 0.002
    assert section.nose_radius() == pytest.approx(0.0158, rel=1e-9)


def test_physical_section_reaches_the_skin_end_to_end(tmp_path):
    component = resolved({"physical": PHYSICAL}, tmp_path)
    models = section_models(component["sections"])
    assert len(models) == 2
    assert models[0]["upper"] == models[1]["upper"]
    assert len(lifting_surface(component)) > 0
    # Another nose radius gives another skin: the physical input is consumed, not skipped.
    other = resolved({"physical": {**PHYSICAL, "nose_radius": 0.03}}, tmp_path)
    assert section_models(other["sections"])[0]["upper"] != models[0]["upper"]


@pytest.mark.parametrize("missing", ["nose_radius", "boat_tail_deg", "thickness_at"])
def test_physical_section_without_a_number_is_refused_with_the_input(tmp_path, missing):
    physical = {k: v for k, v in PHYSICAL.items() if k != missing}
    with pytest.raises(InputValueError, match=f"sections.physical: .*{missing}"):
        resolved({"physical": physical}, tmp_path)
    with pytest.raises(InputValueError, match=missing):  # the section law refuses it alike
        section_from_physical(physical)


def test_physical_section_impossible_and_conflicting_inputs_are_refused(tmp_path):
    with pytest.raises(InputValueError, match="no section has"):
        section_from_physical({**PHYSICAL, "thickness_at": 0.001}, degree=5)
    with pytest.raises(InputValueError, match="boat_tail_deg"):
        resolved({"physical": {**PHYSICAL, "boat_tail_deg": 95.0}}, tmp_path)
    with pytest.raises(InputValueError, match="given together"):
        resolved({"physical": PHYSICAL}, tmp_path, naca_stations(["NACA0012", "NACA0012"]))


def test_s1_records_for_a_fitted_section(tmp_path):
    table = resolved({}, tmp_path, naca_stations(["NACA0012", "NACA0012"]))["sections"]
    section = fit_or_read(table)[0]
    records = {r["id"]: r for r in section_checks(section)}
    value = records["section.residual"]["value"]
    assert 0 < value <= 5e-4
    assert records["section.residual"]["status"] == ("pass" if value <= 1e-4 else "fail")
    assert records["section.te_thickness"]["value"] == pytest.approx(0.00252)
    nose = records["section.one_nose_radius"]["value"]
    assert nose[0] == pytest.approx(nose[1])
    assert nose[0] == pytest.approx(1.1019 * 0.12**2, rel=0.1)
    loose = {r["id"]: r for r in section_checks(section, residual_limit=1e-3)}
    assert loose["section.residual"]["status"] == "pass"
    tight = {r["id"]: r for r in section_checks(section, residual_limit=1e-12)}
    assert tight["section.residual"]["status"] == "fail"


def test_s1_skips_the_residual_without_ordinates_and_counts_inflections():
    smooth = CstSection((0.2, 0.2, 0.2), (-0.2, -0.2, -0.2), 0.0)
    records = {r["id"]: r for r in section_checks(smooth)}
    assert records["section.residual"]["status"] == "skipped"
    assert records["section.inflection"]["value"] == [1, 1]
    wavy = CstSection((0.2, 0.0, 0.3, 0.0, 0.2), (-0.2, 0.0, -0.3, 0.0, -0.2), 0.0)
    counts = section_checks(wavy)[3]["value"]
    assert counts[0] >= 3
    assert counts[0] == counts[1]


def test_s3_shape_unchanged_and_a_wrong_elevation_is_caught(tmp_path):
    items = [
        {"eta": 0.0, "naca": "0012"},
        {"eta": 1.0, "cst_upper": [0.2, 0.3, 0.1], "cst_lower": [-0.2, -0.1, -0.1], "te": 0.0},
    ]
    table = resolved({}, tmp_path, stations(items))["sections"]
    record = shape_unchanged(table)
    assert record["status"] == "pass"
    assert record["value"] <= 1e-9
    # A zero-padded "elevation" is a different section, and compat measures it.
    section = CstSection((0.2, 0.3, 0.1), (-0.2, -0.1, -0.1), 0.0)
    padded = CstSection((0.2, 0.3, 0.1, 0.0), (-0.2, -0.1, -0.1, 0.0), 0.0)
    assert section.compat([padded]) > 1e-3
