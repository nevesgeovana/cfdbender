"""RP-1: retain reported wind coefficients when sideslip makes geometry loads incomplete."""

import csv

import pytest

from cfdbender.matrix._family_polar import polar_files
from cfdbender.matrix.exports import _cell, summary_point


@pytest.mark.parametrize(
    "solver,native", [("su2", {"CL": 0.8, "CD": 0.1}), ("mock", {"CZ": 0.8, "CX": 0.1})]
)
def test_wind_coefficients_survive_unknown_side_force(tmp_path, solver, native):
    row = {
        "pol": "1001",
        "point_id": "1001/DP",
        "condition": {"ALPHA": 4.0, "BETA": 8.0},
        "result": {"coefficients": {k: {"value": v} for k, v in native.items()}, "references": {}},
    }
    point = summary_point(row, solver)

    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    polar_files(tmp_path, [point], solver, write=write, cell=_cell)
    for name in ("FAM-ACOEF-P1001.csv", "P1001-TOTAL-ACOEF.csv", "P1001-AIRFRAME-ACOEF.csv"):
        with (tmp_path / "polars" / name).open(encoding="utf-8", newline="") as stream:
            result = next(csv.DictReader(stream))
        assert float(result["CDW"]) == pytest.approx(0.1)
        assert float(result["CLW"]) == pytest.approx(0.8)
        assert result["CYW"] == "NA"
