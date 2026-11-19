"""The rotor table has pyflightstream's file name and columns; what nothing fills is absent."""

from __future__ import annotations

import json

import pytest
from test_su2_rotor_coefficients import collected, expected

HEAD = [
    "POL",
    "ROTOR",
    "ALPHA",
    "BETA",
    "MACH",
    "RE",
    "VINF",
    "VREF",
    "ALT",
    "RHO",
    "TEMP",
    "MU",
    "J",
    "J_CLOCK",
    "RPM_CLOCK",
    "SREF",
    "CREF",
]
ROTOR = [
    "RPM_ROTOR",
    "DIAMETER_ROTOR",
    "J_ROTOR",
    "CT_ROTOR",
    "CQ_ROTOR",
    "CP_ROTOR",
    "ETA_ROTOR",
    "CN_ROTOR",
    "CS_ROTOR",
    "CMN_ROTOR",
    "CMS_ROTOR",
]


def table(tmp_path, copies=6):
    from cfdbender.matrix.exports import write_tables

    result, plan = collected(tmp_path, copies)
    point = plan.points[0]
    rows = [
        {
            "point_id": point.point_id,
            "pol": point.pol,
            "condition": dict(point.condition),
            "status": "COMPLETED",
            "result": result.to_dict(),
        }
    ]
    return write_tables(tmp_path, "inputs/matrices/dpw.fs", rows, "su2"), point


def test_the_rotor_table_is_the_polar_family_file_with_pyflightstreams_columns(tmp_path):
    paths, point = table(tmp_path)
    found = [p for p in paths if p.name.endswith("_rotor.csv")]
    assert [p.relative_to(tmp_path).as_posix() for p in found] == [
        "post/dpw/polars/P5001-ROTOR_rotor.csv"
    ]
    assert not any(p.name == "rotor.csv" for p in paths)
    header, line = found[0].read_text(encoding="ascii").splitlines()
    assert header.split(",") == [*HEAD, *ROTOR]
    cells = dict(zip(header.split(","), line.split(","), strict=True))
    row, stream = expected(k=6.0)
    for column in ("CT", "CQ", "CP", "ETA", "RPM"):
        assert float(cells[f"{column}_ROTOR"]) == pytest.approx(row[column], rel=1e-9), column
    assert float(cells["J"]) == pytest.approx(0.4, rel=1e-12)
    assert float(cells["J_CLOCK"]) == pytest.approx(0.4, rel=1e-9)
    assert float(cells["RPM_CLOCK"]) == pytest.approx(row["RPM"], rel=1e-9)
    assert float(cells["VINF"]) == pytest.approx(stream["speed_m_s"], rel=1e-9)
    assert float(cells["RHO"]) == pytest.approx(stream["density_kg_m3"], rel=1e-9)
    assert (cells["POL"], cells["ROTOR"], cells["ALT"], cells["RE"]) == (
        "5001",
        "ROTOR",
        "NA",
        "NA",
    )
    for column in ("CN", "CS", "CMN", "CMS"):  # six copies: the in-plane parts cancel
        assert float(cells[f"{column}_ROTOR"]) == pytest.approx(0.0, abs=1e-9)
    assert point.point_id.endswith("J+040"), "the point name codes J as pyflightstream does"


def test_the_manifest_names_every_absent_column(tmp_path):
    from cfdbender.matrix.products import _entry

    paths, _ = table(tmp_path)
    found = next(p for p in paths if p.name.endswith("_rotor.csv"))
    entry = _entry(tmp_path, found)
    assert entry["kind"] == "rotor" and entry["polar"] == "5001" and entry["rotor"] == "ROTOR"
    assert entry["absent_columns"] == [
        "BREF",
        "ETAW_ROTOR",
        "MTIP_ROTOR",
        "MHEL_ROTOR",
        "MADV_ROTOR",
    ]
    json.dumps(entry)
