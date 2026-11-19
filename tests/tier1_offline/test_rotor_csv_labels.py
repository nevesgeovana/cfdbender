"""Rotor aliases remain single CSV labels and round-trip through the manifest reader."""

from __future__ import annotations

import csv

import pytest

from cfdbender.matrix._rotor_table import absent_columns, rotor_table
from cfdbender.matrix._unsteady_polar import unsteady_polars
from cfdbender.matrix.exports import _cell, _write


@pytest.mark.parametrize("alias", ["rotor,one", "rotor\u00e9"])
def test_rotor_alias_round_trips_in_csv_and_manifest(tmp_path, alias):
    rows = [
        {
            "pol": "1",
            "result": {
                "rotor": {"rotor": alias, "row": {"CT": 1.0, "RPM": 600.0}},
                "time_average": {"FIRST_STEP": 1, "LAST_STEP": 2, "STEPS": 2, "means": {"CL": 1.0}},
            },
        }
    ]
    (path,) = rotor_table(tmp_path, rows, write=_write, cell=_cell)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert f"CT_{alias}" in reader.fieldnames
        row = next(reader)
        assert row["ROTOR"] == alias
        assert row[f"CT_{alias}"] == "1.0"
    try:
        absent = absent_columns(path)
    except UnicodeError as error:
        pytest.fail(f"manifest cannot read the writer's UTF-8 table: {error}")
    assert f"ETAW_{alias}" in absent
    assert f"CT_{alias}" not in absent
    (polar,) = unsteady_polars(tmp_path, rows, write=_write, cell=_cell)
    with polar.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert f"RPM_{alias}" in reader.fieldnames
        assert next(reader)[f"RPM_{alias}"] == "600.0"
