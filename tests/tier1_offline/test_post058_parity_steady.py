"""P058-11: cfdbender's polar files agree with pyflightstream 0.58 on the shared steady polar.

The reference files under ``tests/data/post058/`` were written once by pyflightstream's
own post stage (the 0.58 line, commit 60d7fdc02) on a synthetic campaign and copied here
unchanged but for the line ends: polar 6001, two steady points (ALPHA -2 and 0), two
airframe families W and B and no rotor. ``products-statement.json`` holds the units and
frames that the same post wrote into its ``products.json``, and the SHA-256 of each copied
file as written.

The test feeds cfdbender's writer the reference run block (the condition and references
of each point) and the reference DIM loads of each family, and compares what cfdbender
writes cell by cell: the run block, FAM-DIM, FAM-ACOEF, and the TOTAL group's DIM and
ACOEF formed from the summed loads. The reference prints six significant figures, so a
number agrees within a relative 2e-5 or an absolute 1e-6. The drag split CD0 and CDI is a
quantity of FlightStream's loads export that no cfdbender solver states, so it is not
compared. Out of scope: the reference's unsteady rotor polar 6002 (unsteady rows are a
later release) and its group files g01 and g02 (cfdbender declares no groups).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from cfdbender.matrix._family_columns import Family, acoef_columns, dim_columns, rcoef_columns
from cfdbender.matrix._family_polar import Point, polar_files
from cfdbender.matrix._run_block import run_block_columns
from cfdbender.matrix.exports import _cell

DATA = Path(__file__).resolve().parents[1] / "data/post058"
FAMILIES = (Family("W", 1), Family("B", 2))
DRAG_SPLIT = ("CD0", "CDI")
BLOCK_NUMBERS = (
    "ALPHA", "BETA", "MACH", "RE", "VINF", "VREF", "RHO", "TEMP", "SREF", "CREF", "BREF",
    "XMOM", "YMOM", "ZMOM",
)  # fmt: skip
LOADS = ("FX_MRP", "FY_MRP", "FZ_MRP", "MX_MRP", "MY_MRP", "MZ_MRP")


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(text):
    return None if text == "NA" else float(text)


def same(ours, reference, where):
    if reference == "NA" or ours == "NA":
        assert ours == reference, where
        return
    try:
        expected = float(reference)
    except ValueError:
        assert ours == reference, where
        return
    got = float(ours)
    assert math.isclose(got, expected, rel_tol=2e-5, abs_tol=1e-6), (where, got, expected)


def points():
    """The reference points as cfdbender's writer reads them: condition and family loads."""
    rows = read(DATA / "FAM-DIM-P6001.csv")
    by_point: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_point.setdefault(row["POINT"], []).append(row)
    out = []
    for name, members in by_point.items():
        first = members[0]
        condition = {key: number(first[key]) for key in BLOCK_NUMBERS}
        condition["MU"] = float(first["MU"])
        loads = {
            int(row["FAMILY_ID"]): (
                tuple(float(row[k]) for k in LOADS[:3]),
                tuple(float(row[k]) for k in LOADS[3:]),
            )
            for row in members
        }
        out.append(
            Point(
                pol="6001", name=name, condition=condition, identifiers={}, native={}, loads=loads
            )
        )
    return out


@pytest.fixture
def written(tmp_path):
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        return path

    polar_files(tmp_path, points(), "parity", write=write, cell=_cell, families=FAMILIES)
    return tmp_path / "polars"


@pytest.mark.parametrize(
    ("name", "drop"),
    [
        ("FAM-DIM-P6001.csv", ()),
        ("FAM-ACOEF-P6001.csv", DRAG_SPLIT),
        ("P6001-TOTAL-DIM.csv", ()),
        ("P6001-TOTAL-ACOEF.csv", DRAG_SPLIT),
    ],
)
def test_every_cell_agrees_with_the_reference(written, name, drop):
    reference = read(DATA / name)
    ours = read(written / name)
    assert [c for c in reference[0] if c not in drop] == list(ours[0]), name
    assert len(ours) == len(reference)
    for index, (row, expected) in enumerate(zip(ours, reference, strict=True)):
        for column, value in row.items():
            same(value, expected[column], (name, index, column))


def test_the_headers_are_the_reference_headers():
    statement = json.loads((DATA / "products-statement.json").read_bytes())
    reference = read(DATA / "FAM-ACOEF-P6001.csv")[0]
    block = list(run_block_columns())
    assert list(reference)[: len(block)] == block
    # The statement covers the reference matrix, whose polar 6002 declares rotor 0.
    assert [c["name"] for c in statement["run_block"]] == list(run_block_columns((), (0,)))
    kinds = statement["kinds"]
    assert [c["name"] for c in kinds["FAM-DIM"]] == [c.name for c in dim_columns()]
    assert [c["name"] for c in kinds["FAM-ACOEF"]] == [
        c.name for c in acoef_columns((0,), own=DRAG_SPLIT)
    ]
    assert [c["name"] for c in kinds["ACOEF"]] == [
        c.name for c in acoef_columns((0,), own=DRAG_SPLIT, entity="group")
    ]


@pytest.mark.parametrize("kind", ["FAM-DIM", "FAM-ACOEF", "DIM", "ACOEF"])
def test_units_frames_and_signs_agree_with_the_reference(kind):
    statement = json.loads((DATA / "products-statement.json").read_bytes())
    entity = "group" if not kind.startswith("FAM") else "family"
    ours = {
        c.name: c
        for c in (
            dim_columns(entity) if kind.endswith("DIM") else acoef_columns((0,), entity=entity)
        )
    }
    compared = 0
    for column in statement["kinds"][kind]:
        if column["name"] not in ours:
            assert column["name"] in DRAG_SPLIT
            continue
        spec = ours[column["name"]]
        assert (spec.unit, spec.frame) == (column["unit"], column["frame"]), column["name"]
        assert spec.sign == column.get("sign"), column["name"]
        compared += 1
    assert compared == len(ours)


def test_the_run_block_units_agree_except_the_stated_altitude():
    statement = json.loads((DATA / "products-statement.json").read_bytes())
    from cfdbender.matrix._run_block import RUN_BLOCK_SPEC

    ours = {spec.name.replace("<id>", "0"): spec.unit for spec in RUN_BLOCK_SPEC}
    for column in statement["run_block"]:
        if column["name"] == "ALT":
            # cfdbender states ALT as the matrix writes it, ALTFT in feet.
            assert (ours["ALT"], column["unit"]) == ("ft", "m")
            continue
        assert ours[column["name"]] == column["unit"], column["name"]
    assert set(statement["frames"]) == {"E", "B", "S", "W"}


@pytest.mark.parametrize(("kind", "entity"), [("FAM-RCOEF", "family"), ("RCOEF", "group")])
def test_the_rotor_columns_agree_with_the_reference_statement(kind, entity):
    statement = json.loads((DATA / "products-statement.json").read_bytes())
    ours = rcoef_columns((0,), entity)
    assert [c["name"] for c in statement["kinds"][kind]] == [c.name for c in ours]
    for column, spec in zip(statement["kinds"][kind], ours, strict=True):
        assert (spec.unit, spec.frame, spec.sign) == (
            column["unit"],
            column["frame"],
            column.get("sign"),
        ), column["name"]
