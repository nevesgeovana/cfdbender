"""P058-9: the sections table of a point opens with the run block and keys rows by family."""

from __future__ import annotations

import csv
import json

import qprop_workspace as qw
from test_matrix_selection import three

from cfdbender.matrix._section_tables import qprop_radial
from cfdbender.matrix.cli import main

RADIAL = (
    "#  V(m/s)  rpm\n#  20  6000\n#\n#  radius   chord   beta   Cl   Cd\n"
    "   0.05   0.03   50.0   0.4   0.02\n   0.10   0.02   40.0   0.5   0.02\n\n# end\n"
)


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_the_radial_table_is_read_by_its_header_names():
    table = qprop_radial(RADIAL)
    assert table is not None and table.names == ("radius", "chord", "beta", "Cl", "Cd")
    assert table.rows == ((0.05, 0.03, 50.0, 0.4, 0.02), (0.10, 0.02, 40.0, 0.5, 0.02))
    assert qprop_radial("#  radius   chord   beta\n") is None


def test_a_qprop_point_writes_its_sections_table(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDBENDER_QPROP_FAKE_RADIAL", "1")
    qw.workspace(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    post = tmp_path / "post/study"
    files = sorted((post / "sections/P1001").iterdir())
    assert len(files) == 1 and files[0].name.endswith("_sections.csv")
    rows = read(files[0])
    polar = read(post / "polars/FAM-ACOEF-P1001.csv")
    block = list(polar[0])[: list(polar[0]).index("FAMILY")]
    header = list(rows[0])
    assert header[: len(block)] == block
    assert header[len(block) :] == [
        "FAMILY", "FAMILY_ID", "STEP", "PLANE", "ROTOR_ID", "AZIMUTH", "radius", "chord", "beta",
    ]  # fmt: skip
    assert {(r["FAMILY"], r["FAMILY_ID"], r["PLANE"], r["ROTOR_ID"]) for r in rows} == {
        ("BLADES", "1", "radial", "0")
    }
    assert rows[0]["STEP"] == rows[0]["AZIMUTH"] == "NA"
    stations = json.loads(
        next((tmp_path / "sims/sim_1001/datapoints").iterdir())
        .joinpath("sections.json")
        .read_bytes()
    )["sections"]
    assert len(rows) == len(stations)
    data = json.loads((post / "products.json").read_bytes())
    spec = {c["name"]: c for c in data["kinds"]["sections"]["columns"]}
    assert spec["radius"]["unit"] == "m" and spec["beta"]["unit"] == "deg"
    assert any(p["kind"] == "sections" for p in data["products"])


def test_a_point_without_a_distribution_is_a_named_skip_and_an_airframe_writes_none(tmp_path):
    qw.workspace(tmp_path / "q")
    assert main(["run", "study", "--workspace", str(tmp_path / "q")]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path / "q")]) == 0
    data = json.loads((tmp_path / "q/post/study/products.json").read_bytes())
    assert any(s["product"].startswith("sections/") for s in data["skipped"])
    assert data["complete"] is True
    three(tmp_path / "m")
    assert main(["run", "study", "--workspace", str(tmp_path / "m")]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path / "m")]) == 0
    assert not (tmp_path / "m/post/study/sections").exists()
