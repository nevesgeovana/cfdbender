"""P058-7: products.json states the units, frames, kinds, families, groups and polars."""

from __future__ import annotations

import json

import qprop_workspace as qw
from test_matrix_selection import three

from cfdbender.matrix.cli import main


def collected(root):
    assert main(["run", "study", "--workspace", str(root)]) == 0
    assert main(["collect", "study", "--workspace", str(root)]) == 0
    post = root / "post/study"
    return post, json.loads((post / "products.json").read_bytes())


def test_kinds_state_every_written_header_with_units_and_frames(tmp_path):
    three(tmp_path)
    post, data = collected(tmp_path)
    assert data["schema_version"] == 2 and data["complete"] is True
    assert {"polars", "time_history", "reductions"} <= set(data["tree"])
    block = [entry["name"] for entry in data["run_block"]]
    assert all({"name", "unit", "source"} <= set(entry) for entry in data["run_block"])
    by_name = {entry["name"]: entry for entry in data["run_block"]}
    assert by_name["VINF"]["unit"] == "m/s" and by_name["ALPHA"]["unit"] == "deg"
    entries = [p for p in data["products"] if p["path"].startswith("post/study/polars/")]
    assert entries and all("kind" in p and "polar" in p for p in entries)
    for entry in entries:
        header = (tmp_path / entry["path"]).read_text(encoding="utf-8").splitlines()[0]
        names = header.split(",")
        assert names[: len(block)] == block, entry["path"]
        columns = data["kinds"][entry["kind"]]["columns"]
        assert names[len(block) :] == [c["name"] for c in columns], entry["path"]
        assert all("unit" in c and "frame" in c for c in columns)
    acoef = {c["name"]: c for c in data["kinds"]["FAM-ACOEF"]["columns"]}
    assert acoef["CDW"] == {
        "name": "CDW",
        "unit": "-",
        "frame": "W",
        "quantity": "drag coefficient",
        "definition": "docs/results.md#acoef",
        "sign": "positive aft",
    }
    dim = {c["name"]: c for c in data["kinds"]["FAM-DIM"]["columns"]}
    assert dim["FX_MRP"]["unit"] == "N" and dim["MY_MRP"]["unit"] == "N m"
    assert set(data["frames"]) == {"E", "B", "S", "W", "mesh"}
    assert data["frames"]["E"]["axes"] == "x aft, y right, z up"
    assert data["families"] == [{"FAMILY_ID": 1, "FAMILY": "BODY", "ROTOR_ID": None}]
    assert [(g["GROUP"], g["GROUP_ID"], g["members"]) for g in data["groups"]] == [
        ("TOTAL", 0, [1]),
        ("AIRFRAME", 1, [1]),
    ]
    assert data["polars"] == {p: {"run_type": "steady"} for p in ("1001", "1002", "1003")}
    assert [s["product"] for s in data["skipped"]] == [
        "polars/P1001-ROTOR",
        "polars/P1002-ROTOR",
        "polars/P1003-ROTOR",
    ]
    assert data["migrated"] == {}


def test_a_rotor_solver_states_its_rotor_frame_and_the_empty_airframe(tmp_path):
    qw.workspace(tmp_path)
    _post, data = collected(tmp_path)
    assert data["frames"]["R0"] == {
        "rotor": "PROPELLER",
        "from": "B",
        "dcm": "C2(theta) C3(psi)",
        "pitch_deg": 0.0,
        "toe_deg": 0.0,
        "sense": 1,
    }
    assert {"FAM-RCOEF", "RCOEF"} <= set(data["kinds"])
    rcoef = {c["name"]: c for c in data["kinds"]["FAM-RCOEF"]["columns"]}
    assert rcoef["P"]["unit"] == "kW" and rcoef["CTR0"]["frame"] == "R0"
    assert rcoef["T"]["sign"] == "positive forward"
    assert data["families"] == [{"FAMILY_ID": 1, "FAMILY": "BLADES", "ROTOR_ID": 0}]
    # The stand-in QPROP prints no radial row, so its sections table is a named skip too.
    assert [s["product"].split("/")[0] for s in data["skipped"]] == ["polars", "sections"]
    assert data["skipped"][0]["product"] == "polars/P1001-AIRFRAME"
    assert data["complete"] is True
