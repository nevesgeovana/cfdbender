"""Export refusals explain the check and the geometry that needs attention.

Reproduction: cfd-geom run wing.geom.toml with the multi-component fixture.
An area read-back failure must carry its numbers and a corrective action.
"""

import json

import pytest

from cfdbender.geometry import _geom_export as export


@pytest.mark.parametrize("difference", [0.01], ids=["P0600-area-refusal"])
def test_p0600_fr3_area_failure_explains_component_numbers_and_action(
    tmp_path, monkeypatch, difference
):
    manifest = {
        "iges": "panel.igs",
        "face_orders": {file: ["F0000001"] for file in ("panel.igs", "panel.face-0.igs")},
        "pieces": [
            {
                "component": "panel",
                "file": "panel.igs",
                "source_area_m2": 1.0,
                "free_edges": 0,
                "face_records": [
                    {
                        "name": "panel_0",
                        "iges": "panel.face-0.igs",
                        "area_m2": 1.0,
                        "area_error_m2": 0.0,
                        "iges_identity": "F0000001",
                        "centroid_m": [0.5, 0.5, 0],
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(
        export,
        "_inspect",
        lambda *args: {
            "areas": [1.0 + difference],
            "area_errors": [0.0],
            "edges": [],
            "pid": 123,
            "identities": ["F0000001"],
            "centroids": [[0.5, 0.5, 0]],
            "names": ["panel_0" if args[0].name == "panel.face-0.igs" else "panel"],
            "property_names": ["panel_0" if args[0].name == "panel.face-0.igs" else "panel"],
        },
    )
    result = export.read_back(tmp_path, manifest, 0.1)
    check = next(c for c in result["checks"] if c["id"] == "area.read_back")
    assert check["status"] == "failed"
    assert check["measured"] == pytest.approx(difference)
    note = check["note"]
    for value in ("area.read_back", "panel", "panel_0", str(check["measured"]), "0.001"):
        assert value in note
    assert "trim" in note.lower() and "rebuild" in note.lower()
    assert json.loads((tmp_path / "manifest.json").read_text())["pieces"]


@pytest.mark.parametrize("measured", [7], ids=["P0600-stage-refusal"])
def test_p0600_fr3_stage_failure_without_note_still_explains_its_report(tmp_path, measured):
    from cfdbender.geometry import pipeline
    from cfdbender.geometry.stages import check_entry, stage_row

    def failing(context):
        return pipeline.StageOutcome(
            checks=[
                check_entry(
                    "gate.grid", "failed", measured, 0, unit="defects", where={"component": "panel"}
                )
            ]
        )

    report = pipeline.run_geometry(
        {"components": {"panel": {"type": "body"}}},
        out=tmp_path,
        until="S12.G",
        stages=[pipeline.Stage(stage_row("S12.G"), failing)],
    )
    assert report["status"] == "failed"
    for text in ("S12.G", "gate.grid", "panel", "7", "0", "rebuild"):
        assert text in report["message"]


def test_readback_failures_name_the_affected_component(tmp_path, monkeypatch):
    manifest = {
        "iges": "panel.igs",
        "face_orders": {file: ["F0000001"] for file in ("panel.igs", "panel.face-0.igs")},
        "pieces": [
            {
                "component": "panel",
                "file": "panel.igs",
                "source_area_m2": 0.5,
                "free_edges": 1,
                "face_records": [
                    {
                        "name": "panel_0",
                        "iges": "panel.face-0.igs",
                        "area_m2": 1.0,
                        "area_error_m2": 0.0,
                        "iges_identity": "F0000001",
                        "centroid_m": [0.5, 0.5, 0],
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(
        export,
        "_inspect",
        lambda *args: {
            "areas": [1.0],
            "area_errors": [0.0],
            "edges": [],
            "pid": 123,
            "identities": ["F0000001"],
            "centroids": [[0.5, 0.5, 0]],
            "names": ["panel_0" if args[0].name == "panel.face-0.igs" else "panel"],
            "property_names": ["panel_0" if args[0].name == "panel.face-0.igs" else "panel"],
        },
    )
    result = export.read_back(tmp_path, manifest, 0.1)
    for check in result["checks"]:
        if check["status"] == "failed":
            assert "panel" in check["note"], check
