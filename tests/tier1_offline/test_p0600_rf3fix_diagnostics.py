"""Every export/reader failure explains the component, numbers and action.

Reproduction: a stage returns a short kernel error and no outputs.sources;
run_geometry must still produce a complete diagnostic for the selected parts.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cfdbender.geometry import _geom_export as export
from cfdbender.geometry import pipeline
from cfdbender.geometry.stages import check_entry, stage_row


@pytest.mark.parametrize(
    "stage",
    ["S11", "S11.V", "S11.P", "S11.R", "S11.S", "S12.C", "S12.G", "S12"],
    ids=lambda s: f"P0600-RF3-5-{s}",
)
@pytest.mark.parametrize("note", ["kernel failed", ""], ids=["short-note", "no-note"])
def test_p0600_rf3fix_every_failure_has_full_message(tmp_path, stage, note):
    def failing(context):
        return pipeline.StageOutcome(
            checks=[check_entry("test.failure", "failed", 7, 0, note=note)]
        )

    report = pipeline.run_geometry(
        {"components": {"panel": {"type": "body"}}},
        out=tmp_path,
        until=stage,
        stages=[pipeline.Stage(stage_row(stage), failing)],
    )
    assert report["status"] == "failed"
    message = report["message"]
    for text in ("panel", "measured 7", "limit", "0", "rebuild"):
        assert text in message, message
    if note:
        assert note in message


@pytest.mark.parametrize("kind", ["free", "triple", "sliver"], ids=lambda s: f"P0600-RF3-5-{s}")
def test_p0600_rf3fix_sew_failures_name_components(tmp_path, monkeypatch, kind):
    points = [[0, 0, 0], [1, 0, 0]]
    topology = {
        "areas": [1e-12 if kind == "sliver" else 1.0],
        "perimeters": [4.0],
        "edges": [{"samples": points, "faces": [0] if kind == "free" else [0, 1, 2]}]
        if kind != "sliver"
        else [],
        "pid": 123,
    }
    monkeypatch.setattr(export, "_inspect", lambda *args: topology)
    result = export.sew_back(
        tmp_path,
        {
            "iges": "assembly.igs",
            "input_free_edges": [],
            "input_slivers": 0,
            "pieces": [{"component": "panel", "face_records": [{"name": "panel_0"}]}],
        },
    )
    failures = [c for c in result["checks"] if c["status"] == "failed"]
    assert failures
    for check in failures:
        for text in ("panel", str(check["measured"]), str(check["limit"]), "rebuild"):
            assert text in check["note"], check


@pytest.mark.parametrize(
    "stage",
    ["S11", "S11.V", "S11.P", "S11.R", "S11.S", "S12.C", "S12.G", "S12"],
    ids=lambda s: f"P0600-RF3-5-raised-{s}",
)
def test_p0600_rf3fix_kernel_failure_has_full_message(tmp_path, stage):
    from cfdbender._errors import InputArtifactError

    def failing(context):
        raise InputArtifactError("kernel fault", kind="geometry")

    report = pipeline.run_geometry(
        {"components": {"panel": {"type": "body"}}},
        out=tmp_path,
        until=stage,
        stages=[pipeline.Stage(stage_row(stage), failing)],
    )
    for text in ("panel", "kernel fault", "measured 1", "limit", "0", "rebuild"):
        assert text in report["message"], report


def _complete_failures(result):
    failures = [c for c in result["checks"] if c["status"] == "failed"]
    assert failures
    for check in failures:
        for text in ("panel", str(check["measured"]), str(check["limit"]), "rebuild"):
            assert text in check["note"], check


@pytest.mark.parametrize("note", ["rebuild was attempted"], ids=["P0600-RF3-5-action-is-explicit"])
def test_p0600_rf3fix_a_rebuild_mention_is_not_a_corrective_action(tmp_path, note):
    def failing(context):
        return pipeline.StageOutcome(
            checks=[check_entry("test.failure", "failed", 7, 0, note=note)]
        )

    report = pipeline.run_geometry(
        {"components": {"panel": {"type": "body"}}},
        out=tmp_path,
        until="S11.R",
        stages=[pipeline.Stage(stage_row("S11.R"), failing)],
    )
    assert "configuration, then rebuild." in report["message"], report


@pytest.mark.parametrize("stage", ["write", "profile"], ids=lambda s: f"P0600-RF3-5-{s}")
def test_p0600_rf3fix_export_checks_name_components(tmp_path, monkeypatch, stage):
    from cfdbender.geometry import _geom_gate, _iges_write

    manifest = {
        "iges": "assembly.igs",
        "pieces": [
            {
                "component": "panel",
                "file": "panel.igs",
                "faces": 1,
                "layout": "panel.layout.json",
                "face_records": [
                    {
                        "name": "panel_0",
                        "base": "outer",
                        "iges": "panel.face-0.igs",
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(
        _iges_write, "read_iges", lambda *args: {"entities": [{"type": 186, "id": 1}]}
    )
    if stage == "write":
        result = export.write_checks(tmp_path, manifest)
    else:
        (tmp_path / "panel.layout.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(export, "sidecar_identity", lambda *args: "source")
        monkeypatch.setattr(_geom_gate, "read_bases", lambda *args: {"panel_0": {}})
        monkeypatch.setattr(_geom_gate, "base_displacement", lambda *args: 1e-4)
        result = export.profile_checks(
            tmp_path, manifest, {"sources": {"panel": {"source_surface_id": "source"}}}
        )
    _complete_failures(result)


@pytest.mark.parametrize("stage", ["convert", "audit"], ids=lambda s: f"P0600-RF3-5-{s}")
def test_p0600_rf3fix_reader_checks_name_components(tmp_path, monkeypatch, stage):
    from cfdbender.geometry import _geom_reader as reader
    from cfdbender.geometry import _iges_write

    manifest = {"iges": "assembly.igs", "pieces": [{"component": "panel", "faces": 1}]}
    if stage == "convert":
        monkeypatch.setattr(_iges_write, "read_iges", lambda *args: {"entities": []})
        result = reader.convert(tmp_path, manifest)
    else:
        mesh = SimpleNamespace(
            verts=np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)]), families={"panel_0": [[0, 1, 2]]}
        )
        monkeypatch.setattr(reader, "read_obj", lambda *args: mesh)
        (tmp_path / "S12-boundary.json").write_text(
            json.dumps({"boundary": [], "faces": {"panel_0": 1}, "validated": False}),
            encoding="utf-8",
        )
        result = reader.audit(tmp_path, manifest, {"components": {"panel": {"type": "body"}}})
    _complete_failures(result)
