"""Geometry: the runner of the pure stages of the geometry pipeline.

Written from the requirement (a failed parameter set returns a report naming its stage and
its check, so a loop can continue) and the public signatures of
:mod:`cfdbender.geometry.pipeline`. A failure is planted in a stage function of
the table; the reports and the stage records are read back from disk.
"""

from __future__ import annotations

import json

import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry import pipeline as _geom_run
from cfdbender.geometry._records import check_entry, exit_code, geometry_report
from cfdbender.geometry.pipeline import StageOutcome
from cfdbender.geometry.stage_table import STAGE_ROWS

BODY = {"components": {"FUS": {"type": "body", "length_mm": 2000.0}}}


def _failing_at(stage_id, *, when_length=None, checks=None):
    """A stage function that fails when the body length equals ``when_length`` (always if None)."""

    def run(context):
        length = context.resolved["components"]["FUS"]["length"]["value"]
        if when_length is not None and length != when_length:
            return StageOutcome(checks=[check_entry("law.monotonic", "passed", 0.0, 1.0)])
        return StageOutcome(
            checks=checks or [check_entry("law.monotonic", "failed", 2.5, 1.0, where={"x": 3})]
        )

    return run


def test_geom_run_a_planted_failure_stops_the_run_and_names_stage_and_check(tmp_path, monkeypatch):
    monkeypatch.setitem(_geom_run.STAGE_FUNCTIONS, "S4", _failing_at("S4"))
    report = _geom_run.run_geometry(BODY, out=tmp_path, until="S9")
    assert report["status"] == "failed"
    assert (report["stage"], report["check"]) == ("S4", "law.monotonic")
    assert (report["measured"], report["limit"], report["where"]) == (2.5, 1.0, {"x": 3})
    assert [s["stage"] for s in report["stages"]] == ["S0", "S1", "S2", "S3", "S4"]
    on_disk = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert on_disk == report
    assert (tmp_path / "S4.json").is_file() and not (tmp_path / "S5.json").exists()


def test_geom_run_the_first_failed_check_of_a_stage_is_the_one_named(tmp_path, monkeypatch):
    checks = [
        check_entry("law.fewest_stations", "passed", 0.0, 1.0),
        check_entry("law.ruled_twist", "failed", 9.0, 1.0),
        check_entry("law.monotonic", "failed", 5.0, 1.0),
    ]
    monkeypatch.setitem(_geom_run.STAGE_FUNCTIONS, "S4", _failing_at("S4", checks=checks))
    report = _geom_run.run_geometry(BODY, out=tmp_path, until="S9")
    assert report["check"] == "law.ruled_twist"


def _every_stage_passing(monkeypatch, *, omit=(), skip=()):
    """Give every stage a function passing each check of its row, but ``omit`` and ``skip``."""
    for row in STAGE_ROWS:

        def run(context, row=row):
            checks = [
                check_entry(cid, "skipped", note="no original")
                if cid in skip
                else check_entry(cid, "passed", 0.0, 1.0)
                for cid, _ in row.checks
                if cid not in omit
            ]
            return StageOutcome(checks=checks)

        monkeypatch.setitem(_geom_run.STAGE_FUNCTIONS, row.stage, run)


def test_geom_run_until_ends_the_run_and_a_clean_run_is_accepted(tmp_path, monkeypatch):
    _every_stage_passing(monkeypatch, skip=("skin.distance",))
    monkeypatch.setitem(_geom_run.STAGE_FUNCTIONS, "S6", _failing_at("S6"))
    report = _geom_run.run_geometry(BODY, out=tmp_path, until="S5")
    assert report["status"] == "accepted" and len(report["stages"]) == 6
    assert report["not_run"] == []
    assert not (tmp_path / "S6.json").exists()
    _every_stage_passing(monkeypatch, skip=("skin.distance",))
    report = _geom_run.run_geometry(BODY, out=tmp_path / "full")
    assert report["status"] == "accepted" and exit_code(report) == 0, report["not_run"]


def test_geom_run_a_run_through_an_unbuilt_stage_is_incomplete_never_accepted(tmp_path):
    """a check that never ran is named, and the run is not read as accepted."""
    report = _geom_run.run_geometry(BODY, out=tmp_path, until="S3")
    assert report["status"] == "incomplete" and exit_code(report) == 3
    assert "S1 section.residual" in report["not_run"]
    assert "S3 compat.common_vector" in report["not_run"]
    assert not any(item.startswith("S0 ") for item in report["not_run"])
    on_disk = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "incomplete" and on_disk["not_run"] == report["not_run"]


def test_geom_run_a_check_a_built_stage_omits_is_recorded_and_named(tmp_path, monkeypatch):
    _every_stage_passing(monkeypatch, omit=("law.fewest_stations",), skip=("skin.distance",))
    report = _geom_run.run_geometry(BODY, out=tmp_path, until="S6")
    assert report["status"] == "incomplete"
    assert report["not_run"] == ["S4 law.fewest_stations"]
    record = json.loads((tmp_path / "S4.json").read_text(encoding="utf-8"))
    entry = next(c for c in record["checks"] if c["id"] == "law.fewest_stations")
    assert entry["status"] == "skipped" and "not measured" in entry["note"]
    assert record["status"] == "passed"


def test_geom_run_an_accepted_report_with_checks_not_run_is_refused():
    with pytest.raises(InputValueError, match="not accepted"):
        geometry_report("accepted", "p", not_run=["S1 section.residual"])


def test_geom_run_an_unbuilt_stage_is_recorded_skipped_and_the_report_says_so(tmp_path):
    report = _geom_run.run_geometry(BODY, out=tmp_path, until="S3")
    assert [s["status"] for s in report["stages"]] == ["passed", "skipped", "skipped", "skipped"]
    assert any("S1 Section law: stage not built" in w for w in report["warnings"])
    record = json.loads((tmp_path / "S1.json").read_text(encoding="utf-8"))
    assert {c["status"] for c in record["checks"]} == {"skipped"} and record["checks"][0]["note"]


@pytest.mark.parametrize(
    ("kwargs", "text"),
    [
        ({"overrides": {"FUS.lenght_mm": 3000.0}}, "unknown parameter"),
        ({"overrides": {"WING.length_mm": 3000.0}}, "declared component"),
        ({"overrides": {"length_mm": 3000.0}}, "declared component"),
        ({"configuration": "nope"}, "configuration 'nope': unknown"),
        ({"variant": "nope"}, "variant 'nope': unknown"),
    ],
)
def test_geom_run_a_bad_input_is_refused_before_any_stage_runs(tmp_path, kwargs, text):
    report = _geom_run.run_geometry(BODY, out=tmp_path, until="S9", **kwargs)
    assert report["status"] == "refused" and text in report["message"]
    assert report["stage"] == "S0" and not list(tmp_path.glob("S*.json"))


WING = {
    "components": {
        "W": {
            "type": "lifting_surface",
            "planform": {"sref_m2": 9.0, "aspect_ratio": 9.0, "taper": 0.6},
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
        }
    }
}


@pytest.mark.parametrize(
    ("dotted", "refused"),
    [
        ("W.planform.sref_m2", False),  # a published key, translated, so absent from the result
        ("W.planform.sref_mm2", True),  # an unknown key of a known table
        ("W.planfrom.sref_m2", True),  # an unknown table: the reviewed code ran without it
        ("W.surfce.span_axis", True),
    ],
)
def test_geom_run_an_override_on_an_unknown_key_or_table_is_refused(tmp_path, dotted, refused):
    plain = _geom_run.run_geometry(WING, out=tmp_path / "a", until="S0")
    report = _geom_run.run_geometry(WING, out=tmp_path / "b", until="S0", overrides={dotted: 10.0})
    if refused:
        assert report["status"] == "refused"
        assert "unknown parameter, nothing was run" in report["message"]
    else:
        assert report["status"] != "refused"
        assert report["parameter_id"] != plain["parameter_id"]  # the override was applied


def test_geom_run_an_override_reaches_a_published_single_surface_file(tmp_path):
    """A [surface] file (the proposal's DOE form) is one component named by surface.name."""
    path = tmp_path / "lifting_doe_wing.toml"
    path.write_text(
        '[surface]\nname = "Wing"\n[planform]\nsref_m2 = 9.0\naspect_ratio = 9.0\ntaper = 0.6\n'
        '[[stations]]\neta = 0.0\nsection = { kind = "naca", code = "0012" }\n',
        encoding="utf-8",
    )
    plain = _geom_run.run_geometry(path, out=tmp_path / "a", until="S0")
    varied = _geom_run.run_geometry(
        path, out=tmp_path / "b", until="S0", overrides={"Wing.planform.sref_m2": 9.9}
    )
    assert varied["status"] != "refused", varied.get("message")
    assert varied["parameter_id"] != plain["parameter_id"]
    misspelt = _geom_run.run_geometry(
        path, out=tmp_path / "c", until="S0", overrides={"Wing.planfrom.sref_m2": 9.9}
    )
    assert misspelt["status"] == "refused" and "unknown parameter" in misspelt["message"]


def test_geom_run_an_unknown_until_and_an_empty_file_are_refused(tmp_path):
    with pytest.raises(InputValueError, match="stage 'S99'"):
        _geom_run.run_geometry(BODY, out=tmp_path, until="S99")
    empty = tmp_path / "empty.geom.toml"
    empty.write_text("", encoding="utf-8")
    report = _geom_run.run_geometry(empty, out=tmp_path / "o")
    assert report["status"] == "refused" and "components" in report["message"]


def test_geom_run_an_override_changes_the_identity_and_a_repeat_keeps_it(tmp_path):
    a = _geom_run.run_geometry(BODY, out=tmp_path / "a", until="S0")
    b = _geom_run.run_geometry(BODY, out=tmp_path / "b", until="S0")
    c = _geom_run.run_geometry(
        BODY, out=tmp_path / "c", until="S0", overrides={"FUS.length_mm": 3000.0}
    )
    assert a["parameter_id"] == b["parameter_id"] != c["parameter_id"]


def test_geom_run_control_sub_steps_are_recorded_not_measured(tmp_path):
    """The control-surface sub-steps of S10 belong to a solver set-up: skipped, never accepted."""
    report = _geom_run.run_geometry(BODY, out=tmp_path, until="S10.F0")
    assert report["status"] == "incomplete" and exit_code(report) == 3
    assert any(item.startswith("S10.F0 ") for item in report["not_run"])
    record = json.loads((tmp_path / "S10.F0.json").read_text(encoding="utf-8"))
    assert {c["note"] for c in record["checks"]} == {_geom_run._CONTROL_NOTE}
    built = _geom_run.built_stages()
    assert {"S0", "S11", "S12", "S13"} <= built and not any(s.startswith("S10.F") for s in built)
    assert not list(tmp_path.glob("*.igs"))


def test_geom_run_default_until_is_the_last_stage():
    import inspect

    default = inspect.signature(_geom_run.run_geometry).parameters["until"].default
    assert default == "S13" == STAGE_ROWS[-1].stage
