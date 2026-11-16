"""The ``cfd-geom`` verbs, driven through the real entry point.

A planted failure is a stage function replaced in the table; the exit code, the printed report
and the files on disk are read back. A misspelt key, a swapped input, an empty file and a
repeated override are refused at the verb the user calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfdbender.geometry import pipeline as _geom_run
from cfdbender.geometry._records import check_entry
from cfdbender.geometry.cli import main
from cfdbender.geometry.pipeline import StageOutcome

BODY = '[components.FUS]\ntype = "body"\nlength_mm = 2000.0\n'


@pytest.mark.parametrize("verb", ["validate", "run"])
def test_fr1_missing_config_explains_correction(tmp_path, capsys, verb):
    """Reproduce: cfd-geom validate missing.geom.toml and cfd-geom run it."""
    missing = tmp_path / "missing.geom.toml"
    code, _, error = geom(["--workspace", tmp_path, verb, missing], capsys)
    assert code == 2
    assert "provide an existing geometry configuration file" in error
    assert missing.name in error


def test_fr2_refused_run_artifacts_are_documented(tmp_path, capsys):
    """Reproduce: cfd-geom run missing.geom.toml --out output."""
    output = tmp_path / "output"
    code, _, _ = geom(["run", tmp_path / "missing.geom.toml", "--out", output], capsys)
    assert code == 2
    assert (output / "report.json").is_file()
    assert (output / "geom-run.log").is_file()
    page = (Path(__file__).resolve().parents[2] / "docs/cli-geometry-mesh.md").read_text(
        encoding="utf-8"
    )
    assert "a refused `cfd-geom run` can write" in page


def geom(args, capsys):
    capsys.readouterr()
    try:
        code = main([str(a) for a in args])
    except SystemExit as error:
        code = error.code
    out = capsys.readouterr()
    return code, out.out, out.err


def printed(stdout):
    return json.JSONDecoder().raw_decode(stdout[stdout.index("{") :])[0]


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / "demo.geom.toml"
    path.write_text(BODY, encoding="utf-8")
    return path


def _plant(monkeypatch, stage="S4", when_length=None):
    def run(context):
        length = context.resolved["components"]["FUS"]["length"]["value"]
        bad = when_length is None or length == when_length
        check = check_entry(
            "law.monotonic", "failed" if bad else "passed", 2.0 if bad else 0.0, 1.0
        )
        return StageOutcome(checks=[check])

    monkeypatch.setitem(_geom_run.STAGE_FUNCTIONS, stage, run)


def test_geom_cli_run_stops_at_a_planted_failure_and_names_the_stage_and_check(
    demo, tmp_path, capsys, monkeypatch
):
    _plant(monkeypatch)
    code, out, _ = geom(["run", demo, "--out", tmp_path / "o", "--until", "S9"], capsys)
    report = printed(out)
    assert code == 1
    assert (report["stage"], report["check"]) == ("S4", "law.monotonic")
    assert [s["stage"] for s in report["stages"]][-1] == "S4"
    assert not (tmp_path / "o" / "S5.json").exists()


def test_geom_cli_run_overrides_a_value_and_changes_the_identity(demo, tmp_path, capsys):
    base = printed(geom(["run", demo, "--out", tmp_path / "a", "--until", "S0"], capsys)[1])
    moved = geom(
        ["run", demo, "--out", tmp_path / "b", "--until", "S0", "--set", "FUS.length_mm=3000.0"],
        capsys,
    )
    assert moved[0] == 0 and printed(moved[1])["parameter_id"] != base["parameter_id"]


@pytest.mark.parametrize(
    ("extra", "text"),
    [
        (["--set", "FUS.lenght_mm=3000.0"], "unknown parameter"),
        (["--set", "FUS.length_mm"], "component.key=value"),
        (["--set", "FUS.length_mm=3000", "--set", "FUS.length_mm=4000"], "given twice"),
        (["--set", "WING.length_mm=3000"], "declared component"),
        (["--until", "S99"], "S99"),
        (["--variant", "nope"], "variant 'nope'"),
        (["--configuration", "nope"], "configuration 'nope'"),
    ],
)
def test_geom_cli_run_refuses_bad_input_with_code_2_and_a_message(
    demo, tmp_path, capsys, extra, text
):
    code, _, err = geom(["run", demo, "--out", tmp_path / "o", *extra], capsys)
    assert code == 2 and text in err
    assert not (tmp_path / "o" / "S0.json").exists()


def test_geom_cli_run_refuses_an_empty_file_and_a_missing_one(tmp_path, capsys):
    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    assert geom(["run", empty, "--out", tmp_path / "o"], capsys)[0] == 2
    assert geom(["run", tmp_path / "none.toml", "--out", tmp_path / "o"], capsys)[0] == 2


def test_geom_cli_doe_continues_after_a_failed_set_and_names_each_set(
    demo, tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("CFDBENDER_GMSH_ISOLATE", "0")
    _plant(monkeypatch, when_length=3.0)
    sets = tmp_path / "sets.toml"
    sets.write_text(
        "[[sets]]\nname = 'low'\n'FUS.length_mm' = 2500.0\n"
        "[[sets]]\nname = 'mid'\n'FUS.length_mm' = 3000.0\n"
        "[[sets]]\n'FUS.length_mm' = 3500.0\n",
        encoding="utf-8",
    )
    code, out, _ = geom(["doe", demo, sets, "--out", tmp_path / "d", "--until", "S9"], capsys)
    result = printed(out)
    assert code == 1 and result["accepted"] is False
    rows = result["sets"]
    assert [r["name"] for r in rows] == ["low", "mid", "set-0003"]
    assert [r["report"]["status"] for r in rows] == ["incomplete", "failed", "incomplete"]
    assert (rows[1]["report"]["stage"], rows[1]["report"]["check"]) == ("S4", "law.monotonic")


def test_geom_cli_doe_refuses_a_misspelt_table_and_an_empty_file(demo, tmp_path, capsys):
    for body in ("[[set]]\n'FUS.length_mm' = 3000.0\n", "", "sets = []\n"):
        sets = tmp_path / "sets.toml"
        sets.write_text(body, encoding="utf-8")
        code, _, err = geom(["doe", demo, sets, "--out", tmp_path / "d"], capsys)
        assert code == 2 and "[[sets]]" in err
    swapped = geom(["doe", tmp_path / "sets.toml", demo, "--out", tmp_path / "d"], capsys)
    assert swapped[0] == 2


def test_geom_cli_doe_reports_a_refused_set_and_still_runs_the_next(demo, tmp_path, capsys):
    sets = tmp_path / "sets.toml"
    sets.write_text(
        "[[sets]]\n'FUS.nope' = 1\n[[sets]]\n'FUS.length_mm' = 3500.0\n", encoding="utf-8"
    )
    code, out, _ = geom(["doe", demo, sets, "--out", tmp_path / "d", "--until", "S2"], capsys)
    statuses = [r["report"]["status"] for r in printed(out)["sets"]]
    assert code == 2 and statuses == ["refused", "incomplete"]


def test_geom_cli_001_run_through_unbuilt_stages_exits_3_and_doe_ranks_failed_over_it(
    demo, tmp_path, capsys, monkeypatch
):
    code, out, _ = geom(["run", demo, "--out", tmp_path / "o", "--until", "S3"], capsys)
    report = printed(out)
    assert code == 3 and report["status"] == "incomplete"
    assert "S2 identity.listed" in report["not_run"]
    monkeypatch.setenv("CFDBENDER_GMSH_ISOLATE", "0")
    _plant(monkeypatch, when_length=3.0)
    sets = tmp_path / "sets.toml"
    sets.write_text(
        "[[sets]]\n'FUS.length_mm' = 2500.0\n[[sets]]\n'FUS.length_mm' = 3000.0\n", "utf-8"
    )
    code, out, _ = geom(["doe", demo, sets, "--out", tmp_path / "d", "--until", "S9"], capsys)
    assert code == 1 and printed(out)["accepted"] is False
    sets.write_text("[[sets]]\n'FUS.length_mm' = 2500.0\n", "utf-8")
    code, out, _ = geom(["doe", demo, sets, "--out", tmp_path / "e", "--until", "S9"], capsys)
    assert code == 3 and printed(out)["accepted"] is False


def test_geom_cli_004_doe_reads_an_unquoted_dotted_key_as_its_override(demo, tmp_path, capsys):
    """``FUS.length_mm = 3000.0`` unquoted is a nested table in TOML; it is the same override."""
    sets = tmp_path / "sets.toml"
    sets.write_text(
        "[[sets]]\nFUS.length_mm = 3000.0\n[[sets]]\n'FUS.length_mm' = 3000.0\n", "utf-8"
    )
    code, out, _ = geom(["doe", demo, sets, "--out", tmp_path / "d", "--until", "S0"], capsys)
    reports = [r["report"] for r in printed(out)["sets"]]
    assert code == 0 and [r["status"] for r in reports] == ["accepted", "accepted"]
    assert reports[0]["parameter_id"] == reports[1]["parameter_id"]
    sets.write_text("[[sets]]\nFUS.length_mm = 3000.0\n'FUS.length_mm' = 3500.0\n", "utf-8")
    code, _, err = geom(["doe", demo, sets, "--out", tmp_path / "e", "--until", "S0"], capsys)
    assert code == 2 and "FUS.length_mm given twice" in err


def test_geom_cli_validate_prints_every_value_with_its_source(demo, tmp_path, capsys):
    code, out, _ = geom(["validate", demo], capsys)
    values = {v["path"]: v for v in printed(out)["values"]}
    assert code == 0
    assert values["components.FUS.length"]["value"] == 2.0
    assert values["components.FUS.length"]["source"] == "given"
    bad = tmp_path / "bad.toml"
    bad.write_text(BODY.replace("2000.0", '"long"'), encoding="utf-8")
    assert geom(["validate", bad], capsys)[0] == 2


def test_geom_cli_configurations_and_stages_list_what_they_hold(demo, capsys):
    code, out, _ = geom(["configurations", demo], capsys)
    assert code == 0 and printed(out)["components"] == ["FUS"]
    code, out, _ = geom(["stages"], capsys)
    rows = printed(out)["stages"]
    steps = [f"S8.{piece}{n}" for piece in "BF" for n in (1, 2, 3)]  # G6R sub-steps of S8
    junctions = [  # G7R sub-steps of S10, in the order they run
        "S10.D1", "S10.C5", "S10.C6", "S10.C8", "S10.C9", "S10.D2", "S10.C7", "S10.D3",
        "S10.D4", "S10.E1", "S10.D5", "S10.D6", "S10.D8", "S10.D8b", "S10.C1", "S10.C2",
        "S10.C3", "S10.C4", "S10.D9", "S10.D10", "S10.E2",
    ]  # fmt: skip
    controls = [f"S10.F{i}" for i in range(8)]
    fowler = [f"S10.FW{i}" for i in range(1, 5)]
    propellers = {"S1.P", "S4.P", "S5.P", "S6.P", "S8.P", "S11.PU", "S11.PC", "S11.PS"}
    exports = ["S11.V", "S11.PU", "S11.P", "S11.PC", "S11.R", "S11.S", "S11.PS"]
    reader = ["S12.C", "S12.G", "S12", "S13.R", "S13"]
    assert [r["stage"] for r in rows] == (
        [
            "S0",
            "S1",
            "S1.P",
            "S2",
            "S3",
            "S4",
            "S4.P",
            "S5",
            "S5.P",
            "S6",
            "S6.P",
            "S7",
            "S8.P",
            "S8",
        ]
        + steps
        + ["S9", "S10"]
        + junctions
        + fowler
        + controls
        + ["S11"]
        + exports
        + reader
    )
    built = {r["stage"] for r in rows if r["built"]}
    assert built == {
        "S0",
        "S8",
        "S10",
        "S11",
        *propellers,
        *exports,
        *reader,
        *junctions,
        *steps,
    }
    # the Fowler and control-surface sub-steps belong to a solver set-up: listed, not built
    assert not built & {*fowler, *controls}
    assert all(r["checks"] for r in rows)


def test_geom_cli_an_unknown_verb_is_refused_and_the_moved_verbs_are_listed(capsys):
    assert geom(["generate", "x"], capsys)[0] == 2
    _, _, err = geom(["--help"], capsys)
    code, out, _ = geom(["--help"], capsys)
    assert code == 0
    for verb in (
        "run",
        "doe",
        "validate",
        "schema",
        "configurations",
        "stages",
        "identify",
        "check",
        "iges-convert",
    ):
        assert verb in out
    assert "split-symmetry" not in out and "add-control-surface" not in out


def test_geom_cli_python_dash_m_runs_the_geometry_command_line():
    """`python -m cfdbender.geometry.cli` reaches `main` and returns its exit code."""
    import os
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "cfdbender.geometry.cli", "validate", "no-such-file.toml"],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ),
    )
    assert done.returncode == 2, done.stderr
    assert "no-such-file.toml" in done.stderr
