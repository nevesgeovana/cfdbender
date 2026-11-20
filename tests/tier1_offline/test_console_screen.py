"""The console of cfd-matrix looks like pyflightstream's: opening block, banner, point lines, table.

Chatter, blocks and the signature go to stderr; the result of a command stays on stdout.
"""

from __future__ import annotations

import re

import pytest
from test_matrix_columns_all_solvers import build

import cfdbender
from cfdbender.matrix.cli import main
from cfdbender.matrix.receipt import write_receipt
from cfdbender.workspace.planning import plan_workspace

BANNER_WINGS = ("--(+)-----(_)-----(+)--", "--(+)--(+)--(_)--(+)--(+)--")


@pytest.fixture
def root(tmp_path):
    made = build(tmp_path, "mock")
    matrix = made / "inputs/matrices/full.fs"
    matrix.write_text(matrix.read_text(encoding="ascii").replace("| 2h | 1.0 |", "| 2h | - |"))
    write_receipt(plan_workspace(made, "full"))
    return made


def _run(root, capsys, *options):
    code = main(["run", "full", "--workspace", str(root), "--local", *options])
    streams = capsys.readouterr()
    return code, streams.out, streams.err


def test_every_matrix_command_opens_with_its_title_purpose_and_workspace(root, capsys):
    for argv, live in (
        (["plan", "full"], False),
        (["run", "full", "--local"], True),
        (["collect", "full"], True),
        (["post", "full"], True),
        (["status"], False),
    ):
        main([*argv, "--workspace", str(root)])
        err = capsys.readouterr().err
        lines = err.splitlines()
        assert lines[0] == f"cfd-matrix {argv[0]}", argv
        assert lines[1].startswith("  purpose: "), argv
        at = lines.index(f"  workspace: {root.resolve()}")
        if live:
            assert re.fullmatch(
                rf"  live log: logs/{argv[0]}-\d{{8}}T\d{{6}}Z\.log", lines[at + 1]
            ), argv
            assert lines[at + 2] == ""
        else:
            assert lines[at + 1] == "", argv


def test_quiet_leaves_out_the_opening_block_and_the_chatter(root, capsys):
    code, out, err = _run(root, capsys, "--log-level", "quiet")
    assert code == 0
    assert "purpose:" not in err and "[run]" not in err and "-> " not in err
    assert "geoversegoddess sees you" in err


def test_run_says_its_stage_banner_points_table_and_rows_on_stderr(root, capsys):
    code, out, err = _run(root, capsys)
    assert code == 0
    lines = err.splitlines()
    started = lines.index(f"[run] started: {root.resolve()}")
    banner = next(i for i, line in enumerate(lines) if "point(s) to run" in line)
    assert started < banner
    assert lines[banner - 2] == "             _______"
    assert lines[banner - 1] == "                |"
    wing, _, text = lines[banner].partition("   cfdbender ")
    assert wing.strip() in BANNER_WINGS
    assert text == f"{cfdbender.__version__}: campaign full, 2 point(s) to run"
    start = next(i for i, line in enumerate(lines) if line.startswith("  -> "))
    match = re.fullmatch(r"  -> (\S+)  \[steady\]  building and running  \(1 of 2\)", lines[start])
    assert match, lines[start]
    point = match.group(1)
    assert lines[start + 1] == f"     {point}  COMPLETED"
    assert lines[start + 2].startswith("  -> ") and "(2 of 2)" in lines[start + 2]
    assert lines[start + 3].endswith("  COMPLETED")
    table = lines[start + 4 : start + 10]
    assert table[0] == "  +-----------+--------+"
    assert table[1] == "  | status    | points |"
    assert table[3] == "  | COMPLETED |      2 |"
    assert re.fullmatch(r"  2 point\(s\) in \d+:\d\d:\d\d", table[5])
    assert lines[start + 10] == "row 7001: all 2 executed"
    assert lines[start + 11] == "[run] finished; outcomes={'COMPLETED': 2}"
    # The per-point chatter is not a result: standard output carries none of it.
    assert "building and running" not in out and "point(s) in" not in out


def test_the_live_log_receives_the_same_lines(root, capsys):
    _run(root, capsys)
    (log,) = (root / "logs").glob("run-*.log")
    text = log.read_text(encoding="utf-8")
    for needle in ("cfd-matrix run", "[run] started:", "building and running", "[run] finished"):
        assert needle in text


def test_no_banner_and_no_table_when_nothing_will_run(root, capsys):
    assert _run(root, capsys)[0] == 0
    code, out, err = _run(root, capsys, "--resume")
    assert code == 0
    assert "point(s) to run" not in err
    assert "building and running" not in err
    assert "[run] finished" in err


def test_a_failed_point_reads_its_status_word_and_the_run_is_failed(root, capsys):
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text(encoding="ascii").replace("converged", "failed"))
    write_receipt(plan_workspace(root, "full"))
    code, out, err = _run(root, capsys)
    assert code == 2
    assert re.search(r"^     \S+  FAILED\w*", err, re.M)
    assert re.search(r"\[run\] failed; outcomes=\{'FAILED\w*': 2\}", err)


def test_collect_post_and_plan_results_stay_on_stdout(root, capsys):
    assert _run(root, capsys)[0] == 0
    main(["collect", "full", "--workspace", str(root)])
    streams = capsys.readouterr()
    assert re.search(r"^collected \d+, failed \d+, outstanding \d+$", streams.out, re.M)
    main(["post", "full", "--workspace", str(root)])
    streams = capsys.readouterr()
    assert re.search(r"^post read \d+, failed \d+, outstanding \d+$", streams.out, re.M)
    main(["plan", "full", "--workspace", str(root)])
    streams = capsys.readouterr()
    assert "matrix: " in streams.out and "solver: mock" in streams.out
    assert "point(s) planned" in streams.out
