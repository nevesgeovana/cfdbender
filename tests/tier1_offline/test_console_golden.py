"""Golden console transcripts of plan, run, collect and post on a synthetic workspace.

Each file under ``tests/fixtures/console/`` is the exit code, standard output and standard
error of one command, the workspace path, the version, the clock times and the log stamps
replaced by tokens and the random choices seeded. They pin the format: the opening block, the
stage lines, the banner, the point lines, the status table, the row lines, the signature box
and the split of the streams.

To rewrite them after a deliberate format change, run with ``CFDBENDER_UPDATE_GOLDEN=1`` and
review the diff.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

import pytest
from test_matrix_columns_all_solvers import build

import cfdbender
from cfdbender import _cli
from cfdbender.matrix.cli import main

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "console"
COMMANDS = ("plan", "run", "collect", "post", "status", "status-points")
#: The arguments of each command after its name; a status takes no matrix.
ARGUMENTS = {
    "plan": ["full"],
    "run": ["full", "--local"],
    "collect": ["full"],
    "post": ["full"],
    "status": [],
    "status-points": ["--points"],
}


def normal(text: str, root: Path) -> str:
    """Replace what differs between machines and runs with tokens."""
    for form in (str(root.resolve()), root.resolve().as_posix()):
        text = text.replace(form, "<ROOT>")
    text = text.replace(cfdbender.__version__, "<V>")
    text = re.sub(r"written \d{4}-\d\d-\d\dT[\d:]+\+00:00", "written <TIME>", text)
    text = re.sub(r"(?m)(?<=  )[\d.]+(?= +mock-1  local$)", "<WALL>", text)
    text = re.sub(r"\b\d+:\d\d:\d\d\b", "<CLOCK>", text)
    text = re.sub(
        r"(logs[/\\](?:run|collect|post|plan))-\d{8}T\d{6}Z(-\d+)?\.log", r"\1-<STAMP>.log", text
    )
    # The width of the WALL column follows the time measured, so the table's spacing does too.
    text = "\n".join(
        re.sub(r" {2,}", "  ", line) if "<WALL>" in line or (line.startswith("SIM  ") and "WALL" in line) else line
        for line in text.split("\n")
    )
    return text.replace("\r\n", "\n")


@pytest.fixture(scope="module")
def transcripts(tmp_path_factory):
    import sys
    from io import StringIO

    root = build(tmp_path_factory.mktemp("golden"), "mock")
    matrix = root / "inputs/matrices/full.fs"
    matrix.write_text(matrix.read_text(encoding="ascii").replace("| 2h | 1.0 |", "| 2h | - |"))
    for moved in (root / "inputs/geometries").iterdir():
        moved.rename(root / "inputs/mesh" / moved.name)
    (root / "inputs/geometries").rmdir()
    found = {}
    patch = pytest.MonkeyPatch()
    for command in COMMANDS:
        argv = [command.split("-")[0], *ARGUMENTS[command], "--workspace", str(root)]
        patch.setattr(_cli, "_RNG", random.Random(20261120))
        out, err = StringIO(), StringIO()
        real = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = main(argv)
        finally:
            sys.stdout, sys.stderr = real
        found[command] = normal(
            f"exit {code}\n--- stdout\n{out.getvalue()}--- stderr\n{err.getvalue()}", root
        )
    patch.undo()
    return found


@pytest.mark.parametrize("command", COMMANDS)
def test_the_console_transcript_of_each_command_is_pinned(transcripts, command):
    path = GOLDEN / f"{command}.console"
    if os.environ.get("CFDBENDER_UPDATE_GOLDEN"):
        GOLDEN.mkdir(parents=True, exist_ok=True)
        path.write_text(transcripts[command], encoding="utf-8", newline="\n")
    assert transcripts[command] == path.read_text(encoding="utf-8")
