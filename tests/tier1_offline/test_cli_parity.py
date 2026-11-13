"""Every command and option cfdbender shares with pyflightstream is accepted by its parsers.

The list in tests/data/cli_parity_common.json holds each command and option of the
pyflightstream matrix and workspace programs that is not specific to one solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from cfdbender.matrix.cli import build_parser as matrix_parser
from cfdbender.workspace.cli import build_parser as workspace_parser

ROWS = json.loads(
    (Path(__file__).parents[1] / "data/cli_parity_common.json").read_text(encoding="utf8")
)


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _find(command: str) -> argparse.ArgumentParser:
    program, _, name = command.partition(" ")
    root = {"cfd-matrix": matrix_parser()[0], "cfd-workspace": workspace_parser()}[program]
    if not name:
        return root
    subs = _subparsers(root)
    assert name in subs, f"{command}: not a command of {program}"
    return subs[name]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['command']} {r['option']}".strip())
def test_every_common_command_and_option_is_accepted(row):
    parser = _find(row["command"])
    option = row["option"]
    if not option:
        return
    if option.startswith("--"):
        assert option in parser._option_string_actions, f"{row['command']} lacks {option}"
    else:
        name = option.strip("<>")
        assert any(a.dest == name and not a.option_strings for a in parser._actions), (
            f"{row['command']} lacks the argument {name}"
        )


def test_the_common_list_is_complete_for_the_commands_it_names():
    assert len(ROWS) > 100
    assert {r["command"].split(" ")[0] for r in ROWS} == {"cfd-matrix", "cfd-workspace"}
