"""docs/cli.md lists every command and option the two argument parsers accept."""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

from cfdbender.matrix.cli import build_parser as matrix_parser
from cfdbender.workspace.cli import build_parser as workspace_parser

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "cli.md"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "build_cli_page", ROOT / "tools/build_cli_page.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _commands() -> dict[str, argparse.ArgumentParser]:
    found: dict[str, argparse.ArgumentParser] = {}
    for prog, parser in (("cfd-workspace", workspace_parser()), ("cfd-matrix", matrix_parser()[0])):
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    found[f"{prog} {name}"] = sub
    return found


def _sections(page: str) -> dict[str, str]:
    """Split the page into the text under each `### \\`program command\\`` heading."""
    parts = re.split(r"^### `([^`]+)`$", page, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _missing(page: str) -> list[str]:
    """Name every command, positional argument and option the page does not carry."""
    sections = _sections(page)
    absent: list[str] = []
    for command, parser in _commands().items():
        if command not in sections:
            absent.append(command)
            continue
        section = sections[command].split("\n## ")[0]
        cells = set(re.findall(r"^\| `([^`]+)` \|", section, flags=re.M))
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            if action.option_strings:
                names = list(action.option_strings)
            elif action.metavar is None and action.choices is not None:
                names = [str(choice) for choice in action.choices]
            else:
                names = [str(action.metavar or action.dest.upper())]
            for name in names:
                if not any(re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", c) for c in cells):
                    absent.append(f"{command} {name}")
    return absent


def test_cli_page_names_every_command_and_option():
    assert _missing(PAGE.read_text(encoding="utf-8")) == []


def test_cli_page_check_detects_a_dropped_option_and_a_dropped_command():
    page = PAGE.read_text(encoding="utf-8")
    row = next(line for line in page.splitlines() if line.startswith("| `--force-rerun-all`"))
    assert _missing(page.replace(row + "\n", "")) == ["cfd-matrix run --force-rerun-all"]
    assert "cfd-matrix sync" in _missing(page.replace("### `cfd-matrix sync`", "### `sync`"))
    positional = next(line for line in page.splitlines() if line.startswith("| `RECIPE`"))
    assert _missing(page.replace(positional + "\n", "")) == ["cfd-matrix free-space RECIPE"]


def test_cli_page_is_what_the_parsers_give():
    tool = _tool()
    page = PAGE.read_text(encoding="utf-8")
    assert tool.render(page) == page, "run python tools/build_cli_page.py"


def test_cli_page_stage_table_holds_every_command():
    tool = _tool()
    staged = {f"{prog} {name}" for _, entries in tool.STAGES for prog, name in entries}
    assert staged == set(_commands())
