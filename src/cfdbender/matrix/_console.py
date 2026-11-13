"""Console options shared by the cfdbender commands: warnings, verbosity and row output.

``--warning-off`` hides console warnings but keeps every one in ``logs/warnings.log``.
``--verbose`` and ``--log-level quiet|normal|verbose`` pick how much a command says:
quiet prints errors and the final summary line, normal adds progress and the
per-point lines, verbose adds details. ``--json`` and ``--csv`` print rows for a
reader program; a nested cell is JSON in CSV.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from cfdbender.core.settings import WorkspaceError

LEVELS = ("quiet", "normal", "verbose")


def add_warning_option(parser: argparse.ArgumentParser) -> None:
    """Add the console-only warning switch to a program's root parser."""
    parser.add_argument(
        "--warning-off",
        dest="console_warnings",
        action="store_false",
        default=True,
        help="hide console warnings while keeping every one in logs/warnings.log",
    )


def add_console_options(parser: argparse.ArgumentParser) -> None:
    """Add ``--verbose`` and ``--log-level`` to a command."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print details too; the same as --log-level verbose",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=LEVELS,
        default=None,
        help="console verbosity: quiet prints errors and the final summary, normal adds "
        "progress and per-point lines (default), verbose adds details",
    )


def add_runs_option(parser: argparse.ArgumentParser) -> None:
    """Add ``--runs``, the run manifest to read instead of runs.json."""
    parser.add_argument(
        "--runs",
        default="runs.json",
        metavar="NAME",
        help="another run manifest in the workspace root (default: runs.json)",
    )


def add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive ``--json`` and ``--csv`` row outputs."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="print a JSON document of the rows")
    group.add_argument("--csv", action="store_true", help="print CSV; nested cells are JSON")


def level_of(args: argparse.Namespace) -> str:
    """Return the console level the arguments choose."""
    chosen = getattr(args, "log_level", None)
    if getattr(args, "verbose", False):
        if chosen not in (None, "verbose"):
            raise WorkspaceError(f"--verbose contradicts --log-level {chosen}")
        return "verbose"
    return chosen or "normal"


class Console:
    """Print by level: ``say`` at normal and verbose, ``detail`` at verbose, ``final`` always."""

    def __init__(self, level: str = "normal") -> None:
        self.level = level

    def say(self, text: str = "") -> None:
        """Print a progress or per-point line unless the level is quiet."""
        if self.level != "quiet":
            print(text)

    def detail(self, text: str = "") -> None:
        """Print a detail line at verbose only."""
        if self.level == "verbose":
            print(text)

    def final(self, text: str = "") -> None:
        """Print the final summary line at every level."""
        print(text)


@contextmanager
def held_warnings(root: Path | None, *, show: bool) -> Iterator[None]:
    """Hold warnings during a command, print them together at the end, log them always."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            seen = list(dict.fromkeys(str(w.message) for w in caught))
            if seen:
                if root is not None and root.is_dir():
                    try:
                        logs = root / "logs"
                        logs.mkdir(exist_ok=True)
                        stamp = datetime.now(UTC).isoformat()
                        with (logs / "warnings.log").open("a", encoding="utf-8") as stream:
                            stream.writelines(f"{stamp} {text}\n" for text in seen)
                    except OSError:
                        pass
                if show:
                    print(f"Warnings ({len(seen)})", file=sys.stderr)
                    for text in seen:
                        print(f"  [warning] {text}", file=sys.stderr)


def _cell(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def emit_rows(
    rows: Sequence[Mapping[str, object]], columns: Sequence[str], args: argparse.Namespace
) -> None:
    """Print rows as JSON, CSV or an aligned table, by the arguments."""
    if getattr(args, "json", False):
        print(json.dumps(list(rows), indent=2, sort_keys=True))
        return
    if getattr(args, "csv", False):
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(c)) for c in columns])
        print(buffer.getvalue(), end="")
        return
    table = [[_cell(row.get(c)) for c in columns] for row in rows]
    widths = [max([len(c), *(len(r[i]) for r in table)]) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True)).rstrip())
    for line in table:
        print("  ".join(v.ljust(w) for v, w in zip(line, widths, strict=True)).rstrip())
