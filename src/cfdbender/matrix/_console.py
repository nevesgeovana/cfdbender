"""Console options shared by the cfdbender commands: warnings, verbosity and row output.

Every warning is written to ``logs/warnings.log``; ``--warning-off`` only hides it from
the console. A running ``run``, ``collect`` or ``post`` (``plan`` only previews) also writes
the lines it
shows to ``logs/<command>-<stamp>.log`` and its start and end to ``logs/activity.log`` and
``logs/activity.log.jsonl``, which the ``log`` verb reads. Only a folder that is a workspace
gets a ``logs/`` folder.
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
from typing import TextIO

from cfdbender.core.settings import WorkspaceError

LEVELS = ("quiet", "normal", "verbose")
#: One text for the options every verb shares, so no two verbs word them differently.
WORKSPACE_HELP = "campaign workspace (default: the current directory)"
SIMS_HELP = (
    "only these simulations, by POL, space or comma separated "
    "(--sims 1001 1002 or --sims 1001,1002); an id the matrix lacks is refused"
)


def add_warning_option(parser: argparse.ArgumentParser) -> None:
    """Add the console-only warning switch to a program's root parser."""
    parser.add_argument(
        "--warning-off",
        dest="console_warnings",
        action="store_false",
        default=True,
        help="hide console warnings; they are kept in logs/warnings.log",
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


def is_workspace(root: Path) -> bool:
    """Return True when the folder holds a workspace's init.toml, runs.json or inputs/."""
    return any((root / name).exists() for name in ("init.toml", "runs.json", "inputs"))


class LiveLog:
    """The file ``logs/<command>-<stamp>.log``, receiving every line the console shows."""

    def __init__(self, path: Path, stream: TextIO) -> None:
        self.path = path
        self._stream: TextIO | None = stream

    def write(self, text: str) -> None:
        """Append lines; a log that cannot be written is dropped, never an error."""
        if self._stream is None:
            return
        try:
            self._stream.write(text.rstrip("\n") + "\n")
            self._stream.flush()
        except (OSError, ValueError):
            self._stream = None

    def close(self) -> None:
        """Close the file."""
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
            self._stream = None


def open_live_log(root: Path, command: str) -> LiveLog | None:
    """Open ``logs/<command>-<stamp>.log`` in a workspace; None elsewhere or on failure."""
    if not is_workspace(root):
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = root / "logs"
    try:
        folder.mkdir(exist_ok=True)
        for attempt in range(1, 100):
            name = f"{command}-{stamp}.log" if attempt == 1 else f"{command}-{stamp}-{attempt}.log"
            try:
                stream = (folder / name).open("x", encoding="utf-8", newline="\n")
            except FileExistsError:
                continue
            log = LiveLog(folder / name, stream)
            log.write(f"# {command} started {datetime.now(UTC).isoformat()}")
            return log
    except OSError:
        return None
    return None


def record_activity(
    root: Path, stage: str, event: str, message: str = "", **details: object
) -> None:
    """Append one event to ``logs/activity.log.jsonl`` and ``logs/activity.log``.

    Observing a command must not change its result, so a write error is dropped.
    """
    if not is_workspace(root):
        return
    stamp = datetime.now(UTC).isoformat()
    entry = {"timestamp": stamp, "stage": stage, "event": event, "message": message, **details}
    try:
        folder = root / "logs"
        folder.mkdir(exist_ok=True)
        with (folder / "activity.log.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        with (folder / "activity.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{stamp} [{stage}] {event}: {message}\n")
    except OSError:
        pass


def read_activity(root: Path) -> tuple[list[dict[str, object]], int]:
    """Return the readable events of ``logs/activity.log.jsonl`` and how many lines were not."""
    path = root / "logs" / "activity.log.jsonl"
    if not path.is_file():
        return [], 0
    events: list[dict[str, object]] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if isinstance(entry, dict) and isinstance(entry.get("timestamp"), str):
            events.append(entry)
        else:
            skipped += 1
    return events, skipped


class Console:
    """Print by level: ``say`` at normal and verbose, ``detail`` at verbose, ``final`` always.

    Every line a level shows is also written to the live log, when there is one.
    """

    def __init__(self, level: str = "normal", log: LiveLog | None = None) -> None:
        self.level = level
        self.log = log

    def _tee(self, text: str) -> None:
        if self.log is not None:
            self.log.write(text)

    def say(self, text: str = "") -> None:
        """Print a progress or per-point line unless the level is quiet."""
        if self.level != "quiet":
            print(text, flush=True)
            self._tee(text)

    def detail(self, text: str = "") -> None:
        """Print a detail line at verbose only."""
        if self.level == "verbose":
            print(text, flush=True)
            self._tee(text)

    def final(self, text: str = "") -> None:
        """Print the final summary line at every level."""
        print(text, flush=True)
        self._tee(text)

    def error(self, text: str = "") -> None:
        """Print an error line to stderr at every level."""
        print(text, file=sys.stderr)
        self._tee(text)


@contextmanager
def held_warnings(root: Path | None, *, show: bool, log: LiveLog | None = None) -> Iterator[None]:
    """Hold warnings during a command, print them together at the end, log them always.

    ``show`` only decides whether the block reaches the console; every warning goes
    to ``logs/warnings.log`` of a workspace, and the block to the live log.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            seen = list(dict.fromkeys(str(w.message) for w in caught))
            if seen:
                if root is not None and is_workspace(root):
                    try:
                        logs = root / "logs"
                        logs.mkdir(exist_ok=True)
                        stamp = datetime.now(UTC).isoformat()
                        with (logs / "warnings.log").open(
                            "a", encoding="utf-8", newline="\n"
                        ) as stream:
                            stream.writelines(f"{stamp} {text}\n" for text in seen)
                    except OSError:
                        pass
                lines = [f"Warnings ({len(seen)})", *(f"  [warning] {text}" for text in seen)]
                if log is not None:
                    for line in lines:
                        log.write(line)
                if show:
                    for line in lines:
                        print(line, file=sys.stderr)


def _cell(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def status_word(status: object, classification: object, error: object) -> str:
    """Return the word a console shows: a failed point reads FAILED_*, never COMPLETED.

    The records keep COMPLETED with classification failed; only the console changes, as
    pyflightstream's status words say how a point failed.
    """
    if status == "COMPLETED" and classification == "failed":
        head = str(error or "").split(":", 1)[0]
        return head if head.startswith("FAILED") else "FAILED"
    return str(status)


def clock(value: object) -> str:
    """Return an ISO timestamp as local time to the second; any other text is unchanged."""
    text = str(value)
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return text
    if stamp.tzinfo is None:
        return text
    return stamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _shown(row: Mapping[str, object], column: str) -> str:
    value = row.get(column)
    if column == "status" and "classification" in row:
        return status_word(value, row.get("classification"), row.get("error"))
    if column in ("time", "when") and value:
        return clock(value)
    return _cell(value)


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
    table = [[_shown(row, c) for c in columns] for row in rows]
    widths = [max([len(c), *(len(r[i]) for r in table)]) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True)).rstrip())
    for line in table:
        print("  ".join(v.ljust(w) for v, w in zip(line, widths, strict=True)).rstrip())
