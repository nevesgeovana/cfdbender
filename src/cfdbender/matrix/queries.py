"""Read-only verbs over runs.json: status, show, log, trace, history and diff.

They read the run manifest and the files beside it and change nothing. A sim is a
POL; a run is a recorded point, named by its point id (``POL/DP-name``), and an
archived or deleted earlier record of it is named ``POL/DP-name@<archive folder>``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from cfdbender.core.plan import shown_digest
from cfdbender.core.refusals import warn_or_refuse
from cfdbender.core.registry import (
    RegistryError,
    RunRecord,
    RunRegistry,
    decode_record,
    encode_record,
)
from cfdbender.matrix._console import (
    SIMS_HELP,
    WORKSPACE_HELP,
    add_output_options,
    add_runs_option,
    clock,
    emit_rows,
    level_of,
    read_activity,
    status_word,
)
from cfdbender.matrix._journal import MARKS, STORAGE, read_journal
from cfdbender.matrix._tree_migration import refuse_retired
from cfdbender.matrix.selection import split_ids


@dataclass(frozen=True)
class Run:
    """One recorded run: the current record of a point, or an archived or deleted one."""

    run_id: str
    record: RunRecord
    kind: str
    when: str | None
    archive: str | None

    @property
    def point_id(self) -> str:
        """Return the point id the run belongs to."""
        return self.record.point_id


def matrix_stem(matrix: str) -> str:
    """Return the stem of a recorded matrix path or a matrix name."""
    return Path(matrix).name.removesuffix(".fs")


def load_runs(registry: RunRegistry) -> list[Run]:
    """Return every run of the manifest: current records first, then history, oldest first."""
    runs = [Run(r.point_id, r, "current", r.created_at, None) for r in registry.read()]
    for entry in registry.history():
        raw = entry.get("record")
        record = decode_record(raw, registry.root)
        if "archive" in entry:
            folder = str(entry["archive"])
            runs.append(
                Run(
                    f"{record.point_id}@{Path(folder).name}",
                    record,
                    # An earlier segment of a point continued in place (run --continue).
                    "continued" if entry.get("continued") is True else "archived",
                    str(entry.get("archived_at")),
                    folder,
                )
            )
        else:
            when = str(entry.get("deleted_at"))
            runs.append(Run(f"{record.point_id}@deleted-{when}", record, "deleted", when, None))
    return runs


def find_runs(runs: list[Run], target: str, point: str | None = None) -> list[Run]:
    """Return the runs a target names: a sim (POL), a point id or a run id.

    Raises
    ------
    RegistryError
        When the target or the ``--point`` name matches no run.
    """
    if "/" in target or "@" in target:
        found = [r for r in runs if target in (r.run_id, r.point_id)]
    else:
        found = [r for r in runs if r.record.point.pol == target]
    if not found:
        raise RegistryError(f"{target!r}: no recorded sim, point or run carries that name")
    if point is not None:
        found = [r for r in found if point in (r.point_id, r.point_id.split("/", 1)[1], r.run_id)]
        if not found:
            raise RegistryError(f"--point {point}: no run of {target!r} carries that name")
    return found


def _words(record: RunRecord) -> set[str]:
    words = {record.status}
    if record.classification:
        words.add(record.classification.upper())
        if record.classification == "failed":
            words.add("FAILED")
    return words


def _wanted(record: RunRecord, patterns: list[str]) -> bool:
    return not patterns or any(
        fnmatch.fnmatchcase(word, pattern.upper())
        for word in _words(record)
        for pattern in patterns
    )


def _open(args: argparse.Namespace) -> RunRegistry:
    return RunRegistry(Path(args.workspace).resolve(), args.runs)


def _say_empty(rows: list[Any], records: list[RunRecord], args: argparse.Namespace) -> None:
    """Say why a table of the console is empty, so a header alone never stands for an answer."""
    if rows or getattr(args, "json", False) or getattr(args, "csv", False):
        return
    if not records:
        print("no recorded sim yet: run a matrix first")
    elif args.failed:
        print("no failed point")
    else:
        print("no recorded point matches the selection")


def run_status(args: argparse.Namespace) -> int:
    """List each sim of each matrix with its states, or each point with ``--points``."""
    registry = _open(args)
    records = list(registry.read())
    if args.matrix:
        stem = matrix_stem(args.matrix)
        records = [r for r in records if matrix_stem(r.matrix) == stem]
    sims = split_ids(args.sims)
    if sims:
        records = [r for r in records if r.point.pol in sims]
    if args.per_point:
        rows = [
            {
                "sim": r.point.pol,
                "matrix": matrix_stem(r.matrix),
                "point": r.point_id,
                "status": r.status,
                "classification": r.classification,
                "error": r.error,
            }
            for r in records
            if _wanted(r, args.statuses) and (not args.failed or r.classification == "failed")
        ]
        emit_rows(rows, ["sim", "matrix", "point", "status", "classification", "error"], args)
        _say_empty(rows, records, args)
        return 0
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.point.pol, matrix_stem(record.matrix)), []).append(record)
    rows2: list[dict[str, object]] = []
    for (pol, stem), members in sorted(groups.items()):
        if args.statuses and not any(_wanted(m, args.statuses) for m in members):
            continue
        failed = sum(m.classification == "failed" for m in members)
        if args.failed and not failed:
            continue
        states: dict[str, int] = {}
        for member in members:
            states[member.status] = states.get(member.status, 0) + 1
        rows2.append(
            {
                "sim": pol,
                "matrix": stem,
                "points": len(members),
                "status": ", ".join(f"{k}={v}" for k, v in sorted(states.items())),
                "failed": failed,
            }
        )
    emit_rows(rows2, ["sim", "matrix", "points", "status", "failed"], args)
    _say_empty(rows2, records, args)
    return 0


def _outcome(root: Path, run: Run) -> dict[str, Any]:
    """Return the collected result of a current run from its matrix summary, or {}."""
    if run.kind != "current":
        return {}
    summary = root / "post" / matrix_stem(run.record.matrix) / "campaign_summary.json"
    try:
        points = json.loads(summary.read_text(encoding="utf-8")).get("points", [])
    except (OSError, ValueError):
        return {}
    for point in points:
        if isinstance(point, dict) and point.get("point_id") == run.point_id:
            result = point.get("result")
            return result if isinstance(result, dict) else {}
    return {}


def _shown_condition(record: RunRecord) -> str:
    return ", ".join(f"{key}={value:g}" for key, value in record.point.condition.items())


def _card(root: Path, run: Run) -> list[str]:
    """Return a run's card: outcome first, then where it ran and what it measured."""
    record = run.record
    word = status_word(record.status, record.classification, record.error)
    head = f"{run.point_id}  {word}"
    if record.classification:
        head += f" {record.classification}"
    if run.kind != "current":
        head += f"  ({run.kind})"
    lines = [head, f"  matrix {matrix_stem(record.matrix)}  solver {record.solver}"]
    if record.solver_version:
        lines[-1] += f" {record.solver_version}"
    if record.point.condition:
        lines.append(f"  condition {_shown_condition(record)}")
    result = _outcome(root, run)
    history = result.get("history") or []
    if history:
        lines.append(
            f"  iterations {history[-1]['iteration']}  residual {history[-1]['residual']:.3g}"
        )
    coefficients = result.get("coefficients") or {}
    if coefficients:
        shown = ", ".join(f"{name}={item['value']:.6g}" for name, item in coefficients.items())
        lines.append(f"  coefficients {shown}")
    lines.append(f"  folder {record.point.directory}")
    if record.started_at and record.completed_at:
        lines.append(f"  ran {clock(record.started_at)} to {clock(record.completed_at)}")
    if record.error:
        lines.append(f"  error: {record.error}")
    return lines


def run_show(args: argparse.Namespace) -> int:
    """Print an outcome-first card for each matching run; the full record under --verbose."""
    registry = _open(args)
    runs = find_runs(load_runs(registry), args.target, args.point)
    rows = [{"run": r.run_id, "kind": r.kind, **encode_record(r.record)} for r in runs]
    if args.json or args.csv:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        emit_rows(rows, columns, args)
        return 0
    if level_of(args) == "verbose":
        from cfdbender.matrix._console import _cell

        for row in rows:
            print(f"run {row['run']} ({row['kind']})")
            for key, value in row.items():
                if key not in ("run", "kind") and value not in (None, "", [], {}):
                    print(f"  {key}: {_cell(value)}")
        return 0
    if len(runs) > 1 and "/" not in args.target and "@" not in args.target:
        print(f"sim {args.target}: {len(runs)} runs")
        for run in runs:
            word = status_word(run.record.status, run.record.classification, run.record.error)
            print(f"  {run.run_id}  {word} {run.record.classification or ''}".rstrip())
        print("name one run for its card, or add --verbose for the full records")
        return 0
    for run in runs:
        print("\n".join(_card(registry.root, run)))
    return 0


def run_history(args: argparse.Namespace) -> int:
    """Print the current and earlier records of a sim or point, with their archives."""
    runs = find_runs(load_runs(_open(args)), args.target)
    rows = [
        {
            "run": r.run_id,
            "kind": r.kind,
            "when": r.when,
            "status": r.record.status,
            "classification": r.record.classification,
            "archive": r.archive,
        }
        for r in runs
    ]
    emit_rows(rows, ["run", "kind", "when", "status", "classification", "archive"], args)
    return 0


def _flat(prefix: str, value: object, out: dict[str, object]) -> None:
    if isinstance(value, Mapping):
        for key, inner in value.items():
            _flat(f"{prefix}.{key}" if prefix else str(key), inner, out)
    else:
        out[prefix] = value


def run_diff(args: argparse.Namespace) -> int:
    """Compare two runs field by field: their inputs, outputs and results."""
    runs = load_runs(_open(args))
    pair = []
    for name in (args.run_a, args.run_b):
        match = [r for r in runs if name == r.run_id]
        if not match:
            raise RegistryError(f"{name!r}: no run carries that id (see history)")
        pair.append(match[0])
    flat: list[dict[str, object]] = []
    for run in pair:
        values: dict[str, object] = {}
        _flat("", encode_record(run.record), values)
        flat.append(values)
    rows = [
        {"field": key, "a": shown_digest(flat[0].get(key)), "b": shown_digest(flat[1].get(key))}
        for key in sorted(set(flat[0]) | set(flat[1]))
        if flat[0].get(key) != flat[1].get(key)
    ]
    if not (args.json or args.csv) and level_of(args) != "verbose":
        # Hashes and output paths differ whenever two points do; they stay under --verbose.
        noise = ("inputs_sha256.", "outputs_sha256.")
        hidden = [r for r in rows if str(r["field"]).startswith(noise) or r["field"] == "outputs"]
        rows = [r for r in rows if r not in hidden]
        if hidden:
            rows.append(
                {"field": "hashes, output paths", "a": f"{len(hidden)} differ", "b": "--verbose"}
            )
    if args.json or args.csv:
        emit_rows(rows, ["field", "a", "b"], args)
    elif not rows:
        print(f"{args.run_a} and {args.run_b} agree in every field")
    else:
        emit_rows(rows, ["field", "a", "b"], args)
    return 0 if not rows else 1


def run_trace(args: argparse.Namespace) -> int:
    """Follow a product file to the run, inputs and hashes that made it.

    A retired product (the campaign sweep table) is refused by name.
    """
    refuse_retired(args.product)
    registry = _open(args)
    root = registry.root
    runs = load_runs(registry)
    if args.run:
        match = [r for r in runs if args.run == r.run_id]
        if not match:
            raise RegistryError(f"--run {args.run}: no run carries that id")
        chosen = match
        product = args.product
    else:
        product = Path(args.product).as_posix()
        chosen = [r for r in runs if r.kind == "current" and product in r.record.point.outputs]
        if not chosen:
            parts = product.split("/")
            if len(parts) >= 3 and parts[0] == "post":
                chosen = [
                    r
                    for r in runs
                    if r.kind == "current" and matrix_stem(r.record.matrix) == parts[1]
                ]
        if not chosen:
            raise RegistryError(f"{product}: no recorded run made or feeds this product")
    rows: list[dict[str, object]] = []
    for run in chosen:
        record = run.record
        rows.append(
            {
                "run": run.run_id,
                "matrix": record.matrix,
                "status": record.status,
                "classification": record.classification,
                "inputs_sha256": {
                    name: shown_digest(value) for name, value in record.point.inputs_sha256.items()
                },
                "outputs_sha256": {
                    name: shown_digest(value) for name, value in record.outputs_sha256.items()
                },
                "product_sha256": shown_digest(dict(record.outputs_sha256).get(product)),
            }
        )
    target = root / product
    if not args.json and not args.csv:
        print(f"product {product}" + ("" if target.is_file() else " (file not on disk)"))
    if not args.json and not args.csv and level_of(args) != "verbose":
        # The hash map of every input is several hundred columns wide; it stays under --verbose.
        rows = [
            {**row, "inputs_sha256": f"{len(cast(dict[str, str], row['inputs_sha256']))} files"}
            for row in rows
        ]
    emit_rows(
        rows,
        ["run", "matrix", "status", "classification", "inputs_sha256", "product_sha256"],
        args,
    )
    return 0


_DURATION = re.compile(r"(\d+)([smhd])")


def _since(text: str) -> datetime:
    match = _DURATION.fullmatch(text)
    if match:
        unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[match.group(2)]
        return datetime.now(UTC) - timedelta(**{unit: int(match.group(1))})
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError as error:
        raise RegistryError(
            f"--since {text}: give an ISO time or a duration such as 2h, 7d"
        ) from error
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def _events(root: Path, runs: list[Run]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for run in runs:
        record = run.record
        base = {"run": run.run_id, "sim": record.point.pol}
        events.append({**base, "time": record.created_at, "stage": "run", "event": "registered"})
        if record.started_at:
            events.append({**base, "time": record.started_at, "stage": "run", "event": "started"})
        if record.completed_at:
            failed = record.classification == "failed" or bool(record.error)
            events.append(
                {
                    **base,
                    "time": record.completed_at,
                    "stage": "finish",
                    "event": f"finished {record.classification}",
                    "detail": record.error,
                    "problem": failed,
                }
            )
        if run.kind != "current":
            events.append(
                {
                    **base,
                    "time": run.when,
                    "stage": {"archived": "archive", "deleted": "delete", "continued": "continue"}[
                        run.kind
                    ],
                    "event": run.kind,
                    "detail": run.archive,
                }
            )
    for entry in read_journal(root, MARKS):
        point = str(entry.get("point_id"))
        events.append(
            {
                "run": point,
                "sim": point.split("/", 1)[0],
                "time": entry.get("at"),
                "stage": "mark",
                "event": f"marked {entry.get('verdict')}",
                "detail": entry.get("reason"),
                "problem": entry.get("verdict") == "failed",
            }
        )
    # The commands' own start and end, from logs/activity.log.jsonl (stage "command").
    activity, skipped = read_activity(root)
    for entry in activity:
        events.append(
            {
                "run": None,
                "sim": None,
                "time": entry["timestamp"],
                "stage": "command",
                "event": f"{entry.get('stage')} {entry.get('event')}",
                "detail": entry.get("message") or None,
                "problem": entry.get("event") == "failed",
            }
        )
    if skipped:
        warn_or_refuse(
            f"{skipped} line(s) of logs/activity.log.jsonl could not be read and are left out; "
            "the other events are shown",
            UserWarning,
            stacklevel=2,
        )
    return events


def run_log(args: argparse.Namespace) -> int:
    """List the events recorded for runs, or the storage calls or a matrix's post log."""
    registry = _open(args)
    root = registry.root
    if args.storage:
        events = [
            {
                "run": None,
                "sim": None,
                "time": e.get("at"),
                "stage": e.get("command"),
                "event": e.get("event", "call"),
                "detail": e.get("detail"),
            }
            for e in read_journal(root, STORAGE)
        ]
    elif args.post:
        stem = matrix_stem(args.post)
        path = root / "post" / stem / "post.log"
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        # The post log carries no stamps of its own; its rows take the time it was last written.
        written = (
            datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            if path.is_file()
            else None
        )
        events = [
            {
                "run": None,
                "sim": None,
                "time": written,
                "stage": "post",
                "event": line,
                "problem": "warning" in line.lower() or "refus" in line.lower(),
            }
            for line in lines
        ]
    else:
        events = _events(root, load_runs(registry))
    sims = split_ids(args.sims)
    if sims:
        events = [e for e in events if e["sim"] in sims]
    if args.run:
        events = [e for e in events if e["run"] == args.run or str(e["run"]).startswith(args.run)]
    if args.stage:
        events = [e for e in events if e["stage"] == args.stage]
    if args.since:
        limit = _since(args.since)
        events = [
            e for e in events if e["time"] and datetime.fromisoformat(str(e["time"])) >= limit
        ]
    if args.problems:
        events = [e for e in events if e.get("problem")]
    if args.open_only:
        finished = {e["run"] for e in events if e["stage"] == "finish"}
        events = [e for e in events if e["stage"] == "run" and e["run"] not in finished]
    events.sort(key=lambda e: str(e["time"] or ""))
    emit_rows(events, ["time", "run", "stage", "event", "detail"], args)
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".", help=WORKSPACE_HELP)
    add_runs_option(parser)
    add_output_options(parser)


def register(commands: Any) -> dict[str, Any]:
    """Add the matrix-level read verb (status) to the cfd-matrix subparsers."""
    status = commands.add_parser(
        "status",
        help="list every recorded sim per matrix with its states, filterable",
        allow_abbrev=False,
    )
    status.add_argument("--sims", nargs="+", metavar="POL", help=SIMS_HELP)
    status.add_argument("--matrix", help="only this recorded matrix")
    status.add_argument(
        "--status",
        dest="statuses",
        action="append",
        default=[],
        metavar="WORD",
        help="only sims with a point in this status or classification (repeatable; a "
        "trailing * matches every word it begins)",
    )
    status.add_argument("--failed", action="store_true", help="only sims with a failed point")
    status.add_argument("--points", dest="per_point", action="store_true", help="one row per point")
    _common(status)
    return {"status": run_status}


def register_workspace(commands: Any) -> dict[str, Any]:
    """Add the read-only verbs on what the workspace holds to the cfd-workspace subparsers."""
    show = commands.add_parser(
        "show", help="print one sim, point or run record in full", allow_abbrev=False
    )
    show.add_argument("target", help="a sim (POL), a point id (POL/DP-name) or a run id")
    show.add_argument("--point", help="only the run of this point name within the target")
    history = commands.add_parser(
        "history",
        help="print the history of a sim or point: reruns, archives and deletions",
        allow_abbrev=False,
    )
    history.add_argument("target", help="a sim (POL) or a point id (POL/DP-name)")
    log = commands.add_parser(
        "log",
        help="list the recorded events of runs, storage calls or a post log",
        allow_abbrev=False,
    )
    source = log.add_mutually_exclusive_group()
    source.add_argument("--post", metavar="MATRIX", help="the saved post log of a matrix")
    source.add_argument("--storage", action="store_true", help="list the recorded storage calls")
    log.add_argument("--sims", nargs="+", metavar="POL", help=SIMS_HELP)
    log.add_argument("--run", help="only events naming this run id")
    log.add_argument("--stage", help="only this stage (storage: command)")
    log.add_argument("--since", help="events since an ISO time or duration, such as 2h or 7d")
    log.add_argument("--problems", action="store_true", help="only problems and failed outcomes")
    log.add_argument(
        "--open",
        dest="open_only",
        action="store_true",
        help="runs started without a finish in the records; never asks a scheduler",
    )
    trace = commands.add_parser(
        "trace",
        help="follow a product file to the run, inputs and hashes that made it",
        allow_abbrev=False,
    )
    trace.add_argument("product", help="a product path in the workspace, such as post/<matrix>/...")
    trace.add_argument("--run", help="trace this run id instead of looking the product up")
    diff = commands.add_parser(
        "diff", help="compare two runs field by field; exit 1 when they differ", allow_abbrev=False
    )
    diff.add_argument("run_a", help="a run id (see history)")
    diff.add_argument("run_b", help="a run id (see history)")
    for parser in (show, history, log, trace, diff):
        _common(parser)
    return {
        "show": run_show,
        "history": run_history,
        "log": run_log,
        "trace": run_trace,
        "diff": run_diff,
    }
