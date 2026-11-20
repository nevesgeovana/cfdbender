"""Continue recorded time-accurate points in place with a grown run: ``cfd-matrix run --continue``.

The user raises the run length of a finished (or killed) time-accurate row, ``REVOLUTIONS`` or
``TIME_ITERATIONS``, plans the matrix and runs it with ``--continue``. Each selected point then
continues from its latest restart files instead of starting again: its record names the solver
iteration it restarts at and every segment of the run, the earlier record joins the history of
``runs.json`` (``continued`` true) and the files the new run writes again (configuration,
history, logs, job files) move to ``continued/<n>/`` in the point folder first, so nothing is
overwritten.

The option is explicit because a recorded point refuses a second run unless an option says
what to do with it (``--resume``, ``--force-rerun``, ``--restart``): a grown row that silently
continued would make ``--resume``, which skips recorded points, change behaviour. Refused,
changing nothing: a point with no record or not ``COMPLETED``, a point that is not
time-accurate, a row whose other keys changed (only the run length may grow; the scheduler's
``WALLTIME``, ``NCPUS`` and ``hpc.`` overrides may change too), a run length that fell, and
whatever the adapter refuses (no restart files, nothing left to march).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import InputSession, Point, RunPlan, workspace_path
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry, encode_record
from cfdbender.core.run_types import TIME_ACCURATE

#: The row variables that set the length of a time-accurate run, the one thing that may grow.
RUN_LENGTH_KEYS = ("REVOLUTIONS", "TIME_ITERATIONS")
#: The folder of a point that keeps an earlier run's evidence, ``continued/<n>``.
CONTINUED = "continued"
#: Point fields that may differ between the recorded point and the continued one: the run
#: length's consequences, the scheduler's resources, and what the run itself fills in.
_FREE_FIELDS = frozenset(
    {
        "variables",
        "inputs_sha256",
        "outputs",
        "actions",
        "walltime",
        "ncpus",
        "restart_sha256",
        "restart_directory",
        "continuation",
        "origins",
    }
)


def _scheduler_key(key: str) -> bool:
    return key.startswith("hpc.") or key == "cluster"


def _number(value: object) -> float | None:
    try:
        return float(str(value))
    except ValueError:
        return None


def changed_keys(recorded: Point, planned: Point, matrix: str) -> list[str]:
    """Name what differs between a recorded point and its re-planned row, run length aside."""
    changed = [
        field.name
        for field in dataclasses.fields(Point)
        if field.name not in _FREE_FIELDS
        and getattr(recorded, field.name) != getattr(planned, field.name)
    ]
    inputs = {k: v for k, v in recorded.inputs_sha256.items() if k != matrix}
    if inputs != {k: v for k, v in planned.inputs_sha256.items() if k != matrix}:
        changed += sorted(
            name
            for name in inputs.keys() | planned.inputs_sha256.keys()
            if name != matrix and inputs.get(name) != planned.inputs_sha256.get(name)
        )
    for key in sorted(recorded.variables.keys() | planned.variables.keys()):
        if key in RUN_LENGTH_KEYS or _scheduler_key(key):
            continue
        if recorded.variables.get(key) != planned.variables.get(key):
            changed.append(key)
    return changed


def _earlier_segments(recorded: Point, first_step: int, number: int) -> list[dict[str, object]]:
    """Return the segments run before, the latest closed at the continuation's first step."""
    before = cast(list[Mapping[str, object]], recorded.continuation.get("segments", []))
    segments = [dict(segment) for segment in before] or [{"first_step": 1, "folder": "."}]
    segments[-1]["last_step"] = first_step - 1
    segments[-1]["folder"] = f"{CONTINUED}/{number}"
    # A run that kept no step (a continuation that marched nothing) is no segment.
    return [s for s in segments if cast(int, s["last_step"]) >= cast(int, s["first_step"])]


def prepare(plan: RunPlan, adapter: object) -> tuple[RunPlan, dict[str, RunRecord]]:
    """Return the plan with each point continued, and the records they replace; write nothing.

    Raises
    ------
    RegistryError
        A point the continuation refuses, named with the reason (see the module docstring).
    """
    resolve = getattr(adapter, "continuation", None)
    if resolve is None:
        raise RegistryError(f"solver {plan.solver!r} cannot continue a point (--continue)")
    records = {r.point_id: r for r in RunRegistry(plan.root).read()}
    session = InputSession(plan.root)
    points: list[Point] = []
    earlier: dict[str, RunRecord] = {}
    for point in plan.points:
        who = f"--continue {point.point_id}"
        record = records.get(point.point_id)
        if record is None:
            raise RegistryError(f"{who}: no record to continue; run the point first")
        if record.status != "COMPLETED":
            raise RegistryError(
                f"{who}: the point is {record.status}; only a completed point continues "
                "(collect a cluster job first: a job killed by its walltime is collected failed)"
            )
        if record.solver != plan.solver or record.matrix != plan.matrix:
            raise RegistryError(f"{who}: the record belongs to another matrix or solver")
        if point.run_type not in TIME_ACCURATE:
            raise RegistryError(
                f"{who}: only a time-accurate point (WORKFLOW unsteady or unsteady_rotor) "
                f"continues with more steps; this row is {point.run_type}"
            )
        if changed := changed_keys(record.point, point, plan.matrix):
            raise RegistryError(
                f"{who}: {', '.join(changed)} changed since the point ran. A continuation "
                f"continues the same run with more {' or '.join(RUN_LENGTH_KEYS)} and nothing "
                f"else; put the row back, or run the point again with --force-rerun "
                f"{point.point_id}"
            )
        for key in RUN_LENGTH_KEYS:
            old, new = _number(record.point.variables.get(key)), _number(point.variables.get(key))
            if old is not None and new is not None and new < old:
                raise RegistryError(
                    f"{who}: {key} fell from {old:g} to {new:g}; a continuation only adds steps"
                )
        context = load_point(plan.root, point, solver=plan.solver, session=session)
        try:
            found = cast(Mapping[str, object], resolve(context))
        except (OSError, ValueError) as failure:
            raise RegistryError(f"{who}: {failure}") from failure
        number = len(cast(list[object], record.point.continuation.get("segments", []))) or 1
        first, last = cast(int, found["first_step"]), cast(int, found["last_step"])
        moved = list(cast(list[str], found["evidence"]))
        descriptor = (record.submission or {}).get("descriptor")
        if isinstance(descriptor, str) and descriptor.startswith(point.directory + "/"):
            name = descriptor.removeprefix(point.directory + "/")
            if "/" not in name and workspace_path(plan.root, descriptor).is_file():
                moved.append(name)
        continued = replace(
            point,
            outputs=tuple(cast(list[str], found["outputs"])),
            restart_sha256=record.point.restart_sha256,
            restart_directory=record.point.restart_directory,
            continuation={
                "restart_iter": found["restart_iter"],
                "kept": f"{CONTINUED}/{number}",
                "moved": list(dict.fromkeys(moved)),
                "segments": [
                    *_earlier_segments(record.point, first, number),
                    {"first_step": first, "last_step": last, "folder": "."},
                ],
            },
        )
        if workspace_path(plan.root, f"{point.directory}/{CONTINUED}/{number}").exists():
            raise RegistryError(
                f"{who}: {point.directory}/{CONTINUED}/{number} exists already; it holds an "
                "earlier run's evidence and is never overwritten"
            )
        points.append(continued)
        earlier[point.point_id] = record
    return replace(plan, points=tuple(points)), earlier


def begin(plan: RunPlan, earlier: Mapping[str, RunRecord]) -> Path:
    """Register the continued points and move each earlier run's evidence aside.

    ``runs.json`` is copied to ``archive/<stamp>/`` first; each earlier record joins the
    history with ``continued`` true, and the continued point is registered PLANNED in its
    place. Then the files the new run writes again move to ``continued/<n>/``.
    """
    registry = RunRegistry(plan.root)
    with registry.transaction():
        folder = registry.archive_manifest()
        now = datetime.now(UTC).isoformat()
        entries = tuple(
            {
                "archived_at": now,
                "archive": folder.relative_to(plan.root).as_posix(),
                "record": encode_record(earlier[point.point_id]),
                "continued": True,
            }
            for point in plan.points
        )
        changes = {
            point.point_id: RunRecord(point, plan.matrix, plan.solver, "PLANNED", now)
            for point in plan.points
        }
        registry.replace_records(changes, history_entries=entries, archive_first=False)
        for point in plan.points:
            base = workspace_path(plan.root, point.directory)
            kept = base / str(point.continuation["kept"])
            kept.mkdir(parents=True, exist_ok=False)
            for name in cast(list[str], point.continuation["moved"]):
                source = base / name
                if source.is_file():
                    os.replace(source, kept / name)
    return folder
