"""Collect saved results once and atomically rebuild matrix summaries.

A SUBMITTED point is read and completed from numerical evidence once the files
its job writes are present: the job_outputs recorded at submission (from the
adapter's JOB_OUTPUTS), or every declared output when none were recorded. Outputs only
a local run writes (execution.json) may be absent. PLANNED, RUNNING and
incomplete submitted points remain outstanding.
Collection never launches work or rewrites finished evidence. A summary is a
derived product and can be rebuilt from the registry and adapter outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import SolverAdapter, load_point
from cfdbender.core.plan import workspace_path
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry
from cfdbender.core.settings import read_init
from cfdbender.matrix.exports import write_tables
from cfdbender.matrix.selection import chosen
from cfdbender.solvers import available_solvers, load_solver


@dataclass(frozen=True)
class CollectReport:
    """One sweep's counts and derived summary paths; failures take exit precedence.

    ``tables`` are the CSV tables written beside each summary: campaign_sweep.csv
    and polars/P<POL>.csv per matrix.
    """

    collected: int
    failed: int
    outstanding: int
    summaries: tuple[Path, ...]
    tables: tuple[Path, ...] = ()

    @property
    def exit_code(self) -> int:
        """Return 1 for collection failures, 3 for outstanding work, or 0."""
        return 1 if self.failed else 3 if self.outstanding else 0


def _verify_outputs(root: Path, record: RunRecord) -> None:
    for name, expected in record.outputs_sha256.items():
        path = workspace_path(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RegistryError(f"{record.point_id}: output changed after execution: {name}")


def _verify_submission(root: Path, record: RunRecord) -> None:
    if record.submission is None:
        return
    for key, digest in (("descriptor", "descriptor_sha256"), ("config_path", "config_sha256")):
        path = workspace_path(root, str(record.submission[key]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.submission[digest]:
            raise RegistryError(f"{record.point_id}: submission {key} changed")


def _job_outputs(record: RunRecord) -> tuple[str, ...]:
    if record.submission is not None and "job_outputs" in record.submission:
        return cast(tuple[str, ...], record.submission["job_outputs"])
    return record.point.outputs


def _write_summary(root: Path, matrix: str, solver: str, rows: list[dict[str, object]]) -> Path:
    target = workspace_path(root, f"post/{Path(matrix).stem}/campaign_summary.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            {"schema_version": 1, "matrix": matrix, "solver": solver, "points": rows},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + chr(10)
    ).encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".collect-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _previous_rows(root: Path, matrix: str) -> dict[str, dict[str, object]]:
    path = root / "post" / Path(matrix).stem / "campaign_summary.json"
    try:
        points = json.loads(path.read_bytes())["points"]
        return {str(row["point_id"]): dict(row) for row in points}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def collect_workspace(
    root: str | Path,
    matrix: str | None = None,
    *,
    sims: Sequence[str] = (),
    points: Sequence[str] = (),
) -> CollectReport:
    """Read saved results through the declared adapter and rebuild neutral summaries.

    Parameters
    ----------
    root : str or Path
        Workspace containing init.toml and runs.json.
    matrix : str, optional
        Matrix stem or filename. Omit to collect all recorded matrices.
    sims : sequence of str, optional
        POL ids to collect; empty collects every simulation.
    points : sequence of str, optional
        Point names (DP-name or POL/DP-name) to collect; empty collects every point.
        Points outside the selection are neither read nor counted; the summary
        keeps their rows from the previous summary, or their identity rows.

    Returns
    -------
    CollectReport
        Successful reads, failed points, outstanding points and summary paths.
        Summaries retain identity rows even when a point has no readable result.

    Raises
    ------
    RegistryError
        No matching records, a solver mismatch or an invalid matrix selection.
    OSError
        A summary cannot be written. Existing summaries remain intact.
    """
    target = Path(root).resolve()
    config = read_init(target, available_solvers=available_solvers())
    adapter = cast(SolverAdapter, load_solver(config.solver))
    if (
        matrix is not None
        and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]*", matrix.removesuffix(".fs")) is None
    ):
        raise RegistryError(f"{matrix!r}: name a .fs matrix in inputs/matrices/")
    selected = None if matrix is None else f"inputs/matrices/{matrix.removesuffix('.fs')}.fs"
    registry = RunRegistry(target)
    records = tuple(r for r in registry.read() if selected is None or r.matrix == selected)
    if not records:
        raise RegistryError("no recorded points for this collection")
    if any(r.solver != config.solver for r in records):
        raise RegistryError("init.toml solver differs from the recorded solver")
    keep = chosen((r.point for r in records), sims, points)
    previous = {name: _previous_rows(target, name) for name in {r.matrix for r in records}}
    grouped: dict[str, list[dict[str, object]]] = {}
    collected = failed = outstanding = 0
    for record in records:
        row: dict[str, object] = {
            "point_id": record.point_id,
            "pol": record.point.pol,
            "aircraft": record.point.aircraft,
            "configuration": record.point.configuration,
            "description": record.point.description,
            "condition": dict(record.point.condition),
            "status": record.status,
            "classification": record.classification,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
            "inputs_sha256": dict(record.point.inputs_sha256),
            "outputs_sha256": dict(record.outputs_sha256),
            "error": record.error,
            "result": None,
        }
        if record.submission is not None:
            row["submission"] = dict(record.submission)
        if record.point_id not in keep:
            row = previous[record.matrix].get(record.point_id, row)
            grouped.setdefault(record.matrix, []).append(row)
            continue
        grouped.setdefault(record.matrix, []).append(row)
        if record.status not in ("COMPLETED", "SUBMITTED"):
            outstanding += 1
            continue
        try:
            if record.status == "SUBMITTED" and any(
                not workspace_path(target, name).is_file() for name in _job_outputs(record)
            ):
                outstanding += 1
                continue
            point = load_point(target, record.point, solver=config.solver)
            _verify_submission(target, record)
            measured = record
            if record.status == "SUBMITTED":
                measured = replace(
                    record,
                    outputs_sha256={
                        name: hashlib.sha256(workspace_path(target, name).read_bytes()).hexdigest()
                        for name in record.point.outputs
                        if workspace_path(target, name).is_file()
                    },
                )
            _verify_outputs(target, measured)
            result = adapter.collect(point)
            _verify_outputs(target, measured)
            _verify_submission(target, record)
            if record.status == "SUBMITTED":
                record = registry.transition(
                    record.point_id,
                    "COMPLETED",
                    classification=result.classification,
                    error="FAILED_EXECUTION: numerical evidence failed during collect"
                    if result.classification == "failed"
                    else None,
                )
                row.update(
                    status=record.status,
                    classification=record.classification,
                    completed_at=record.completed_at,
                    outputs_sha256=dict(record.outputs_sha256),
                    error=record.error,
                )
            if result.classification != record.classification:
                raise RegistryError(
                    f"{record.point_id}: collected classification differs from record"
                )
            row["result"] = result.to_dict()
            if result.classification == "failed":
                failed += 1
            else:
                collected += 1
        except (OSError, ValueError) as error:
            row["error"] = str(error)
            failed += 1
    summaries = tuple(
        _write_summary(target, name, config.solver, rows) for name, rows in sorted(grouped.items())
    )
    tables = tuple(
        path for name, rows in sorted(grouped.items()) for path in write_tables(target, name, rows)
    )
    return CollectReport(collected, failed, outstanding, summaries, tables)
