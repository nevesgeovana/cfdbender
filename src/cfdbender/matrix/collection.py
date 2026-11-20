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
import stat
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import SolverAdapter, load_point
from cfdbender.core.convergence import SOLVERS as JUDGED
from cfdbender.core.executables import build_mismatch
from cfdbender.core.plan import (
    InputSession,
    PlanError,
    matrix_homes,
    new_post_file_allowed,
    workspace_path,
)
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry
from cfdbender.core.settings import read_init
from cfdbender.matrix._declared import matrix_products
from cfdbender.matrix._journal import verdicts
from cfdbender.matrix._tree_migration import warn_retired
from cfdbender.matrix._unsteady_polar import left_out, plots_absent
from cfdbender.matrix.batch import STATUS_FILE, batch_exit_written, batch_status
from cfdbender.matrix.exports import _write as _write_product
from cfdbender.matrix.exports import write_polar_files
from cfdbender.matrix.polar2d import judge_rows, write_polars
from cfdbender.matrix.products import _check_metadata, write_products
from cfdbender.matrix.selection import chosen
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.workspace.planning import plan_workspace


@dataclass(frozen=True)
class CollectReport:
    """One sweep's counts and derived summary paths; failures take exit precedence.

    ``tables`` are the CSV tables written beside each summary: the polar files and the
    sections tables of each matrix (the campaign sweep table is retired), plus the
    XFOIL-format 2D polars of SU2 and OpenFOAM ALPHA sweeps. ``products`` are
    the products.json and post.log written beside each summary.
    """

    collected: int
    failed: int
    outstanding: int
    summaries: tuple[Path, ...]
    tables: tuple[Path, ...] = ()
    products: tuple[Path, ...] = ()
    strays: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        """Return 1 for collection failures, 3 for outstanding work, or 0."""
        return 1 if self.failed else 3 if self.outstanding else 0


def _verify_outputs(root: Path, record: RunRecord, session: InputSession) -> None:
    files = session.files()
    for name, expected in record.outputs_sha256.items():
        path = workspace_path(root, name)
        if not files.output_agrees(path, record.point.inputs_sha256, expected):
            raise RegistryError(f"{record.point_id}: output changed after execution: {name}")


def _verify_submission(root: Path, record: RunRecord) -> None:
    if record.submission is None:
        return
    for key, digest in (("descriptor", "descriptor_sha256"), ("config_path", "config_sha256")):
        path = workspace_path(root, str(record.submission[key]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.submission[digest]:
            raise RegistryError(f"{record.point_id}: submission {key} changed")


def _ended_reason(root: Path, record: RunRecord, status: int) -> str:
    """Say why a run the batch script ended with ``status`` left no output: its stderr.log line."""
    text = f"the batch script reported exit status {status}"
    try:
        log = (workspace_path(root, record.point.directory) / "stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return text
    first = next((line.strip() for line in log.splitlines() if line.strip()), "")
    return f"{text}: {first}" if first else text


def _job_outputs(record: RunRecord, root: Path | None = None) -> tuple[str, ...]:
    markers = (
        cast(tuple[str, ...], record.submission.get("job_end_files", ()))
        if record.submission is not None
        else ()
    )
    if root is not None:
        completion = getattr(load_solver(record.solver), "job_completion_files", None)
        if completion is not None and (files := completion(root, record.point)) is not None:
            return (*files, *markers)
    if record.submission is not None and "job_outputs" in record.submission:
        return (*cast(tuple[str, ...], record.submission["job_outputs"]), *markers)
    return (*record.point.outputs, *markers)


#: The error of a point sealed by ``collect --ended``: its job ended and wrote no job.exit.
ENDED_ERROR = "FAILED_EXECUTION: job ended without job.exit (declared with --ended)"


def _ended_points(root: Path, records: Sequence[RunRecord], ended: Sequence[str]) -> set[str]:
    """Check the points ``collect --ended`` names, before anything is written.

    Raises
    ------
    RegistryError
        A name no record of the collection carries, a point that is not SUBMITTED, or one
        whose folder holds job.exit (its job ended and said how: a plain collect reads it).
    """
    by_id = {r.point_id: r for r in records}
    found: set[str] = set()
    for name in ended:
        record = by_id.get(name)
        if record is None:
            raise RegistryError(
                f"--ended {name}: no recorded point of this collection has that id "
                "(write it POL/DP-name, as cfd-matrix status --points shows it)"
            )
        if record.status != "SUBMITTED":
            raise RegistryError(
                f"--ended {name}: the point is {record.status}; only a SUBMITTED point whose "
                "job ended without job.exit is declared ended"
            )
        if (workspace_path(root, record.point.directory) / STATUS_FILE).exists():
            raise RegistryError(
                f"--ended {name}: its folder holds {STATUS_FILE}, so its job said how it "
                "ended; collect it without --ended"
            )
        found.add(name)
    return found


def _write_summary(root: Path, matrix: str, solver: str, rows: list[dict[str, object]]) -> Path:
    target = workspace_path(root, f"post/{Path(matrix).stem}/campaign_summary.json")
    data = json.dumps(
        {"schema_version": 1, "matrix": matrix, "solver": solver, "points": rows},
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + chr(10)
    return _write_product(target, data)


def _unsettled(root: Path, records: Sequence[RunRecord], settle: float) -> set[str]:
    """Return the submitted points whose job outputs changed between two observations."""

    def observe() -> dict[str, tuple[object, ...] | None]:
        seen: dict[str, tuple[object, ...] | None] = {}
        for record in records:
            if record.status != "SUBMITTED":
                continue
            states = []
            for name in _job_outputs(record, root):
                path = workspace_path(root, name)
                try:
                    observed = path.stat()
                    state = (
                        (observed.st_size, observed.st_mtime_ns)
                        if stat.S_ISREG(observed.st_mode)
                        else None
                    )
                except OSError:
                    state = None
                states.append(state)
            seen[record.point_id] = None if None in states else tuple(states)
        return seen

    first = observe()
    if first:
        time.sleep(settle)
    second = observe()
    return {
        point_id
        for point_id, state in second.items()
        if state is None or state != first.get(point_id)
    }


def _snapshot(root: Path, grouped: dict[str, list[dict[str, object]]]) -> dict[Path, bytes]:
    found: dict[Path, bytes] = {}
    for name in grouped:
        folder = root / "post" / Path(name).stem
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file() and not path.name.startswith(".collect-"):
                    found[path] = path.read_bytes()
    return found


def _archive_replaced(root: Path, before: dict[Path, bytes]) -> None:
    """Copy each product the collect replaced to archive/<stamp>/, leaving unchanged ones."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = root / "archive" / stamp
    suffix = 1
    while target.exists():
        target = root / "archive" / f"{stamp}-{suffix}"
        suffix += 1
    for path, data in before.items():
        if not path.is_file() or path.read_bytes() != data:
            destination = target / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)


def _previous_rows(root: Path, matrix: str) -> dict[str, dict[str, object]]:
    path = root / "post" / Path(matrix).stem / "campaign_summary.json"
    try:
        points = json.loads(path.read_bytes())["points"]
        return {str(row["point_id"]): dict(row) for row in points}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _strays(
    root: Path, grouped: Mapping[str, Sequence[Mapping[str, object]]]
) -> dict[str, tuple[str, ...]]:
    """Return, per matrix stem, the recorded points its current plan no longer holds.

    A matrix that cannot be planned now (removed or invalid) reports none: the
    comparison needs a plan, and a refusal here belongs to ``plan``.
    """
    found: dict[str, tuple[str, ...]] = {}
    for name, rows in sorted(grouped.items()):
        stem = Path(name).name.removesuffix(".fs")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current = {point.point_id for point in plan_workspace(root, stem).points}
        except (PlanError, OSError, ValueError):
            continue
        gone = tuple(str(row["point_id"]) for row in rows if row["point_id"] not in current)
        if gone:
            found[stem] = gone
    return found


def collect_workspace(
    root: str | Path,
    matrix: str | None = None,
    *,
    sims: Sequence[str] = (),
    points: Sequence[str] = (),
    runs: str = "runs.json",
    archive: bool = False,
    settle: float | None = None,
    pickup: bool = True,
    ended: Sequence[str] = (),
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
    runs : str, optional
        The run manifest in the workspace root to read (default runs.json).
    archive : bool, optional
        Keep a copy of each replaced summary, table or polar under archive/<stamp>/.
    settle : float, optional
        Seconds between the two observations that decide a submitted point's outputs
        are settled (size and modification time stable); None skips the check.
    pickup : bool, optional
        Read a SUBMITTED point whose job outputs are present and seal its record
        COMPLETED. False leaves every submitted point outstanding and changes no
        record; the post stage passes it.
    ended : sequence of str, optional
        Point ids (``POL/DP-name``) of SUBMITTED points whose cluster job ended without
        writing job.exit (killed by its walltime, cancelled): each is sealed COMPLETED with
        a failed classification and :data:`ENDED_ERROR`, its files kept (``collect --ended``).

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
    try:
        homes = None if matrix is None else matrix_homes(matrix)
    except PlanError as error:
        raise RegistryError(str(error)) from error
    registry = RunRegistry(target, runs)
    records = tuple(r for r in registry.read() if homes is None or r.matrix in homes)
    if homes is not None and len({r.matrix for r in records}) > 1:
        raise RegistryError(
            f"matrix {matrix!r} is recorded from both {homes[0]} and {homes[1]}; "
            "name the one to collect with its folder, or keep one file"
        )
    if not records:
        raise RegistryError("no recorded points for this collection")
    if any(r.solver != config.solver for r in records):
        raise RegistryError("init.toml solver differs from the recorded solver")
    sealing = _ended_points(target, records, ended)
    keep = chosen((r.point for r in records), sims, points) | sealing
    previous = {name: _previous_rows(target, name) for name in {r.matrix for r in records}}
    unsettled = _unsettled(target, records, settle) if settle is not None else set()
    marked = verdicts(target, records)
    grouped: dict[str, list[dict[str, object]]] = {}
    collected = failed = outstanding = 0
    failed_ids: set[str] = set()
    post_files: dict[str, dict[str, bytes]] = {}
    session = InputSession(target)
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
        if record.point_id in sealing:
            # Declared ended (--ended): sealed failed, every file of the point kept as it is.
            record = registry.transition(
                record.point_id, "COMPLETED", classification="failed", error=ENDED_ERROR
            )
        if record.status == "COMPLETED" and record.error == ENDED_ERROR:
            row.update(
                status=record.status,
                classification=record.classification,
                completed_at=record.completed_at,
                error=record.error,
            )
            failed += 1
            failed_ids.add(record.point_id)
            continue
        if (
            record.status not in ("COMPLETED", "SUBMITTED")
            or record.point_id in unsettled
            or (record.status == "SUBMITTED" and not pickup)
        ):
            outstanding += 1
            continue
        try:
            if record.status == "SUBMITTED" and any(
                not workspace_path(target, name).is_file() for name in _job_outputs(record, target)
            ):
                exit_status = batch_exit_written(target, record)
                if exit_status is None or exit_status == 0:
                    outstanding += 1
                    continue
                # The job script ended this run with a failure and left no solver output (a
                # run it did not start, or one that died): it is failed, never outstanding.
                record = registry.transition(
                    record.point_id,
                    "COMPLETED",
                    classification="failed",
                    error="FAILED_EXECUTION: " + _ended_reason(target, record, exit_status),
                )
                row.update(
                    status=record.status,
                    classification=record.classification,
                    completed_at=record.completed_at,
                    outputs_sha256=dict(record.outputs_sha256),
                    error=record.error,
                )
                failed += 1
                failed_ids.add(record.point_id)
                continue
            point = load_point(target, record.point, solver=config.solver, session=session)
            _verify_submission(target, record)
            measured = record
            if record.status == "SUBMITTED":
                measured = replace(
                    record,
                    outputs_sha256={
                        name: session.files().digest_output(
                            workspace_path(target, name), record.point.inputs_sha256
                        )
                        for name in record.point.outputs
                        if workspace_path(target, name).is_file()
                    },
                )
            _verify_outputs(target, measured, session)
            result = adapter.collect(point)
            mismatch = build_mismatch(record.point.solver_build, result.solver_version)
            reported = batch_status(target, record)
            if reported:
                mismatch = mismatch or (
                    f"the batch script reported exit status {reported} for this run"
                )
            if mismatch:
                result = replace(result, classification="failed")
            _verify_outputs(target, measured, session)
            _verify_submission(target, record)
            if record.status == "SUBMITTED":
                record = registry.transition(
                    record.point_id,
                    "COMPLETED",
                    classification=result.classification,
                    solver_version=result.solver_version,
                    error="FAILED_EXECUTION: "
                    + (mismatch or "numerical evidence failed during collect")
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
            # mark-converged is a verdict on a completed point: the record carries it and
            # the result keeps the class the numerical evidence gives.
            # mark-failed is the same kind of verdict: the row keeps the mark's reason.
            judged = record.classification in ("converged", "failed") and (
                marked.get(record.point_id) == record.classification
            )
            if result.classification != record.classification and not judged:
                raise RegistryError(
                    f"{record.point_id}: collected classification differs from record"
                )
            row["result"] = result.to_dict()
            if new_post_file_allowed():
                # the digest and the copy are those of the bytes the point was parsed from
                name = f"inputs/pproc/{record.point.post_id}.toml"
                files = session.files()
                data = files.read_bytes(name)  # the first-read bytes the point parsed
                files.track(name)
                digest = files.digests()[name]
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"{name} changed while it was read")
                row["inputs_sha256"] = {**cast(dict[str, str], row["inputs_sha256"]), name: digest}
                post_files.setdefault(record.matrix, {})[digest] = data
            if record.classification == "failed":
                failed += 1
                failed_ids.add(record.point_id)
            else:
                collected += 1
        except (OSError, ValueError) as error:
            row["error"] = str(error)
            failed += 1
            failed_ids.add(record.point_id)
    before = _snapshot(target, grouped) if archive else {}
    summaries: list[Path] = []
    tables: list[Path] = []
    products: list[Path] = []
    for name, rows in sorted(grouped.items()):
        _check_metadata(workspace_path(target, f"post/{Path(name).stem}"), name, config.solver)
        for digest, data in post_files.get(name, {}).items():
            kept = workspace_path(target, f"post/{Path(name).stem}/pproc/{digest}.toml")
            if not kept.is_file() or kept.read_bytes() != data:
                kept.parent.mkdir(parents=True, exist_ok=True)
                temporary = kept.with_name(".collect-" + kept.name)
                temporary.write_bytes(data)
                os.replace(temporary, kept)
        summary = _write_summary(target, name, config.solver, rows)
        migrated = warn_retired(workspace_path(target, f"post/{Path(name).stem}"))
        by_rule = judge_rows(target, rows, config.solver) if config.solver in JUDGED else None
        polar = write_polar_files(target, name, rows, config.solver, by_rule)
        written = polar.paths
        if by_rule is not None:
            written += write_polars(target, name, rows, config.solver, by_rule)
        declared, details, declared_skips = matrix_products(
            target,
            name,
            rows,
            {r.point_id: r.point for r in records if r.matrix == name and r.point_id in keep},
            adapter,
            config.solver,
            session,
        )
        written += declared
        summaries.append(summary)
        tables.extend(written)
        products.extend(
            write_products(
                target,
                name,
                config.solver,
                rows,
                summary,
                written,
                polar,
                migrated,
                details=details,
                declared_skips=(
                    *declared_skips,
                    *left_out(rows, Path(name).stem),
                    *plots_absent(rows, Path(name).stem),
                ),
            )
        )
    if archive:
        _archive_replaced(target, before)
    strays = _strays(target, grouped)
    # A point the matrix lost is reported as a stray, not counted among the failures.
    failed -= len(failed_ids & {point for gone in strays.values() for point in gone})
    return CollectReport(
        collected,
        failed,
        outstanding,
        tuple(summaries),
        tuple(tables),
        tuple(products),
        strays,
    )
