"""``cfd-matrix run --batch N``: rows grouped into scheduler jobs with a generated shell script.

The flag, its meaning and its split are those of pyflightstream (``run --batch N``): N is the
number of jobs, the rows (POL, with their points) are cut into CONTIGUOUS runs in matrix order
by the optimal partition (``cfdbender.hpc._batch_split``), a row is never cut, and a batch
lives in ``sims/batch/<matrix>_b<ID>``. What pyflightstream does with a solver script,
this module does with a generated shell script: one block per run, in matrix order, each in
its own point folder, with no warm start and no file passed from one run to the next.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import SolverAdapter, load_point
from cfdbender.core.matrix import HPC_OVERRIDE_PREFIX
from cfdbender.core.plan import InputSession, Point, RunPlan, workspace_path
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry
from cfdbender.core.restart import PENDING_FILE, check_restart
from cfdbender.hpc import HpcProfile, SubmissionPlan, plan_submission
from cfdbender.hpc._batch_split import RowUnit, split_rows
from cfdbender.hpc._overrides import ORIGIN, apply_overrides, is_script_path, row_overrides
from cfdbender.hpc._plan import _clock
from cfdbender.hpc._submit import submit
from cfdbender.solvers import load_solver

from .submission import (
    _uses_cluster,
    job_outputs,
    point_values,
    staged_config,
    warn_low_ram,
)

#: The folder of the batches, under the workspace root (pyflightstream: ``sims/batch``).
BATCH_DIR = "sims/batch"
#: The file each run of a batch writes in its own folder: the exit status of its command.
STATUS_FILE = "job.exit"
#: How a script the row names as its application is started on the job's machine.
INTERPRETERS = {".py": "python3", ".sh": "sh", ".bash": "bash"}
_BATCH_FOLDER = re.compile(r"^(?P<matrix>.+)_b(?P<id>\d+)$")


def batch_count(text: str) -> int:
    """Parse the N of ``--batch N``: a whole number of at least 1 (pyflightstream's text)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"batch (CLI: --batch) takes a whole number, not {text!r}"
        ) from None
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"batch (CLI: --batch) takes a whole number of at least 1, not {value}"
        )
    return value


@dataclass(frozen=True)
class BatchPlan:
    """One batch: the rows it holds, its script and the descriptor that submits it."""

    batch_id: int
    label: str
    folder: str
    script: str
    sims: tuple[str, ...]
    points: tuple[Point, ...]
    script_text: str
    submission: SubmissionPlan
    profile: HpcProfile
    overrides: Mapping[str, Mapping[str, object]]
    requested: int


def next_batch_id(root: Path, matrix_stem: str) -> int:
    """Return one more than the largest existing batch number of the matrix, else 1."""
    largest = 0
    base = root / BATCH_DIR
    for folder in base.iterdir() if base.is_dir() else ():
        found = _BATCH_FOLDER.match(folder.name)
        if folder.is_dir() and found is not None and found["matrix"] == matrix_stem:
            largest = max(largest, int(found["id"]))
    return largest + 1


def _relative(target: Path, start: Path) -> str:
    return os.path.relpath(target, start).replace(os.sep, "/")


def _command(
    plan: RunPlan,
    adapter: object,
    context: object,
    point: Point,
    overrides: Mapping[str, object],
    folder: Path,
    config_name: str,
    resolved_application: str,
) -> list[str]:
    """Return one run's command line: the row's own, as the single job would run it.

    A point that runs in stages (the adapter's ``job_wrapper``, a script of its folder) runs
    the row's command under it, without the configuration file: the script appends the file of
    each stage, so a wrapper reads it as its last argument, as it does today.
    """
    if "application_id" in overrides:
        raw = str(overrides["application_id"]).strip()
        if is_script_path(raw):
            relative = _relative(Path(resolved_application), folder)
            interpreter = INTERPRETERS.get(Path(raw).suffix.lower())
            command = [*([interpreter] if interpreter else []), relative, config_name]
        else:
            command = [raw, config_name]
        wrapper = getattr(adapter, "job_wrapper", None)
        prefix = list(cast(list[str], wrapper(context))) if wrapper is not None else []
        return [*prefix, *command[:-1]] if prefix else command
    hook = getattr(adapter, "job_command", None)
    if hook is None:
        raise ValueError(
            f"row {point.pol}: solver {plan.solver!r} names no command line a batch script can "
            f"run; state {HPC_OVERRIDE_PREFIX}application_id in the row"
        )
    return list(cast(list[str], hook(context)))


#: Exit status a dependent run records when the RESTART source of its job did not end with 0.
SOURCE_FAILED = 97
#: Exit status a dependent run records when the source's restart file could not be staged.
STAGING_FAILED = 98
#: The line of a dependent block that submission replaces with the staging of its solution.
STAGE_MARK = "@@STAGE {point_id}@@"


def _staging(point: Point, relative: str, pairs: Mapping[str, str]) -> str:
    """Return the shell that copies each source file to its staged name, or fails with 98.

    ``pairs`` maps each staged name to the source file, relative to the script's folder. A
    source file that is missing or empty, or a copy that fails, records 98 and one line naming
    the file in the dependent's stderr.log; nothing is staged after the first failure. When
    ``sha256sum`` exists on the node the digests of the staged files go to
    ``restart-staged.sha256``; nothing reads that file.
    """
    quoted = shlex.quote(relative)
    lines = []
    for name, source in pairs.items():
        message = shlex.quote(
            f"RESTART {point.restart_from}: the source wrote no restart file "
            f"{posixpath.basename(source)}; the dependent point was not run"
        )
        lines += [
            '  if [ "$rc" -eq 0 ]; then',
            f"    if [ -s {shlex.quote(source)} ] && cp {shlex.quote(source)} "
            f"{shlex.quote(posixpath.join(relative, name))}; then :; else",
            f"      printf '%s\\n' {message} > {quoted}/stderr.log",
            f"      rc={STAGING_FAILED}",
            "    fi",
            "  fi",
        ]
    names = " ".join(shlex.quote(name) for name in pairs)
    lines += [
        '  if [ "$rc" -eq 0 ] && command -v sha256sum > /dev/null 2>&1; then',
        f"    ( cd {quoted} && sha256sum {names} > restart-staged.sha256 ) 2> /dev/null || :",
        "  fi",
    ]
    return chr(10).join(lines)


def _stage_pairs(point_folder: Path, relative: str) -> dict[str, str]:
    """Read the staged names and source paths recorded for a dependent, script-relative."""
    document = json.loads((point_folder / PENDING_FILE).read_text(encoding="ascii"))
    return {
        str(name): posixpath.normpath(posixpath.join(relative, str(path)))
        for name, path in dict(document["stage"]).items()
    }


def _dependent_block(
    point: Point, relative: str, command: Sequence[str], source_relative: str
) -> str:
    """Return the block of a run whose RESTART source ran earlier in the same script.

    The source's exit status is read from its job.exit; only when it is 0 is the solution
    staged and the run started. The staging is plain POSIX shell: submission replaces the mark
    line with :func:`_staging`. Otherwise the run is not started, its job.exit records 97
    (source failed) or 98 (staging failed) and one line in its stderr.log says why; the script
    goes on to the next run.
    """
    quoted = shlex.quote(relative)
    ended = shlex.quote(f"{source_relative}/{STATUS_FILE}")
    status = shlex.quote(f"{relative}/{STATUS_FILE}")
    temporary = shlex.quote(f"{relative}/{STATUS_FILE}.tmp")
    line = " ".join(shlex.quote(part) for part in command)
    head = shlex.quote(f"RESTART {point.restart_from}: the source ended with status ")
    tail = shlex.quote("; this run was not started")
    return chr(10).join(
        [
            f"# {point.point_id}: RESTART from {point.restart_from}, staged when it ends",
            f"src=$(cat {ended} 2>/dev/null || echo missing)",
            "rc=0",
            'if [ "$src" != 0 ]; then',
            f"  printf '%s%s%s\\n' {head} \"$src\" {tail} > {quoted}/stderr.log",
            f"  rc={SOURCE_FAILED}",
            "else",
            STAGE_MARK.format(point_id=point.point_id),
            '  if [ "$rc" -eq 0 ]; then',
            f"    ( cd {quoted} && {line} > stdout.log 2> stderr.log )",
            "    rc=$?",
            "  fi",
            "fi",
            f"printf '%s\\n' \"$rc\" > {temporary} && mv {temporary} {status}",
            '[ "$rc" -eq 0 ] || failed=1',
        ]
    )


def _block(
    point: Point, relative: str, command: Sequence[str], source_relative: str | None = None
) -> str:
    if source_relative is not None:
        return _dependent_block(point, relative, command, source_relative)
    quoted = shlex.quote(relative)
    status = shlex.quote(f"{relative}/{STATUS_FILE}")
    temporary = shlex.quote(f"{relative}/{STATUS_FILE}.tmp")
    line = " ".join(shlex.quote(part) for part in command)
    return chr(10).join(
        [
            f"# {point.point_id}",
            f"( cd {quoted} && {line} > stdout.log 2> stderr.log )",
            "rc=$?",
            f"printf '%s\\n' \"$rc\" > {temporary} && mv {temporary} {status}",
            '[ "$rc" -eq 0 ] || failed=1',
        ]
    )


def script_text(label: str, requested: int, blocks: Sequence[str], restarts: bool = False) -> str:
    """Return the shell script of a batch: LF line endings, one block per run in order."""
    head = [
        "#!/bin/sh",
        f"# Generated by cfd-matrix run --batch {requested}: batch {label}, {len(blocks)} run(s).",
        "# One block per run, in matrix order, each in its own point folder. No warm start: no",
        "# restart or solution file passes from one run to the next. A failed run does not stop",
        "# the next; the script exits non-zero when any run failed.",
        *(
            [
                "# A run whose RESTART names an earlier run of this script is staged from it when",
                "# that run ends with status 0, and is not started otherwise (status 97 or 98).",
            ]
            if restarts
            else []
        ),
        "set -u",
        'cd "$(dirname "$0")" || exit 2',
        "failed=0",
        "",
    ]
    body = (chr(10) * 2).join(blocks)
    return chr(10).join(head) + chr(10) + body + chr(10) * 2 + 'exit "$failed"' + chr(10)


def _job_values(
    point: Point, overrides: Mapping[str, object], variables: Mapping[str, object]
) -> dict[str, str]:
    """Return the values of a row that belong to the whole job: all but its application."""
    found = {name: str(value) for name, value in overrides.items() if name != "application_id"}
    if "cluster" not in found and "cluster" in variables:
        found["cluster"] = str(variables["cluster"])
    return found


def _agree(label: str, rows: Sequence[tuple[Point, dict[str, str]]]) -> dict[str, str]:
    """Refuse rows of one batch that disagree on a job-level value, naming both."""
    names = list(dict.fromkeys(name for _, values in rows for name in values))
    first_point, first = rows[0]
    for name in names:
        for point, values in rows[1:]:
            if values.get(name) != first.get(name):
                shown = name if name == "cluster" else HPC_OVERRIDE_PREFIX + name
                raise ValueError(
                    f"rows {first_point.pol} and {point.pol} of batch {label} disagree on "
                    f"{shown} ({first.get(name)!r} and {values.get(name)!r}); a batch is one "
                    "scheduler job with one value of it, so state it alike in every row it "
                    "holds, or run without --batch"
                )
    return first


def batch_plans(
    plan: RunPlan,
    profiles: tuple[HpcProfile, ...],
    n: int,
    *,
    completed: frozenset[str] = frozenset(),
) -> tuple[tuple[BatchPlan, ...], tuple[str, ...]]:
    """Group the plan's rows into batches and render each batch's script and descriptor.

    Returns
    -------
    tuple of BatchPlan and tuple of str
        The batches in matrix order, and the warnings of the split.
    """
    session = InputSession(plan.root)
    adapter = load_solver(plan.solver)
    solver_type = str(getattr(adapter, "SOLVER_TYPE", plan.solver))
    config_name = staged_config(adapter)
    stem = Path(plan.matrix).stem
    by_row: dict[str, list[tuple[Point, HpcProfile]]] = {}
    for point, profile in zip(plan.points, profiles, strict=True):
        by_row.setdefault(point.pol, []).append((point, profile))
    contexts = {
        point.point_id: load_point(plan.root, point, solver=plan.solver, session=session)
        for point in plan.points
    }
    units = []
    for order, (pol, members) in enumerate(by_row.items(), 1):
        point, profile = members[0]
        context = contexts[point.point_id]
        ncpus = point.ncpus if point.ncpus is not None else context.setup.get("ncpus")
        build = point.solver_build or context.setup.get("solver_version")
        weight = 0.0
        for member, _ in members:
            seconds = _clock(pol, member.walltime, True)[0]
            weight += seconds if seconds is not None else 1.0
        units.append(RowUnit(pol, order, (profile.path.as_posix(), str(ncpus), str(build)), weight))
    runs, warnings = split_rows(units, n)
    first_id = next_batch_id(plan.root, stem)
    batches = []
    for index, run in enumerate(runs):
        batch_id = first_id + index
        label = f"{stem}_b{batch_id}"
        sims = tuple(unit.sim_id for unit in run)
        members = [item for sim in sims for item in by_row[sim]]
        base = members[0][1]
        job_rows = [
            (p, _job_values(p, row_overrides(p.variables), p.variables)) for p, _ in members
        ]
        agreed = _agree(label, job_rows)
        job_overrides = {
            k: v for k, v in row_overrides(members[0][0].variables).items() if k != "application_id"
        }
        profile, overridden = apply_overrides(
            base, job_overrides, root=plan.root, row=members[0][0].pol
        )
        folder = workspace_path(plan.root, f"{BATCH_DIR}/{label}")
        script_name = f"BATCH-{sims[0]}-{sims[-1]}.sh"
        blocks, values_list = [], []
        applications: dict[str, Mapping[str, object]] = {}
        uses_cluster = _uses_cluster(profile)
        cluster = overridden.get("cluster", agreed.get("cluster"))
        if uses_cluster and cluster is None:
            raise ValueError(f"row {members[0][0].pol}: state cluster: <name> in VAR_NAMES_VALUES")
        members = in_job_order(members, completed)
        by_id = {p.point_id: p for p, _ in members}
        for point, _ in members:
            overrides = row_overrides(point.variables)
            resolved, _ignored = apply_overrides(base, overrides, root=plan.root, row=point.pol)
            point_folder = workspace_path(plan.root, point.directory)
            context = contexts[point.point_id]
            command = _command(
                plan,
                adapter,
                context,
                point,
                overrides,
                point_folder,
                config_name,
                resolved.application_id,
            )
            source = by_id.get(point.restart_from or "") if point.restart_in_job else None
            source_relative = (
                _relative(workspace_path(plan.root, source.directory), folder)
                if source is not None
                else None
            )
            blocks.append(_block(point, _relative(point_folder, folder), command, source_relative))
            values = point_values(
                plan,
                point,
                context,
                config_name,
                solver_type,
                cluster if uses_cluster else None,
            )
            values.update(
                script_path=script_name,
                stdout_log=(folder / "stdout.log").as_posix(),
                stderr_log=(folder / "stderr.log").as_posix(),
                batch=n,
                batch_id=batch_id,
            )
            values_list.append(values)
            if overrides:
                applications[point.point_id] = overrides
        submission = plan_submission(
            values_list, profile, workdir=folder, job_name=label, grouped=True
        )
        batches.append(
            BatchPlan(
                batch_id,
                label,
                f"{BATCH_DIR}/{label}",
                f"{BATCH_DIR}/{label}/{script_name}",
                sims,
                tuple(p for p, _ in members),
                script_text(label, n, blocks, any(p.restart_in_job for p, _ in members)),
                submission,
                base,
                applications,
                n,
            )
        )
    return tuple(batches), tuple(warnings)


def in_job_order(
    members: Sequence[tuple[Point, HpcProfile]], completed: frozenset[str]
) -> list[tuple[Point, HpcProfile]]:
    """Put each RESTART source of a job before the points it feeds, and mark those points.

    A point whose ``restart_from`` names another point of the same job, not already in
    ``completed``, is fed from it at run time: it is marked ``restart_in_job`` with the
    source's folder as ``restart_directory``. Runs of one job share no state, so moving a
    source ahead of its dependents cannot change a result; the order inside the job is
    otherwise the matrix order.

    Raises
    ------
    ValueError
        Points that feed each other.
    """
    ids = {p.point_id: p for p, _ in members}

    def feeds(point: Point) -> Point | None:
        source = ids.get(point.restart_from or "")
        return None if source is None or source.point_id in completed else source

    marked = [
        (
            replace(p, restart_in_job=True, restart_directory=src.directory)
            if (src := feeds(p)) is not None
            else p,
            prof,
        )
        for p, prof in members
    ]
    placed: list[tuple[Point, HpcProfile]] = []
    waiting = list(marked)
    while waiting:
        done = {p.point_id for p, _ in placed}
        ready = next(
            (
                item
                for item in waiting
                if not item[0].restart_in_job or item[0].restart_from in done
            ),
            None,
        )
        if ready is None:
            first = waiting[0][0]
            raise ValueError(
                f"RESTART of {first.point_id} names {first.restart_from}, which waits on it "
                "in the same job; points that feed each other cannot run in sequence"
            )
        waiting.remove(ready)
        placed.append(ready)
    return placed


def grouping_receipt(batches: Sequence[BatchPlan], n: int) -> dict[str, object]:
    """Return the ``grouping`` block of the plan receipt: the split ``--batch N`` decided."""
    return {
        "mode": "batch",
        "requested": n,
        "jobs": [
            {
                "batch_id": batch.batch_id,
                "label": batch.label,
                "dir": batch.folder,
                "script": batch.script,
                "sims": list(batch.sims),
                "points": [p.point_id for p in batch.points],
            }
            for batch in batches
        ],
    }


def grouped_options_refusal(
    *, local: bool, restart: bool, resume: bool, force_rerun: object
) -> str | None:
    """Say why ``--batch`` may not be combined with an option, or return None."""
    if local:
        return (
            "--batch groups rows into scheduler jobs; a batch is a scheduler job, so it "
            "cannot be combined with --local"
        )
    given = [
        flag
        for flag, present in (
            ("--force-rerun", bool(force_rerun)),
            ("--resume", resume),
            ("--restart", restart),
        )
        if present
    ]
    if given:
        return (
            f"a grouped run (--batch) does not take {', '.join(given)}"
            "; run the points one by one for that."
        )
    return None


def batch_exit_written(root: Path, record: RunRecord) -> int | None:
    """Return the exit status the batch script wrote for this run, or None when it wrote none."""
    job = record.submission.get("job") if record.submission is not None else None
    if not isinstance(job, Mapping):
        return None
    try:
        text = (workspace_path(root, record.point.directory) / STATUS_FILE).read_text("ascii")
        return int(text.strip())
    except (OSError, ValueError):
        return None


def batch_status(root: Path, record: RunRecord) -> int | None:
    """Return the exit status a batch's script wrote for this run, or None outside a batch."""
    job = record.submission.get("job") if record.submission is not None else None
    if not isinstance(job, Mapping):
        return None
    try:
        text = (workspace_path(root, record.point.directory) / STATUS_FILE).read_text("ascii")
        return int(text.strip())
    except (OSError, ValueError):
        return 1


def _source_cfg(point: Point, translated: Mapping[str, bytes]) -> str | None:
    """Return the rendering of the earlier run that feeds ``point`` in its job, or None."""
    if not point.restart_in_job or point.restart_from is None:
        return None
    return translated[point.restart_from].decode("ascii")


def submit_batches(
    plan: RunPlan, batches: Sequence[BatchPlan], adapter: SolverAdapter
) -> tuple[RunRecord, ...]:
    """Preflight every point, stage its inputs, write each batch's script and submit it once."""
    warn_low_ram()
    session = InputSession(plan.root)
    registry = RunRegistry(plan.root, session=session)
    previous = {r.point_id: r for r in registry.read()}
    config_name = staged_config(adapter)
    stage = getattr(adapter, "stage", None)
    for batch in batches:
        if workspace_path(plan.root, batch.folder).exists():
            raise RegistryError(f"{batch.folder}: existing output; refusing overwrite")
    translated: dict[str, bytes] = {}
    for point in (p for batch in batches for p in batch.points):
        recorded = previous.get(point.point_id)
        if recorded is not None:
            if recorded.status != "PLANNED":
                raise RegistryError(f"point {point.point_id!r} already recorded {recorded.status}")
            if (
                recorded.point != point
                or recorded.matrix != plan.matrix
                or recorded.solver != plan.solver
            ):
                raise RegistryError(
                    f"point {point.point_id!r}: registered proposal differs; plan again"
                )
        for name in (point.directory, *point.outputs):
            if workspace_path(plan.root, name).exists():
                raise RegistryError(f"{name}: existing output; refusing overwrite")
        context = load_point(
            plan.root,
            point,
            solver=plan.solver,
            session=session,
            restart_source_cfg=_source_cfg(point, translated),
        )
        check_restart(context, adapter)
        translated[point.point_id] = adapter.translate(context).encode("ascii")
    missing = tuple(p for batch in batches for p in batch.points if p.point_id not in previous)
    if missing:
        registry.record(replace(plan, points=missing))
    records: list[RunRecord] = []
    for batch in batches:
        folder = workspace_path(plan.root, batch.folder)
        folder.mkdir(parents=True, exist_ok=False)
        staged: list[Point] = []
        ascii_input: dict[str, bytes] = {}
        for point in batch.points:
            registry.transition(point.point_id, "RUNNING")
            point_folder = workspace_path(plan.root, point.directory)
            config = point_folder / config_name
            if stage is None:
                point_folder.mkdir(parents=True, exist_ok=False)
                with config.open("xb") as stream:
                    stream.write(translated[point.point_id])
                ascii_input[point.point_id] = translated[point.point_id]
            else:
                try:
                    stage(
                        load_point(
                            plan.root,
                            point,
                            solver=plan.solver,
                            session=session,
                            restart_source_cfg=_source_cfg(point, translated),
                        )
                    )
                    ascii_input[point.point_id] = config.read_bytes()
                except (OSError, ValueError) as failure:
                    records.append(
                        registry.transition(
                            point.point_id,
                            "COMPLETED",
                            classification="failed",
                            error=f"FAILED_STAGING: {failure}; nothing was submitted",
                        )
                    )
                    continue
            staged.append(point)
        if not staged:
            continue
        text = batch.script_text
        for point in batch.points:
            if not point.restart_in_job:
                continue
            point_folder = workspace_path(plan.root, point.directory)
            mark = STAGE_MARK.format(point_id=point.point_id)
            if point in staged:
                relative = _relative(point_folder, folder)
                shell = _staging(point, relative, _stage_pairs(point_folder, relative))
            else:
                shell = f"  rc={STAGING_FAILED}"
            text = text.replace(mark, shell)
        script = workspace_path(plan.root, batch.script)
        with script.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
        script.chmod(0o755)
        job = batch.submission
        result = submit(job)
        descriptor = job.descriptor_path.relative_to(plan.root).as_posix()
        for point in staged:
            point_folder = workspace_path(plan.root, point.directory)
            config = point_folder / config_name
            submission: dict[str, object] = {
                "descriptor": descriptor,
                "descriptor_sha256": hashlib.sha256(job.descriptor_path.read_bytes()).hexdigest(),
                "config_path": config.relative_to(plan.root).as_posix(),
                "config_sha256": hashlib.sha256(ascii_input[point.point_id]).hexdigest(),
                "profile": batch.profile.path.relative_to(plan.root).as_posix(),
                "command": job.command,
                "stdout_path": f"{batch.folder}/submit.stdout.txt",
                "stderr_path": f"{batch.folder}/submit.stderr.txt",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "return_code": result.return_code,
                "timeout_s": result.timeout_s,
                "timed_out": result.timed_out,
                "job": {
                    "mode": "batch",
                    "batch": batch.requested,
                    "batch_id": batch.batch_id,
                    "label": batch.label,
                    "dir": batch.folder,
                    "script": batch.script,
                    "points": len(batch.points),
                    "status_file": STATUS_FILE,
                },
                "job_end_files": [
                    f"{point.directory}/{STATUS_FILE}",
                    *(f"{point.directory}/{name}" for name in batch.profile.job_end_files),
                ],
            }
            overrides = row_overrides(point.variables)
            chosen = overrides.get("cluster", point.variables.get("cluster"))
            if chosen is not None and _uses_cluster(batch.profile):
                submission["cluster"] = chosen
            if overrides:
                submission["hpc_overrides"] = {"origin": ORIGIN, "values": overrides}
            if (names := job_outputs(adapter, point)) != point.outputs:
                submission["job_outputs"] = list(names)
            failed = result.return_code != 0 and not result.timed_out
            records.append(
                registry.transition(
                    point.point_id,
                    "COMPLETED" if failed else "SUBMITTED",
                    submission=submission,
                    classification="failed" if failed else None,
                    error=(
                        f"FAILED_EXECUTION: submit command exited {result.return_code}: "
                        f"{result.stderr}"
                    )
                    if failed
                    else None,
                )
            )
    return tuple(records)
