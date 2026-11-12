"""Bind solver-neutral points to the profile's descriptor and command.

Every job runs in its point folder. Before the submit command runs, the
adapter's stage hook writes into that folder every file the solver reads, as a
local run stages them (SU2: su2.cfg and the mesh or its link; OpenFOAM: the case
tree and Allrun; PyFR: case.ini and mesh.msh; QPROP: propeller.prop, qcon.def
and sections.json; XFOIL: xfoil.in and airfoil.dat). The adapter's STAGED_CONFIG
names the file the solver reads; it is config_path and script_path for the
profile. stdout_log and stderr_log name stdout.log and stderr.log in the point
folder, where a profile directs the job's channels so collect can read them. An
adapter without a stage hook has its translated ASCII saved as solver.in. The row's
NCPUS and SOLVER_BUILD columns supply ncpus and solver_version; where a row
states none, optional top-level setup values solver_version and ncpus answer.
solver_type is the setup's value, else the adapter's solver name (SU2 for
su2). No scheduler command or resource value is guessed here.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from cfdbender.core.adapter import SolverAdapter, load_point
from cfdbender.core.matrix import MatrixError, parse_matrix
from cfdbender.core.plan import InputFiles, Point, RunPlan, workspace_path
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry
from cfdbender.core.run import check_available_ram
from cfdbender.hpc import HpcProfile, SubmissionPlan, plan_submission, resolve_hpc_profile
from cfdbender.hpc._submit import submit
from cfdbender.solvers import load_solver

_FALLBACK_CONFIG = "solver.in"


def staged_config(adapter: object) -> str:
    """Return the point-folder file name a job's solver reads for this adapter.

    Parameters
    ----------
    adapter : object
        Solver adapter module.

    Returns
    -------
    str
        STAGED_CONFIG when the adapter stages its inputs, else solver.in.
    """
    if callable(getattr(adapter, "stage", None)):
        return str(adapter.STAGED_CONFIG)  # type: ignore[attr-defined]
    return _FALLBACK_CONFIG


def select_profile(root: Path, name: str | None) -> HpcProfile | None:
    """Resolve this call's profile against explicitly stated setup versions.

    Parameters
    ----------
    root : Path
        Workspace root containing inputs/hpc and inputs/setups.
    name : str or None
        Explicit profile selector or the only-profile rule.

    Returns
    -------
    HpcProfile or None
        Selected declaration, or None when no profile exists or was requested.
    """
    versions: set[str] = set()
    if (root / "inputs/hpc").is_dir():
        files = InputFiles(root)
        for path in sorted((root / "inputs/setups").glob("*.toml")):
            version = files.read_toml(path.relative_to(root).as_posix()).get("solver_version")
            if isinstance(version, str) and version.strip():
                versions.add(version)
        for path in sorted((root / "inputs/matrices").glob("*.fs")):
            try:
                rows = parse_matrix(path.read_bytes(), path, active_only=False)
            except (OSError, MatrixError):
                # Planning that matrix names its own refusal; it states no build here.
                continue
            versions.update(row.solver_build for row in rows if row.solver_build)
    return resolve_hpc_profile(root / "inputs", name, solver_versions=versions)


def job_outputs(adapter: object, point: Point) -> tuple[str, ...]:
    """Return the declared outputs a cluster job writes, by the adapter's JOB_OUTPUTS.

    Parameters
    ----------
    adapter : object
        Solver adapter module; without JOB_OUTPUTS every declared output counts.
    point : Point
        Planned point and its declared outputs.

    Returns
    -------
    tuple of str
        Workspace-relative outputs collect waits for before reading the point.
    """
    names = getattr(adapter, "JOB_OUTPUTS", None)
    if names is None:
        return point.outputs
    prefix = point.directory.rstrip("/") + "/"
    return tuple(name for name in point.outputs if name.removeprefix(prefix) in tuple(names))


def submission_plans(plan: RunPlan, profile: HpcProfile) -> tuple[SubmissionPlan, ...]:
    """Preflight all descriptor substitutions without writing any job files.

    Parameters
    ----------
    plan : RunPlan
        Validated matrix and input digests.
    profile : HpcProfile
        This invocation's selected profile.

    Returns
    -------
    tuple of SubmissionPlan
        One descriptor and argv per point, in matrix order.
    """
    jobs = []
    adapter = load_solver(plan.solver)
    solver_type = str(getattr(adapter, "SOLVER_TYPE", plan.solver))
    name = staged_config(adapter)
    for point in plan.points:
        context = load_point(plan.root, point, solver=plan.solver)
        folder = workspace_path(plan.root, point.directory)
        config = (folder / name).as_posix()
        values: dict[str, object] = {
            "row_id": point.pol,
            "point": Path(point.directory).name,
            "workdir": folder.as_posix(),
            "config_path": config,
            "script_path": config,
            "stdout_log": (folder / "stdout.log").as_posix(),
            "stderr_log": (folder / "stderr.log").as_posix(),
            "walltime": point.walltime,
        }
        for key in ("solver_version", "ncpus", "solver_type"):
            value = context.setup.get(key)
            if value is not None and value != "":
                values[key] = value
        # The row's columns win over the setup, as in a pyflightstream matrix.
        if point.ncpus is not None:
            values["ncpus"] = point.ncpus
        if point.solver_build is not None:
            values["solver_version"] = point.solver_build
        values.setdefault("solver_type", solver_type)
        jobs.append(plan_submission([values], profile))
    return tuple(jobs)


def submit_plan(
    plan: RunPlan, jobs: tuple[SubmissionPlan, ...], profile: HpcProfile, adapter: SolverAdapter
) -> tuple[RunRecord, ...]:
    """Preflight every point, stage its inputs, then submit it through its profile.

    Parameters
    ----------
    plan : RunPlan
        Matrix with validated input digests.
    jobs : tuple of SubmissionPlan
        One already-rendered job per point.
    profile : HpcProfile
        The selected profile, recorded with every submission.
    adapter : SolverAdapter
        Translates and stages inputs; it never executes the solver here.

    Returns
    -------
    tuple of RunRecord
        SUBMITTED points or failed COMPLETED points. A staging failure is a
        failed COMPLETED point that was never submitted. A timed-out submit remains
        SUBMITTED because the scheduler may have accepted it. Interruption
        before the submission is recorded leaves RUNNING, preventing a retry.
    """
    check_available_ram()
    registry = RunRegistry(plan.root)
    previous = {r.point_id: r for r in registry.read()}
    config_name = staged_config(adapter)
    stage = getattr(adapter, "stage", None)
    translated = []
    for point, job in zip(plan.points, jobs, strict=True):
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
        reserved = {config_name.casefold(), "submit.stdout.txt", "submit.stderr.txt"}
        outputs = {workspace_path(plan.root, name).name.casefold() for name in point.outputs}
        if stage is not None:
            outputs.discard(config_name.casefold())
        if job.descriptor_path.name.casefold() in reserved | outputs:
            raise RegistryError(f"{job.descriptor_path.name}: descriptor collides with a job file")
        if outputs & {"submit.stdout.txt", "submit.stderr.txt"}:
            raise RegistryError("submit capture files collide with declared outputs")
        context = load_point(plan.root, point, solver=plan.solver)
        translated.append(adapter.translate(context).encode("ascii"))
    missing = tuple(point for point in plan.points if point.point_id not in previous)
    if missing:
        registry.record(replace(plan, points=missing))
    records = []
    for point, job, ascii_input in zip(plan.points, jobs, translated, strict=True):
        check_available_ram()
        registry.transition(point.point_id, "RUNNING")
        folder = workspace_path(plan.root, point.directory)
        config = folder / config_name
        if stage is None:
            folder.mkdir(parents=True, exist_ok=False)
            with config.open("xb") as stream:
                stream.write(ascii_input)
        else:
            try:
                stage(load_point(plan.root, point, solver=plan.solver))
                ascii_input = config.read_bytes()
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
        result = submit(job)
        submission: dict[str, object] = {
            "descriptor": job.descriptor_path.relative_to(plan.root).as_posix(),
            "descriptor_sha256": hashlib.sha256(job.descriptor_path.read_bytes()).hexdigest(),
            "config_path": config.relative_to(plan.root).as_posix(),
            "config_sha256": hashlib.sha256(ascii_input).hexdigest(),
            "profile": profile.path.relative_to(plan.root).as_posix(),
            "command": job.command,
            "stdout_path": (folder / "submit.stdout.txt").relative_to(plan.root).as_posix(),
            "stderr_path": (folder / "submit.stderr.txt").relative_to(plan.root).as_posix(),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "return_code": result.return_code,
            "timeout_s": result.timeout_s,
            "timed_out": result.timed_out,
        }
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
                    f"FAILED_EXECUTION: submit command exited {result.return_code}: {result.stderr}"
                )
                if failed
                else None,
            )
        )
    return tuple(records)
