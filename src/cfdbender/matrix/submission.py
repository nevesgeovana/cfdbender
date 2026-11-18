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
import string
import warnings
from dataclasses import replace
from pathlib import Path

from cfdbender.core.adapter import SolverAdapter, load_point
from cfdbender.core.matrix import HPC_OVERRIDE_PREFIX, MatrixError, parse_matrix
from cfdbender.core.plan import InputFiles, InputSession, Point, RunPlan, workspace_path
from cfdbender.core.refusals import warn_or_refuse, warn_unused_cluster
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry
from cfdbender.core.restart import check_restart
from cfdbender.core.run import FLOOR_GIB, free_ram_gib
from cfdbender.hpc import HpcProfile, SubmissionPlan, plan_submission, resolve_hpc_profile
from cfdbender.hpc._overrides import ORIGIN, apply_overrides, row_overrides
from cfdbender.hpc._submit import submit
from cfdbender.solvers import load_solver

_FALLBACK_CONFIG = "solver.in"


def _uses_cluster(profile: HpcProfile) -> bool:
    pending = [str(part) for part in (*profile.fields.values(), *profile.submit)]
    while pending:
        for _, name, spec, _ in string.Formatter().parse(pending.pop()):
            if name == "cluster":
                return True
            if spec:
                pending.append(spec)
    return False


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
        for path in sorted(
            p for home in (root / "inputs/matrices", root) for p in home.glob("*.fs")
        ):
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
    selector = getattr(adapter, "job_output_files", None)
    if selector is not None:
        names = selector(point)
    if names is None:
        return point.outputs
    prefix = point.directory.rstrip("/") + "/"
    return tuple(name for name in point.outputs if name.removeprefix(prefix) in tuple(names))


def point_profiles(plan: RunPlan, name: str | None) -> tuple[HpcProfile | None, ...]:
    """Resolve each point's row selector before the invocation's default."""
    cache: dict[str | None, HpcProfile | None] = {}
    for point in plan.points:
        selector = point.variables.get("hpc", name)
        if selector is not None and not isinstance(selector, str):
            raise ValueError(f"row {point.pol}: hpc must name a profile")
        if selector not in cache:
            cache[selector] = select_profile(plan.root, selector)
    found = tuple(
        cache[str(value) if value is not None else None]
        for point in plan.points
        for value in (point.variables.get("hpc", name),)
    )
    for point, chosen in zip(plan.points, found, strict=True):
        if chosen is None and (names := row_overrides(point.variables)):
            raise ValueError(
                f"row {point.pol}: {', '.join(HPC_OVERRIDE_PREFIX + n for n in names)} override "
                "an HPC profile, and no profile is selected; state hpc: <profile> in the row "
                "or pass --hpc"
            )
    return found


def warn_ignored_overrides(plan: RunPlan) -> None:
    """Warn once that a local run ignores the rows' ``hpc.<name>`` entries."""
    rows = [p.pol for p in plan.points if row_overrides(p.variables)]
    if rows:
        warn_or_refuse(
            f"--local ignores the {HPC_OVERRIDE_PREFIX}<name> entries of row(s) "
            f"{', '.join(rows)}; they override an HPC profile, and nothing is submitted",
            UserWarning,
            stacklevel=2,
        )


def submission_plans(
    plan: RunPlan, profile: HpcProfile | tuple[HpcProfile, ...]
) -> tuple[SubmissionPlan, ...]:
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
    session = InputSession(plan.root)
    adapter = load_solver(plan.solver)
    solver_type = str(getattr(adapter, "SOLVER_TYPE", plan.solver))
    name = staged_config(adapter)
    profiles = (profile,) * len(plan.points) if isinstance(profile, HpcProfile) else profile
    for point, base in zip(plan.points, profiles, strict=True):
        profile, overridden = apply_overrides(
            base, row_overrides(point.variables), root=plan.root, row=point.pol
        )
        uses_cluster = _uses_cluster(profile)
        cluster = overridden.get("cluster", point.variables.get("cluster"))
        if uses_cluster and cluster is None:
            raise ValueError(f"row {point.pol}: state cluster: <name> in VAR_NAMES_VALUES")
        if cluster is not None and not uses_cluster:
            warn_unused_cluster(point.pol)
        context = load_point(plan.root, point, solver=plan.solver, session=session)
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
        if cluster is not None and uses_cluster:
            values["cluster"] = cluster
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


def warn_low_ram() -> None:
    """Warn, never refuse, when this host's free RAM is below the floor or unknown.

    A submitted job runs on the cluster, not on the host that submits it, so the memory
    of the submitting host (often a shared login node) is no reason to refuse.
    """
    try:
        available = free_ram_gib()
    except (OSError, ValueError, IndexError):
        message = (
            "cannot measure free RAM on this host; submitting anyway because the job runs "
            "on the cluster"
        )
    else:
        if available >= FLOOR_GIB:
            return
        message = (
            f"free RAM {available:.3f} GiB on this host is below the {FLOOR_GIB} GiB floor; "
            "submitting anyway because the job runs on the cluster"
        )
    warnings.warn(message, UserWarning, stacklevel=2)


def submit_plan(
    plan: RunPlan,
    jobs: tuple[SubmissionPlan, ...],
    profile: HpcProfile | tuple[HpcProfile, ...],
    adapter: SolverAdapter,
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
    warn_low_ram()
    session = InputSession(plan.root)
    registry = RunRegistry(plan.root, session=session)
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
        context = load_point(plan.root, point, solver=plan.solver, session=session)
        check_restart(context, adapter)
        translated.append(adapter.translate(context).encode("ascii"))
    missing = tuple(point for point in plan.points if point.point_id not in previous)
    if missing:
        registry.record(replace(plan, points=missing))
    records = []
    profiles = (profile,) * len(plan.points) if isinstance(profile, HpcProfile) else profile
    for point, job, ascii_input, profile in zip(
        plan.points, jobs, translated, profiles, strict=True
    ):
        registry.transition(point.point_id, "RUNNING")
        folder = workspace_path(plan.root, point.directory)
        config = folder / config_name
        if stage is None:
            folder.mkdir(parents=True, exist_ok=False)
            with config.open("xb") as stream:
                stream.write(ascii_input)
        else:
            try:
                stage(load_point(plan.root, point, solver=plan.solver, session=session))
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
        overrides = row_overrides(point.variables)
        chosen = overrides.get("cluster", point.variables.get("cluster"))
        if chosen is not None and _uses_cluster(profile):
            submission["cluster"] = chosen
        if overrides:
            submission["hpc_overrides"] = {"origin": ORIGIN, "values": overrides}
        if (names := job_outputs(adapter, point)) != point.outputs:
            submission["job_outputs"] = list(names)
        if profile.job_end_files:
            submission["job_end_files"] = [
                f"{point.directory}/{name}" for name in profile.job_end_files
            ]
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
