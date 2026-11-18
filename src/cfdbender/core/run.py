"""Execute injected adapters point by point, preserving each outcome in runs.json.

Preflight every selected point before writing. A failed execution is recorded
and the next point runs; an unexpected programming error propagates, leaving
RUNNING evidence rather than pretending the solver failed. A recorded point
refuses overwrite here; cfd-matrix run --resume and --force-rerun decide, before
this runs, which points to skip and which to archive and register again.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from cfdbender.core.adapter import SolverAdapter, load_point
from cfdbender.core.executables import bind_executable, build_mismatch
from cfdbender.core.plan import InputSession, RunPlan, workspace_path
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry
from cfdbender.core.restart import check_restart

FLOOR_GIB = 0.5


def free_ram_gib() -> float:
    """Return available host RAM in GiB; the one measurement every RAM guard reads.

    Windows uses GlobalMemoryStatusEx; Linux reads MemAvailable from procfs.
    No solver or optional monitoring dependency is imported.

    Raises
    ------
    OSError
        Available memory cannot be measured.
    """
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page",
                    "free_page",
                    "virtual",
                    "free_virtual",
                    "ext",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return float(status.available) / 2**30
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise OSError("MemAvailable missing")


def check_available_ram() -> None:
    """Refuse local work if available memory cannot be measured above 0.5 GiB.

    Raises
    ------
    RegistryError
        Available memory is unknown or below the required floor.
    """
    try:
        available = free_ram_gib()
    except (OSError, ValueError, IndexError) as error:
        raise RegistryError("cannot measure free RAM; local run refused") from error
    if available < FLOOR_GIB:
        raise RegistryError(f"free RAM {available:.3f} GiB is below the 0.5 GiB floor")


def judge_environment(
    plan: RunPlan,
    adapter: SolverAdapter,
    *,
    check_ram: Callable[[], None] | None = None,
) -> None:
    """Refuse, changing nothing, when this machine cannot run the plan's points now.

    The same judgment ``execute_plan`` makes before its first record (free RAM, each
    row's executable, the adapter's preflight), offered separately so that a caller
    about to archive recorded points can ask first.

    Parameters
    ----------
    plan : RunPlan
        The points about to run.
    adapter : SolverAdapter
        The workspace's adapter.
    check_ram : callable, optional
        The RAM guard; ``check_available_ram`` by default.

    Raises
    ------
    RegistryError
        Free RAM below the floor, an undeclared build, a missing executable or a busy host.
    """
    (check_ram or check_available_ram)()
    preflight = getattr(adapter, "preflight", None)
    session = InputSession(plan.root)
    for point in plan.points:
        try:
            context = bind_executable(
                load_point(plan.root, point, solver=plan.solver, session=session), adapter
            )
            check_restart(context, adapter)
            if preflight is not None:
                preflight(context)
        except (OSError, ValueError) as failure:
            raise RegistryError(str(failure)) from failure


def execute_plan(
    plan: RunPlan,
    adapter: SolverAdapter,
    *,
    check_ram: Callable[[], None] = check_available_ram,
    on_event: Callable[[str, int, int, RunRecord | None, str], None] | None = None,
    one_run_at_a_time: bool = False,
) -> tuple[RunRecord, ...]:
    """Run every point locally, recording failures and continuing to the next point.

    Parameters
    ----------
    plan : RunPlan
        Validated plan, including the adapter's validate_row hook and input digests.
    adapter : SolverAdapter
        Injected adapter; core never discovers or imports a solver.
    check_ram : callable, optional
        Guard called before writing, before every point and throughout adapter execution.
    on_event : callable, optional
        Called as ``on_event("start", index, total, None, point_id)`` when a point starts and
        ``on_event("end", index, total, record, point_id)`` when it ends.

    one_run_at_a_time : bool, optional
        Record whether this run holds the machine lease.

    Returns
    -------
    tuple of RunRecord
        Completed records, including failed points and their diagnoses.

    Raises
    ------
    RegistryError
        A recorded point, changed input, existing output or threatened memory floor.
    """
    check_ram()
    session = InputSession(plan.root)
    registry = RunRegistry(plan.root, session=session)
    previous = {r.point_id: r for r in registry.read()}
    prepared = []
    for point in plan.points:
        recorded = previous.get(point.point_id)
        if recorded is not None:
            if recorded.status != "PLANNED":
                raise RegistryError(
                    f"point {point.point_id!r} already recorded {recorded.status}; use --resume "
                    "to run only the points not yet recorded, or --force-rerun POINT to redo one"
                )
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
        # A row's SOLVER_BUILD selects its executable here, refusing an undeclared build.
        context = bind_executable(
            load_point(plan.root, point, solver=plan.solver, session=session), adapter
        )
        check_restart(context, adapter)
        prepared.append(context)
    # The environment is judged once, before any record: a missing executable, a busy host
    # slot or a RAM floor refuses the run and leaves the records as they were.
    preflight = getattr(adapter, "preflight", None)
    if preflight is not None:
        for context in prepared:
            try:
                preflight(context)
            except (OSError, ValueError) as failure:
                raise RegistryError(str(failure)) from failure
    missing = tuple(p for p in plan.points if p.point_id not in previous)
    if missing:
        registry.record(replace(plan, points=missing))
    completed = []
    total = len(prepared)
    for index, context in enumerate(prepared, 1):
        check_ram()
        registry.transition(context.point.point_id, "RUNNING", one_run_at_a_time=one_run_at_a_time)
        if on_event is not None:
            on_event("start", index, total, None, context.point.point_id)
        error = None
        version = None
        try:
            result = adapter.execute(context, check_ram=check_ram)
            version = str(getattr(result, "solver_version", "")) or None
            classification = adapter.classify(result)
            if classification not in ("converged", "iteration_limit", "failed"):
                raise ValueError(f"unknown classification {classification!r}")
            # A row's SOLVER_BUILD must be the solver that ran: the printed version decides.
            if mismatch := build_mismatch(context.point.solver_build, version or ""):
                classification = "failed"
                error = f"FAILED_EXECUTION: {mismatch}"
            elif classification == "failed":
                error = f"FAILED_EXECUTION: exit code {result.exit_code}; numerical evidence failed"
            elif any(
                not workspace_path(plan.root, name).is_file() for name in context.point.outputs
            ):
                classification = "failed"
                error = "FAILED_INCOMPLETE_OUTPUT: missing declared output"
        except (OSError, ValueError) as failure:
            classification = "failed"
            error = f"FAILED_EXECUTION: {failure}"
        record = registry.transition(
            context.point.point_id,
            "COMPLETED",
            classification=classification,
            error=error,
            solver_version=version,
        )
        completed.append(record)
        if on_event is not None:
            on_event("end", index, total, record, context.point.point_id)
    return tuple(completed)
