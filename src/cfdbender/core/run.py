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
from cfdbender.core.executables import bind_executable
from cfdbender.core.plan import RunPlan, workspace_path
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry


def check_available_ram() -> None:
    """Refuse local work if available memory cannot be measured above 0.5 GiB.

    Windows uses GlobalMemoryStatusEx; Linux reads MemAvailable from procfs.
    No solver or optional monitoring dependency is imported.

    Raises
    ------
    RegistryError
        Available memory is unknown or below the required floor.
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
            raise RegistryError("cannot measure free RAM; local run refused")
        available = status.available
    else:
        try:
            line = next(
                line
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
            available = int(line.split()[1]) * 1024
        except (OSError, StopIteration, ValueError, IndexError) as error:
            raise RegistryError("cannot measure free RAM; local run refused") from error
    if available < 0.5 * 2**30:
        raise RegistryError(f"free RAM {available / 2**30:.3f} GiB is below the 0.5 GiB floor")


def execute_plan(
    plan: RunPlan,
    adapter: SolverAdapter,
    *,
    check_ram: Callable[[], None] = check_available_ram,
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
    registry = RunRegistry(plan.root)
    previous = {r.point_id: r for r in registry.read()}
    prepared = []
    for point in plan.points:
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
        # A row's SOLVER_BUILD selects its executable here, refusing an undeclared build.
        prepared.append(bind_executable(load_point(plan.root, point, solver=plan.solver), adapter))
    missing = tuple(p for p in plan.points if p.point_id not in previous)
    if missing:
        registry.record(replace(plan, points=missing))
    completed = []
    for context in prepared:
        check_ram()
        registry.transition(context.point.point_id, "RUNNING")
        error = None
        try:
            result = adapter.execute(context, check_ram=check_ram)
            classification = adapter.classify(result)
            if classification not in ("converged", "iteration_limit", "failed"):
                raise ValueError(f"unknown classification {classification!r}")
            if classification == "failed":
                error = f"FAILED_EXECUTION: exit code {result.exit_code}; numerical evidence failed"
            elif any(
                not workspace_path(plan.root, name).is_file() for name in context.point.outputs
            ):
                classification = "failed"
                error = "FAILED_INCOMPLETE_OUTPUT: missing declared output"
        except (OSError, ValueError) as failure:
            classification = "failed"
            error = f"FAILED_EXECUTION: {failure}"
        completed.append(
            registry.transition(
                context.point.point_id,
                "COMPLETED",
                classification=classification,
                error=error,
            )
        )
    return tuple(completed)
