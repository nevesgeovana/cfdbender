"""Decide which planned points a run executes again: resume, force-rerun, archive.

Without options a recorded point refuses the run, as before. ``resume`` skips
COMPLETED and SUBMITTED points and runs the rest. A point left RUNNING by a
local run that crashed is run again only when no solver process holds this
host's process slot; a RUNNING point of a cluster submission, whose job may
exist, is run again only when named by ``force``. ``force`` names recorded
points to redo, leaving the other recorded points untouched, and ``force_all``
redoes every recorded point of the plan; each
is archived first (see RunRegistry.archive) and registered again.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from cfdbender.core.plan import Point, RunPlan
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry
from cfdbender.solvers._host._single import process_lease


@dataclass(frozen=True)
class RerunDecision:
    """The plan to execute, the points skipped with their reasons, and any archive."""

    plan: RunPlan
    skipped: tuple[tuple[str, str], ...]
    archive: Path | None


def host_slot_free() -> bool:
    """Return True when no solver process holds this host's process slot."""
    try:
        with process_lease("probe"):
            return True
    except ValueError:
        return False


def _named(plan: RunPlan, names: Sequence[str]) -> set[str]:
    chosen: set[str] = set()
    for name in names:
        matches = [
            p.point_id
            for p in plan.points
            if name in (p.point_id, p.point_id.split("/", 1)[1], Path(p.directory).name)
        ]
        if not matches:
            raise RegistryError(f"--force-rerun {name}: no point of this plan carries that name")
        if len(matches) > 1:
            raise RegistryError(
                f"--force-rerun {name}: names {len(matches)} points; give POL/DP-name"
            )
        chosen.add(matches[0])
    return chosen


def decide(
    plan: RunPlan,
    *,
    resume: bool = False,
    force: Sequence[str] = (),
    force_all: bool = False,
    submitting: bool = False,
    slot_free: Callable[[], bool] | None = None,
) -> RerunDecision:
    """Archive the points to redo and return the plan the run executes.

    Parameters
    ----------
    plan : RunPlan
        Validated plan, already narrowed by any selection.
    resume : bool, optional
        Skip recorded COMPLETED and SUBMITTED points instead of refusing.
    force : sequence of str, optional
        Recorded points to redo, by POL/DP-name or DP-name.
    force_all : bool, optional
        Redo every recorded point of the plan.
    submitting : bool, optional
        True for a cluster submission: a RUNNING point is then never presumed gone.
    slot_free : callable, optional
        Reports whether the host's solver process slot is free; host_slot_free by default.

    Returns
    -------
    RerunDecision
        Points to execute in plan order, skipped points and the archive folder.

    Raises
    ------
    RegistryError
        Conflicting options, a name no recorded point carries, or an archive failure.
    """
    if force_all and (force or resume):
        raise RegistryError("--force-rerun-all cannot be combined with --resume or --force-rerun")
    named = _named(plan, force)
    slot_free = slot_free or host_slot_free
    registry = RunRegistry(plan.root)
    recorded: dict[str, RunRecord] = {r.point_id: r for r in registry.read()}
    if unrecorded := sorted(n for n in named if n not in recorded):
        raise RegistryError(f"--force-rerun {unrecorded[0]}: the point has no record to redo")
    redo: list[str] = []
    skipped: list[tuple[str, str]] = []
    run: list[Point] = []
    for point in plan.points:
        record = recorded.get(point.point_id)
        if record is None or record.status == "PLANNED":
            run.append(point)
        elif force_all or point.point_id in named:
            redo.append(point.point_id)
            run.append(point)
        elif not resume and not named:
            run.append(point)  # the run refuses it, naming its status
        elif not resume:
            skipped.append((point.point_id, f"{record.status}, not named by --force-rerun"))
        elif record.status == "RUNNING" and not submitting and slot_free():
            redo.append(point.point_id)
            run.append(point)
        elif record.status == "RUNNING":
            skipped.append((point.point_id, "RUNNING and its process may be alive"))
        else:
            skipped.append((point.point_id, record.status))
    archive = registry.archive(tuple(redo)) if redo else None
    return RerunDecision(replace(plan, points=tuple(run)), tuple(skipped), archive)
