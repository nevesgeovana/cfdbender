"""Decide which planned points a run executes again: resume, force-rerun, archive.

Without options a recorded point refuses the run, as before. ``resume`` skips
COMPLETED and SUBMITTED points and runs the rest. A point left RUNNING is
skipped because its process or cluster job may still exist. Name it by ``force``
after confirming it has stopped. ``force`` names recorded
points to redo, leaving the other recorded points untouched, and ``force_all``
redoes every recorded point of the plan; each
is archived first (see RunRegistry.archive) and registered again. ``force_all``
yields to every recorded state and lists, in ``RerunDecision.live``, the points whose
job may still be alive; it never cancels one.
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
    live: tuple[tuple[str, str], ...] = ()


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
    before_archive: Callable[[RunPlan], None] | None = None,
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
        Reports whether a RUNNING point can be redone; defaults to False.
    before_archive : callable, optional
        Called with the plan to execute just before any point is archived; a refusal it
        raises leaves every record and folder as it was.

    Returns
    -------
    RerunDecision
        Points to execute in plan order, skipped points, the archive folder and, under
        ``force_all``, the points redone while SUBMITTED or RUNNING.

    Raises
    ------
    RegistryError
        Conflicting options, a name no recorded point carries, or an archive failure.
    """
    if force_all and (force or resume):
        raise RegistryError("--force-rerun-all cannot be combined with --resume or --force-rerun")
    named = _named(plan, force)
    slot_free = slot_free or (lambda: False)
    registry = RunRegistry(plan.root)
    recorded: dict[str, RunRecord] = {r.point_id: r for r in registry.read()}
    if unrecorded := sorted(n for n in named if n not in recorded):
        raise RegistryError(f"--force-rerun {unrecorded[0]}: the point has no record to redo")
    redo: list[str] = []
    live: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    run: list[Point] = []
    for point in plan.points:
        record = recorded.get(point.point_id)
        if record is None or record.status == "PLANNED":
            run.append(point)
        elif force_all or point.point_id in named:
            if force_all and record.status in ("SUBMITTED", "RUNNING"):
                live.append((point.point_id, record.status))
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
        elif record.classification == "failed":
            # pyflightstream's --resume also skips a recorded point; the line says how to redo it.
            skipped.append(
                (
                    point.point_id,
                    f"{record.status} failed; redo it with --force-rerun {point.point_id}",
                )
            )
        else:
            skipped.append((point.point_id, record.status))
    chosen = replace(plan, points=tuple(run))
    if redo and before_archive is not None:
        before_archive(chosen)
    archive = registry.archive(tuple(redo)) if redo else None
    return RerunDecision(chosen, tuple(skipped), archive, tuple(live))
