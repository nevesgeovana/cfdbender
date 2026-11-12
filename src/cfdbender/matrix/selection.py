"""Select simulations (POL) and points of a plan or of recorded points.

``--sims`` takes POL ids, space or comma separated; ``--points`` takes point
names, ``DP-name`` or ``POL/DP-name``. An id or name the plan or the records do
not carry is refused before anything runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from cfdbender.core.plan import Point, RunPlan
from cfdbender.core.registry import RegistryError


def split_ids(values: Sequence[str] | None) -> tuple[str, ...]:
    """Split space- and comma-separated command values into ids, keeping order."""
    if not values:
        return ()
    found = [part.strip() for value in values for part in value.split(",")]
    if any(not part for part in found):
        raise RegistryError("an empty id in --sims or --points")
    return tuple(dict.fromkeys(found))


def _matches(point: Point, name: str) -> bool:
    return name in (point.point_id, point.point_id.split("/", 1)[1])


def chosen(points: Iterable[Point], sims: Sequence[str], names: Sequence[str]) -> set[str]:
    """Return the point ids that the selection keeps, refusing unknown ids.

    Parameters
    ----------
    points : iterable of Point
        Planned or recorded points.
    sims : sequence of str
        POL ids; empty keeps every simulation.
    names : sequence of str
        Point names; empty keeps every point of the kept simulations.

    Returns
    -------
    set of str
        Point ids.
    """
    pool = list(points)
    if unknown := [sim for sim in sims if all(p.pol != sim for p in pool)]:
        raise RegistryError(f"--sims {unknown[0]}: no simulation with that POL")
    kept = [p for p in pool if not sims or p.pol in sims]
    if unknown := [name for name in names if not any(_matches(p, name) for p in kept)]:
        raise RegistryError(f"--points {unknown[0]}: no selected point carries that name")
    return {p.point_id for p in kept if not names or any(_matches(p, n) for n in names)}


def select_plan(plan: RunPlan, sims: Sequence[str], names: Sequence[str]) -> RunPlan:
    """Return the plan narrowed to the selection, in matrix order."""
    if not sims and not names:
        return plan
    keep = chosen(plan.points, sims, names)
    return replace(plan, points=tuple(p for p in plan.points if p.point_id in keep))
