"""Prepare restart requests before archiving their original evidence."""

import hashlib
from dataclasses import replace

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import InputSession, RunPlan
from cfdbender.core.registry import RegistryError, RunRegistry
from cfdbender.core.restart import check_restart


def prepare(plan: RunPlan, adapter: object, *, own: bool, local: bool = True) -> RunPlan:
    """Resolve sources and prove their solutions exist before a destructive rerun step."""
    records = {r.point_id: r for r in RunRegistry(plan.root).read()}
    points = []
    session = InputSession(plan.root)
    for point in plan.points:
        source = point.point_id if own else point.restart_from
        if source is None:
            points.append(point)
            continue
        if source not in records:
            raise RegistryError(f"RESTART {source}: no recorded point; run the source point first")
        record = records[source]
        if record.solver != plan.solver:
            raise RegistryError(f"RESTART {source}: recorded solver differs from {plan.solver}")
        interrupted_local = (
            own and local and record.status == "RUNNING" and record.submission is None
        )
        if record.status != "COMPLETED" and not interrupted_local:
            raise RegistryError(f"RESTART {source}: source is {record.status}; wait for completion")
        updated = replace(point, restart_from=source, restart_directory=record.point.directory)
        context = load_point(plan.root, updated, solver=plan.solver, session=session)
        files = check_restart(context, adapter)
        updated = replace(
            updated,
            restart_sha256={
                name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
            },
        )
        points.append(updated)
    return replace(plan, points=tuple(points))
