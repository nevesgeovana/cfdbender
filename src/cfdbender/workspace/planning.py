"""Compose registered solver row hooks with the independent core planner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import capabilities
from cfdbender.core.plan import PlanError, RowValidator, RunPlan, plan_matrix
from cfdbender.core.refusals import warn_or_refuse
from cfdbender.core.run_types import TIME_ACCURATE, UNSTEADY_ROTOR
from cfdbender.core.settings import read_init
from cfdbender.solvers import available_solvers, load_solver


def plan_workspace(root: str | Path, name: str) -> RunPlan:
    """Load the declared solver's validate_row hook and produce a read-only plan.

    Parameters
    ----------
    root : str or Path
        Workspace root, containing a single solver declaration in init.toml.
    name : str
        Filename or stem in inputs/matrices/.

    Returns
    -------
    RunPlan
        Validated points and proposed work, with no execution or file writes.

    Raises
    ------
    PlanError
        A registered adapter has no callable validate_row hook.
    """
    config = read_init(root, available_solvers=available_solvers())
    module = load_solver(config.solver)
    hook = getattr(module, "validate_row", None)
    if not callable(hook):
        raise PlanError(f"solver {config.solver!r} has no validate_row planning hook")
    plan = plan_matrix(
        root,
        name,
        validators={config.solver: cast(RowValidator, hook)},
        accepts_numerics="numerics_file" in capabilities(module),
    )
    if "restart" not in capabilities(module) and any(p.restart_from for p in plan.points):
        warn_or_refuse(
            f"solver {config.solver!r} does not support restart; RESTART entries skipped"
        )
        plan = replace(plan, points=tuple(replace(p, restart_from=None) for p in plan.points))
    return _skip_precursors(plan, config.solver, module)


def _skip_precursors(plan: RunPlan, solver: str, module: object) -> RunPlan:
    """Warn and skip an ``MRF`` RESTART entry on a row it does not apply to.

    The keyword asks a time-accurate sliding rotor row for a steady frozen-rotor stage before
    it. On a row that is not time-accurate the solve is steady already and the entry changes
    nothing, so it is skipped with one warning per row; a solver without the capability skips
    it as it skips a RESTART it cannot take. A time-accurate row that cannot take the stage is
    refused by its solver's row hook, because skipping would start it cold, not as asked.
    """
    rows = {p.pol: p for p in plan.points if p.precursor}
    keep: dict[str, bool] = {}
    for pol, point in rows.items():
        if not getattr(module, "PRECURSOR", False):
            if point.run_type in TIME_ACCURATE:
                raise PlanError(
                    f"POL {pol}: RESTART MRF: solver {solver!r} has no frozen-rotor "
                    "precursor stage, and skipping it would start the time-accurate row cold"
                )
            why = f"solver {solver!r} has no frozen-rotor precursor stage"
        elif point.run_type != UNSTEADY_ROTOR:
            why = f"WORKFLOW {point.run_type} is not a time-accurate sliding rotor row"
        else:
            keep[pol] = True
            continue
        warn_or_refuse(f"POL {pol}: RESTART MRF is skipped: {why}", UserWarning, stacklevel=3)
        keep[pol] = False
    if not rows or all(keep.values()):
        return plan
    return replace(
        plan,
        points=tuple(
            replace(p, precursor=False) if p.precursor and not keep[p.pol] else p
            for p in plan.points
        ),
    )
