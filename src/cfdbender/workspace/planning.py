"""Compose registered solver row hooks with the independent core planner."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from cfdbender.core.plan import PlanError, RowValidator, RunPlan, plan_matrix
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
    return plan_matrix(root, name, validators={config.solver: cast(RowValidator, hook)})
