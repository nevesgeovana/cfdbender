"""Split the rows of a matrix into the jobs of ``--batch N``, as pyflightstream does.

Ported from pyflightstream ``run/_batch_split.py`` (``_partition``, ``split_polars``). A
row (POL) with its points is the unit and is never cut. Rows are grouped by what one job
cannot mix (the profile, the processor count and the solver build); each group gets at
least one job and is cut into CONTIGUOUS runs of rows in matrix order, by the optimal
linear partition that minimises the longest run. The jobs left over after one per group go,
one at a time, to the group whose longest run is longest while a row can still be split off.
No overhead of a solver instance is priced here: that figure is FlightStream's own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RowUnit:
    """One row as the split sees it.

    Attributes
    ----------
    sim_id : str
        The row's POL.
    order : int
        Its position in the matrix, from 1; the split keeps this order.
    group : tuple of str
        What one job cannot mix; rows of different groups never share a job.
    weight : float
        The seconds the row weighs: its points' WALLTIME cells, else one per point.
    """

    sim_id: str
    order: int
    group: tuple[str, ...]
    weight: float


def _partition(weights: Sequence[float], jobs: int) -> list[int]:
    """Return the cut indices of the optimal contiguous partition into ``jobs`` runs.

    Minimises the longest run (the classical linear partition, by dynamic programming);
    ties take the earliest cut, so the result is deterministic.
    """
    count = len(weights)
    prefix = [0.0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    longest_by: dict[int, dict[int, float]] = {1: {u: prefix[u] for u in range(1, count + 1)}}
    cut_by: dict[int, dict[int, int]] = {}
    for runs in range(2, jobs + 1):
        longest_by[runs] = {}
        cut_by[runs] = {}
        for upto in range(runs, count - (jobs - runs) + 1):
            choice = (math.inf, runs - 1)
            for cut in range(runs - 1, upto):
                longest = max(longest_by[runs - 1][cut], prefix[upto] - prefix[cut])
                if longest < choice[0]:
                    choice = (longest, cut)
            longest_by[runs][upto], cut_by[runs][upto] = choice
        del longest_by[runs - 1]
    cuts: list[int] = []
    upto = count
    for runs in range(jobs, 1, -1):
        cut = cut_by[runs][upto]
        cuts.append(cut)
        upto = cut
    return sorted(cuts)


def _runs(units: Sequence[RowUnit], jobs: int) -> list[tuple[RowUnit, ...]]:
    cuts = _partition([unit.weight for unit in units], jobs)
    edges = [0, *cuts, len(units)]
    return [tuple(units[a:b]) for a, b in zip(edges, edges[1:], strict=False)]


def _longest(units: Sequence[RowUnit], jobs: int) -> float:
    return max(sum(unit.weight for unit in run) for run in _runs(units, jobs))


def split_rows(units: Sequence[RowUnit], n: int) -> tuple[list[tuple[RowUnit, ...]], list[str]]:
    """Cut the rows into about ``n`` batch jobs.

    Parameters
    ----------
    units : sequence of RowUnit
        The rows, in matrix order.
    n : int
        The number of jobs asked for, at least 1.

    Returns
    -------
    tuple of list of tuple of RowUnit and list of str
        The jobs in matrix order of their first row, and the warnings: ``n`` above the rows
        (only what is needed is used) and ``n`` below the number of groups (one job per group).
    """
    groups: dict[tuple[str, ...], list[RowUnit]] = {}
    for unit in units:
        groups.setdefault(unit.group, []).append(unit)
    warnings: list[str] = []
    if n > len(units):
        warnings.append(
            f"--batch {n} asks for more jobs than the {len(units)} row(s) there are, and a row "
            f"is never cut, so {len(units)} job(s) are planned."
        )
    if n < len(groups):
        warnings.append(
            f"--batch {n} asks for fewer jobs than the {len(groups)} group(s) of profile, "
            "processor count and build there are, and a job holds one of each, so one "
            "job per group is planned."
        )
    counts = dict.fromkeys(groups, 1)
    for _ in range(max(0, min(n, len(units)) - len(groups))):
        open_groups = [key for key, members in groups.items() if counts[key] < len(members)]
        if not open_groups:
            break
        chosen = max(open_groups, key=lambda key: _longest(groups[key], counts[key]))
        counts[chosen] += 1
    jobs = [run for key, members in groups.items() for run in _runs(members, counts[key])]
    jobs.sort(key=lambda run: run[0].order)
    return jobs, warnings
