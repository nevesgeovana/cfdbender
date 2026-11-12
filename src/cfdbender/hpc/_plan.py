"""Prepare one point or grouped job without writing or submitting anything."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ._profiles import HpcProfile
from ._render import build_submit_command, render_descriptor
from ._walltime import BudgetGroup, BudgetRow, job_walltime, walltime_text

_WALLTIME_TEXT = re.compile(
    r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf(?:inity)?))\s*([a-z]+)\s*",
    re.IGNORECASE,
)
_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_UNITS_GLOSS = "s (seconds), m (minutes), h (hours), d (days)"
_LOCAL_REFUSAL = (
    "a local grouped job needs a limit: state WALLTIME in its rows or timeout_s in "
    "the setup, or give the HPC profile a max_walltime; it would otherwise run unbounded"
)


@dataclass(frozen=True)
class SubmissionPlan:
    """The descriptor text/path, argv and budget of a job that has not been submitted.

    ``walltime`` is the descriptor value (text or seconds); ``walltime_s`` is
    the scheduler budget, possibly absent. ``local_time_limit_s`` gives a later
    local executor its deadline when requested. ``warnings`` are returned for
    the caller to print. ``fits`` and ``shortfall_s`` compare recorded estimates
    with the budget; neither claims a measured runtime. ``submitted`` is false.
    """

    descriptor_text: str
    descriptor_path: Path
    command: list[str]
    walltime: str | int | None
    walltime_s: int | None
    walltime_source: str
    warnings: tuple[str, ...]
    local_time_limit_s: float | None = None
    fits: bool | None = None
    shortfall_s: float | None = None
    submitted: bool = False


def _number(value: object, key: str, row: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"row {row}: {key} {value!r} is not a number of seconds")
    try:
        result = float(value)
    except ValueError:
        raise ValueError(f"row {row}: {key} {value!r} is not a number of seconds") from None
    if not math.isfinite(result) or result < 0 or (not zero and result == 0):
        raise ValueError(
            f"row {row}: {key} {value!r} must be finite and {'nonnegative' if zero else 'positive'}"
        )
    return result


def _clock(row: str, stated: object, grouped: bool) -> tuple[float | None, str | None, bool]:
    if stated is None or str(stated).strip() in ("", "-"):
        return None, None, False
    text = str(stated).strip()
    if text.upper() == "BEST":
        if not grouped:
            raise ValueError(
                f"row {row} states WALLTIME: {text!r}. BEST asks the package for the walltime, "
                "which grouped modes (--batch or --polar-sweep) compute; write a wall clock "
                "with its unit for the default mode."
            )
        return None, None, True
    matched = _WALLTIME_TEXT.fullmatch(text)
    if matched is None and not text[-1:].isalpha():
        raise ValueError(
            f"row {row} states WALLTIME: {text!r}, which carries no unit. Write the unit with "
            "the number, as 240m or 4h: a bare number read as seconds here and as minutes "
            f"on the scheduler. The units are {_UNITS_GLOSS}."
        )
    if matched is None:
        raise ValueError(
            f"row {row} states WALLTIME: {text!r}, not a number followed by {_UNITS_GLOSS}"
        )
    number, unit = matched.group(1), matched.group(2).lower()
    if unit not in _UNITS:
        raise ValueError(
            f"row {row} states WALLTIME: {text!r}; unit {unit!r} is not one of {_UNITS_GLOSS}"
        )
    return _number(number, "WALLTIME", row) * _UNITS[unit], text, False


def _budget(points: Sequence[Mapping[str, object]], grouped: bool) -> BudgetGroup:
    by_row: dict[str, list[Mapping[str, object]]] = {}
    for point in points:
        row = str(point.get("row_id") or point.get("sim") or "?")
        by_row.setdefault(row, []).append(point)
    units = []
    for row, members in by_row.items():
        first = members[0]
        cell, written, best = _clock(row, first.get("walltime"), grouped)
        if any(member.get("walltime") != first.get("walltime") for member in members):
            raise ValueError(f"row {row} has inconsistent WALLTIME values among its points")
        estimates = tuple(
            _number(p["estimate_s"], "estimate_s", row, zero=True)
            if p.get("estimate_s") is not None
            else None
            for p in members
        )
        margins = [_number(p.get("margin_s", 1200), "margin_s", row, zero=True) for p in members]
        timeouts = [
            _number(p["timeout_s"], "timeout_s", row)
            for p in members
            if p.get("timeout_s") is not None
        ]
        units.append(
            BudgetRow(
                sim_id=row,
                run_ids=tuple(
                    str(p.get("run_id") or f"{row}/{p.get('point', '?')}") for p in members
                ),
                point_seconds=estimates,
                walltime_cell_s=cell,
                walltime_cell_text=written,
                best=best,
                margin_s=max(margins),
                timeout_s=max(timeouts) if timeouts else None,
            )
        )
    return BudgetGroup(tuple(units))


def plan_submission(
    points: Sequence[Mapping[str, object]],
    profile: HpcProfile,
    *,
    workdir: str | Path | None = None,
    job_name: str | None = None,
    grouped: bool | None = None,
    local: bool = False,
    estimate_s: float | None = None,
) -> SubmissionPlan:
    """Plan one job using only profile declarations and resolved point values.

    Parameters
    ----------
    points : sequence of mappings
        Each mapping supplies ``row_id``, ``point``, ``solver_version``, ``ncpus``,
        ``workdir`` and ``config_path`` (or ``script_path`` for a prepared group
        launcher). Other descriptor values such as ``solver_type`` and ``cluster``
        are optional unless the template asks for them. ``walltime`` is a row cell
        with s/m/h/d units or BEST. Optional ``estimate_s`` prices a point;
        ``margin_s`` defaults to 1200 and ``timeout_s`` bounds a local point.
        All points in one job must name the same solver version and CPU count.
    profile : HpcProfile
        Descriptor template, version mappings, command and maximum budget.
    workdir : str or Path, optional
        Job folder; defaults to the first point's folder. Supply it for a group
        that runs in a separate folder. No directory is created.
    job_name : str, optional
        Scheduler job name. Defaults to the first row and point joined by '_'.
    grouped : bool, optional
        Defaults to True for multiple points. Set True for a one-point grouped
        plan that may use BEST. This function accepts an already selected group;
        it does not split jobs or construct a solver launcher.
    local : bool, optional
        Compute the deadline a later forced-local executor must enforce.
    estimate_s : float, optional
        Group estimate including caller-measured overheads. Otherwise sum point
        estimates, using cell budgets for gaps. No solver's measured overheads
        are assumed by this solver-independent layer.

    Returns
    -------
    SubmissionPlan
        Descriptor text and planned path, argv, walltime, and warnings. Nothing
        is written, launched, submitted or printed.

    Raises
    ------
    ValueError
        Empty or incompatible points, invalid budgets, a row above the maximum,
        an unbounded local group, or an unresolved template value.
    """
    if not points:
        raise ValueError("cannot plan a submission with no points")
    first = points[0]
    for key in ("solver_version", "ncpus"):
        if any(p.get(key) != first.get(key) for p in points):
            raise ValueError(f"a grouped job holds one {key}; the points name different values")
    is_group = len(points) > 1 if grouped is None else grouped
    if len(points) > 1 and not is_group:
        raise ValueError("multiple points require a grouped job")
    group = _budget(points, is_group)
    maximum = profile.max_walltime_s
    if estimate_s is not None:
        estimate_s = _number(estimate_s, "estimate_s", "job", zero=True)
    else:
        costs = [
            s if s is not None else row.walltime_cell_s
            for row in group.units
            for s in row.point_seconds
        ]
        estimate_s = (
            sum(s for s in costs if s is not None) if all(s is not None for s in costs) else None
        )
    seconds, source, fits, shortfall, warnings, refusal = job_walltime(
        group, estimate_s, max_walltime_s=maximum
    )
    if refusal is not None:
        raise ValueError(refusal)
    local_limit = float(seconds) if local and seconds is not None else None
    if local and seconds is None:
        limits = [row.timeout_s for row in group.units if row.timeout_s is not None]
        if limits:
            local_limit = max(limits) * len(points)
            warnings.append(
                f"run on this machine (--local), its time limit is {local_limit:g} s: "
                "no WALLTIME, so the largest solver timeout_s for each point"
            )
        elif maximum:
            seconds, source, local_limit = int(maximum), "max_walltime", float(maximum)
            warnings.append(
                "no WALLTIME and no timeout_s, so the local job takes the profile's "
                f"walltime of {walltime_text(seconds)}."
            )
        else:
            raise ValueError(_LOCAL_REFUSAL)
    values = {k: v for k, v in first.items() if v is not None and v != ""}
    folder = workdir if workdir is not None else first.get("workdir", first.get("work_dir"))
    if folder is None or str(folder).strip() == "":
        raise ValueError("this job cannot supply workdir")
    path = Path(str(folder)) / profile.descriptor_name
    row_ids = list(dict.fromkeys(row.sim_id for row in group.units))
    values.update(
        workdir=Path(str(folder)).as_posix(),
        work_dir=Path(str(folder)).as_posix(),
        descriptor_path=path.as_posix(),
        sim=row_ids[0],
        first_sim=row_ids[0],
        last_sim=row_ids[-1],
        sims=",".join(row_ids),
    )
    name = job_name or first.get("job_name") or f"{row_ids[0]}_{first.get('point', row_ids[0])}"
    values["job_name"] = name
    values["config_paths"] = ",".join(str(p["config_path"]) for p in points if p.get("config_path"))
    wall: str | int | None = None
    if seconds is not None:
        written = (
            group.units[0].walltime_cell_text
            if not is_group and source == "matrix"
            else walltime_text(seconds)
        )
        wall = seconds if profile.walltime_arithmetic == "seconds" else written
        values.update(walltime=wall, walltime_s=seconds, walltime_written=written)
    else:
        # A raw row's absent wall clock must not format as None or inherit another point's.
        values.pop("walltime", None)
    command = build_submit_command(profile, values, path)
    text = render_descriptor(profile, values)
    return SubmissionPlan(
        text, path, command, wall, seconds, source, tuple(warnings), local_limit, fits, shortfall
    )
