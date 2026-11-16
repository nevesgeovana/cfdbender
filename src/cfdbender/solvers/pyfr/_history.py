"""Reduce the named fluid-force columns over a physical time window."""

import configparser
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.results import Coefficient, ExecutionResult, Iteration, References
from cfdbender.solvers.pyfr._config import config


@dataclass(frozen=True)
class Evidence(ExecutionResult):
    failed: bool = False
    insufficient: bool = False
    budget_exhausted: bool = False


def read(
    folder: Path, point: AdapterPoint
) -> tuple[Evidence, dict[str, Coefficient], References, tuple[float, float]]:
    case = config(point.setup, point.point.condition, point.reference, point.geometry)
    if point.point.restart_from is not None:
        staged = configparser.ConfigParser()
        staged.read(folder / "case.ini", encoding="ascii")
        start = staged.getfloat("solver-time-integrator", "tstart")
        case = replace(case, tend=case.tend + start, window_start=case.window_start + start)
    refs = References(
        1,
        1,
        (0.25, 0, 0),
        "PyFR XY wind axes; q=0.5",
        ("drag along freestream", "lift normal to freestream", "pitch about z"),
    )
    # A cluster job leaves no execution.json; classification then rests on forces.csv.
    execution = folder / "execution.json"
    meta = json.loads(execution.read_bytes()) if execution.is_file() else {}
    code = int(meta["exit_code"]) if meta else 0
    version = str(meta.get("version") or "unknown")
    empty = Evidence(code, (), 1, version, True)
    try:
        with (folder / "forces.csv").open(encoding="ascii", newline="") as stream:
            reader = csv.DictReader(stream)
            if (
                reader.fieldnames is None
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not {"t", "px", "py", "vx", "vy"} <= set(reader.fieldnames)
            ):
                raise ValueError("invalid force columns")
            rows: list[tuple[float, float, float]] = []
            angle = math.radians(case.alpha)
            for row in reader:
                t, px, py, vx, vy = (float(row[key]) for key in ("t", "px", "py", "vx", "vy"))
                if not all(math.isfinite(v) for v in (t, px, py, vx, vy)) or (
                    rows and t <= rows[-1][0]
                ):
                    raise ValueError("invalid force sample")
                fx, fy = px + vx, py + vy
                rows.append(
                    (
                        t,
                        2 * (fy * math.cos(angle) - fx * math.sin(angle)),
                        2 * (fx * math.cos(angle) + fy * math.sin(angle)),
                    )
                )
        # The force plugin writes every nsteps, so the last sample can fall one
        # output interval short of tend; that still means the budget ran out.
        gap = rows[-1][0] - rows[-2][0] if len(rows) > 1 else 0.0
        exhausted = bool(rows) and rows[-1][0] + gap >= case.tend * (1 - 1e-9)
        window = [row for row in rows if case.window_start <= row[0] <= case.tend]
        if not window:
            return (
                Evidence(code, (), 1, version, True),
                {},
                refs,
                (case.window_start, case.tend),
            )
        cl = sum(r[1] for r in window) / len(window)
        cd = sum(r[2] for r in window) / len(window)
        coeff = {"Cl": Coefficient(cl, "y", "force"), "Cd": Coefficient(cd, "x", "force")}
        half = len(window) // 2
        if half < 1 or len(window) - half < 1:
            return (
                Evidence(code, (), 1, version, False, True, exhausted),
                coeff,
                refs,
                (case.window_start, case.tend),
            )
        first, second = window[:half], window[half:]
        diff = [
            abs(sum(r[i] for r in first) / len(first) - sum(r[i] for r in second) / len(second))
            for i in (1, 2)
        ]
        stationary = diff[0] < case.tolerance * abs(cl) and diff[1] < case.tolerance * abs(cd)
        history = (Iteration(1, max(diff), stationary),)
        return (
            Evidence(code, history, 1, version, budget_exhausted=exhausted),
            coeff,
            refs,
            (case.window_start, case.tend),
        )
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return empty, {}, refs, (case.window_start, case.tend)
