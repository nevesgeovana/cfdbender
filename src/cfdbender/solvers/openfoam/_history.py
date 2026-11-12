"""Read ESI forceCoeffs by column name and SIMPLE numerical termination evidence."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.results import Coefficient, ExecutionResult, Iteration, References
from cfdbender.solvers.openfoam._config import config


@dataclass(frozen=True)
class Evidence(ExecutionResult):
    """Retain numerical failure independently of the actual process exit code."""

    evidence_failed: bool = False


def _residuals(log: str) -> dict[int, dict[str, float]]:
    values: dict[int, dict[str, float]] = {}
    current = 0
    for line in log.splitlines():
        if match := re.fullmatch(r"Time = ([0-9]+)", line.strip()):
            current = int(match[1])
            values[current] = {}
        if current and (
            match := re.search(r"Solving for ([A-Za-z0-9_]+), Initial residual = ([^, ]+)", line)
        ):
            residual = float(match[2])
            if not math.isfinite(residual) or residual < 0:
                raise ValueError("non-finite or negative initial residual")
            # residualControl uses initial, not final, residuals. Keep every solve,
            # including pressure correctors, by taking the largest per field.
            values[current][match[1]] = max(residual, values[current].get(match[1], 0))
    return values


def _rows(folder: Path) -> list[dict[str, float]]:
    paths = sorted(folder.glob("postProcessing/forceCoeffs*/0/coefficient.dat"))
    if len(paths) != 1:
        raise ValueError("need exactly one forceCoeffs output")
    headers: list[str] = []
    rows: list[dict[str, float]] = []
    for line in paths[0].read_text(encoding="ascii").splitlines():
        if line.startswith("# Time"):
            headers = line[1:].split()
            if len(set(headers)) != len(headers) or not {"Time", "Cd", "Cl", "CmPitch"} <= set(
                headers
            ):
                raise ValueError("forceCoeffs needs unique Time, Cd, Cl and CmPitch headers")
        elif line.strip() and not line.lstrip().startswith("#"):
            if not headers:
                raise ValueError("forceCoeffs data before # Time header")
            row = dict(zip(headers, map(float, line.split()), strict=True))
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError("forceCoeffs contains non-finite values")
            if row["Time"] < 1 or not row["Time"].is_integer():
                raise ValueError("simpleFoam history requires positive integer Time")
            if rows and row["Time"] <= rows[-1]["Time"]:
                raise ValueError("forceCoeffs Time must increase")
            rows.append(row)
    if not rows:
        raise ValueError("forceCoeffs has no rows")
    return rows


def read(folder: Path, point: AdapterPoint) -> tuple[Evidence, dict[str, Coefficient], References]:
    """Read saved coefficients, preserving references and process exit status."""
    case = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
    )
    # A cluster job leaves no execution.json; classification then rests on the
    # force file and the solver logs alone.
    execution = folder / "execution.json"
    metadata = json.loads(execution.read_bytes()) if execution.is_file() else {"exit_code": 0}
    application_log = (
        (folder / "log.simpleFoam").read_text(encoding="utf-8", errors="replace")
        if (folder / "log.simpleFoam").is_file()
        else ""
    )
    logs = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in sorted(folder.glob("log.*"))
    )
    logs += "\n" + "\n".join(
        (folder / name).read_text(encoding="utf-8", errors="replace")
        for name in ("stdout.log", "stderr.log")
        if (folder / name).is_file()
    )
    version = re.search(r"OpenFOAM[- ](v[0-9]{4})", logs)
    failed = re.search(r"FOAM FATAL (?:IO )?ERROR", logs) is not None
    result = Evidence(
        int(metadata["exit_code"]),
        (),
        case.iterations,
        version[1] if version else "unknown",
        failed,
    )
    try:
        rows = _rows(folder)
        residuals = _residuals(application_log)
        final_iteration = int(rows[-1]["Time"])
        # Retain printed coefficients for diagnostics, but refuse success when
        # the force file and application log describe different final states.
        failed = failed or not residuals or next(reversed(residuals)) != final_iteration
        messages = re.findall(
            r"^\s*SIMPLE solution converged in ([0-9]+) iterations\s*$",
            application_log,
            re.MULTILINE,
        )
        affirmative = bool(messages) and int(messages[-1]) == final_iteration
        required = {"Ux", "Uy", "p"}
        if case.turbulence == "SpalartAllmaras":
            required.add("nuTilda")
        history = []
        for row in rows:
            iteration = int(row["Time"])
            initial = residuals.get(iteration, {})
            residual = max(initial.values(), default=1.0)
            last = row is rows[-1]
            converged = (
                last
                and not failed
                and (
                    affirmative
                    or (
                        required <= initial.keys()
                        and all(v < case.target for v in initial.values())
                    )
                )
            )
            history.append(Iteration(iteration, residual, converged))
        row = rows[-1]
        coefficients = {
            "Cd": Coefficient(row["Cd"], "x", "force"),
            "Cl": Coefficient(row["Cl"], "y", "force"),
            "CmPitch": Coefficient(row["CmPitch"], "z", "moment"),
        }
        return (
            Evidence(
                result.exit_code, tuple(history), case.iterations, result.solver_version, failed
            ),
            coefficients,
            case.references,
        )
    except (OSError, UnicodeError, ValueError, OverflowError):
        return (
            Evidence(result.exit_code, (), case.iterations, result.solver_version, True),
            {},
            case.references,
        )
