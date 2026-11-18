"""Read SU2 CSV and affirmative solver log evidence into the neutral contract.

SU2 residual columns are log10 values and steady Inner_Iter starts at zero.
The neutral Iteration carries linear residual magnitudes and one-based counts.
Coefficient names retain SU2 spelling: CL/CD/CSF use wind z/x/y; CFx/CFy/CFz
and CMx/CMy/CMz use body x/y/z. No wind-to-body rotation is inferred.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from cfdbender.core.results import Coefficient, ExecutionResult, Iteration, References


@dataclass(frozen=True)
class _FramedCoefficient(Coefficient):
    """Retain a per-column frame when the shared dataclass serializes coefficients."""

    axis_system: Literal["wind", "body"]


_AXES: dict[str, tuple[Literal["x", "y", "z"], Literal["force", "moment"]]] = {
    "CL": ("z", "force"),
    "CD": ("x", "force"),
    "CSF": ("y", "force"),
    "CFx": ("x", "force"),
    "CFy": ("y", "force"),
    "CFz": ("z", "force"),
    "CMx": ("x", "moment"),
    "CMy": ("y", "moment"),
    "CMz": ("z", "moment"),
}


def references(cfg: dict[str, str]) -> References:
    return References(
        float(cfg["REF_AREA"]),
        float(cfg["REF_LENGTH"]),
        tuple(float(cfg[f"REF_ORIGIN_MOMENT_{axis}"]) for axis in "XYZ"),  # type: ignore[arg-type]
        "SU2 wind (CD, CSF, CL); SU2 body (CFx, CFy, CFz, CMx, CMy, CMz)",
        ("wind drag / body +x", "wind side force / body +y", "wind lift / body +z"),
    )


def dimensions(folder: Path, *, mesh: bool = True) -> int | None:
    """Return 2 or 3, the dimension of the problem SU2 solved, or None when unstated.

    SU2 prints "Two dimensional problem." or "Three dimensional problem." while it
    reads the mesh; a native mesh states NDIME near its top. The dimension decides
    which body moment is the pitching moment: CMz in two dimensions (the flow is in
    the x-y plane), CMy in three.
    """
    for name in ("stdout.log", "stderr.log"):
        path = folder / name
        if path.is_file():
            found = re.search(
                "(Two|Three) dimensional problem[.]",
                path.read_text(encoding="utf-8", errors="replace"),
            )
            if found:
                return 2 if found[1] == "Two" else 3
    if not mesh:
        return None
    staged = folder / "mesh.su2"
    try:
        with staged.open(encoding="ascii", errors="replace") as stream:
            for _, line in zip(range(200), stream, strict=False):
                stated = re.fullmatch("NDIME[ ]*=[ ]*([23])[ ]*", line.strip())
                if stated:
                    return int(stated[1])
    except OSError:
        return None
    return None


#: SU2's CONV_FIELD names and the history.csv columns they converge on.
CONV_COLUMNS = {
    "RMS_DENSITY": "rms[Rho]",
    "RMS_MOMENTUM-X": "rms[RhoU]",
    "RMS_MOMENTUM-Y": "rms[RhoV]",
    "RMS_MOMENTUM-Z": "rms[RhoW]",
    "RMS_ENERGY": "rms[RhoE]",
    "RMS_NU_TILDE": "rms[nu]",
    "RMS_TKE": "rms[k]",
    "RMS_DISSIPATION": "rms[w]",
    "RMS_PRESSURE": "rms[P]",
    "RMS_VELOCITY-X": "rms[U]",
    "RMS_VELOCITY-Y": "rms[V]",
    "RMS_VELOCITY-Z": "rms[W]",
    "RMS_TEMPERATURE": "rms[T]",
}
#: The name recorded when CONV_FIELD names no residual column (a Cauchy coefficient).
LARGEST_RMS = "largest rms"


def _cfg(folder: Path) -> dict[str, str]:
    lines = (folder / "su2.cfg").read_text().splitlines()
    return {key.strip(): value.strip() for key, value in (line.split("=", 1) for line in lines)}


def _convergence_columns(cfg: dict[str, str], residuals: list[str]) -> tuple[list[str], str]:
    """Return the residual columns SU2 converges on and the name they are recorded under.

    The columns are those of CONV_FIELD (SU2's default RMS_DENSITY). A CONV_FIELD that
    names no residual column, such as a Cauchy criterion on a coefficient, falls back to
    the largest of all rms columns.
    """
    fields = re.split(r"[\s,()]+", cfg.get("CONV_FIELD", "RMS_DENSITY").upper())
    columns = [CONV_COLUMNS[name] for name in fields if CONV_COLUMNS.get(name) in residuals]
    if not columns:
        return residuals, LARGEST_RMS
    return columns, ", ".join(columns)


def residual_field(folder: Path) -> str | None:
    """Return the name of the residual the history records, or None without a history."""
    try:
        with (folder / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
            headers = [key.strip() for key in next(csv.reader(stream, skipinitialspace=True))]
        residuals = [key for key in headers if key.startswith("rms[")]
        return _convergence_columns(_cfg(folder), residuals)[1] if residuals else None
    except (OSError, ValueError, StopIteration):
        return None


def read(folder: Path) -> tuple[ExecutionResult, dict[str, Coefficient], References]:
    cfg = _cfg(folder)
    refs = references(cfg)
    # A cluster job leaves no execution.json: its exit code is not seen here and
    # classification rests on the history and the solver log alone.
    execution = folder / "execution.json"
    metadata = json.loads(execution.read_bytes()) if execution.is_file() else {"exit_code": 0}
    log = chr(10).join(
        path.read_text(encoding="utf-8", errors="replace")
        for name in ("stdout.log", "stderr.log")
        if (path := folder / name).is_file()
    )
    version = re.search(
        "(?:SU2.*?(?:version|Version|v)|Version|Release)[ ]*([0-9]+[.][0-9]+[.][0-9]+)", log
    )
    result = ExecutionResult(
        int(metadata["exit_code"]), (), int(cfg["ITER"]), version[1] if version else "unknown"
    )
    # Affirmative lines only. 'Exit Success' says nothing about numerical convergence.
    # The message counts only when no iteration line follows it: a banner printed before
    # later iterations belongs to an earlier state of the log, not to the last iteration.
    lines = log.splitlines()
    last_iteration = max(
        (n for n, line in enumerate(lines) if re.match(r"\s*[|]\s*\d+\s*[|]", line)),
        default=-1,
    )
    converged = any(
        re.fullmatch(
            "(?:[|* ]*)(?:(?:All )?Convergence criteria (?:were )?satisfied[.!]?|"
            "SU2 has successfully converged[.!]?)(?:[|* ]*)",
            line.strip(),
            re.IGNORECASE,
        )
        is not None
        for line in lines[last_iteration + 1 :]
    )
    history: list[Iteration] = []
    coefficients: dict[str, Coefficient] = {}
    try:
        with (folder / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, skipinitialspace=True)
            headers = [key.strip() for key in next(reader)]
            if len(set(headers)) != len(headers) or "Inner_Iter" not in headers:
                raise ValueError("history.csv needs unique headers and Inner_Iter")
            residuals = [key for key in headers if key.startswith("rms[")]
            if not residuals or not any(key in _AXES for key in headers):
                raise ValueError("history.csv needs residual and coefficient columns")
            # The residual recorded is the one SU2 converges on, so it shows the criterion.
            residuals = _convergence_columns(cfg, residuals)[0]
            for cells in reader:
                row = dict(zip(headers, cells, strict=True))
                # Reject non-finite numerical history, including uncollected coefficient fields.
                if not all(math.isfinite(float(value)) for value in cells):
                    raise ValueError("history.csv contains non-finite values")
                iteration = int(row["Inner_Iter"]) + 1
                if history and iteration <= history[-1].iteration:
                    raise ValueError("history.csv iterations must increase")
                residual = max(10.0 ** float(row[key]) for key in residuals)
                history.append(Iteration(iteration, residual))
                coefficients = {
                    key: _FramedCoefficient(
                        float(row[key]),
                        axis,
                        kind,
                        "wind" if key in ("CL", "CD", "CSF") else "body",
                    )
                    for key, (axis, kind) in _AXES.items()
                    if key in row
                }
        if history and converged:
            history[-1] = replace(history[-1], converged=True)
    except (OSError, ValueError, StopIteration, OverflowError):
        return result, {}, refs
    return replace(result, history=tuple(history)), coefficients, refs


def force_groups(
    folder: Path, requested: Mapping[str, tuple[str, ...]], total: Mapping[str, Coefficient]
) -> Mapping[str, Mapping[str, Coefficient]]:
    """Sum native per-marker coefficients about the same axes and reference origin."""
    if not requested:
        return {}
    with (folder / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        rows = list(reader)
    if not rows or not total:
        raise ValueError("history.csv: missing boundary force results")
    row = {key.strip(): value for key, value in rows[-1].items()}
    result: dict[str, Mapping[str, Coefficient]] = {"TOTAL": dict(total)}
    for name, markers in requested.items():
        coefficients = {}
        for key, coefficient in total.items():
            columns = [f"{key}({marker})" for marker in markers]
            if missing := [column for column in columns if column not in row]:
                raise ValueError(
                    f"history.csv: group {name} lacks {', '.join(missing)}; keep AERO_COEFF_SURF"
                )
            values = [float(row[column]) for column in columns]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"history.csv: group {name} contains non-finite coefficients")
            coefficients[key] = replace(coefficient, value=sum(values))
        result[name] = coefficients
    return result
