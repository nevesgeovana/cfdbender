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


def read(folder: Path) -> tuple[ExecutionResult, dict[str, Coefficient], References]:
    cfg = dict(line.split("=", 1) for line in (folder / "su2.cfg").read_text().splitlines())
    cfg = {key.strip(): value.strip() for key, value in cfg.items()}
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
    converged = any(
        re.fullmatch(
            "(?:[|* ]*)(?:(?:All )?Convergence criteria (?:were )?satisfied[.!]?|"
            "SU2 has successfully converged[.!]?)(?:[|* ]*)",
            line.strip(),
            re.IGNORECASE,
        )
        is not None
        for line in log.splitlines()
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
