"""Read fixed-width XFOIL polars without deriving or rescaling coefficients."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Literal

from cfdbender.core.results import Coefficient, ExecutionResult, Iteration, References

_COLUMNS = {"CL": 1, "CD": 2, "CDp": 3, "CM": 4}


def read_polar(path: Path, alpha: float) -> dict[str, Coefficient]:
    """Read the requested alpha using dashed-header column widths.

    XFOIL prints alpha to three decimal places. A match uses half that printed
    resolution. Malformed, missing, duplicate or non-finite matching evidence
    returns no coefficients. All printed numerical columns must be finite.
    """
    try:
        lines = path.read_text(encoding="ascii").splitlines()
        header = next(i for i, line in enumerate(lines) if line.split()[:1] == ["alpha"])
        if lines[header].split()[:7] != ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]:
            return {}
        rule = lines[header + 1]
        if set(rule) - {" ", "-"}:
            return {}
        spans = [(match.start(), match.end()) for match in re.finditer("-+", rule)]
        if len(spans) < 7 or len(spans) != len(lines[header].split()):
            return {}
        matches: list[list[float]] = []
        for line in lines[header + 2 :]:
            if not line.strip():
                continue
            cells = [float(line[start:end]) for start, end in spans]
            if not all(map(math.isfinite, cells)):
                return {}
            if abs(cells[0] - alpha) <= 0.0005000001:
                matches.append(cells)
        if len(matches) != 1:
            return {}
        axes: dict[str, Literal["x", "y", "z"]] = {"CL": "z", "CD": "x", "CDp": "x", "CM": "y"}
        return {
            key: Coefficient(matches[0][column], axes[key], "moment" if key == "CM" else "force")
            for key, column in _COLUMNS.items()
        }
    except (OSError, UnicodeError, ValueError, StopIteration, IndexError):
        return {}


def references() -> References:
    return References(
        1.0,
        1.0,
        (0.25, 0.0, 0.0),
        "XFOIL wind lift/drag; quarter-chord pitching moment, unit chord and unit span",
        ("positive drag", "positive nose-up pitching moment", "positive lift"),
    )


def read(
    folder: Path, alpha: float, iterations: int
) -> tuple[ExecutionResult, dict[str, Coefficient]]:
    # A cluster job leaves no execution.json; its version is then read from stdout.log.
    execution = folder / "execution.json"
    if execution.is_file():
        metadata = json.loads(execution.read_bytes())
    else:
        log = folder / "stdout.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        line = next((x for x in text.splitlines() if "XFOIL" in x and "ersion" in x), "")
        metadata = {"exit_code": 0, "version_line": line}
    coefficients = read_polar(folder / "polar.txt", alpha)
    # The neutral protocol requires a converged history entry. This is a polar
    # acceptance marker, not a measured viscous iteration or residual: XFOIL's
    # polar contains neither. 1 and 0 are protocol sentinels, not physical data.
    history = (Iteration(1, 0.0, True),) if coefficients else ()
    version_line = metadata.get("version_line") or ""
    version = re.search(r"\bVersion\s+([0-9]+(?:[.][0-9]+)+)", version_line, re.IGNORECASE)
    return ExecutionResult(
        int(metadata["exit_code"]),
        history,
        iterations,
        version[1] if version else "unknown",
    ), coefficients
