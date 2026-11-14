"""Write the CSV tables of a collected matrix beside its campaign summary.

``post/<matrix>/campaign_sweep.csv`` holds one row per recorded point of the
matrix and ``post/<matrix>/polars/P<POL>.csv`` one row per point of each
simulation, in the pyflightstream spelling of a polar file's ``P<sim>`` stem
(cfdbender has no boundary groups, so the stem carries no group suffix). The
columns are POL, point, the fixed context block of pyflightstream (ALPHA, BETA,
MACH, RE in millions, VINF, VREF, ALT, RHO, TEMP, MU, J, J_CLOCK, RPM_CLOCK, then
SREF, CREF, BREF), any other FLIGHT_CONDITION key the matrix states, status,
classification, iterations, residual, wall_time_s, solver_version, the neutral
coefficients CL, CD and CM (NA where the solver reports none) and every coefficient
the solver reported under its own name. A value a point does not have is the token NA;
no cell holds a comma, a double quote or a line break and none is quoted, so
numpy.genfromtxt reads every row to the header's count. Every table of a matrix
has the same columns. The tables are derived and rebuilt by every collect.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from cfdbender.core.flight import FlightConditionError, resolve_flow
from cfdbender.core.neutral import NEUTRAL, ROTOR, ROTOR_NAMES, neutral_values, rotor_values
from cfdbender.core.plan import workspace_path

FIXED = ("status", "classification", "iterations", "residual", "wall_time_s", "solver_version")
NOT_APPLICABLE = "NA"
#: The fixed context block of every table, as pyflightstream's CONTEXT_COLUMNS: the
#: flight condition, then the reference lengths. A value the point lacks is NA.
CONTEXT = (
    "ALPHA",
    "BETA",
    "MACH",
    "RE",
    "VINF",
    "VREF",
    "ALT",
    "RHO",
    "TEMP",
    "MU",
    "J",
    "J_CLOCK",
    "RPM_CLOCK",
    "SREF",
    "CREF",
    "BREF",
)
#: Matrix keys spelled as a context column, folded by case.
_ALIASES = {
    "alpha": "ALPHA",
    "beta": "BETA",
    "mach": "MACH",
    "remi": "RE",
    "tasmps": "VINF",
    "altft": "ALT",
    "rhokgm3": "RHO",
    "tk": "TEMP",
    "mupas": "MU",
    "advance_ratio": "J",
    "rpm": "RPM_CLOCK",
}


#: A key spelled as a context column fills that column; it is never a second column.
_CONTEXT_FOLDED = {name.casefold(): name for name in CONTEXT}


def plain_cell(text: str) -> str:
    """Return text as a cell no reader has to unquote.

    A comma becomes a semicolon, a double quote a single one and a line break a space.
    """
    return (
        text.replace(",", ";")
        .replace('"', "'")
        .replace(chr(13) + chr(10), " ")
        .replace(chr(13), " ")
        .replace(chr(10), " ")
    )


def _cell(value: object) -> str:
    if value is None:
        return NOT_APPLICABLE
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else NOT_APPLICABLE
    text = plain_cell(str(value))
    return text if text.strip() else NOT_APPLICABLE


def _wall(row: Mapping[str, object]) -> float | None:
    start, end = row.get("started_at"), row.get("completed_at")
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _context(
    condition: Mapping[str, object], references: Mapping[str, object] | None
) -> tuple[dict[str, object], dict[str, object]]:
    """Return one point's context cells and the condition keys they did not absorb."""
    cells: dict[str, object] = {}
    extra: dict[str, object] = {}
    for key, value in condition.items():
        folded = str(key).casefold()
        column = _ALIASES.get(folded) or _CONTEXT_FOLDED.get(folded)
        if column is None:
            extra[str(key)] = value
        else:
            cells[column] = value
    reference = references or {}
    cells["SREF"] = reference.get("area")
    cells["CREF"] = reference.get("length")
    numeric = {
        str(key): float(value)
        for key, value in condition.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    length = reference.get("length")
    try:
        state = resolve_flow(
            numeric,
            reference_length=float(length) if isinstance(length, int | float) else None,
            label="FLIGHT_CONDITION",
        )
    except (FlightConditionError, ValueError):
        return cells, extra
    cells.setdefault("MACH", state.mach)
    if "ALT" not in cells and state.density_source == "atmosphere":
        cells["ALT"] = 0.0
    cells["VINF"] = state.velocity
    cells["RHO"] = state.density
    cells["TEMP"] = state.temperature
    cells["MU"] = state.viscosity
    if state.reynolds is not None:
        cells["RE"] = state.reynolds / 1e6
    return cells, extra


def _cells(row: Mapping[str, object], solver: str | None = None) -> dict[str, object]:
    result = cast(Mapping[str, object] | None, row.get("result")) or {}
    history = cast(Sequence[Mapping[str, object]], result.get("history") or ())
    last = history[-1] if history else {}
    context, extra = _context(
        cast(Mapping[str, object], row["condition"]),
        cast(Mapping[str, object] | None, result.get("references")),
    )
    cells: dict[str, object] = {
        "POL": row["pol"],
        "point": str(row["point_id"]).split("/", 1)[1],
        **context,
        **extra,
        "status": row["status"],
        "classification": row.get("classification"),
        "iterations": last.get("iteration"),
        "residual": last.get("residual"),
        "wall_time_s": _wall(row),
        "solver_version": result.get("solver_version"),
    }
    coefficients = cast(Mapping[str, Mapping[str, object]], result.get("coefficients") or {})
    for name, value in coefficients.items():
        cells[name] = value["value"]
    own = {name: cast(float, value["value"]) for name, value in coefficients.items()}
    cells.update(
        neutral_values(
            solver,
            own,
            dimensions=result.get("dimensions"),
            condition=cast(Mapping[str, object], row["condition"]),
        )
    )
    cells.update(rotor_values(solver, own))
    return cells


def _header(rows: Sequence[Mapping[str, object]], solver: str | None = None) -> list[str]:
    extras: dict[str, None] = {}
    coefficients: dict[str, None] = {}
    for row in rows:
        condition = cast(Mapping[str, object], row["condition"])
        extras.update(dict.fromkeys(_context(condition, None)[1]))
        result = cast(Mapping[str, object] | None, row.get("result")) or {}
        names = cast(Mapping[str, object], result.get("coefficients") or {})
        coefficients.update(dict.fromkeys(names))
    rotor = tuple(c for c in ROTOR if c not in CONTEXT) if solver in ROTOR_NAMES else ()
    taken = {"POL", "point", *CONTEXT, *extras, *FIXED, *NEUTRAL, *rotor}
    return [
        "POL",
        "point",
        *CONTEXT,
        *extras,
        *FIXED,
        *NEUTRAL,
        *rotor,
        *(c for c in coefficients if c not in taken),
    ]


def _text(
    header: Sequence[str], rows: Sequence[Mapping[str, object]], solver: str | None = None
) -> str:
    lines = [",".join(header)]
    for cell in (_cells(row, solver) for row in rows):
        lines.append(",".join(_cell(cell.get(name)) for name in header))
    return chr(10).join(lines) + chr(10)


def _write(target: Path, text: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".collect-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def write_tables(
    root: Path,
    matrix: str,
    rows: Sequence[Mapping[str, object]],
    solver: str | None = None,
) -> tuple[Path, ...]:
    """Rebuild campaign_sweep.csv and polars/P<POL>.csv for one matrix.

    Parameters
    ----------
    root : Path
        Workspace root.
    matrix : str
        Recorded matrix path, inputs/matrices/<name>.fs.
    rows : sequence of mapping
        The campaign summary's point rows, in record order.
    solver : str, optional
        Registered solver name; it selects the neutral CL, CD and CM columns.

    Returns
    -------
    tuple of Path
        The written tables, campaign_sweep.csv first.
    """
    folder = workspace_path(root, f"post/{Path(matrix).stem}")
    header = _header(rows, solver)
    written = [_write(folder / "campaign_sweep.csv", _text(header, rows, solver))]
    by_pol: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_pol.setdefault(str(row["pol"]), []).append(row)
    for pol, members in by_pol.items():
        written.append(_write(folder / "polars" / f"P{pol}.csv", _text(header, members, solver)))
    return tuple(written)
