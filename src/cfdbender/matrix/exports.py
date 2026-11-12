"""Write the CSV tables of a collected matrix beside its campaign summary.

``post/<matrix>/campaign_sweep.csv`` holds one row per recorded point of the
matrix and ``post/<matrix>/polars/P<POL>.csv`` one row per point of each
simulation, in the pyflightstream spelling of a polar file's ``P<sim>`` stem
(cfdbender has no boundary groups, so the stem carries no group suffix). The
columns are POL, point, the FLIGHT_CONDITION keys in the order the matrix
states them, status, classification, iterations, residual, wall_time_s,
solver_version and every coefficient the solver reported. A value a point does
not have is an empty cell; every table of a matrix has the same columns. The
tables are derived and rebuilt by every collect.
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from cfdbender.core.plan import workspace_path

FIXED = ("status", "classification", "iterations", "residual", "wall_time_s", "solver_version")


def _wall(row: Mapping[str, object]) -> float | None:
    start, end = row.get("started_at"), row.get("completed_at")
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _cells(row: Mapping[str, object]) -> dict[str, object]:
    result = cast(Mapping[str, object] | None, row.get("result")) or {}
    history = cast(Sequence[Mapping[str, object]], result.get("history") or ())
    last = history[-1] if history else {}
    cells: dict[str, object] = {
        "POL": row["pol"],
        "point": str(row["point_id"]).split("/", 1)[1],
        **cast(Mapping[str, object], row["condition"]),
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
    return cells


def _header(rows: Sequence[Mapping[str, object]]) -> list[str]:
    conditions: dict[str, None] = {}
    coefficients: dict[str, None] = {}
    for row in rows:
        conditions.update(dict.fromkeys(cast(Mapping[str, object], row["condition"])))
        result = cast(Mapping[str, object] | None, row.get("result")) or {}
        names = cast(Mapping[str, object], result.get("coefficients") or {})
        coefficients.update(dict.fromkeys(names))
    return ["POL", "point", *conditions, *FIXED, *(c for c in coefficients if c not in conditions)]


def _text(header: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for cell in (_cells(row) for row in rows):
        writer.writerow(["" if cell.get(name) is None else cell[name] for name in header])
    return stream.getvalue()


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


def write_tables(root: Path, matrix: str, rows: Sequence[Mapping[str, object]]) -> tuple[Path, ...]:
    """Rebuild campaign_sweep.csv and polars/P<POL>.csv for one matrix.

    Parameters
    ----------
    root : Path
        Workspace root.
    matrix : str
        Recorded matrix path, inputs/matrices/<name>.fs.
    rows : sequence of mapping
        The campaign summary's point rows, in record order.

    Returns
    -------
    tuple of Path
        The written tables, campaign_sweep.csv first.
    """
    folder = workspace_path(root, f"post/{Path(matrix).stem}")
    header = _header(rows)
    written = [_write(folder / "campaign_sweep.csv", _text(header, rows))]
    by_pol: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_pol.setdefault(str(row["pol"]), []).append(row)
    for pol, members in by_pol.items():
        written.append(_write(folder / "polars" / f"P{pol}.csv", _text(header, members)))
    return tuple(written)
