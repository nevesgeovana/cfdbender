"""Read neutral records from the public ledger and written polar tables.

Usage: ``read_workspace(root, group="1")[(sim, point)].result``.
No product is regenerated. Body coefficients retain upstream names and signs:
drag and lift oppose positive body x and z. Reference length is the chord;
roll and yaw moments use the separately retained reference span. History holds
only the final iteration evidence recorded by upstream, not an invented trace.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cfdbender import _pyfs
from cfdbender.core.results import Classification, Coefficient, Iteration, NeutralResult, References

# Kept explicit: the upstream enum inventory test must force a mapping decision
# when a release adds a status. Process exit codes never participate.
STATUS_CLASSIFICATIONS: dict[str, Classification | None] = {
    "CONVERGED": "converged",
    "COMPLETED_MAX_ITER": "iteration_limit",
    "MAX_ITER_REACHED": "iteration_limit",
    "FAILED_EXECUTION": "failed",
    "FAILED_SCRIPT": "failed",
    "FAILED_INCOMPLETE_OUTPUT": "failed",
    "FAILED_DIVERGED": "failed",
    "WALLTIME_REACHED": "failed",
    "SUBMITTED": None,
    "FAILED_MARKED": "failed",
    "RAN_MISSING_LOG": "failed",
}

_BODY_COLUMNS: dict[str, tuple[Literal["x", "y", "z"], Literal["force", "moment"]]] = {
    "CDB": ("x", "force"),
    "CYB": ("y", "force"),
    "CLB": ("z", "force"),
    "CRB25": ("x", "moment"),
    "CMB25": ("y", "moment"),
    "CNB25": ("z", "moment"),
}


@dataclass(frozen=True)
class FlightStreamPoint:
    """One effective point and its neutral result, absent while submitted.

    Failed points can have no result when no reference block was recorded.
    ``reference_span`` retains BREF because NeutralResult's one length is CREF.
    """

    status: str
    classification: Classification | None
    result: NeutralResult | None
    reference_span: float | None


def _references(record: dict[str, Any]) -> References:
    ref = record.get("reference")
    if not isinstance(ref, dict):
        raise ValueError(f"{record['run_id']}: missing recorded reference block")
    try:
        return References(
            area=float(ref["SREF"]),
            length=float(ref["CREF"]),
            moment_origin=(float(ref["XMOM"]), float(ref["YMOM"]), float(ref["ZMOM"])),
            axis_system="body",
            axis_directions=("forward", "right", "down"),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{record['run_id']}: incomplete recorded reference block") from error


def _polars(root: Path, matrix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    # Only original polar tables. SUPER, trimmed, interpolated, merged and
    # installed-frame products must not introduce duplicate or altered rows.
    for path in sorted((root / "post" / matrix / "polars").glob("*.csv")):
        if path.name.startswith("P") and re.search(r"(?:_g\d+|_uns_avg)\.csv$", path.name):
            with path.open(encoding="utf-8", newline="") as stream:
                rows.extend(csv.DictReader(stream))
    return rows


def _coefficients(
    rows: list[dict[str, str]], record: dict[str, Any], group: str, *, required: bool
) -> dict[str, Coefficient]:
    state = record.get("point") or {}
    matches = []
    for row in rows:
        if row.get("POL") != str(record["sim_id"]):
            continue
        steady = "GROUP" in row
        if steady and row["GROUP"].lstrip("0") != group.lstrip("0"):
            continue
        if not steady and not any(f"{name}_{group}" in row for name in _BODY_COLUMNS):
            continue
        # All state columns carried by the point must match. A missing column
        # cannot silently pair two different points by their order in the file.
        if all(
            key.upper() in row
            and math.isclose(float(row[key.upper()]), float(value), rel_tol=1e-7, abs_tol=1e-9)
            for key, value in state.items()
        ):
            matches.append(row)
    if not matches and not required:
        return {}
    if len(matches) != 1:
        raise ValueError(
            f"{record['run_id']}: expected one polar row for group {group}, found {len(matches)}"
        )
    row = matches[0]
    return {
        column: Coefficient(float(row[column]), axis, kind)
        for name, (axis, kind) in _BODY_COLUMNS.items()
        for column in [name if "GROUP" in row else f"{name}_{group}"]
        if row.get(column) not in (None, "", "NA", "nan")
    }


def read_workspace(
    root: str | Path, *, group: str = "1"
) -> dict[tuple[str, str], FlightStreamPoint]:
    """Read one neutral record per effective point from a FlightStream workspace.

    Parameters
    ----------
    root : str or Path
        Workspace containing upstream runs.json and post products.
    group : str, optional
        Explicit steady boundary group ("1"), or unsteady plot group ("MRP_TOTAL").

    Returns
    -------
    dict
        (simulation id, point name) to status and NeutralResult. SUBMITTED has
        classification and result None because the core has no pending value.
        Failed points without recorded references have result None.

    Raises
    ------
    ValueError
        Unreadable records, unknown status, or absent or ambiguous numerical
        evidence for a completed point. Coefficients are never recomputed.
    """
    base = Path(root)
    ledger = _pyfs.read_ledger(base)
    if ledger.unreadable:
        raise ValueError(f"unreadable ledger records: {ledger.unreadable}")
    result = {}
    products: dict[str, list[dict[str, str]]] = {}
    for point in ledger.points():
        if point["run_id"] is None:
            continue  # Planned points have no run record to translate.
        record = ledger.card(point["run_id"])
        if record is None:
            raise ValueError(f"missing ledger card for {point['run_id']}")
        status = str(record["status"])
        if status not in STATUS_CLASSIFICATIONS:
            raise ValueError(f"unmapped FlightStream status: {status}")
        classification = STATUS_CLASSIFICATIONS[status]
        neutral = None
        ref = record.get("reference") or {}
        span = float(ref["BREF"]) if "BREF" in ref else None
        if classification is not None and (ref or classification != "failed"):
            references = _references(record)
            matrix = str(point["matrix"])
            if matrix not in products:
                products[matrix] = _polars(base, matrix)
            coefficients = _coefficients(
                products[matrix], record, group, required=classification != "failed"
            )
            iterations, residual = record.get("iterations"), record.get("residual")
            history: tuple[Iteration, ...] = ()
            if iterations is not None and residual is not None:
                history = (Iteration(int(iterations), float(residual), status == "CONVERGED"),)
            neutral = NeutralResult(
                coefficients=coefficients,
                references=references,
                history=history,
                classification=classification,
                solver_version=record.get("fs_build") or record["fs_version_requested"],
            )
        result[(str(point["sim"]), str(point["point"]))] = FlightStreamPoint(
            status, classification, neutral, span
        )
    return result
