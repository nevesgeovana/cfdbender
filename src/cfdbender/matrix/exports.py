"""Write the CSV tables of a collected matrix beside its campaign summary.

Every polar of the matrix gets its family and group files under ``post/<matrix>/polars/``
(:mod:`cfdbender.matrix._family_polar`) and every point whose solver prints a sectional
distribution its ``sections/P<POL>/<point>_sections.csv``
(:mod:`cfdbender.matrix._section_tables`).
This module turns a campaign summary row into the point those writers read: the flight
condition resolved by the rules that prepare the solver input, the references, the
solver's own coefficients and the loads in the geometry frame. A value a point does not
have is the token NA; no cell holds a comma, a double quote or a line break and none is
quoted, so numpy.genfromtxt reads every row to the header's count. The tables are derived
and rebuilt by every collect. The campaign sweep table of 0.5 is retired (removed in 0.6.0).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from cfdbender.core.convergence import Verdict
from cfdbender.core.flight import FlightConditionError, resolve_flow
from cfdbender.core.plan import workspace_path
from cfdbender.matrix._family_columns import SOLVER_ROTORS
from cfdbender.matrix._family_polar import Point, PolarFiles, point_loads, polar_files
from cfdbender.matrix._rotor_table import rotor_table
from cfdbender.matrix._section_tables import section_files
from cfdbender.matrix.polar2d import verdict_cells

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
    "v": "VINF",
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


def _replaceable(target: Path, data: bytes) -> bool:
    """Require a matching recorded digest before replacing a derived table."""
    if target.is_symlink():
        raise ValueError(f"{target}: linked output; refusing overwrite")
    if not target.exists():
        return False
    original = target.read_bytes()
    if original == data:
        return True
    for folder in target.parents:
        if folder.parent.name != "post":
            continue
        manifest = folder / "products.json"
        relative = target.relative_to(folder.parent.parent).as_posix()
        try:
            document = json.loads(manifest.read_bytes())
            entries = document.get("products", [])
            if isinstance(entries, list) and any(
                isinstance(entry, dict)
                and entry.get("path") == relative
                and entry.get("sha256") == hashlib.sha256(original).hexdigest()
                for entry in entries
            ):
                return True
        except (OSError, ValueError, AttributeError):
            pass
        break
    raise ValueError(
        f"{target}: unrecorded or edited output; refusing overwrite; "
        "move it aside and collect again"
    )


def _write(target: Path, text: str) -> Path:
    data = text.encode("utf-8")
    replacing = _replaceable(target, data)
    if replacing and target.read_bytes() == data:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".collect-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replacing:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def summary_point(row: Mapping[str, object], solver: str | None = None) -> Point:
    """Return what one campaign summary row contributes to the polar files.

    The run block takes the resolved flight condition, VREF = VINF, the references (SREF,
    CREF, the moment point; BREF is NA, no solver states a span) and every other condition
    key as an identifier. A rotor's diameter is the reference length; its speed is the
    row's RPM, else ``60 VINF / (J D)`` from the J the solver reports.
    """
    result = cast(Mapping[str, object] | None, row.get("result")) or {}
    references = cast(Mapping[str, object], result.get("references") or {})
    context, extra = _context(cast(Mapping[str, object], row["condition"]), references)
    origin = cast(Sequence[object], references.get("moment_origin") or (None, None, None))
    condition = {
        **context,
        "VREF": context.get("VINF"),
        "XMOM": origin[0],
        "YMOM": origin[1],
        "ZMOM": origin[2],
    }
    coefficients = cast(Mapping[str, Mapping[str, object]], result.get("coefficients") or {})
    native = {name: cast(float, value["value"]) for name, value in coefficients.items()}
    rpm: dict[int, float | None] = {}
    diameter: dict[int, float] = {}
    for rotor in SOLVER_ROTORS.get(solver or "", ()):
        length = references.get("length")
        size = float(length) if isinstance(length, int | float) else math.nan
        diameter[rotor.rotor_id] = size
        stated = context.get("RPM_CLOCK")
        speed = float(stated) if isinstance(stated, int | float) else None
        vinf, advance = context.get("VINF"), native.get("J")
        if speed is None and isinstance(vinf, int | float) and advance and size > 0.0:
            speed = 60.0 * float(vinf) / (float(advance) * size)
        rpm[rotor.rotor_id] = speed
    loads, in_units = point_loads(solver, native, condition, dimensions=result.get("dimensions"))
    return Point(
        pol=str(row["pol"]),
        name=str(row["point_id"]).split("/", 1)[1],
        condition=condition,
        identifiers=extra,
        native=native,
        loads=loads,
        rpm=rpm,
        diameter=diameter,
        in_units=in_units,
    )


def write_polar_files(
    root: Path,
    matrix: str,
    rows: Sequence[Mapping[str, object]],
    solver: str | None = None,
    verdicts: Mapping[str, Verdict | str] | None = None,
) -> PolarFiles:
    """Rebuild the polar files and the sections tables of one matrix.

    See :mod:`cfdbender.matrix._family_polar` and :mod:`cfdbender.matrix._section_tables`;
    the returned paths are the polar files, then the sections tables.
    """
    folder = workspace_path(root, f"post/{Path(matrix).stem}")
    points = [summary_point(row, solver) for row in rows]
    if verdicts is not None:
        points = [
            replace(
                point,
                convergence={
                    "solver_version": (cast(Mapping[str, object], row.get("result") or {})).get(
                        "solver_version"
                    ),
                    **verdict_cells(verdicts.get(str(row["point_id"]))),
                },
            )
            for row, point in zip(rows, points, strict=True)
        ]
    polar = polar_files(
        folder, points, solver, write=_write, cell=_cell, judged=verdicts is not None
    )
    sections, columns, skipped = section_files(
        root, folder, rows, points, polar, solver, write=_write, cell=_cell
    )
    rotors = rotor_table(folder, rows, write=_write, cell=_cell)
    return replace(
        polar,
        paths=(*polar.paths, *sections, *rotors),
        sections=sections,
        sections_columns=columns,
        sections_skipped=skipped,
        mesh_frames=_mesh_frames(root, rows),
    )


def _mesh_frames(
    root: Path, rows: Sequence[Mapping[str, object]]
) -> dict[str, Mapping[str, object]]:
    """Read mesh conversion records from the verified section evidence of each point."""
    frames: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if row.get("result") is None:
            continue
        outputs = cast(Mapping[str, object], row.get("outputs_sha256") or {})
        for name in outputs:
            if not name.endswith("/sections.json"):
                continue
            path = workspace_path(root, name)
            if path.is_file():
                data = json.loads(path.read_bytes())
                if isinstance(data.get("mesh_frame"), dict):
                    frames[str(row["point_id"])] = data["mesh_frame"]
    return frames


def write_tables(
    root: Path,
    matrix: str,
    rows: Sequence[Mapping[str, object]],
    solver: str | None = None,
    verdicts: Mapping[str, Verdict | str] | None = None,
) -> tuple[Path, ...]:
    """Rebuild the polar files and the sections tables of one matrix.

    campaign_sweep.csv is retired (removed in 0.6.0): each polar's TOTAL group file holds
    every point of the polar, and joining several polars is a concatenation of those files.

    Parameters
    ----------
    root : Path
        Workspace root.
    matrix : str
        Recorded matrix path, inputs/matrices/<name>.fs.
    rows : sequence of mapping
        The campaign summary's point rows, in record order.
    solver : str, optional
        Registered solver name; it selects the loads, the families and the rotors.
    verdicts : mapping, optional
        Per-point SU2 or OpenFOAM convergence rule verdicts. Tables carry solver_version
        followed by rule_verdict, rule_iteration and rule_drop_orders.

    Returns
    -------
    tuple of Path
        The written tables.
    """
    return write_polar_files(root, matrix, rows, solver, verdicts).paths
