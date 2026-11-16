"""Write the family and group polar files of one matrix: DIM, ACOEF and RCOEF per polar.

For every polar ``<POL>`` of the matrix, ``post/<matrix>/polars/`` holds
``FAM-DIM-P<POL>.csv``, ``FAM-ACOEF-P<POL>.csv`` and, when the solver states a rotor,
``FAM-RCOEF-P<POL>.csv`` (one row per point and family), then one file set
``P<POL>-<GROUP>-<KIND>.csv`` per group (one row per point). Every file opens with the run
block of :mod:`cfdbender.matrix._run_block`; the columns after it are those of
:mod:`cfdbender.matrix._family_columns`.

DIM states the force and the moment about the moment reference point in the geometry frame
(x aft, y right, z up): an airframe solver's coefficient vectors
(:func:`cfdbender.core.neutral.airframe_load_coefficients`) times ``q SREF`` and
``q SREF CREF`` with ``q = 0.5 RHO VINF^2``; a rotor solver's thrust and torque
(:func:`cfdbender.core.neutral.rotor_loads`), its hub at the moment reference point. ACOEF
turns them into the frames W, S, B and R<id> (:mod:`cfdbender.core.axes`) and then carries
the solver's own coefficients under its own names; RCOEF forms the rotor coefficients
(:mod:`cfdbender.core.rotor_coefficients`). A group's values are formed from the sum of its
members' dimensional loads, never by summing coefficients. A value that does not exist is
NA; a point with no readable result is a row of NA loads.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from cfdbender.core.axes import (
    RotorFrame,
    airframe_axis_coefficients,
    frame_rotations,
    rotor_frame,
)
from cfdbender.core.neutral import airframe_load_coefficients, neutral_values, rotor_loads
from cfdbender.core.rotor_coefficients import rotor_frame_coefficients
from cfdbender.matrix._family_columns import (
    CONVERGENCE_COLUMNS,
    SOLVER_ROTORS,
    ColumnSpec,
    Family,
    Group,
    Rotor,
    acoef_columns,
    column_names,
    dim_columns,
    family_file_name,
    group_file_name,
    polar_groups,
    rcoef_columns,
    solver_families,
)
from cfdbender.matrix._run_block import RotorRun, run_block_columns, run_block_row

Vector = tuple[float, float, float]
NAN = float("nan")
#: Solvers whose points are time-accurate: their rows are of run type unsteady.
UNSTEADY_SOLVERS = frozenset({"pyfr"})


@dataclass(frozen=True)
class Point:
    """What one recorded point contributes to the polar files.

    Attributes
    ----------
    pol, name : str
        Its polar and point name.
    condition : mapping
        The run block's scalar columns (ALPHA to ZMOM).
    identifiers : mapping
        The other condition keys, in order.
    native : mapping
        The solver's own coefficients.
    loads : mapping
        Family id to ``(force, moment about the moment point)`` in the geometry frame.
    rpm : mapping
        Rotor id to its speed signed about its declared axis, or None.
    diameter : mapping
        Rotor id to its diameter.
    convergence : mapping
        Solver version and the per-point polar convergence rule's cells, when judged.
    """

    pol: str
    name: str
    condition: Mapping[str, object]
    identifiers: Mapping[str, object]
    native: Mapping[str, float]
    loads: Mapping[int, tuple[Vector, Vector]]
    rpm: Mapping[int, float | None] = field(default_factory=dict)
    diameter: Mapping[int, float] = field(default_factory=dict)
    in_units: bool = False
    convergence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PolarFiles:
    """The polar files of one matrix and what products.json states about them."""

    paths: tuple[Path, ...]
    families: tuple[Family, ...]
    groups: tuple[Group, ...]
    skipped: tuple[tuple[str, str], ...]
    rotors: tuple[Rotor, ...]
    frames: Mapping[int, RotorFrame]
    headers: Mapping[str, tuple[ColumnSpec, ...]]
    run_block: tuple[str, ...]
    run_types: Mapping[str, str]
    identifiers: tuple[str, ...] = ()
    sections: tuple[Path, ...] = ()
    sections_columns: tuple[ColumnSpec, ...] | None = None
    sections_skipped: tuple[tuple[str, str], ...] = ()
    mesh_frames: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scaled(vector: Vector, scale: float | None) -> Vector:
    if scale is None:
        return (NAN, NAN, NAN)
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def point_loads(
    solver: str | None,
    native: Mapping[str, float],
    condition: Mapping[str, object],
    *,
    dimensions: object = None,
    families: Sequence[Family] = (),
) -> tuple[dict[int, tuple[Vector, Vector]], bool]:
    """Return each family's force and moment about the moment point, and whether in units.

    The airframe loads are dimensional when RHO, VINF and SREF give a positive ``q SREF``
    (the moment needs CREF too); otherwise they are returned in units of ``q SREF`` (force)
    and ``q SREF CREF`` (moment) with the flag True, so the coefficients are still formed
    while the dimensional loads are NA. A rotor solver's loads are dimensional already.
    """
    family = (families or solver_families(solver))[0]
    rotor = rotor_loads(solver, native)
    if rotor is not None:
        return {family.family_id: rotor}, False
    unknown = ((NAN, NAN, NAN), (NAN, NAN, NAN))
    vectors = airframe_load_coefficients(solver, native, dimensions=dimensions, condition=condition)
    if vectors is None:
        return {family.family_id: unknown}, False
    rho, vinf = _number(condition.get("RHO")), _number(condition.get("VINF"))
    sref, cref = _number(condition.get("SREF")), _number(condition.get("CREF"))
    q_area = None if rho is None or vinf is None or sref is None else 0.5 * rho * vinf**2 * sref
    if q_area is None or not math.isfinite(q_area) or q_area <= 0.0:
        return {family.family_id: vectors}, True
    force = _scaled(vectors[0], q_area)
    moment = _scaled(vectors[1], None if cref is None else q_area * cref)
    return {family.family_id: (force, moment)}, False


class _Rows:
    """Collect the rows of one polar's files, keyed by file name."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, object]]] = {}

    def add(self, name: str, row: dict[str, object]) -> None:
        self.rows.setdefault(name, []).append(row)


def _dim(force: Vector, moment: Vector) -> dict[str, object]:
    return {
        "FX_MRP": force[0],
        "FY_MRP": force[1],
        "FZ_MRP": force[2],
        "MX_MRP": moment[0],
        "MY_MRP": moment[1],
        "MZ_MRP": moment[2],
    }


def polar_files(
    folder: Path,
    points: Sequence[Point],
    solver: str | None,
    *,
    write: Callable[[Path, str], Path],
    cell: Callable[[object], str],
    families: Sequence[Family] | None = None,
    judged: bool = False,
) -> PolarFiles:
    """Write every polar's family and group files under ``folder/polars``.

    Parameters
    ----------
    folder : Path
        ``post/<matrix>``.
    points : sequence of Point
        Every recorded point, in record order.
    solver : str or None
        The workspace's solver.
    write : callable
        Writes text to a path atomically and returns the path.
    cell : callable
        Turns one value into its csv cell (NA where it does not exist).
    families : sequence of Family, optional
        The families the points' loads are keyed by, off any rotor; the solver's own
        families when omitted.
    judged : bool, optional
        Carry the solver version and polar rule columns in every family and group table.

    Returns
    -------
    PolarFiles
        The written paths and the statement products.json carries.
    """
    if families is None:
        families = solver_families(solver)
        rotors = SOLVER_ROTORS.get(solver or "", ())
    else:
        rotors = ()
    frames = {r.rotor_id: rotor_frame(r.shaft, r.rotor_id) for r in rotors}
    groups, skipped = polar_groups(families, rotors)
    rotor_ids = tuple(sorted(frames))
    identifiers: dict[str, None] = {}
    own: dict[str, None] = {}
    for point in points:
        identifiers.update(dict.fromkeys(point.identifiers))
        own.update(dict.fromkeys(point.native))
    block = run_block_columns(tuple(identifiers), rotor_ids)
    # A solver coefficient spelled as a run block column (QPROP's J) is stated there.
    own = {name: None for name in own if name not in block}
    run_type = "unsteady" if solver in UNSTEADY_SOLVERS else "steady"
    headers: dict[str, tuple[ColumnSpec, ...]] = {
        "FAM-DIM": dim_columns(),
        "FAM-ACOEF": acoef_columns(rotor_ids, tuple(own)),
        "DIM": dim_columns("group"),
        "ACOEF": acoef_columns(rotor_ids, tuple(own), "group"),
    }
    if rotor_ids:
        headers["FAM-RCOEF"] = rcoef_columns(rotor_ids)
        headers["RCOEF"] = rcoef_columns(rotor_ids, "group")
    if judged:
        headers = {kind: (*columns, *CONVERGENCE_COLUMNS) for kind, columns in headers.items()}
    by_pol: dict[str, list[Point]] = {}
    for point in points:
        by_pol.setdefault(point.pol, []).append(point)
    written: list[Path] = []
    for pol, members in by_pol.items():
        rows = _Rows()
        for index, point in enumerate(members, start=1):
            _point_rows(
                rows, pol, index, point, run_type, identifiers, families, groups, frames, solver
            )
        names = [family_file_name(pol, kind) for kind in ("DIM", "ACOEF", "RCOEF")]
        kinds = {names[0]: "FAM-DIM", names[1]: "FAM-ACOEF", names[2]: "FAM-RCOEF"}
        for group in groups:
            for kind in ("DIM", "ACOEF", "RCOEF"):
                kinds[group_file_name(pol, group.name, kind)] = kind
        for name, kind in kinds.items():
            if name not in rows.rows:
                continue
            header = [*block, *column_names(headers[kind])]
            lines = [",".join(header)]
            lines += [",".join(cell(row.get(h)) for h in header) for row in rows.rows[name]]
            written.append(write(folder / "polars" / name, chr(10).join(lines) + chr(10)))
    return PolarFiles(
        tuple(written),
        tuple(families),
        groups,
        skipped,
        rotors,
        frames,
        headers,
        block,
        {pol: run_type for pol in by_pol},
        tuple(identifiers),
    )


def point_block(
    point: Point,
    index: int,
    run_type: str,
    identifiers: Sequence[str] | Mapping[str, None],
    frames: Mapping[int, RotorFrame],
) -> dict[str, object]:
    """Return the run block row of one point, the ``index``-th of its polar."""
    runs = [
        RotorRun(
            rotor_id,
            point.diameter.get(rotor_id, NAN),
            frame.pitch_deg,
            frame.toe_deg,
            rpm=point.rpm.get(rotor_id),
        )
        for rotor_id, frame in sorted(frames.items())
    ]
    return run_block_row(
        pol=point.pol,
        point_id=index,
        point=point.name,
        run_type=run_type,
        condition=point.condition,
        identifiers={name: point.identifiers.get(name) for name in identifiers},
        rotors=runs,
    )


def _point_rows(
    rows: _Rows,
    pol: str,
    index: int,
    point: Point,
    run_type: str,
    identifiers: Mapping[str, None],
    families: Sequence[Family],
    groups: Sequence[Group],
    frames: Mapping[int, RotorFrame],
    solver: str | None,
) -> None:
    condition = point.condition
    block = point_block(point, index, run_type, identifiers, frames)
    alpha = _number(condition.get("ALPHA")) or 0.0
    beta = _number(condition.get("BETA")) or 0.0
    rotations = frame_rotations(alpha, beta, list(frames.values()))
    rho, vinf = _number(condition.get("RHO")), _number(condition.get("VINF"))
    sref, cref, bref = (_number(condition.get(k)) for k in ("SREF", "CREF", "BREF"))
    q_ref = None if rho is None or vinf is None else 0.5 * rho * vinf * vinf
    unknown: tuple[Vector, Vector] = ((NAN, NAN, NAN), (NAN, NAN, NAN))

    def values(force: Vector, moment: Vector, rotor_id: int | None, kind: str) -> dict[str, object]:
        if kind == "DIM":
            return _dim(*unknown) if point.in_units else _dim(force, moment)
        if kind == "ACOEF" and point.in_units:
            # Loads in units of q SREF and q SREF CREF: the coefficients need no pressure.
            span = None if bref is None or cref is None else bref / cref
            return dict(
                airframe_axis_coefficients(
                    force, moment, rotations, q_ref_pa=1.0, sref_m2=1.0, cref_m=1.0, bref_m=span
                )
            )
        if kind == "ACOEF":
            return dict(
                airframe_axis_coefficients(
                    force,
                    moment,
                    rotations,
                    q_ref_pa=q_ref,
                    sref_m2=sref,
                    cref_m=cref,
                    bref_m=bref,
                )
            )
        frame = frames[cast(int, rotor_id)]
        return {
            "ROTOR_ID": rotor_id,
            **rotor_frame_coefficients(
                force_n=force,
                # A rotor solver's hub is the moment reference point.
                moment_hub_nm=moment,
                frame=frame,
                rotations=rotations,
                rpm=point.rpm.get(frame.rotor_id),
                diameter_m=point.diameter.get(frame.rotor_id, NAN),
                density_kg_m3=rho,
                vinf_m_s=vinf,
                sref_m2=sref,
                cref_m=cref,
                bref_m=bref,
            ),
        }

    every = {f.family_id for f in families}

    def retain_wind(row: dict[str, object]) -> None:
        # A missing wind-axis side force can make every reconstructed geometry
        # component unknown. The reported wind lift and drag remain known.
        if solver in (None, "flightstream", "qprop"):
            return
        native = neutral_values(solver, point.native, condition=condition)
        for stem in ("CD", "CL"):
            if stem in native and row.get(f"{stem}W") == "NA":
                row[f"{stem}W"] = native[stem]

    for family in families:
        force, moment = point.loads.get(family.family_id, unknown)
        identity = {"FAMILY": family.name, "FAMILY_ID": family.family_id}
        for kind in ("DIM", "ACOEF", "RCOEF"):
            if kind == "RCOEF" and family.rotor_id is None:
                continue
            row = {
                **block,
                **identity,
                **values(force, moment, family.rotor_id, kind),
                **point.convergence,
            }
            if kind == "ACOEF" and set(every) == {family.family_id}:
                retain_wind(row)
                row = {**{k: v for k, v in point.native.items()}, **row}
            rows.add(family_file_name(pol, kind), row)
    for group in groups:
        total: Vector = (0.0, 0.0, 0.0)
        about: Vector = (0.0, 0.0, 0.0)
        for member in group.members:
            f, m = point.loads.get(member, unknown)
            total, about = _add(total, f), _add(about, m)
        identity = {"GROUP": group.name, "GROUP_ID": group.group_id}
        for kind in ("DIM", "ACOEF", "RCOEF"):
            if kind == "RCOEF" and group.rotor_id is None:
                continue
            row = {
                **block,
                **identity,
                **values(total, about, group.rotor_id, kind),
                **point.convergence,
            }
            if kind == "ACOEF" and set(group.members) == every:
                if len(every) == 1:
                    retain_wind(row)
                row = {**{k: v for k, v in point.native.items()}, **row}
            rows.add(group_file_name(pol, group.name, kind), row)
