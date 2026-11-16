"""The canonical run block: the columns every polar, group and sections table opens with.

A reader holding one file can tell which run, point and condition each row is of, and
two files of one point join on their first columns. A column that does not apply to a
row is NA, never absent, so the schema is the same for every solver. Units and sources
are stated in ``products.json`` from :data:`RUN_BLOCK_SPEC`, not in the csv.

The block is fixed columns, one column per identifier of the row (in cfdbender, every
FLIGHT_CONDITION key that is not a block column, in the order written), and per-rotor
columns named by the rotor's integer id with no underscore (``RPM0``, ``D0``). The window
columns are NA on a steady row.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

NOT_APPLICABLE = "NA"
#: How :data:`RUN_BLOCK_SPEC` writes the place of the identifier columns.
IDENTIFIER_PLACEHOLDER = "<IDENTIFIER>"
_ROTOR = "<id>"
#: The run types a row names.
RUN_TYPES = ("steady", "unsteady")


@dataclass(frozen=True)
class RunColumn:
    """One column of the run block: header, unit and where the value comes from."""

    name: str
    unit: str
    source: str

    @property
    def per_rotor(self) -> bool:
        """Whether the column is repeated once per rotor."""
        return _ROTOR in self.name


RUN_BLOCK_SPEC: tuple[RunColumn, ...] = (
    RunColumn("POL", "-", "the matrix row's polar"),
    RunColumn("POINT_ID", "-", "1-based position of the point in its polar's sweep order"),
    RunColumn("POINT", "-", "the point name"),
    RunColumn(IDENTIFIER_PLACEHOLDER, "as written", "the row's other condition keys, in order"),
    RunColumn("RUN_TYPE", "-", "steady, or unsteady for a time-accurate solver"),
    RunColumn("ALPHA", "deg", "angle of attack as written"),
    RunColumn("BETA", "deg", "sideslip as written"),
    RunColumn("MACH", "-", "the point's Mach number"),
    RunColumn("RE", "-", "the point's Reynolds number, in millions"),
    RunColumn("VINF", "m/s", "free-stream speed of the point"),
    RunColumn("VREF", "m/s", "reference velocity of the coefficients, VINF"),
    RunColumn("RPM<id>", "rev/min", "speed of rotor <id>, signed about its declared axis"),
    RunColumn("J<id>", "-", "advance ratio rotor <id> ran at, VINF/(abs(n) D<id>)"),
    RunColumn("MTIP<id>", "-", "tip Mach number of rotor <id>"),
    RunColumn("MHEL<id>", "-", "helical tip Mach number of rotor <id>"),
    RunColumn("J", "-", "advance ratio the row requested"),
    RunColumn("J_CLOCK", "-", "advance ratio the clock rotor ran at"),
    RunColumn("RPM_CLOCK", "rev/min", "speed the row states for its rotor"),
    RunColumn("THRUST", "N", "thrust condition the row states"),
    RunColumn("ALT", "ft", "altitude: ALTFT of the row, 0 for the sea-level atmosphere"),
    RunColumn("RHO", "kg/m^3", "density of the point"),
    RunColumn("TEMP", "K", "temperature of the point"),
    RunColumn("MU", "Pa s", "dynamic viscosity, in scientific notation"),
    RunColumn("SREF", "m^2", "reference area"),
    RunColumn("CREF", "m", "reference chord"),
    RunColumn("BREF", "m", "reference span"),
    RunColumn("XMOM", "m", "moment reference point x, mesh frame"),
    RunColumn("YMOM", "m", "moment reference point y, mesh frame"),
    RunColumn("ZMOM", "m", "moment reference point z, mesh frame"),
    RunColumn("D<id>", "m", "diameter of rotor <id>"),
    RunColumn("ADISC<id>", "m^2", "disc area of rotor <id>, pi D^2/4"),
    RunColumn("PITCH<id>", "deg", "pitch of the frame R<id>"),
    RunColumn("TOE<id>", "deg", "toe of the frame R<id>"),
    RunColumn("FIRST_STEP", "-", "first step of the averaging window, 1-based"),
    RunColumn("LAST_STEP", "-", "last step of the averaging window, inclusive"),
    RunColumn("STEPS", "-", "steps in the averaging window"),
    RunColumn("WINDOW_REVS", "-", "clock rotor revolutions the window spans"),
    RunColumn("WINDOW_PASSAGES", "-", "clock rotor blade passages the window spans"),
)
_WINDOW = frozenset({"FIRST_STEP", "LAST_STEP", "STEPS", "WINDOW_REVS", "WINDOW_PASSAGES"})


@dataclass(frozen=True)
class RotorRun:
    """What one rotor contributes to a run block row.

    Attributes
    ----------
    rotor_id : int
        The rotor's id.
    diameter_m : float
        Its diameter.
    pitch_deg, toe_deg : float
        The pitch and toe of its frame R<id>.
    rpm : float or None
        The speed it turned at, signed about its declared axis; None when unknown.
    mtip, mhel : float or None
        Its tip and helical Mach numbers, None where not stated.
    """

    rotor_id: int
    diameter_m: float
    pitch_deg: float
    toe_deg: float
    rpm: float | None = None
    mtip: float | None = None
    mhel: float | None = None


def _expanded(identifiers: Sequence[str], rotor_ids: Sequence[int]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    index = 0
    while index < len(RUN_BLOCK_SPEC):
        spec = RUN_BLOCK_SPEC[index]
        if spec.name == IDENTIFIER_PLACEHOLDER:
            out.extend((str(name), spec.name) for name in identifiers)
            index += 1
        elif not spec.per_rotor:
            out.append((spec.name, spec.name))
            index += 1
        else:
            end = index
            while end < len(RUN_BLOCK_SPEC) and RUN_BLOCK_SPEC[end].per_rotor:
                end += 1
            for rotor in sorted(rotor_ids):
                out.extend(
                    (g.name.replace(_ROTOR, str(rotor)), g.name) for g in RUN_BLOCK_SPEC[index:end]
                )
            index = end
    return out


def run_block_columns(
    identifiers: Sequence[str] = (), rotor_ids: Sequence[int] = ()
) -> tuple[str, ...]:
    """Return the run block's header for the identifiers and rotor ids.

    Raises
    ------
    ValueError
        If an identifier spells a column of the block, or a rotor id repeats or is negative.
    """
    ids = [int(value) for value in rotor_ids]
    if any(value < 0 for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"rotor ids {ids} must be distinct integers of zero or more")
    headers = [header for header, _template in _expanded(identifiers, ids)]
    clash = sorted({str(name) for name in identifiers if headers.count(str(name)) > 1})
    if clash:
        raise ValueError(f"the identifiers {', '.join(clash)} spell run block columns")
    return tuple(headers)


def _value(value: object) -> object:
    if value is None or isinstance(value, bool):
        return NOT_APPLICABLE
    if isinstance(value, int | float):
        return value if math.isfinite(float(value)) else NOT_APPLICABLE
    text = str(value).strip()
    return text if text else NOT_APPLICABLE


def run_block_row(
    *,
    pol: str,
    point_id: int,
    point: str,
    run_type: str,
    condition: Mapping[str, object],
    identifiers: Mapping[str, object] | None = None,
    rotors: Sequence[RotorRun] = (),
) -> dict[str, object]:
    """Return one row of the run block, keyed by header, in block order.

    Parameters
    ----------
    pol, point : str
        The polar and the point name.
    point_id : int
        1-based position of the point in its polar.
    run_type : str
        One of :data:`RUN_TYPES`.
    condition : mapping
        The point's scalar columns (``ALPHA`` to ``ZMOM``), matched by name; a column it
        does not state is NA. A numeric ``MU`` is written in scientific notation.
    identifiers : mapping, optional
        The row's identifier pairs, in order.
    rotors : sequence of RotorRun, optional
        Every rotor of the row's solver.

    Raises
    ------
    ValueError
        If the run type is unknown or the point id is not a positive integer.
    """
    if run_type not in RUN_TYPES:
        raise ValueError(f"run type {run_type!r} is not one of {', '.join(RUN_TYPES)}")
    if isinstance(point_id, bool) or not isinstance(point_id, int) or point_id < 1:
        raise ValueError(f"POINT_ID {point_id!r} must be a 1-based integer")
    stated = dict(identifiers or {})
    by_id = {int(rotor.rotor_id): rotor for rotor in rotors}
    fixed = {"POL": pol, "POINT_ID": point_id, "POINT": point, "RUN_TYPE": run_type}
    vinf = condition.get("VINF")
    per_rotor: dict[int, dict[str, object]] = {}
    for rotor_id, rotor in by_id.items():
        diameter = float(rotor.diameter_m)
        if not math.isfinite(diameter) or diameter <= 0.0:
            diameter = math.nan
        advance: object = None
        if isinstance(rotor.rpm, int | float) and isinstance(vinf, int | float) and rotor.rpm:
            advance = float(vinf) / (abs(float(rotor.rpm)) / 60.0 * diameter)
        per_rotor[rotor_id] = {
            "RPM<id>": rotor.rpm,
            "J<id>": advance,
            "MTIP<id>": rotor.mtip,
            "MHEL<id>": rotor.mhel,
            "D<id>": diameter,
            "ADISC<id>": math.pi * diameter**2 / 4.0,
            "PITCH<id>": rotor.pitch_deg,
            "TOE<id>": rotor.toe_deg,
        }
    row: dict[str, object] = {}
    for header, template in _expanded(tuple(stated), sorted(by_id)):
        value: object
        if template == IDENTIFIER_PLACEHOLDER:
            value = stated[header]
        elif _ROTOR in template:
            value = per_rotor[int(header[len(template.split(_ROTOR)[0]) :])][template]
        elif header in fixed:
            value = fixed[header]
        elif header in _WINDOW and run_type == "steady":
            value = None
        else:
            value = condition.get(header)
        if header == "MU" and isinstance(value, int | float) and not isinstance(value, bool):
            value = f"{float(value):.5e}" if math.isfinite(float(value)) else None
        row[header] = _value(value)
    return row
