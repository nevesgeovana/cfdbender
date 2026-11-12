"""Read named pipe-delimited matrices without any solver dependency.

Required columns are POL, RUN, SET, REF and PPROC. The optional columns are
those of a pyflightstream matrix, with the same names and meanings: HIDDEN,
AIRCRAFT, CONFIGURATION, DESCRIPTION, FLIGHT_CONDITION, SWEEP_VALUES, GEOMETRY,
SYMMETRY, SYMMETRY_LOADS, NCPUS, WALLTIME, SOLVER_BUILD (FS_BUILD is accepted as
its pyflightstream name), WORKFLOW and VAR_NAMES_VALUES. A cell holding ``-``
states nothing. FLIGHT_CONDITION holds comma-separated KEY:number pairs and at
most one KEY:sweep; its values come from SWEEP_VALUES. Known keys match
case-insensitively and keep their pyflightstream spelling (MACH, REmi, TK,
ALPHA, ...); other keys are kept in upper case. VAR_NAMES_VALUES holds
``/``-separated ``KEY: value`` pairs that override keys of the row's setup
file; a value is read as a TOML value when it is one, otherwise as text.
"""

from __future__ import annotations

import difflib
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from cfdbender.core.flight import canonical_key
from cfdbender.core.settings import WorkspaceError

REQUIRED_COLUMNS = ("POL", "RUN", "SET", "REF", "PPROC")
OPTIONAL_COLUMNS = (
    "HIDDEN",
    "AIRCRAFT",
    "CONFIGURATION",
    "DESCRIPTION",
    "FLIGHT_CONDITION",
    "SWEEP_VALUES",
    "GEOMETRY",
    "SYMMETRY",
    "SYMMETRY_LOADS",
    "NCPUS",
    "WALLTIME",
    "SOLVER_BUILD",
    "FS_BUILD",
    "WORKFLOW",
    "VAR_NAMES_VALUES",
)
COLUMNS = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
UNSTATED = ("", "-", "NA")


class MatrixError(WorkspaceError):
    """A matrix or one of its rows cannot be planned safely."""


def _no_variables() -> Mapping[str, object]:
    return MappingProxyType({})


@dataclass(frozen=True)
class MatrixRow:
    """One data row, numbered before filtering RUN, with exact text identifiers.

    ``hidden`` is the HIDDEN cell (whether a solver with a window shows it);
    it never removes a row, which RUN alone decides. ``variables`` holds the
    VAR_NAMES_VALUES overrides of setup keys. Text columns stating ``-`` are
    empty strings, and other unstated columns are None.
    """

    row_number: int
    pol: str
    run: int
    setup_id: str
    reference_id: str
    post_id: str
    condition: Mapping[str, float]
    sweep_key: str | None
    sweep_values: tuple[float, ...]
    walltime: str | None
    description: str
    workflow: str
    hidden: bool = False
    aircraft: str = ""
    configuration: str = ""
    geometry: str | None = None
    ncpus: int | None = None
    solver_build: str | None = None
    variables: Mapping[str, object] = field(default_factory=_no_variables)
    condition_order: tuple[str, ...] = ()


def _number(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise MatrixError(f"{label}: {text!r} must be a finite number") from error
    if not math.isfinite(value):
        raise MatrixError(f"{label}: {text!r} must be a finite number")
    return 0.0 if value == 0 else value


def _artifact_id(text: str, prefix: str, label: str) -> str:
    code = text.removesuffix(".toml").removeprefix(prefix)
    if not code or not code.isascii() or not code.isdecimal():
        raise MatrixError(f"{label}: {text!r} must name {prefix}<digits>.toml")
    return prefix + code


def _condition(
    cells: Mapping[str, str], label: str
) -> tuple[dict[str, float], str | None, tuple[str, ...]]:
    held: dict[str, float] = {}
    swept: str | None = None
    text = cells.get("FLIGHT_CONDITION", "")
    if text in UNSTATED:
        return held, swept, ()
    seen: set[str] = set()
    for pair in text.split(","):
        name, colon, raw = pair.partition(":")
        key, raw = canonical_key(name), raw.strip()
        if not colon or re.fullmatch("[A-Za-z][A-Za-z0-9_]*", key) is None:
            raise MatrixError(f"{label}: FLIGHT_CONDITION needs KEY:number pairs")
        if key in seen:
            raise MatrixError(f"{label}: FLIGHT_CONDITION repeats {key}")
        seen.add(key)
        if raw.casefold() == "sweep":
            if swept is not None:
                raise MatrixError(f"{label}: only one FLIGHT_CONDITION key may sweep")
            swept = key
        else:
            held[key] = _number(raw, f"{label} FLIGHT_CONDITION {key}")
    order = tuple(canonical_key(pair.partition(":")[0]) for pair in text.split(","))
    return held, swept, order


def _value(text: str) -> object:
    try:
        value = tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text
    if not isinstance(value, (str, int, float, bool, list, dict)):
        return text
    return value


def _split_outside(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    depth = start = 0
    for index, char in enumerate(text):
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _variables(text: str, label: str) -> Mapping[str, object]:
    found: dict[str, object] = {}
    if text in UNSTATED:
        return MappingProxyType(found)
    for pair in _split_outside(text, "/"):
        name, colon, raw = pair.partition(":")
        key, raw = name.strip(), raw.strip()
        if not colon:
            raise MatrixError(
                f"{label}: VAR_NAMES_VALUES entry {pair.strip()!r} is not a KEY: value pair; "
                "the cell holds '/'-separated KEY: value entries"
            )
        if re.fullmatch("[A-Za-z_][A-Za-z0-9_]*([.][A-Za-z_][A-Za-z0-9_]*)*", key) is None:
            raise MatrixError(f"{label}: VAR_NAMES_VALUES key {key!r} is not a setup key name")
        if key.upper() in COLUMNS:
            raise MatrixError(
                f"{label}: VAR_NAMES_VALUES states {key}, which is a matrix column; "
                f"state it in the {key.upper()} column only"
            )
        if key in found:
            raise MatrixError(
                f"{label}: VAR_NAMES_VALUES states {key} twice; keep the value the row means"
            )
        if not raw:
            raise MatrixError(f"{label}: VAR_NAMES_VALUES key {key} has no value")
        found[key] = _value(raw)
    return MappingProxyType(found)


def _row(cells: Mapping[str, str], number: int, path: Path) -> MatrixRow:
    pol = cells["POL"]
    label = f"{path.name} data row {number}, POL {pol!r}"
    if re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]*", pol) is None:
        raise MatrixError(f"{label}: POL must be a non-empty filename-safe identifier")
    if cells["RUN"] not in ("0", "1"):
        raise MatrixError(f"{label}: RUN must be 0 or 1")
    held, swept, order = _condition(cells, label)
    values_text = cells.get("SWEEP_VALUES", "")
    values: tuple[float, ...] = ()
    if swept is None and values_text not in UNSTATED:
        raise MatrixError(f"{label}: SWEEP_VALUES needs one FLIGHT_CONDITION key set to sweep")
    if swept is not None:
        if values_text in UNSTATED:
            raise MatrixError(f"{label}: SWEEP_VALUES must name the values of {swept}")
        values = tuple(_number(v.strip(), f"{label} SWEEP_VALUES") for v in values_text.split(","))
        if len(set(values)) != len(values):
            raise MatrixError(f"{label}: repeated SWEEP_VALUES would write the same point")
    walltime = cells.get("WALLTIME", "")
    if walltime in UNSTATED:
        walltime_value = None
    else:
        match = re.fullmatch("([0-9]+(?:[.][0-9]+)?)(s|m|h|d)", walltime)
        if match is None or _number(match[1], f"{label} WALLTIME") <= 0:
            raise MatrixError(f"{label}: WALLTIME needs a positive number and s, m, h or d")
        walltime_value = walltime
    hidden = cells.get("HIDDEN", "")
    if hidden not in (*UNSTATED, "0", "1"):
        raise MatrixError(f"{label}: HIDDEN must be 0, 1 or -")
    for column in ("SYMMETRY", "SYMMETRY_LOADS"):
        if cells.get(column, "") not in UNSTATED:
            raise MatrixError(
                f"{label}: {column} {cells[column]!r} is not read by cfdbender solvers, which "
                "take symmetry from the mesh and the setup; write - in the cell"
            )
    ncpus_text = cells.get("NCPUS", "")
    ncpus: int | None = None
    if ncpus_text not in UNSTATED:
        if not ncpus_text.isascii() or not ncpus_text.isdecimal() or int(ncpus_text) < 1:
            raise MatrixError(f"{label}: NCPUS must be a positive whole number or -")
        ncpus = int(ncpus_text)
    build = cells.get("SOLVER_BUILD", cells.get("FS_BUILD", ""))
    if build not in UNSTATED and re.fullmatch("[A-Za-z0-9][A-Za-z0-9._+-]*", build) is None:
        raise MatrixError(f"{label}: SOLVER_BUILD {build!r} must be a version such as 8.3.0")
    geometry = cells.get("GEOMETRY", "")
    if geometry not in UNSTATED and (
        chr(92) in geometry or any(ord(c) < 32 or c in '<>"|?*' for c in geometry)
    ):
        raise MatrixError(f"{label}: GEOMETRY {geometry!r} must name a file with / separators")
    workflow = cells.get("WORKFLOW", "")

    def text(column: str) -> str:
        value = cells.get(column, "")
        return "" if value in UNSTATED else value

    return MatrixRow(
        row_number=number,
        pol=pol,
        run=int(cells["RUN"]),
        setup_id=_artifact_id(cells["SET"], "s", f"{label} SET"),
        reference_id=_artifact_id(cells["REF"], "r", f"{label} REF"),
        post_id=_artifact_id(cells["PPROC"], "p", f"{label} PPROC"),
        condition=MappingProxyType(held),
        sweep_key=swept,
        sweep_values=values,
        walltime=walltime_value,
        description=text("DESCRIPTION"),
        workflow="steady" if workflow in UNSTATED else workflow,
        hidden=hidden == "1",
        aircraft=text("AIRCRAFT"),
        configuration=text("CONFIGURATION"),
        geometry=None if geometry in UNSTATED else geometry,
        ncpus=ncpus,
        solver_build=None if build in UNSTATED else build,
        variables=_variables(cells.get("VAR_NAMES_VALUES", ""), label),
        condition_order=order,
    )


def parse_matrix(data: bytes, source: Path, *, active_only: bool = True) -> tuple[MatrixRow, ...]:
    """Parse exact input bytes, retaining data row numbers before the RUN filter.

    Parameters
    ----------
    data : bytes
        UTF-8 text with an optional BOM, hash comments and dashed separator rules.
    source : Path
        Source name for refusal messages.
    active_only : bool, optional
        Return only RUN = 1 rows; all rows are checked regardless.

    Returns
    -------
    tuple of MatrixRow
        Rows in source order, with ids kept as text, including leading zeros.
    """
    try:
        lines = data.decode("utf-8-sig").splitlines()
    except UnicodeError as error:
        raise MatrixError(f"{source}: cannot read UTF-8: {error}") from error
    content = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#") and set(line.strip()) - set("-:+| \t")
    ]
    if not content:
        raise MatrixError(f"{source}: missing matrix header: {', '.join(REQUIRED_COLUMNS)}")
    header = [c.strip() for c in content[0].split("|")]
    if any(not c for c in header) or len(set(header)) != len(header):
        raise MatrixError(f"{source}: blank or duplicate matrix headers")
    missing = set(REQUIRED_COLUMNS) - set(header)
    unknown = set(header) - set(COLUMNS)
    if missing or unknown:
        hints = [
            f"{name} (closest known column: {close[0]})" if close else name
            for name in sorted(unknown)
            for close in [difflib.get_close_matches(name, COLUMNS, n=1)]
        ]
        raise MatrixError(
            f"{source}: missing columns: {', '.join(sorted(missing)) or '(none)'}; "
            f"unsupported columns: {', '.join(hints) or '(none)'}"
        )
    if {"SOLVER_BUILD", "FS_BUILD"} <= set(header):
        raise MatrixError(f"{source}: SOLVER_BUILD and FS_BUILD name one column; keep one")
    result: list[MatrixRow] = []
    seen: set[str] = set()
    for number, line in enumerate(content[1:], 1):
        fields = [c.strip() for c in line.split("|")]
        if len(fields) != len(header):
            raise MatrixError(f"{source}: data row {number} needs {len(header)} columns")
        row = _row(dict(zip(header, fields, strict=True)), number, source)
        if row.pol.casefold() in seen:
            raise MatrixError(f"{source}: data row {number} repeats POL {row.pol!r}")
        seen.add(row.pol.casefold())
        if row.run == 1 or not active_only:
            result.append(row)
    return tuple(result)


def read_matrix(path: str | Path, *, active_only: bool = True) -> tuple[MatrixRow, ...]:
    """Read a pipe-delimited .fs matrix without interpreting solver settings.

    Parameters
    ----------
    path : str or Path
        Matrix location. Workspace discovery belongs to the planning layer.
    active_only : bool, optional
        Return only RUN = 1 rows.

    Returns
    -------
    tuple of MatrixRow
        Validated rows. No solver is loaded and no file is written.

    Raises
    ------
    MatrixError
        Named read, schema, identity or cell refusal.
    """
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as error:
        raise MatrixError(f"{source}: cannot read: {error}") from error
    return parse_matrix(data, source, active_only=active_only)
