"""The rotor table: pyflightstream's ``polars/P<sim>-<alias>_rotor.csv``, one row per point.

The file name, the column order and the definitions are pyflightstream's
(``docs/definitions/rotor-coefficients.md`` and ``what-every-product-states.md``)::

    POL  ROTOR  ALPHA  BETA  MACH  RE  VINF  VREF  ALT  RHO  TEMP  MU  J  J_CLOCK  RPM_CLOCK
    SREF  CREF  BREF  RPM_<a>  DIAMETER_<a>  J_<a>  CT_<a>  CQ_<a>  CP_<a>  ETA_<a>  ETAW_<a>
    MTIP_<a>  MHEL_<a>  CN_<a>  CS_<a>  CMN_<a>  CMS_<a>  MADV_<a>

with ``<a>`` the rotor's alias. A column is ABSENT from the file, never written as NA, when
nothing in this package fills it: the columns of :data:`NOT_PORTED` for every adapter, and
the condition columns an adapter's rotor record names under ``absent`` (SU2: ``BREF``). The
products manifest names each absent column. A value that does not exist for one row is NA.

The row is formed by :func:`cfdbender.core.rotor.rotor_row` from the force and moment the
adapter supplies; the condition cells come from the adapter's record and the row.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from cfdbender._textio import csv_writer
from cfdbender.core.plan import _checked_file_name

#: The condition block, in pyflightstream's order.
CONDITION_COLUMNS = (
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
#: The per-rotor columns, before the alias suffix, in pyflightstream's order.
ROTOR_COLUMNS = (
    "RPM",
    "DIAMETER",
    "J",
    "CT",
    "CQ",
    "CP",
    "ETA",
    "ETAW",
    "MTIP",
    "MHEL",
    "CN",
    "CS",
    "CMN",
    "CMS",
    "MADV",
)
#: pyflightstream's rotor columns this package does not compute yet: absent from every file.
NOT_PORTED = ("ETAW", "MTIP", "MHEL", "MADV")
#: What the rotor table's file name ends with.
SUFFIX = "_rotor.csv"


def rotor_file_name(pol: str, alias: str) -> str:
    """Return ``P<pol>-<alias>_rotor.csv``."""
    return _checked_file_name(f"P{pol}-{alias}{SUFFIX}")


def full_header(alias: str) -> tuple[str, ...]:
    """Return every column of pyflightstream's rotor table of ``alias``, in order."""
    return ("POL", "ROTOR", *CONDITION_COLUMNS, *(f"{name}_{alias}" for name in ROTOR_COLUMNS))


def _cells(row: Mapping[str, object], rotor: Mapping[str, object]) -> dict[str, object]:
    condition = cast(Mapping[str, object], row.get("condition") or {})
    cells = dict(cast(Mapping[str, object], rotor.get("condition") or {}))
    values = cast(Mapping[str, object], rotor.get("row") or {})
    alias = str(rotor["rotor"])
    cells.update(
        POL=row["pol"],
        ROTOR=alias,
        ALPHA=condition.get("ALPHA"),
        BETA=condition.get("BETA"),
        ALT=condition.get("ALTFT"),
        J=condition.get("ADVANCE_RATIO"),
        J_CLOCK=values.get("J"),
        RPM_CLOCK=values.get("RPM"),
    )
    for name in ROTOR_COLUMNS:
        if name in values:
            cells[f"{name}_{alias}"] = values[name]
    return cells


def rotor_table(
    folder: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    write: Callable[[Path, str], Path],
    cell: Callable[[object], str],
) -> tuple[Path, ...]:
    """Write ``polars/P<pol>-<alias>_rotor.csv`` for every polar and rotor, or nothing.

    Parameters
    ----------
    folder : Path
        ``post/<matrix>``.
    rows : sequence of mapping
        The campaign summary rows, in record order.
    write, cell : callable
        The writer of a derived table and the cell formatter of the polar files.

    Returns
    -------
    tuple of Path
        The written paths, none when no point has a rotor.
    """
    files: dict[tuple[str, str], list[dict[str, object]]] = {}
    absent: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        result = cast(Mapping[str, object], row.get("result") or {})
        rotor = result.get("rotor")
        if not isinstance(rotor, dict) or not isinstance(rotor.get("row"), dict):
            continue
        key = (str(row["pol"]), str(rotor["rotor"]))
        files.setdefault(key, []).append(_cells(row, rotor))
        absent.setdefault(key, set()).update(str(name) for name in rotor.get("absent", ()))
    written: list[Path] = []
    for (pol, alias), lines in files.items():
        dropped = {*absent[(pol, alias)], *(f"{name}_{alias}" for name in NOT_PORTED)}
        header = [name for name in full_header(alias) if name not in dropped]
        buffer = io.StringIO(newline="")
        writer = csv_writer(buffer)
        writer.writerow(header)
        writer.writerows(
            [line.get(h) if h == "ROTOR" else cell(line.get(h)) for h in header] for line in lines
        )
        written.append(write(folder / "polars" / rotor_file_name(pol, alias), buffer.getvalue()))
    return tuple(written)


def absent_columns(path: Path) -> list[str]:
    """Return the columns of pyflightstream's rotor table that ``path`` leaves out, in order."""
    alias = path.name.removesuffix(SUFFIX).split("-", 1)[1]
    with path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    return [name for name in full_header(alias) if name not in header]
