"""The steady plot table: ``polars/P<sim>_<name>_steady_plots.csv``, one row per steady point.

A steady SU2 point whose post file declares ``[plots]`` gets the columns a time-accurate point
gets in its unsteady polar, under the same names, ``<parameter>_<group>`` (``CL_MRP_TOTAL``,
``FX_MRP_Wing``), from the last row of its history. This is an extension of the sister
library, whose plots are unsteady-only: it lets an airframe solved steady be compared, column
for column, with the same airframe in a time-accurate run. The columns are::

    POL  ITER  <condition block>  ALPHA  BETA  ALT  J  XMOM  YMOM  ZMOM  <plot columns>

The condition columns an adapter cannot fill are absent, as in the unsteady polar. A plot
column or group that cannot be filled is never a column of empty or ``NA`` cells; it is named,
with its reason, in ``products.json``.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from cfdbender._textio import csv_writer
from cfdbender.matrix._rotor_table import CONDITION_COLUMNS
from cfdbender.matrix._unsteady_polar import MOMENT_POINT_COLUMNS

SUFFIX = "_steady_plots.csv"


def steady_plots_name(pol: str, name: str) -> str:
    """Return ``P<pol>_<name>_steady_plots.csv``."""
    return f"P{pol}_{name}{SUFFIX}"


def steady_plot_polars(
    folder: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    write: Callable[[Path, str], Path],
    cell: Callable[[object], str],
) -> tuple[Path, ...]:
    """Write one steady plot table per polar whose points carry plot values, or nothing."""
    by_pol: dict[str, list[dict[str, object]]] = {}
    plots: dict[str, dict[str, None]] = {}
    absent: dict[str, set[str]] = {}
    for row in rows:
        result = cast(Mapping[str, object], row.get("result") or {})
        record = result.get("plots")
        if not isinstance(record, dict) or not isinstance(record.get("values"), dict):
            continue
        if not record["values"]:
            continue
        pol = str(row["pol"])
        condition = cast(Mapping[str, object], row.get("condition") or {})
        cells: dict[str, object] = {"POL": pol, "ITER": record.get("iteration")}
        cells.update(cast(Mapping[str, object], record.get("condition") or {}))
        cells.update(
            ALPHA=condition.get("ALPHA"),
            BETA=condition.get("BETA"),
            ALT=condition.get("ALTFT"),
            J=condition.get("ADVANCE_RATIO"),
        )
        references = cast(Mapping[str, object], result.get("references") or {})
        origin = cast(Sequence[object], references.get("moment_origin") or (None,) * 3)
        cells.update(zip(MOMENT_POINT_COLUMNS, origin, strict=False))
        values = cast(Mapping[str, object], record["values"])
        cells.update(values)
        plots.setdefault(pol, {}).update(dict.fromkeys(values))
        absent.setdefault(pol, set()).update(str(n) for n in record.get("absent_condition", ()))
        by_pol.setdefault(pol, []).append(cells)
    written: list[Path] = []
    for pol, lines in by_pol.items():
        header = [
            "POL",
            "ITER",
            *(name for name in CONDITION_COLUMNS if name not in absent[pol]),
            "ALPHA",
            "BETA",
            "ALT",
            "J",
            *MOMENT_POINT_COLUMNS,
            *plots[pol],
        ]
        buffer = io.StringIO(newline="")
        writer = csv_writer(buffer)
        writer.writerow(header)
        writer.writerows([cell(line.get(h)) for h in header] for line in lines)
        written.append(
            write(folder / "polars" / steady_plots_name(pol, folder.name), buffer.getvalue())
        )
    return tuple(written)
