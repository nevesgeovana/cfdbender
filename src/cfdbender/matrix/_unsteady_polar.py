"""The unsteady polar: pyflightstream's ``polars/P<sim>_<name>_uns_avg.csv``, one row per point.

Each row is the time average of a time-accurate point over its averaging window
(:func:`cfdbender.post.unsteady_point.reduce_point`). The columns follow pyflightstream's
order (``docs/definitions/the-unsteady-polar.md``)::

    POL  FIRST_STEP  LAST_STEP  STEPS  <condition block>  XMOM  YMOM  ZMOM  <history columns>
    RPM_<alias>  DIAMETER_<alias>

``<name>`` is the matrix's stem (PROVISIONAL: pyflightstream names the sweep). The condition
columns an adapter cannot fill are absent, as in the rotor table, and so is ``MESH_FACES``,
which this package does not compute. A point whose history does not cover its window is left
out of the file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from cfdbender.matrix._rotor_table import CONDITION_COLUMNS

WINDOW_COLUMNS = ("FIRST_STEP", "LAST_STEP", "STEPS")
MOMENT_POINT_COLUMNS = ("XMOM", "YMOM", "ZMOM")
SUFFIX = "_uns_avg.csv"


def unsteady_polar_name(pol: str, name: str) -> str:
    """Return ``P<pol>_<name>_uns_avg.csv``."""
    return f"P{pol}_{name}{SUFFIX}"


def unsteady_polars(
    folder: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    write: Callable[[Path, str], Path],
    cell: Callable[[object], str],
) -> tuple[Path, ...]:
    """Write one unsteady polar per polar whose points carry a time average, or nothing."""
    by_pol: dict[str, list[dict[str, object]]] = {}
    plots: dict[str, dict[str, None]] = {}
    rotors: dict[str, dict[str, None]] = {}
    absent: dict[str, set[str]] = {}
    for row in rows:
        result = cast(Mapping[str, object], row.get("result") or {})
        averaged = result.get("time_average")
        if not isinstance(averaged, dict) or not isinstance(averaged.get("means"), dict):
            continue
        pol = str(row["pol"])
        condition = cast(Mapping[str, object], row.get("condition") or {})
        cells: dict[str, object] = {"POL": pol}
        cells.update({key: averaged.get(key) for key in WINDOW_COLUMNS})
        cells.update(cast(Mapping[str, object], averaged.get("condition") or {}))
        cells.update(
            ALPHA=condition.get("ALPHA"),
            BETA=condition.get("BETA"),
            ALT=condition.get("ALTFT"),
            J=condition.get("ADVANCE_RATIO"),
        )
        references = cast(Mapping[str, object], result.get("references") or {})
        origin = cast(Sequence[object], references.get("moment_origin") or (None,) * 3)
        cells.update(zip(MOMENT_POINT_COLUMNS, origin, strict=False))
        means = cast(Mapping[str, object], averaged["means"])
        cells.update(means)
        plots.setdefault(pol, {}).update(dict.fromkeys(means))
        rotor = result.get("rotor")
        if isinstance(rotor, dict) and isinstance(rotor.get("row"), dict):
            alias = str(rotor["rotor"])
            values = cast(Mapping[str, object], rotor["row"])
            cells.update(
                {
                    "J_CLOCK": values.get("J"),
                    "RPM_CLOCK": values.get("RPM"),
                    f"RPM_{alias}": values.get("RPM"),
                    f"DIAMETER_{alias}": values.get("DIAMETER"),
                }
            )
            rotors.setdefault(pol, {}).update({f"RPM_{alias}": None, f"DIAMETER_{alias}": None})
        absent.setdefault(pol, set()).update(str(n) for n in averaged.get("absent", ()))
        by_pol.setdefault(pol, []).append(cells)
    written: list[Path] = []
    for pol, lines in by_pol.items():
        header = [
            "POL",
            *WINDOW_COLUMNS,
            *(name for name in CONDITION_COLUMNS if name not in absent[pol]),
            *MOMENT_POINT_COLUMNS,
            *plots[pol],
            *rotors.get(pol, {}),
        ]
        text = [",".join(header), *(",".join(cell(line.get(h)) for h in header) for line in lines)]
        name = unsteady_polar_name(pol, folder.name)
        written.append(write(folder / "polars" / name, chr(10).join(text) + chr(10)))
    return tuple(written)
