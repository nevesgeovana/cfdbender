"""The sections table of a point: its sectional distribution, one file per point.

``post/<matrix>/sections/P<POL>/<point>_sections.csv`` opens with the run block, then ``FAMILY``,
``FAMILY_ID``, ``STEP``, ``PLANE``, ``ROTOR_ID`` and ``AZIMUTH``, then the solver's own
sectional columns in the order it prints them. Every family the distribution cuts is in
the one file of its point, keyed by ``FAMILY`` and ``FAMILY_ID``. A solver states its
distribution where it prints one: QPROP's radial table (one row per station along the
blade, plane ``radial``, rotor 0). A solver that prints no distribution writes no sections
file; a QPROP point whose output holds no radial row is a named skip. No solver of this
release prints a pressure distribution, so no Cp file is written.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cfdbender.matrix._family_columns import ColumnSpec, column_names, sections_columns
from cfdbender.matrix._family_polar import Point, PolarFiles, point_block

#: The units of QPROP's radial columns that it names without one.
_QPROP_UNITS = {"radius": "m", "chord": "m", "beta": "deg", "Wa(m/s)": "m/s"}


@dataclass(frozen=True)
class Distribution:
    """One point's sectional distribution: column names and rows, as the solver prints them."""

    names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]


def qprop_radial(text: str) -> Distribution | None:
    """Return QPROP's radial table: the header naming radius and chord, then numeric rows.

    A row may be printed behind ``#``. The table ends at the first line that is not a row
    of as many numbers as the header names; None when no row follows the header.
    """
    lines = text.splitlines()
    for number, line in enumerate(lines):
        names = line.lstrip("#").split()
        if names[:1] != ["radius"] or "chord" not in names:
            continue
        rows: list[tuple[float, ...]] = []
        for follower in lines[number + 1 :]:
            cells = follower.lstrip("#").split()
            if not cells:
                if rows:
                    break
                continue
            try:
                values = tuple(float(cell) for cell in cells)
            except ValueError:
                break
            if len(values) != len(names) or not all(map(math.isfinite, values)):
                break
            rows.append(values)
        return Distribution(tuple(names), tuple(rows)) if rows else None
    return None


#: The solvers that print a distribution: the reader and the stdout file it reads.
SECTION_READERS: dict[str, tuple[str, Callable[[str], Distribution | None]]] = {
    "qprop": ("stdout.log", qprop_radial),
}


def export_columns(solver: str | None, names: Sequence[str]) -> tuple[ColumnSpec, ...]:
    """Return the specification of a solver's sectional columns, in its order."""
    units = _QPROP_UNITS if solver == "qprop" else {}
    return tuple(
        ColumnSpec(name, units.get(name, "-"), "section", "sectional value as the solver prints it")
        for name in names
    )


def section_files(
    root: Path,
    folder: Path,
    rows: Sequence[Mapping[str, object]],
    points: Sequence[Point],
    polar: PolarFiles,
    solver: str | None,
    *,
    write: Callable[[Path, str], Path],
    cell: Callable[[object], str],
) -> tuple[tuple[Path, ...], tuple[ColumnSpec, ...] | None, tuple[tuple[str, str], ...]]:
    """Write ``sections/P<POL>/<point>_sections.csv`` for each point with a distribution.

    Parameters
    ----------
    root, folder : Path
        The workspace and ``post/<matrix>``.
    rows, points : sequence
        The campaign summary rows and their points, in record order.
    polar : PolarFiles
        The polar files written for the same points: their run block and frames.
    solver : str or None
        The workspace's solver.
    write, cell : callable
        As :func:`cfdbender.matrix._family_polar.polar_files`.

    Returns
    -------
    tuple
        The written paths, the column specification after the run block (None when no
        file was written), and ``(product, reason)`` for each point skipped.
    """
    reader = SECTION_READERS.get(solver or "")
    if reader is None:
        return (), None, ()
    filename, parse = reader
    tables: list[tuple[Point, int, Distribution]] = []
    skipped: list[tuple[str, str]] = []
    positions: dict[str, int] = {}
    for row, point in zip(rows, points, strict=True):
        positions[point.pol] = positions.get(point.pol, 0) + 1
        outputs = cast(Mapping[str, object], row.get("outputs_sha256") or {})
        path = next((p for p in outputs if str(p).endswith(f"/{filename}")), None)
        text = ""
        if row.get("result") is not None and path is not None and (root / str(path)).is_file():
            data = (root / str(path)).read_bytes()
            if hashlib.sha256(data).hexdigest() == outputs[path]:
                text = data.decode("utf-8", errors="replace")
        distribution = parse(text) if text else None
        if distribution is None:
            skipped.append(
                (
                    f"sections/P{point.pol}/{point.name}_sections.csv",
                    f"the verified output of {point.pol}/{point.name} is unavailable "
                    "or holds no sectional distribution",
                )
            )
            continue
        tables.append((point, positions[point.pol], distribution))
    if not tables:
        return (), None, tuple(skipped)
    names: dict[str, None] = {}
    for _point, _index, distribution in tables:
        names.update(dict.fromkeys(distribution.names))
    spec = sections_columns(export_columns(solver, tuple(names)))
    header = [*polar.run_block, *column_names(spec)]
    written: list[Path] = []
    for point, index, distribution in tables:
        head = point_block(
            point, index, polar.run_types[point.pol], polar.identifiers, polar.frames
        )
        lines = [",".join(header)]
        for values in distribution.rows:
            cells = {
                **head,
                "FAMILY": "BLADES",
                "FAMILY_ID": 1,
                "STEP": None,
                "PLANE": "radial",
                "ROTOR_ID": 0,
                "AZIMUTH": None,
                **dict(zip(distribution.names, values, strict=True)),
            }
            lines.append(",".join(cell(cells.get(h)) for h in header))
        target = folder / "sections" / f"P{point.pol}" / f"{point.name}_sections.csv"
        written.append(write(target, chr(10).join(lines) + chr(10)))
    return tuple(written), spec, tuple(skipped)
