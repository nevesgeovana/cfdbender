"""The files of a volume cut and of probes, written from interpolated volume fields.

Two products, named, placed and laid out as the declarations they follow write them:

* ``fields/<point>_vsec.vtk`` or ``fields/<point>_vsec.dat`` for a volume cut, with
  ``_step_<STEP>`` before the extension for each step of a series. The topology is a vertex cloud:
  the sampled positions, in metres in the reference (inertial) frame, and one array per column.
  ``vtk`` is a legacy ASCII polydata file; ``tecplot`` has one single-point zone per sample. A
  companion ``<file>.provenance.json`` records the source, the frame, the step and the counts.
* ``probes/<point>_probes.csv`` (``probes/<point>_probes_history.csv`` for a history) with one row
  per probe point and per step. A row opens with ``POL, PROBE, X, Y, Z, FRAME, STEP``, then the
  condition columns the caller states, then the value columns. ``PROBE`` counts the probe points
  from 1 across the entries; ``X, Y, Z`` are the sampled positions in the reference frame, in
  metres, and ``FRAME`` says ``REFERENCE``; ``STEP`` is the solver step, ``NA`` for none.

What a column holds is the caller's table ``{column: field name or callable on the fields}``. A
column whose field the volume file does not carry, or whose callable raises ``KeyError``, is
absent from the product and named in the report; nothing is invented for it. A point in no cell has
no value: a cut leaves the point out of its file, and a probe table writes the row with an empty
cell for every value, so the points that could not be filled are never mistaken for zeros. The
files are byte-stable: the same inputs give the same bytes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cfdbender._textio import write_csv, write_json, write_text
from cfdbender.post.volume import VolumeError, VolumeSampler
from cfdbender.post.volume_frames import Frame
from cfdbender.post.vtu import UnstructuredGrid, read_vtu

__all__ = [
    "NOT_APPLICABLE",
    "POSITION_FRAME_LABEL",
    "ColumnSource",
    "ProbeEntry",
    "ProbeReport",
    "SectionReport",
    "StepReport",
    "VolumeStep",
    "write_probe_table",
    "write_volume_section",
]

#: What the ``STEP`` column says when the product has no step.
NOT_APPLICABLE = "NA"

#: The ``FRAME`` column of a probe table: the sampled positions are inertial coordinates.
POSITION_FRAME_LABEL = "REFERENCE"

ColumnSource = str | Callable[[Mapping[str, NDArray[np.float64]]], NDArray[np.float64]]


class VolumeStep:
    """One volume file of a series: its step, its time and the grid it holds.

    Parameters
    ----------
    step : int or None
        The solver step; None for a product with no step (a steady solution).
    time : float or None
        The physical time, recorded in the provenance, never written as a column.
    source : path or UnstructuredGrid
        The ``.vtu`` file, read when first needed, or a grid already read.
    """

    def __init__(
        self, step: int | None, time: float | None, source: str | Path | UnstructuredGrid
    ) -> None:
        self.step = step
        self.time = time
        self.source = source
        self._sampler: VolumeSampler | None = None

    def sampler(self) -> VolumeSampler:
        """Return the sampler of this volume, reading and indexing it on first use."""
        if self._sampler is None:
            grid = (
                self.source if isinstance(self.source, UnstructuredGrid) else read_vtu(self.source)
            )
            self._sampler = VolumeSampler(grid)
        return self._sampler

    def release(self) -> None:
        """Drop the index so that the memory of a large volume can be reused."""
        self._sampler = None

    def source_name_and_hash(self) -> tuple[str | None, str | None]:
        """Return the file name and SHA-256 of a source file, ``(None, None)`` for a grid."""
        if isinstance(self.source, UnstructuredGrid):
            return None, None
        path = Path(self.source)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return path.name, digest.hexdigest()


@dataclass(frozen=True)
class ProbeEntry:
    """Probe points declared in one frame.

    Attributes
    ----------
    points : numpy.ndarray
        Shape ``(n, 3)``, in the frame's local coordinates, in metres.
    frame : Frame
        The frame that places them at a step.
    """

    points: NDArray[np.float64]
    frame: Frame


@dataclass(frozen=True)
class StepReport:
    """What one step of a product holds.

    Attributes
    ----------
    step : int or None
        The step.
    n_points : int
        The points laid for this step.
    n_found : int
        The points that lie in a cell.
    absent_columns : tuple of str
        The columns that the volume of this step could not fill.
    skipped_cells : int
        The cells of the volume that the search does not read.
    file : pathlib.Path or None
        The file written for the step, None when a cut found no point.
    """

    step: int | None
    n_points: int
    n_found: int
    absent_columns: tuple[str, ...]
    skipped_cells: int
    file: Path | None = None

    @property
    def n_missing(self) -> int:
        """The points that lie in no cell."""
        return self.n_points - self.n_found


@dataclass(frozen=True)
class SectionReport:
    """The result of :func:`write_volume_section`."""

    steps: tuple[StepReport, ...]

    @property
    def files(self) -> tuple[Path, ...]:
        """The data files written, in step order."""
        return tuple(report.file for report in self.steps if report.file is not None)


@dataclass(frozen=True)
class ProbeReport:
    """The result of :func:`write_probe_table`.

    Attributes
    ----------
    file : pathlib.Path
        The table written.
    steps : tuple of StepReport
        One entry per step.
    absent_columns : tuple of str
        The columns left out because some step could not fill them.
    n_rows : int
        The rows of the table.
    """

    file: Path
    steps: tuple[StepReport, ...]
    absent_columns: tuple[str, ...]
    n_rows: int
    listed: dict[str, list[str]] = field(default_factory=dict)


def _resolve(
    columns: Mapping[str, ColumnSource],
    fields: Mapping[str, NDArray[np.float64]],
    n_points: int,
) -> tuple[dict[str, NDArray[np.float64]], list[str]]:
    """Fill each column from the interpolated fields; the unfillable ones are returned by name."""
    out: dict[str, NDArray[np.float64]] = {}
    absent: list[str] = []
    for name, source in columns.items():
        try:
            values = fields[source] if isinstance(source, str) else source(fields)
        except KeyError:
            absent.append(name)
            continue
        array = np.asarray(values, dtype=np.float64)
        if array.shape not in ((n_points,), (n_points, 3)):
            raise VolumeError(
                f"column {name!r} has shape {array.shape}; it holds one scalar or one 3-vector "
                f"per point ({n_points} points)"
            )
        out[name] = array
    return out, absent


def _check_names(columns: Mapping[str, ColumnSource]) -> None:
    for name in columns:
        if not name or any(ch.isspace() or ch in ',"' for ch in name):
            raise VolumeError(f"column name {name!r} is empty or holds a space, comma or quote")


def _step_suffix(step: int | None) -> str:
    return "" if step is None else f"_step_{step}"


def _number(value: float) -> str:
    return repr(float(value))


def _vtk_text(points: NDArray[np.float64], columns: Mapping[str, NDArray[np.float64]]) -> str:
    n = len(points)
    lines = [
        "# vtk DataFile Version 3.0",
        "cfdbender volume section",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {n} float",
    ]
    lines += [" ".join(f"{c:.9e}" for c in row) for row in points]
    lines.append(f"VERTICES {n} {2 * n}")
    lines += [f"1 {i}" for i in range(n)]
    if columns:
        lines.append(f"POINT_DATA {n}")
        for name, array in columns.items():
            if array.ndim == 1:
                lines += [f"SCALARS {name} float 1", "LOOKUP_TABLE default"]
                lines += [f"{v:.9e}" for v in array]
            else:
                lines.append(f"VECTORS {name} float")
                lines += [" ".join(f"{c:.9e}" for c in row) for row in array]
    return "\n".join(lines) + "\n"


def _tecplot_text(points: NDArray[np.float64], columns: Mapping[str, NDArray[np.float64]]) -> str:
    names = ["X", "Y", "Z"]
    table = [points[:, 0], points[:, 1], points[:, 2]]
    for name, array in columns.items():
        if array.ndim == 1:
            names.append(name)
            table.append(array)
        else:
            for axis, part in zip("xyz", array.T, strict=True):
                names.append(f"{name}_{axis}")
                table.append(part)
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise VolumeError(f"the Tecplot variables {repeated} would be written more than once")
    lines = [
        'TITLE = "cfdbender volume section"',
        "VARIABLES = " + " ".join(f'"{n}"' for n in names),
    ]
    matrix = np.column_stack(table)
    for index, row in enumerate(matrix, 1):
        lines.append(f'ZONE T="Sample {index}", I=1, J=1, K=1, ZONETYPE=ORDERED, DATAPACKING=POINT')
        lines.append(" ".join(f"{float(v):.17g}" for v in row))
    return "\n".join(lines) + "\n"


def write_volume_section(
    directory: str | Path,
    point: str,
    points: NDArray[np.float64],
    frame: Frame,
    volumes: Sequence[VolumeStep],
    columns: Mapping[str, ColumnSource],
    *,
    format: str = "vtk",
    description: Mapping[str, object] | None = None,
) -> SectionReport:
    """Write ``fields/<point>_vsec[_step_<STEP>].vtk`` (or ``.dat``) for each volume.

    Parameters
    ----------
    directory : path
        The folder that holds ``fields/``.
    point : str
        The point label of the file name.
    points : numpy.ndarray
        The laid points, local to ``frame`` (see :mod:`cfdbender.post.volume_layouts`).
    frame : Frame
        Places the points at each step.
    volumes : sequence of VolumeStep
        The ordered series: one file per step; a single step of None gives a file with no step.
    columns : mapping
        ``{column: field name or callable on the interpolated fields}``; a column the volume
        cannot fill is absent. A column is written as a vector when its values are ``(n, 3)``.
    format : {'vtk', 'tecplot'}
        The file format.
    description : mapping, optional
        Declaration facts (shape, plane, ...) copied into each provenance record.

    Returns
    -------
    SectionReport
        Per step, the points laid, found, the absent columns and the file.

    Raises
    ------
    VolumeError
        If the format is unknown or a column name or shape cannot be written.
    """
    if format not in ("vtk", "tecplot"):
        raise VolumeError(f"the format is {format!r}; it is 'vtk' or 'tecplot'")
    _check_names(columns)
    extension = ".vtk" if format == "vtk" else ".dat"
    folder = Path(directory) / "fields"
    folder.mkdir(parents=True, exist_ok=True)
    reports: list[StepReport] = []
    for volume in volumes:
        sampler = volume.sampler()
        placed = frame.place(points, volume.step)
        result = sampler.sample(placed)
        found = result.found
        resolved, absent = _resolve(columns, result.values, len(placed))
        file: Path | None = None
        if result.n_found:
            kept = {name: values[found] for name, values in resolved.items()}
            text = (_vtk_text if format == "vtk" else _tecplot_text)(placed[found], kept)
            file = folder / f"{point}_vsec{_step_suffix(volume.step)}{extension}"
            write_text(file, text)
            source_name, source_hash = volume.source_name_and_hash()
            write_json(
                file.with_name(file.name + ".provenance.json"),
                {
                    **dict(description or {}),
                    "absent_columns": absent,
                    "columns": list(kept),
                    "coordinate_units": "m",
                    "declared_points": int(len(placed)),
                    "frame": frame.name,
                    "frame_rotating": frame.rotating,
                    "removed_points": int(result.n_missing),
                    "sample_count": int(result.n_found),
                    "skipped_cells": sampler.skipped_cells,
                    "source_file": source_name,
                    "source_sha256": source_hash,
                    "step": volume.step,
                    "time": volume.time,
                    "topology": "vertex-cloud",
                },
                sort_keys=True,
            )
        reports.append(
            StepReport(
                volume.step,
                int(len(placed)),
                int(result.n_found),
                tuple(absent),
                sampler.skipped_cells,
                file,
            )
        )
    return SectionReport(tuple(reports))


def write_probe_table(
    directory: str | Path,
    point: str,
    entries: Sequence[ProbeEntry],
    volumes: Sequence[VolumeStep],
    columns: Mapping[str, ColumnSource],
    *,
    pol: str = "",
    condition: Mapping[str, str | float] | None = None,
    history: bool = False,
) -> ProbeReport:
    """Write ``probes/<point>_probes.csv``: one row per probe point and per step.

    Parameters
    ----------
    directory : path
        The folder that holds ``probes/``.
    point : str
        The point label of the file name.
    entries : sequence of ProbeEntry
        The declared probe points, each in its own frame; ``PROBE`` counts them across entries.
    volumes : sequence of VolumeStep
        The ordered series; rows are written step by step in this order.
    columns : mapping
        ``{column: field name or callable}``; every value column is a scalar. A column that any
        volume cannot fill is absent from the table and named in the report.
    pol : str
        The ``POL`` cell of every row.
    condition : mapping, optional
        The condition block: ``{column: value}``, written after ``STEP``, one value for all rows.
    history : bool, default False
        Write ``<point>_probes_history.csv``, the name of a per-step history beside a last-step
        table.

    Returns
    -------
    ProbeReport

    Raises
    ------
    VolumeError
        If a value column is not a scalar per point or a name cannot be written.
    """
    _check_names(columns)
    cond = dict(condition or {})
    _check_names({name: name for name in cond})
    n_total = sum(len(entry.points) for entry in entries)
    series: list[
        tuple[int | None, NDArray[np.float64], dict[str, NDArray[np.float64]], NDArray[np.bool_]]
    ] = []
    reports: list[StepReport] = []
    absent_any: dict[str, list[str]] = {}
    for volume in volumes:
        sampler = volume.sampler()
        placed = np.vstack([entry.frame.place(entry.points, volume.step) for entry in entries])
        result = sampler.sample(placed)
        resolved, absent = _resolve(columns, result.values, n_total)
        for name, values in resolved.items():
            if values.ndim != 1:
                raise VolumeError(f"probe column {name!r} is a vector; a table column is a scalar")
        for name in absent:
            absent_any.setdefault(name, []).append(str(volume.step))
        series.append((volume.step, placed, resolved, result.found))
        reports.append(
            StepReport(
                volume.step, n_total, int(result.n_found), tuple(absent), sampler.skipped_cells
            )
        )
    written = [name for name in columns if name not in absent_any]
    header = ["POL", "PROBE", "X", "Y", "Z", "FRAME", "STEP", *cond, *written]
    cond_cells = [
        value if isinstance(value, str) else _number(float(value)) for value in cond.values()
    ]
    rows: list[list[object]] = []
    for step, placed, resolved, found in series:
        for index in range(n_total):
            row: list[object] = [
                pol,
                index + 1,
                _number(placed[index, 0]),
                _number(placed[index, 1]),
                _number(placed[index, 2]),
                POSITION_FRAME_LABEL,
                NOT_APPLICABLE if step is None else step,
                *cond_cells,
            ]
            for name in written:
                row.append(_number(resolved[name][index]) if found[index] else "")
            rows.append(row)
    folder = Path(directory) / "probes"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{point}_probes{'_history' if history else ''}.csv"
    write_csv(target, rows, header=header)
    return ProbeReport(target, tuple(reports), tuple(absent_any), len(rows), absent_any)
