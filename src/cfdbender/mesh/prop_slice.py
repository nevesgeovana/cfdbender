"""Slice a propeller OBJ into blade sections and the propeller definition.

Pipeline role: the reader and writer around the pure slicer
:mod:`cfdbender.geometry._propeller_slice`. An OBJ is read as its polygons, fan-triangulated, and
cut at radial stations. The folder written holds the propeller definition ``propeller.toml``
that rebuilds the propeller, one normalized profile per station under ``sections/``, the
table ``table.csv`` and the record ``prop-slice.json``.
"""

from __future__ import annotations

import io
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender._textio import csv_writer, write_text
from cfdbender.geometry import Propeller
from cfdbender.geometry._propeller_slice import (
    BladeFrame,
    BladeSurface,
    definition_text,
    profile_text,
    slice_blade,
)
from cfdbender.geometry.blade_sections import parse_obj

__all__ = ["OBJ_UNITS", "TABLE_COLUMNS", "slice_propeller"]

#: Millimetres per unit of an OBJ, which states no unit of its own.
OBJ_UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}
#: The unit words the other geometry verbs take, read as the short names above.
_UNIT_WORDS = {"millimeter": "mm", "centimeter": "cm", "meter": "m", "inch": "in"}
#: The columns of ``table.csv``, in order.
TABLE_COLUMNS = (
    "r_R",
    "c_R",
    "beta_deg",
    "t_c",
    "camber",
    "te_gap_c",
    "skew_deg",
    "rake_x_R",
    "airfoil",
)
#: The outputs ``out`` receives; any of them already there is refused without overwrite.
_OUTPUTS = ("propeller.toml", "table.csv", "prop-slice.json", "sections")


def _refuse(text: str) -> InputArtifactError:
    return InputArtifactError(f"{text}. Nothing was written.", kind="geometry")


def _obj_surface(path: Path, unit: str, frame: BladeFrame) -> BladeSurface:
    vertices, triangles = parse_obj(path.read_text(encoding="utf-8", errors="replace"))
    if not triangles:
        raise _refuse(f"{path}: no blade found, the OBJ holds no face")
    scale = OBJ_UNITS[unit]
    return BladeSurface(np.asarray(vertices, float) * scale, np.asarray(triangles), frame)


def _vector(value: Sequence[float] | None, default: tuple[float, ...], name: str) -> list[float]:
    flag = "--" + name.replace("_", "-")
    try:
        vector = [float(v) for v in (default if value is None else value)]
    except (TypeError, ValueError):
        vector = []
    if len(vector) != 3:
        raise InputValueError(
            f"{name} (CLI: {flag}): got {value!r}, expected three numbers such as "
            f"{' '.join(f'{v:g}' for v in default)}"
        )
    return vector


def slice_propeller(
    geometry: str | Path,
    *,
    blades: int,
    out: str | Path,
    axis: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    blade_dir: Sequence[float] | None = None,
    stations: Sequence[float] | None = None,
    unit: str = "mm",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Cut one blade of a propeller at radial stations and write its propeller definition.

    Parameters
    ----------
    geometry : str or Path
        The propeller, an OBJ; read only.
    blades : int
        The number of blades; the blades found about the axis must match it.
    out : str or Path
        The folder to write: ``propeller.toml``, ``sections/sNN.dat``, ``table.csv`` and
        ``prop-slice.json``.
    axis : sequence of float, optional
        The thrust direction, the builder's ``frame.thrust_dir``; default ``(-1, 0, 0)``.
    origin : sequence of float, optional
        A point of the shaft in mm, the builder's ``frame.pitch_axis_point_mm``; default 0.
    blade_dir : sequence of float, optional
        The reference blade direction, the builder's ``frame.blade_dir``; default
        ``(0, 0, 1)``. Blade 0 is the blade nearest it, and skew is measured from it.
    stations : sequence of float, optional
        r/R of the sections, at least four, increasing, up to 1; default 21 equal steps from
        the blade root to the tip, and 0.7.
    unit : str, optional
        The length unit of the OBJ: ``mm`` (default), ``cm``, ``m`` or ``in``, or the words
        ``MILLIMETER``, ``CENTIMETER``, ``METER`` and ``INCH``, in any case; the record keeps
        the short name.
    overwrite : bool, default False
        Replace ``propeller.toml``, ``table.csv``, ``prop-slice.json`` and ``sections/``
        when ``out`` already holds any of them; without it they are refused.

    Returns
    -------
    dict
        The record written to ``prop-slice.json``: the inputs, the measured radius, root,
        rotation and blade count, the files, and the table without the profile points.

    Raises
    ------
    InputArtifactError
        The file is missing, its suffix is not .obj, no blade is found, or ``out`` already
        holds an output and ``overwrite`` is not given.
    InputValueError
        The blade count is not a positive integer or does not match the blades found, or a
        station or a direction is invalid.
    """
    source = Path(geometry)
    if not source.is_file():
        raise _refuse(f"{source}: no such geometry file; name an existing .obj")
    suffix = source.suffix.lower()
    if suffix != ".obj":
        raise _refuse(f"{source}: unsupported geometry suffix {suffix!r}; give an .obj")
    if type(blades) is not int or blades < 1:
        raise InputValueError(f"blades: expected a positive integer, got {blades!r}")
    given = unit
    unit = str(unit).strip().lower()
    unit = _UNIT_WORDS.get(unit, unit)
    if unit not in OBJ_UNITS:
        raise InputValueError(
            f"unit (CLI: --unit): got {given!r}, expected one of mm, cm, m, in or "
            "MILLIMETER, CENTIMETER, METER, INCH (any case)"
        )
    folder = Path(out)
    if source.resolve().is_relative_to((folder / "sections").resolve()):
        raise _refuse(
            f"{source}: input is inside the output sections directory; choose another "
            "out (CLI: --out) folder to keep the input"
        )
    present = [name for name in _OUTPUTS if (folder / name).exists()]
    if present and not overwrite:
        raise _refuse(
            f"{folder}: already holds {', '.join(present)}; name another out (CLI: --out) "
            "folder or pass overwrite (CLI: --overwrite) to replace them"
        )
    arguments = {
        "axis": _vector(axis, (-1.0, 0.0, 0.0), "axis"),
        "origin": _vector(origin, (0.0, 0.0, 0.0), "origin"),
        "blade_dir": _vector(blade_dir, (0.0, 0.0, 1.0), "blade_dir"),
        "stations": None if stations is None else [float(s) for s in stations],
    }
    frame = BladeFrame(arguments["origin"], arguments["axis"], arguments["blade_dir"])
    result = slice_blade(
        _obj_surface(source, unit, frame), blades=blades, stations=arguments["stations"]
    )
    return _write(source, folder, frame, result, {"blades": blades, "unit": unit, **arguments})


def _write(
    source: Path, folder: Path, frame: BladeFrame, result: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any]:
    sections = folder / "sections"
    if sections.is_dir():
        shutil.rmtree(sections)
    sections.mkdir(parents=True, exist_ok=True)
    names = []
    for i, row in enumerate(result["table"]):
        label = f"s{i:02d}"
        write_text(sections / f"{label}.dat", profile_text(row, label))
        names.append(f"sections/{label}.dat")
        row["airfoil"] = label
    write_text(
        folder / "propeller.toml",
        definition_text(
            result, name=source.stem.upper() or "PROPELLER", frame=frame, profile_files=names
        ),
    )
    Propeller.from_toml(folder / "propeller.toml")
    buffer = io.StringIO()
    writer = csv_writer(buffer)
    writer.writerow(TABLE_COLUMNS)
    for row in result["table"]:
        writer.writerow([row[key] if key == "airfoil" else repr(row[key]) for key in TABLE_COLUMNS])
    write_text(folder / "table.csv", buffer.getvalue())
    record = {
        "geometry": str(source),
        "inputs": inputs,
        "measured": result["measured"],
        "files": {
            "definition": "propeller.toml",
            "table": "table.csv",
            "sections": names,
        },
        "table": [
            {key: value for key, value in row.items() if key != "profile"}
            for row in result["table"]
        ],
    }
    write_text(folder / "prop-slice.json", json.dumps(record, indent=2))
    return record
