"""Scale an OBJ mesh uniformly about a point into a new output folder.

Pipeline role: the mesh operation behind ``cfd-mesh scale``. Every
vertex ``p`` of the OBJ becomes ``c + factor * (p - c)`` for the scale centre ``c`` (the origin
unless ``about`` is given), a homothety: lengths grow by the factor, angles and the face
topology do not change. The file is rewritten line by line as bytes and only the three
coordinates of each ``v`` line change; face order, group and component names, normals, texture
coordinates, comments and line endings stay byte for byte, because a solver's results depend on
the face order. The scaled OBJ is written under the input's own name into an output folder the
verb creates (``<stem>_scaled<factor>`` beside the input by default), with a JSON record beside
it: the input digest, the factor, the centre, the vertex and face counts and the bounding boxes
before and after. The input is never overwritten, and an output folder that already holds a file
of that name is refused unless ``overwrite`` is given.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from cfdbender._digest import file_sha256
from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender._textio import write_bytes, write_json

__all__ = ["scale_mesh"]

_VERTEX = re.compile(rb"^([ \t]*v[ \t]+)(\S+)([ \t]+)(\S+)([ \t]+)(\S+)(.*)$", re.DOTALL)
_FACE = re.compile(rb"^[ \t]*f[ \t]")


def parse_factor(factor: object) -> float:
    """Return the factor as a finite positive float, or refuse it by name."""
    try:
        value = float(factor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InputValueError(
            f"scale: the factor {factor!r} is not a number. Give a positive number such as 0.5 "
            "(a length multiplier); no file was written."
        ) from None
    if not math.isfinite(value) or value <= 0.0:
        raise InputValueError(
            f"scale: the factor {factor!r} is not a positive finite number. A zero collapses "
            "the mesh to a point and a negative factor mirrors it; give a factor above zero, "
            "such as 0.5. No file was written."
        )
    return value


def parse_about(about: object) -> tuple[float, float, float]:
    """Return the scale centre as three finite floats from ``x,y,z`` text or a sequence."""
    parts = about.split(",") if isinstance(about, str) else list(about)  # type: ignore[call-overload]
    try:
        point = tuple(float(p) for p in parts)
    except (TypeError, ValueError):
        point = ()
    if len(point) != 3 or not all(math.isfinite(c) for c in point):
        raise InputValueError(
            f"scale: the scale centre {about!r} is not three finite numbers. Give it as "
            "x,y,z, for example 0,0,1.5. No file was written."
        )
    return point[0], point[1], point[2]


def _scaled_line(line: bytes, factor: float, about: tuple[float, ...]) -> tuple[bytes, list[float]]:
    """Return one ``v`` line with its three coordinates scaled, and the new coordinates."""
    match = _VERTEX.match(line)
    if match is None:
        raise InputValueError(
            f"scale: the vertex line {line[:60]!r} does not hold three coordinates. Repair "
            "the OBJ so every v line has x y z; no file was written."
        )
    head, tail = match.group(1, 7)
    old = [match.group(i) for i in (2, 4, 6)]
    try:
        coords = [float(c) for c in old]
    except ValueError:
        raise InputValueError(
            f"scale: the vertex line {line[:60]!r} holds a coordinate that is not a number. "
            "Repair the OBJ; no file was written."
        ) from None
    new = [c0 + factor * (c - c0) for c, c0 in zip(coords, about, strict=True)]
    if not all(math.isfinite(c) for c in new):
        raise InputValueError(
            f"scale: the vertex line {line[:60]!r} is not finite after scaling by {factor!r}. "
            "Use a smaller factor or repair the OBJ; no file was written."
        )
    seps = (match.group(3), match.group(5))
    text = head + repr(new[0]).encode() + seps[0] + repr(new[1]).encode()
    return text + seps[1] + repr(new[2]).encode() + tail, new


def _bounds(points: list[list[float]]) -> dict[str, list[float]]:
    columns = list(zip(*points, strict=True))
    return {"min": [min(c) for c in columns], "max": [max(c) for c in columns]}


def _rewrite(data: bytes, factor: float, about: tuple[float, ...]) -> tuple[bytes, dict[str, Any]]:
    """Return the scaled bytes and the counts and bounding boxes before and after."""
    lines = data.split(b"\n")
    before: list[list[float]] = []
    after: list[list[float]] = []
    faces = 0
    for index, line in enumerate(lines):
        if line.lstrip(b" \t").startswith((b"v ", b"v\t")):
            lines[index], moved = _scaled_line(line, factor, about)
            after.append(moved)
            before.append([float(c) for c in _VERTEX.match(line).group(2, 4, 6)])  # type: ignore[union-attr]
        elif _FACE.match(line):
            faces += 1
    if not before:
        raise InputValueError(
            "scale: the OBJ holds no vertex line, so there is nothing to scale. Give the OBJ "
            "of a mesh; no file was written."
        )
    facts = {
        "vertices": len(before),
        "faces": faces,
        "bounding_box_before": _bounds(before),
        "bounding_box_after": _bounds(after),
    }
    return b"\n".join(lines), facts


def _output_folder(source: Path, factor: float, out: str | Path | None) -> Path:
    if out is not None:
        return Path(out)
    return source.parent / f"{source.stem}_scaled{factor:g}"


def _check_targets(source: Path, targets: list[Path], overwrite: bool) -> None:
    for target in targets:
        if target.exists() and target.resolve() == source.resolve():
            raise InputValueError(
                f"scale: the output {str(target)!r} is the input file itself and the input is "
                "never overwritten. Give out (CLI: --out) a different folder; no file was written."
            )
        if target.exists() and not overwrite:
            raise InputValueError(
                f"scale: {str(target)!r} already exists in the output folder. Remove it, give "
                "another out (CLI: --out) folder or pass overwrite (CLI: --overwrite) to "
                "replace it; no file was written."
            )


def scale_mesh(
    mesh: str | Path,
    factor: float | str,
    *,
    out: str | Path | None = None,
    about: str | tuple[float, float, float] | None = None,
    overwrite: bool = False,
    name: str | None = None,
) -> dict[str, Any]:
    """Scale every vertex of an OBJ by ``factor`` about a point into a new folder.

    Parameters
    ----------
    mesh : str or Path
        The source OBJ, read only.
    factor : float or str
        The uniform length multiplier, finite and above zero (text such as ``"0.5"`` is read).
    out : str or Path, optional
        The output folder, created with its parents. The scaled OBJ goes there under the input
        file name, with ``<name>.scale.json`` beside it. Default: ``<stem>_scaled<factor>``
        beside the input.
    about : str or tuple of float, optional
        The scale centre ``x,y,z``; default the origin.
    overwrite : bool, default False
        Replace a same-name file already in the output folder instead of refusing.
    name : str, optional
        The file name of the scaled OBJ in the output folder; default the input file name.

    Returns
    -------
    dict
        The record also written to JSON: input name and sha256, factor, centre, output paths,
        vertex and face counts and the bounding boxes before and after.

    Raises
    ------
    InputValueError
        The factor or centre is unusable, the OBJ holds no usable vertex, or the output would
        overwrite a file without ``overwrite``; nothing is written.
    InputArtifactError
        The mesh file does not exist.
    """
    value = parse_factor(factor)
    centre = parse_about(about) if about is not None else (0.0, 0.0, 0.0)
    source = Path(mesh)
    if source.suffix.lower() == ".fsm":
        raise InputValueError(
            "scale: a saved simulation (.fsm) is not accepted here. Give an OBJ as mesh "
            "(CLI: --mesh); no file was written."
        )
    if not source.is_file():
        raise InputArtifactError(
            f"scale: the mesh file {str(source)!r} does not exist. Give the path of an OBJ "
            "as mesh (CLI: --mesh); no file was written."
        )
    folder = _output_folder(source, value, out)
    out_name = name or source.name
    target, record_path = folder / out_name, folder / f"{out_name}.scale.json"
    _check_targets(source, [target, record_path], overwrite)
    data = source.read_bytes()
    scaled, facts = _rewrite(data, value, centre)
    record = {
        "input": source.name,
        "input_sha256": file_sha256(source),
        "factor": value,
        "about": list(centre),
        "output": str(target),
        **facts,
    }
    folder.mkdir(parents=True, exist_ok=True)
    write_bytes(target, scaled)
    write_json(record_path, record)
    record["record"] = str(record_path)
    return record
