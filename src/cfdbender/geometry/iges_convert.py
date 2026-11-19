"""Rewrite one or more named IGES files as one IGES of B-spline faces, names kept.

Each file is imported into one OpenCASCADE model (the gmsh kernel), every face is
converted to a NURBS surface, and one IGES is written holding only B-spline surfaces and curves,
the form the solver's CAD import reads. Each component keeps its name: its faces sit on one level
and one 406 form 15 NAME names that level, so the solver and
:func:`cfdbender.geometry._iges_names.read_iges_names` read the same components back. The
first file's length unit is kept and every other file is scaled to it. A compatible
single NURBS source retains its coefficients and trims, with a second native import
checking its faces and names before writing. A record
``<out>.json`` beside the output lists the faces of each component, the digest of every source,
the surface types before and after, and the read-back.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cfdbender._textio as _textio
from cfdbender._digest import file_sha256
from cfdbender._errors import InputArtifactError
from cfdbender._lengths import METRES_PER_UNIT
from cfdbender.geometry._iges_names import (
    DEFAULT_NAME_ENCODING,
    flag_curves_on_surfaces,
    name_faces,
    read_iges_names,
    require_name_encoding,
)
from cfdbender.mesh._gmsh import (
    IMPORT_UNIT,
    gmsh_module,
    import_iges,
    import_named,
    session,
)
from cfdbender.mesh._gmsh_worker import run_isolated

__all__ = ["convert_iges"]

_PURPOSE = "Converting an IGES file to B-spline faces"


def _types(gmsh: Any, faces: Sequence[int]) -> dict[str, int]:
    return dict(Counter(gmsh.model.getType(2, t) for t in faces))


def _centres(gmsh: Any, faces: Sequence[int]) -> list[tuple[float, ...]]:
    return [tuple(gmsh.model.occ.getCenterOfMass(2, t)) for t in faces]


def convert_iges(
    sources: Sequence[str | Path],
    out: str | Path,
    *,
    name_encoding: str = DEFAULT_NAME_ENCODING,
) -> dict[str, Any]:
    """Write the faces of ``sources`` as one IGES of NURBS faces with their component names.

    Parameters
    ----------
    sources : sequence of str or Path
        IGES files, read only; their faces are named by the NAME of their level.
    out : str or Path
        The IGES to write; ``<out>.json`` is written beside it.
    name_encoding : str, optional
        The text encoding of the NAME bytes, read and written; default ``"latin-1"``.

    Returns
    -------
    dict
        The record written to ``<out>.json``.

    Raises
    ------
    MissingExtraError
        The gmsh library cannot load (``KernelUnavailableError``).
    InputArtifactError
        A source cannot be named face by face, the written file does not read back with the
        same components, or the geometry kernel failed (it runs in a worker process of its
        own, so a native fault ends only that worker, is reported with its cause, and the
        conversion is not retried); nothing was written.
    """
    require_name_encoding(name_encoding)
    target = Path(out)
    if target.suffix.lower() not in (".igs", ".iges"):
        raise InputArtifactError(
            f"{target}: the output of iges-convert is an IGES file (.igs or .iges). Nothing was "
            "written.",
            kind="geometry",
        )
    gmsh_module(_PURPOSE)
    return run_isolated(
        f"{__name__}:_convert",
        {
            "sources": [str(s) for s in sources],
            "out": str(target),
            "name_encoding": name_encoding,
        },
        _convert,
    )


def _convert(sources: Sequence[str | Path], out: str | Path, name_encoding: str) -> dict[str, Any]:
    """Run the conversion in this process; :func:`convert_iges` calls it in a worker."""
    target = Path(out)
    with session(_PURPOSE) as gmsh:
        model = import_named(gmsh, sources, name_encoding=name_encoding)
        before = _types(gmsh, model.faces)
        retained = _retained_source(sources, model, before, name_encoding)
        if retained is not None:
            return _retain(gmsh, retained, target, model, before, name_encoding=name_encoding)
        gmsh.model.occ.convertToNURBS([(2, t) for t in model.faces])
        gmsh.model.occ.synchronize()
        after = _types(gmsh, model.faces)
        centres = _centres(gmsh, model.faces)
        size = _diagonal(gmsh)
        # The import is in millimeters; the model now holds the first file's unit, which the
        # writer labels as millimeters, so the unit flag is set back on the named copy below.
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "faces.igs"
            gmsh.write(str(raw))
            gmsh.clear()
            back = _read_back(gmsh, raw, len(model.faces))
            if len(back) != len(model.faces):
                raise InputArtifactError(
                    f"{target}: the written file does not read back with its "
                    f"{len(model.faces)} faces, so no face can carry its name. Nothing was "
                    "written.",
                    kind="geometry",
                )
            distance = max(
                (
                    max(abs(a - b) for a, b in zip(x, y, strict=True))
                    for x, y in zip(centres, back, strict=True)
                ),
                default=0.0,
            )
            # The writer rounds coordinates; a face read back out of order moves by a face.
            tolerance = 1e-4 * max(size, 1.0)
            if distance > tolerance:
                raise InputArtifactError(
                    f"{target}: the written faces do not read back in the order they were "
                    f"written (largest center distance {distance:g} {model.unit}), so no face "
                    "can carry its name. Nothing was written.",
                    kind="geometry",
                )
            named = Path(folder) / "named.igs"
            name_faces(raw, named, model.names, unit=model.unit, name_encoding=name_encoding)
            # The solver's CAD import reads the parameter-space flags the writer leaves wrong.
            staged = flag_curves_on_surfaces(named, Path(folder) / "flagged.igs")
            again = read_iges_names(staged, name_encoding=name_encoding)
            if list(again.faces) != model.names:
                raise InputArtifactError(
                    f"{target}: the named file reads back other components than were written. "
                    "Nothing was written.",
                    kind="geometry",
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            _textio.write_bytes(target, staged.read_bytes())
    return _conversion_record(
        target, model, before, after, distance=distance, name_encoding=name_encoding
    )


def _conversion_record(target, model, before, after, *, distance, name_encoding, retained=False):
    readback = read_iges_names(target, name_encoding=name_encoding)
    record = {
        "output": target.name,
        "sha256": file_sha256(target),
        "unit": model.unit,
        "sources": model.sources,
        "notes": model.notes,
        "faces": len(model.faces),
        "components": readback.components,
        "surface_types_before": before,
        "surface_types_after": after,
        "readback": {
            "components": readback.components,
            "unit": readback.unit,
            "same_components": list(readback.faces) == model.names,
            "largest_centre_distance": distance,
        },
        "import_unit": IMPORT_UNIT,
        "metres_per_unit": float(METRES_PER_UNIT[model.unit]),
    }
    if retained:
        record["readback"]["retained_compatible_nurbs"] = True
    _textio.write_text(Path(str(target) + ".json"), json.dumps(record, indent=2) + "\n")
    return record


def _nurbs_document(document):
    """Recognize the bounded representation already accepted by the native writer."""
    allowed = {102, 110, 124, 126, 128, 142, 144, 402, 406}
    if not {e["type"] for e in document["entities"]} <= allowed:
        return False
    by_id = {e["id"]: e for e in document["entities"]}
    faces = [e for e in document["entities"] if e["type"] == 144]
    loops = [e["parameters"] for e in document["entities"] if e["type"] == 142]
    return (
        bool(faces)
        and bool(loops)
        and all(
            by_id[int(e["parameters"][1])]["type"] == 128 and int(e["parameters"][2]) == 1
            for e in faces
        )
        and all(p[5] == "1" and int(p[3]) in by_id and int(p[4]) in by_id for p in loops)
    )


def _retained_source(sources, model, before, name_encoding):
    from ._iges_write import read_iges

    if len(sources) != 1 or before != {"BSpline surface": len(model.faces)}:
        return None
    source = Path(sources[0])
    names = read_iges_names(source, name_encoding=name_encoding)
    if names.unit != model.unit or list(names.faces) != model.names:
        return None
    try:
        document = read_iges(source)
    except UnicodeError:
        return None
    return source if _nurbs_document(document) else None


def _retain(gmsh, source, target, model, before, *, name_encoding):
    """Preserve ready NURBS geometry and validate a second independent import."""
    centres, size = _centres(gmsh, model.faces), _diagonal(gmsh)
    with tempfile.TemporaryDirectory() as folder:
        staged = flag_curves_on_surfaces(source, Path(folder) / "retained.igs")
        gmsh.clear()
        readback = import_named(gmsh, [staged], name_encoding=name_encoding)
        if len(readback.faces) != len(model.faces) or readback.names != model.names:
            raise InputArtifactError(
                f"{target}: the retained file does not read back with its original faces "
                "and names. Nothing was written.",
                kind="geometry",
            )
        after = _types(gmsh, readback.faces)
        back = _centres(gmsh, readback.faces)
        distance = max(
            (
                max(abs(a - b) for a, b in zip(x, y, strict=True))
                for x, y in zip(centres, back, strict=True)
            ),
            default=0.0,
        )
        if distance > 1e-4 * max(size, 1.0):
            raise InputArtifactError(
                f"{target}: the retained faces do not read back in their original order. "
                "Nothing was written.",
                kind="geometry",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        _textio.write_bytes(target, staged.read_bytes())
    return _conversion_record(
        target, model, before, after, distance=distance, name_encoding=name_encoding, retained=True
    )


def _diagonal(gmsh: Any) -> float:
    box = gmsh.model.getBoundingBox(-1, -1)
    return float(sum((box[i + 3] - box[i]) ** 2 for i in range(3)) ** 0.5)


def _read_back(gmsh: Any, path: Path, expected: int) -> list[tuple[float, ...]]:
    """Import a written IGES into an empty model and return the center of each face."""
    faces = import_iges(gmsh, path)
    gmsh.model.occ.synchronize()
    centres = _centres(gmsh, faces) if len(faces) == expected else []
    gmsh.clear()
    return centres
