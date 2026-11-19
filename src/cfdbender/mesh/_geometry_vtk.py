"""The preview of a mesh: a legacy VTK file colored by boundary.

:func:`write_preview_vtk` writes the surface of an OBJ with one cell array ``boundary``
(the family's position, counted from 1, so ParaView colors each surface) and a field
``names`` listing the families in order. :func:`write_sweep_vtk` overlays one control surface
of several variants in one file, a cell array ``variant`` telling them apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cfdbender._textio as _textio
from cfdbender._errors import InputArtifactError
from cfdbender.mesh._refine._obj import read_obj

__all__ = ["PREVIEW_SUFFIX", "SWEEP_SUFFIX", "write_preview_vtk", "write_sweep_vtk"]

#: The suffix of the preview file: ``<name>.preview.vtk``.
PREVIEW_SUFFIX = ".preview.vtk"

#: The suffix of the overlaid preview of a sweep: ``<name>.sweep.vtk``.
SWEEP_SUFFIX = ".sweep.vtk"

Point = tuple[float, float, float]


def _from_the_mesh(
    obj: Path, names: Sequence[str]
) -> tuple[list[Point], list[tuple[int, ...]], list[int], list[tuple[int, int]]]:
    mesh = read_obj(obj)
    vertices = [(float(x), float(y), float(z)) for x, y, z in mesh.verts.tolist()]
    faces: list[tuple[int, ...]] = []
    boundary_of: list[int] = []
    for position, (family, family_faces) in enumerate(mesh.families.items(), start=1):
        boundary_id = list(names).index(family) + 1 if family in names else position
        for face in family_faces:
            faces.append(tuple(int(index) for index in face))
            boundary_of.append(boundary_id)
    return vertices, faces, boundary_of, []


def write_preview_vtk(target: Path, *, obj: Path, names: Sequence[str]) -> Path:
    """Write the preview of one built body from its OBJ and return its path.

    Parameters
    ----------
    target : Path
        The ``<name>.preview.vtk`` to write.
    obj : Path
        The exported mesh; its families are the boundaries.
    names : sequence of str
        The boundaries, in order.
    """
    vertices, faces, boundary_of, lines = _from_the_mesh(obj, names)
    rows = [
        "# vtk DataFile Version 3.0",
        "cfdbender geometry preview",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(vertices)} double",
        *(f"{x!r} {y!r} {z!r}" for x, y, z in vertices),
        f"POLYGONS {len(faces)} {sum(len(face) + 1 for face in faces)}",
        *(" ".join(str(value) for value in (len(face), *face)) for face in faces),
    ]
    if lines:
        rows += [f"LINES {len(lines)} {3 * len(lines)}", *(f"2 {a} {b}" for a, b in lines)]
    rows += [
        f"CELL_DATA {len(faces) + len(lines)}",
        "SCALARS boundary int 1",
        "LOOKUP_TABLE default",
        # POLYDATA numbers its cells lines first, then polygons: a line is boundary 0.
        *("0" for _ in lines),
        *(str(boundary_id) for boundary_id in boundary_of),
        "FIELD names 1",
        f"names 1 {len(names)} string",
        *names,
    ]
    _textio.write_lines(target, rows)
    return target


def _surface_of(obj: Path, surface: str) -> tuple[list[Point], list[tuple[int, ...]]]:
    """Return the faces of the families named after one control surface, and their vertices."""
    mesh = read_obj(obj)
    vertices = tuple((float(x), float(y), float(z)) for x, y, z in mesh.verts.tolist())
    faces = [
        tuple(int(index) for index in face)
        for family, family_faces in mesh.families.items()
        if family == surface or family.startswith(surface + "_")
        for face in family_faces
    ]
    used = sorted({index for face in faces for index in face})
    number = {index: new for new, index in enumerate(used)}
    return [vertices[i] for i in used], [tuple(number[i] for i in face) for face in faces]


def write_sweep_vtk(
    target: Path, variants: Sequence[tuple[int, str, Path]], surface: str
) -> Path | None:
    """Overlay the control surface of every built variant in one file .

    Parameters
    ----------
    target : Path
        The ``<name>.sweep.vtk`` to write.
    variants : sequence of (ordinal, tag, OBJ)
        The variants built, in the order of ``angle_deg``, carrying their original
        one-based positions even when only a subset is built.
    surface : str
        The swept control surface; its boundaries are those named after it.

    Returns
    -------
    Path or None
        The file, or None when no variants were supplied.

    Raises
    ------
    InputArtifactError
        A variant cannot be read or has no faces of the selected surface.
    """
    points: list[Point] = []
    faces: list[tuple[int, ...]] = []
    boundary_of: list[int] = []
    tags: list[str] = []
    for ordinal, tag, obj in variants:
        vertices, local = _surface_of(obj, surface)
        if not local:
            raise InputArtifactError(f"variant {tag}: no faces of control surface {surface}")
        tags.append(tag)
        offset = len(points)
        points += vertices
        faces += [tuple(offset + index for index in face) for face in local]
        boundary_of += [ordinal] * len(local)
    if not tags:
        return None
    rows = [
        "# vtk DataFile Version 3.0",
        f"cfdbender sweep preview of {surface}",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} double",
        *(f"{x!r} {y!r} {z!r}" for x, y, z in points),
        f"POLYGONS {len(faces)} {sum(len(face) + 1 for face in faces)}",
        *(" ".join(str(value) for value in (len(face), *face)) for face in faces),
        f"CELL_DATA {len(faces)}",
        "SCALARS variant int 1",
        "LOOKUP_TABLE default",
        *(str(boundary_id) for boundary_id in boundary_of),
        "FIELD tags 1",
        f"tags 1 {len(tags)} string",
        *tags,
    ]
    _textio.write_lines(target, rows)
    return target
