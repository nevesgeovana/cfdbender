"""Read a shell mesh from an ANSA NASTRAN export, with the FE perimeters ANSA writes beside it.

prepare-to-refine reads this in place of an OBJ when the patches must follow
the pre-processor's meshing macros: an OBJ carries only the families, while a
NASTRAN export written with the FE perimeters on (``comment_output_fe_perimeters``)
carries, in the comment ``$ANSA_FE_PERIMETER``, every element side that lies on
a perimeter. Cutting each family along those sides gives the macros.

Read: ``GRID``, ``CQUAD4``, ``CTRIA3`` and ``PSHELL`` in small (8 column),
large (``*``, 16 column) or free (comma) field, and the property name ANSA
writes as a ``$<name>`` comment right before each ``PSHELL``. Every other card
is read past. The perimeter block is base64 text over ``$`` lines: a 4-byte
big-endian length, then zlib data holding a big-endian 32-bit count and, per
side, a 32-bit element id and a 16-bit side number (1 to 4, side k joining an
element's nodes k and k+1). That layout is read from the files ANSA writes; a
block that does not read is warned and skipped, and the families are then cut
at their sharp edges as for an OBJ.
"""

from __future__ import annotations

import base64
import binascii
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy

from cfdbender._errors import GeometryWarning, InputArtifactError, InputValueError, warn
from cfdbender._lengths import scale
from cfdbender.mesh._refine._obj import KIND, Faces, ObjMesh

#: The suffixes read as a NASTRAN export.
SUFFIXES = (".nas", ".bdf", ".dat", ".nastran")
#: The comment that opens ANSA's perimeter block.
PERIMETER_TAG = "$ANSA_FE_PERIMETER;"
#: The shells read, with their number of nodes.
SHELLS = {"CQUAD4": 4, "CTRIA3": 3}

_EXPONENT = re.compile(r"(?<=[0-9.])([+-])(?=\d)")


@dataclass(frozen=True)
class NastranMesh:
    """A NASTRAN shell mesh as the refinement reads it, and the perimeter edges ANSA wrote.

    ``perimeter`` holds each perimeter edge as a pair of vertex indices of
    ``mesh`` (low first); it is empty when the export carries no block.
    """

    mesh: ObjMesh
    perimeter: frozenset[tuple[int, int]]


def is_nastran(path: str | Path) -> bool:
    """Return whether the path names a NASTRAN export by its suffix."""
    return Path(path).suffix.lower() in SUFFIXES


def real(text: str) -> float:
    """Return a NASTRAN real, whose exponent letter may be left out (``1.5-3`` is 1.5e-3)."""
    text = text.strip().upper().replace("D", "E")
    if "E" not in text:
        text = _EXPONENT.sub(r"E\1", text)
    return float(text)


def _fields(line: str, wide: bool) -> list[str]:
    """Return the data fields of one card line (the name or continuation field dropped)."""
    if "," in line:
        return [f.strip() for f in line.split(",")[1:]]
    width = 16 if wide else 8
    body = line[8:72]
    return [body[k : k + width].strip() for k in range(0, len(body), width)]


def _cards(lines: list[str]) -> list[tuple[str, list[str], str | None]]:
    """Return each card's name, its fields over its continuation lines, and the comment before."""
    cards: list[tuple[str, list[str], str | None]] = []
    comment: str | None = None
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("$"):
            comment = line[1:].strip() or comment
            continue
        if line[0] in "*+ " and cards:
            name, fields, before = cards[-1]
            cards[-1] = (name, fields + _fields(line, name.endswith("*")), before)
            continue
        name = (line.split(",")[0] if "," in line else line[:8]).strip().upper()
        cards.append((name, _fields(line, name.endswith("*")), comment))
        comment = None
    return cards


def _perimeter_block(lines: list[str]) -> str | None:
    """Return the perimeter comment's text over its ``$`` lines, or None when there is none."""
    for number, line in enumerate(lines):
        if line.startswith(PERIMETER_TAG):
            parts = [line[len(PERIMETER_TAG) :]]
            for more in lines[number + 1 :]:
                if not more.startswith("$") or more.startswith("$$"):
                    break
                parts.append(more[1:])
            return "".join(p.strip() for p in parts)
    return None


def _side_records(segment: str) -> list[tuple[int, int]] | None:
    """Return the (element id, side) records of one block, or None for a block of another layout."""
    raw = base64.b64decode(segment + "=" * (-len(segment) % 4))
    declared = struct.unpack(">I", raw[:4])[0]
    body = zlib.decompress(raw[4:])
    count = struct.unpack(">I", body[:4])[0]
    if len(body) != declared or len(body) != 4 + 6 * count:
        return None
    return [struct.unpack(">IH", body[4 + 6 * k : 10 + 6 * k]) for k in range(count)]


def decode_sides(text: str) -> list[tuple[int, int]]:
    """Return the (element id, side) records of every side block of the comment.

    The comment holds ``;``-separated fields: group numbers and base64 blocks.
    The side blocks are the ones whose body is a count and 6-byte records; the
    other blocks (lists of 4-byte ids) are read past, and so is a field that is not
    a zlib block at all. Raise ValueError when no side block reads.
    """
    sides: list[tuple[int, int]] = []
    found = False
    for segment in text.split(";"):
        if len(segment) < 8:
            continue
        try:
            records = _side_records(segment)
        except (binascii.Error, zlib.error, struct.error):
            continue
        if records is not None:
            found = True
            sides.extend(records)
    if not found:
        raise InputValueError(
            "no block of element sides; check that the file is a refined-mesh export"
        )
    return sides


def _grids(cards) -> tuple[dict[int, int], numpy.ndarray]:
    ids: dict[int, int] = {}
    points: list[list[float]] = []
    for name, fields, _ in cards:
        if name.rstrip("*") == "GRID":
            ids[int(fields[0])] = len(points)
            points.append([real(fields[k]) for k in (2, 3, 4)])
    return ids, numpy.array(points, dtype=float).reshape(-1, 3)


def _properties(cards) -> dict[int, str]:
    names: dict[int, str] = {}
    for name, fields, comment in cards:
        if name.rstrip("*") == "PSHELL":
            pid = int(fields[0])
            names[pid] = comment if comment and not comment.startswith("ANSA") else f"PID_{pid}"
    return names


def _shells(cards, ids: dict[int, int], path: Path) -> dict[int, tuple[int, list[int]]]:
    shells: dict[int, tuple[int, list[int]]] = {}
    for name, fields, _ in cards:
        size = SHELLS.get(name.rstrip("*"))
        if size is None:
            continue
        try:
            nodes = [ids[int(g)] for g in fields[2 : 2 + size]]
        except (KeyError, ValueError) as error:
            raise InputArtifactError(
                f"{path}: element {fields[0]} names a node that is not a GRID ({error}). "
                "Fix the file and run again. Nothing was written.",
                kind=KIND,
            ) from error
        shells[int(fields[0])] = (int(fields[1]), nodes)
    return shells


def _perimeter_edges(lines, shells, path: Path) -> set[tuple[int, int]]:
    text = _perimeter_block(lines)
    if text is None:
        warn(
            f"{path}: holds no {PERIMETER_TAG.rstrip(';')} block (export with the FE perimeters "
            "on); "
            "the families are cut at their sharp edges, as for an OBJ",
            GeometryWarning,
            stacklevel=3,
        )
        return set()
    try:
        sides = decode_sides(text)
    except (ValueError, binascii.Error, zlib.error, struct.error) as error:
        warn(
            f"{path}: the {PERIMETER_TAG.rstrip(';')} block does not read ({error}); skipped, "
            "and the "
            "families are cut at their sharp edges, as for an OBJ",
            GeometryWarning,
            stacklevel=3,
        )
        return set()
    edges: set[tuple[int, int]] = set()
    unknown = 0
    for element, side in sides:
        if element not in shells or not 1 <= side <= len(shells[element][1]):
            unknown += 1
            continue
        nodes = shells[element][1]
        a, b = nodes[side - 1], nodes[side % len(nodes)]
        edges.add((min(a, b), max(a, b)))
    if unknown:
        warn(
            f"{path}: {unknown} perimeter side(s) name no shell of the file; skipped",
            GeometryWarning,
            stacklevel=3,
        )
    return edges


def read_nastran(
    path: str | Path, *, unit: str = "MILLIMETER", obj_unit: str = "METER"
) -> NastranMesh:
    """Read a NASTRAN shell mesh, its families by property name, and ANSA's perimeter edges.

    Parameters
    ----------
    path : str or pathlib.Path
        The NASTRAN export.
    unit : str
        The length unit of the file's coordinates (default MILLIMETER, ANSA's default).
    obj_unit : str
        The length unit the mesh is returned in (default METER).

    Returns
    -------
    NastranMesh
        The mesh (families in property order, faces in element order, vertices in
        GRID order, unused ones dropped) and the perimeter edges.

    Raises
    ------
    InputArtifactError
        The file is not readable, holds no shell, a shell names a missing GRID, or
        a unit has no length scale.
    """
    source = Path(path)
    factor = scale(unit.upper(), obj_unit.upper())
    if factor is None:
        raise InputArtifactError(
            f"{source}: the unit {unit} or {obj_unit} names no length scale. Nothing was written.",
            kind=KIND,
        )
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise InputArtifactError(
            f"{source}: not readable ({error}). Nothing was written.", kind=KIND
        ) from error
    cards = _cards(lines)
    ids, verts = _grids(cards)
    shells = _shells(cards, ids, source)
    if not shells:
        raise InputArtifactError(
            f"{source}: holds no CQUAD4 or CTRIA3 shell. Nothing was written.", kind=KIND
        )
    names = _properties(cards)
    families: dict[int, Faces] = {}
    for _, (pid, nodes) in sorted(shells.items()):
        families.setdefault(pid, []).append(nodes)
    named: dict[str, Faces] = {}
    for pid in sorted(families):
        named.setdefault(names.get(pid, f"PID_{pid}"), []).extend(families[pid])
    edges = _perimeter_edges(lines, shells, source)
    # The vertices keep the GRID order; a GRID no shell uses is dropped.
    order = sorted({v for faces in named.values() for f in faces for v in f})
    renumber = {old: new for new, old in enumerate(order)}
    families_out = {n: [[renumber[v] for v in f] for f in faces] for n, faces in named.items()}
    perimeter = frozenset(
        (min(renumber[a], renumber[b]), max(renumber[a], renumber[b])) for a, b in edges
    )
    header = [f"# from {source.name} (NASTRAN, {unit.upper()} written in {obj_unit.upper()})"]
    return NastranMesh(ObjMesh(verts[order] * factor, families_out, header), perimeter)
