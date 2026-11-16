"""Trailing-edge points and node lists written from an OBJ's edges.

Two plain-text products leave this module. The *points file* of a trailing edge holds a
length-unit word on its first line and then one ``x,y,z`` row per mid-point of the mesh
edges on the trailing edge. The *node list* holds the end vertices of those edges as an
``Id, Name, X, Y, Z`` listing, the ids being the vertex numbers of the OBJ, in the form a
mesh-preparation tool exports a node set.

The refine and trailing-edge verbs share these writers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._errors import InputArtifactError
from cfdbender._points import write_trailing_edge_points as write_trailing_edge_points

__all__ = [
    "QUAD_FAMILY_SHARE",
    "node_listing_rows",
    "quad_families",
    "write_level_trailing_edges",
    "write_node_listing",
    "write_trailing_edge_points",
]

#: The share of quadrilaterals above which a family counts as a quad family: a structured
#: family may hold a few triangles at a singular point, while a triangle family holds none.
QUAD_FAMILY_SHARE = 0.8

#: The first line of a node list, the column names a mesh-preparation tool exports.
LISTING_HEADER = "#  Id, Name, X , Y, Z"


def quad_families(families: Mapping[str, Any]) -> list[str]:
    """Return the families whose faces are quadrilaterals for :data:`QUAD_FAMILY_SHARE`.

    Parameters
    ----------
    families : mapping of str to list of faces
        Each family's faces, each a sequence of vertex indices.

    Returns
    -------
    list of str
        The quad families, in the order of ``families``; a family with no face is not one.
    """
    return [
        name
        for name, faces in families.items()
        if faces and sum(len(face) == 4 for face in faces) >= QUAD_FAMILY_SHARE * len(faces)
    ]


def node_listing_rows(
    verts: NDArray[numpy.float64], found: Mapping[str, set[tuple[int, int]]]
) -> list[str]:
    """Return the rows of a node list for the end vertices of each family's edges.

    The nodes of a family are listed once, in order of their second coordinate, with the
    vertex number of the OBJ (1-based) as the id and the family as the name.
    """
    rows = [LISTING_HEADER]
    for name, edges in found.items():
        for v in sorted({v for e in edges for v in e}, key=lambda v: float(verts[v, 1])):
            px, py, pz = (repr(float(c)) for c in verts[v])
            rows.append(f"{v + 1}, {name}, {px}, {py}, {pz}")
    return rows


def write_node_listing(
    path: Path, verts: NDArray[numpy.float64], found: Mapping[str, set[tuple[int, int]]]
) -> Path:
    """Write the node list of the end vertices of ``found`` and return its path."""
    if not math.isfinite(float(numpy.asarray(verts, dtype=float).sum())):
        raise InputArtifactError(
            "the vertices carry a coordinate that is not a finite number", kind="trailing_edges"
        )
    _textio.write_text(path, "\n".join(node_listing_rows(verts, found)) + "\n")
    return path


def write_level_trailing_edges(
    verts: NDArray[numpy.float64],
    found: dict[str, set[tuple[int, int]]],
    folder: Path,
    stem: str,
) -> list[Path]:
    """Write a refined level's trailing-edge node lists, all families and each family.

    ``<stem>.nodes.txt`` lists the end vertices of every family's trailing-edge edges and
    ``<stem>.<family>.nodes.txt`` those of one family, as ``Id, Name, X, Y, Z`` rows whose
    ids are the vertex numbers of the level's OBJ. Families with no edge write nothing; an
    empty ``found`` writes nothing.

    Returns
    -------
    list of pathlib.Path
        The files written: the list of every family, then each family's.
    """
    found = {name: edges for name, edges in found.items() if edges}
    if not found:
        return []
    written = [write_node_listing(folder / f"{stem}.nodes.txt", verts, found)]
    for name, edges in found.items():
        written.append(
            write_node_listing(folder / f"{stem}.{name}.nodes.txt", verts, {name: edges})
        )
    return written
