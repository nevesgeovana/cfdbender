"""Surface mesh files: read a mesh, one OBJ group, weld coincident vertices, cut by a plane.

Pipeline role: below the geometry and mesh layers, imported by the ones that need a surface
mesh and importing nothing from them. ``trimesh`` is a base dependency of cfdbender; it is
imported at call time so that importing :mod:`cfdbender` does not pay for it, and its absence
is an incomplete installation, answered by the interpreter's own ``ModuleNotFoundError``. The
OBJ group reader keeps the face order of the file, because a panel solver can depend on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import InputValueError

__all__ = ["mesh_reader", "obj_group_surface", "plane_cut_pieces", "read_mesh", "weld"]


def mesh_reader() -> Any:
    """Return the ``trimesh`` module, imported at call time.

    The ONE accessor. Every site in this package that needs the reader
    calls this rather than writing its own deferred import, so there is
    a single place where the import cost is paid and a single place a
    later decision about the reader has to be made.

    Returns
    -------
    module
        The imported ``trimesh`` module.

    Raises
    ------
    ModuleNotFoundError
        If trimesh is not installed. Deliberately not translated into a
        refusal of this package's own: see this module's docstring, it
        is a declared runtime dependency and its absence is an
        incomplete installation rather than a missing extra.

    Examples
    --------
    >>> from cfdbender.mesh.objio import mesh_reader
    >>> mesh_reader().__name__
    'trimesh'
    """
    import trimesh

    return trimesh


def read_mesh(path: str | Path) -> Any:
    """Load one surface mesh file through :func:`mesh_reader`.

    Parameters
    ----------
    path : str or pathlib.Path
        Mesh file (``.obj`` or ``.stl``), in the simulation's own length
        units and reference frame.

    Returns
    -------
    trimesh.Trimesh
        The loaded mesh. Nothing here inspects it; the caller states
        what it needs of it (watertightness, a spanwise extent) and
        refuses in its own vocabulary.

    Notes
    -----
    ``load_mesh`` is the typed loader; the ``load(force="mesh")`` form is
    the deprecated content-dependent compatibility wrapper.
    """
    return mesh_reader().load_mesh(str(path))


def obj_group_surface(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    r"""Read the faces of one object or group of an OBJ, as vertices and 0-based triangles.

    Polygons are split into a fan of triangles. The vertices are those the group's
    faces use. A coupled blade is its group of the imported OBJ.

    Parameters
    ----------
    path : pathlib.Path
        The OBJ file.
    name : str
        The ``o`` or ``g`` name of the group, a boundary name of the mesh.

    Returns
    -------
    tuple of numpy.ndarray
        The vertices ``(v, 3)`` and the 0-based triangles ``(f, 3)``.

    Raises
    ------
    InputValueError
        If the file holds no face of ``name``.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     obj = Path(folder) / "blade.obj"
    ...     _ = obj.write_text("g blade\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    ...     vertices, triangles = obj_group_surface(obj, "blade")
    >>> vertices.shape, triangles.tolist()
    ((3, 3), [[0, 1, 2]])
    """
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    current = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        if words[0] in ("o", "g") and len(words) > 1:
            current = words[1]
        elif words[0] == "v" and len(words) >= 4:
            vertices.append([float(word) for word in words[1:4]])
        elif words[0] == "f" and current == name:
            index = [int(word.split("/", 1)[0]) for word in words[1:]]
            index = [i - 1 if i > 0 else len(vertices) + i for i in index]
            triangles.extend([index[0], index[k], index[k + 1]] for k in range(1, len(index) - 1))
    if not triangles:
        raise InputValueError(f"{path.name} holds no face of {name!r}; the blade cannot be placed")
    faces = np.asarray(triangles, dtype=np.int64)
    used, inverse = np.unique(faces, return_inverse=True)
    return np.asarray(vertices, dtype=float)[used], inverse.reshape(-1, 3)


def weld(vertices: Any, faces: Any, fraction: float = 1.0e-9) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices and drop the triangles that collapse.

    A seam written twice (two vertices at one point) would otherwise open the surface
    for a cut that follows the mesh's edges.

    Parameters
    ----------
    vertices : array_like, shape (v, 3)
        The vertices.
    faces : array_like, shape (f, 3)
        The 0-based triangles.
    fraction : float, optional
        Vertices closer than this fraction of the mesh's extent are one vertex.

    Returns
    -------
    tuple of numpy.ndarray
        The merged vertices and the triangles indexing them.
    """
    points = np.asarray(vertices, dtype=float)
    span = float(np.max(np.ptp(points, axis=0))) or 1.0
    keys = np.round(points / (span * fraction))
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    triangles = inverse.reshape(-1)[np.asarray(faces, dtype=np.int64)]
    a, b, c = triangles.T
    return points[first], triangles[(a != b) & (b != c) & (a != c)]


def plane_cut_pieces(
    points: np.ndarray, triangles: np.ndarray, height: float
) -> list[tuple[np.ndarray, bool]] | None:
    """Chain the cut of a mesh by the plane z = ``height`` into pieces.

    Each crossing triangle links the two of its edges the plane crosses, so a piece
    follows the mesh's own connectivity: a loop where the surface closes around the
    plane, an open chain where it ends. A vertex on the plane counts as above it.

    Parameters
    ----------
    points : numpy.ndarray, shape (v, 3)
        The vertices.
    triangles : numpy.ndarray, shape (f, 3)
        The 0-based triangles.
    height : float
        The plane's z.

    Returns
    -------
    list of (numpy.ndarray, bool) or None
        Each piece's x, y points in order and whether it closes; None where an edge is
        crossed by more than two triangles, so the pieces branch.
    """
    depth = points[:, 2] - height
    above = depth >= 0.0
    corners = above[triangles]
    crossing = triangles[corners.any(axis=1) & ~corners.all(axis=1)]
    if not len(crossing):
        return []
    start, end = crossing, np.roll(crossing, -1, axis=1)
    count = len(points)
    keys = np.minimum(start, end) * count + np.maximum(start, end)
    edges, links = np.unique(keys[above[start] != above[end]], return_inverse=True)
    first, second = edges // count, edges % count
    share = depth[first] / (depth[first] - depth[second])
    xy = (points[first] + share[:, None] * (points[second] - points[first]))[:, :2]
    chains = _chains(links.reshape(-1, 2), len(edges))
    return None if chains is None else [(xy[chain], closed) for chain, closed in chains]


def _chains(links: np.ndarray, count: int) -> list[tuple[list[int], bool]] | None:
    """Walk the links into chains, open ones from their ends first; None at a branch."""
    neighbours: list[list[int]] = [[] for _ in range(count)]
    for u, v in links.tolist():
        neighbours[u].append(v)
        neighbours[v].append(u)
    if any(len(near) > 2 for near in neighbours):
        return None
    seen = [False] * count
    ends = [i for i in range(count) if len(neighbours[i]) == 1]
    chains = []
    for start in [*ends, *range(count)]:
        if seen[start]:
            continue
        chain, previous, current = [start], -1, start
        seen[start] = True
        while step := [k for k in neighbours[current] if k != previous and not seen[k]]:
            previous, current = current, step[0]
            seen[current] = True
            chain.append(current)
        chains.append((chain, len(neighbours[start]) == 2))
    return chains
