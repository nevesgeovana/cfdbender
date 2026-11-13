"""The one mesh resolver: canonical inputs/mesh/, with the legacy folder names still read.

A workspace keeps its meshes and geometry files in ``inputs/mesh/``. Workspaces made
earlier used ``inputs/geometry/``, ``inputs/geometries/`` or ``inputs/meshes/``;
those are read too, in that order after ``inputs/mesh/``, and a legacy name is
reported once per process. Every solver's GEOMETRY and mesh lookup goes through here.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

MESH_FOLDER = "inputs/mesh"
LEGACY_MESH_FOLDERS = ("inputs/geometry", "inputs/geometries", "inputs/meshes")
#: Folders searched for a mesh name, canonical first.
MESH_FOLDERS = (MESH_FOLDER, *LEGACY_MESH_FOLDERS)
#: Path prefixes a setup may use to name a mesh file.
MESH_PREFIXES = tuple(f"{folder}/" for folder in MESH_FOLDERS)

_warned: set[str] = set()


def _note_legacy(folder: str) -> None:
    if folder in _warned:
        return
    _warned.add(folder)
    warnings.warn(
        f"{folder}/ is a legacy mesh folder name; move its files to {MESH_FOLDER}/",
        UserWarning,
        stacklevel=3,
    )


def mesh_folder_of(name: str) -> str | None:
    """Return the mesh folder a workspace-relative path starts with, or None."""
    for folder in MESH_FOLDERS:
        if name.startswith(folder + "/"):
            return folder
    return None


def resolve_mesh_path(root: Path, name: str) -> str:
    """Return the existing file a mesh path names, searching the mesh folders in order.

    Parameters
    ----------
    root : Path
        Resolved workspace root.
    name : str
        Workspace-relative path under any mesh folder name.

    Returns
    -------
    str
        The path in the first folder (``inputs/mesh`` first) that holds the file;
        ``name`` unchanged when it names no mesh folder or no folder holds it.
    """
    folder = mesh_folder_of(name)
    if folder is None:
        return name
    leaf = name[len(folder) + 1 :]
    for candidate in MESH_FOLDERS:
        if (root / candidate / leaf).is_file():
            if candidate != MESH_FOLDER:
                _note_legacy(candidate)
            return f"{candidate}/{leaf}"
    return name


def mesh_candidates(root: Path, cell: str) -> list[str]:
    """Return the workspace-relative files a bare GEOMETRY name or stem can mean.

    The first mesh folder holding a match answers alone, so a canonical file wins
    over a legacy file of the same name.
    """
    for folder in MESH_FOLDERS:
        base = root / folder
        if (base / cell).is_file():
            found = [f"{folder}/{cell}"]
        elif base.is_dir():
            found = [
                p.relative_to(root).as_posix()
                for p in sorted(base.glob(f"{cell}.*"))
                if p.is_file()
            ]
        else:
            found = []
        if found:
            if folder != MESH_FOLDER:
                _note_legacy(folder)
            return found
    return []


def resolve_setup_paths(root: Path, setup: Mapping[str, object]) -> Mapping[str, object]:
    """Return the setup with every top-level mesh path pointing at the file that exists."""
    result = dict(setup)
    for key, value in setup.items():
        if isinstance(value, str) and mesh_folder_of(value) is not None:
            result[key] = resolve_mesh_path(root, value)
    return result
