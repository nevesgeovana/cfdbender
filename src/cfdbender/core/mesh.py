"""The one mesh resolver: canonical inputs/mesh/, with the legacy folder names still read.

A workspace keeps its meshes and geometry files in ``inputs/mesh/``. Workspaces made
earlier used ``inputs/geometry/``, ``inputs/geometries/`` or ``inputs/meshes/``;
those are read too, in that order after ``inputs/mesh/``, and a legacy name is
reported once per process. Every solver's MESH and mesh lookup goes through here.

Generated geometry lives in ``inputs/geometry/<name>/``, a folder holding the sidecar
``geometry.json``. Such a folder is not a legacy mesh folder: a path inside it is used as
named, with no legacy warning and never replaced by a same-named file of ``inputs/mesh/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from cfdbender.core.refusals import warn_or_refuse

MESH_FOLDER = "inputs/mesh"
LEGACY_MESH_FOLDERS = ("inputs/geometry", "inputs/geometries", "inputs/meshes")
#: Folders searched for a mesh name, canonical first.
MESH_FOLDERS = (MESH_FOLDER, *LEGACY_MESH_FOLDERS)
#: Path prefixes a setup may use to name a mesh file.
MESH_PREFIXES = tuple(f"{folder}/" for folder in MESH_FOLDERS)
#: The folder whose named subfolders hold generated geometry.
GEOMETRY_FOLDER = "inputs/geometry"
#: The sidecar that marks ``inputs/geometry/<name>/`` as generated geometry.
GEOMETRY_SIDECAR = "geometry.json"
#: The sidecar a cfd-mesh verb writes beside each mesh, recording how it was made.
PROVENANCE_SUFFIX = ".provenance.json"
#: Files of a mesh folder that describe a mesh and are never a mesh themselves.
SIDECAR_SUFFIXES = (PROVENANCE_SUFFIX, ".boundaries.toml")

_warned: set[str] = set()


def _note_legacy(folder: str) -> None:
    if folder in _warned:
        return
    warn_or_refuse(
        f"{folder}/ is a legacy mesh folder name; move its files to {MESH_FOLDER}/",
        UserWarning,
        stacklevel=3,
    )
    _warned.add(folder)


def _note_both(folder: str, leaf: str) -> None:
    key = f"{folder}/{leaf}"
    if key in _warned:
        return
    warn_or_refuse(
        f"{key} and {MESH_FOLDER}/{leaf} both exist; the setup names {key}, so that file is used",
        UserWarning,
        stacklevel=3,
    )
    _warned.add(key)


def mesh_folder_of(name: str) -> str | None:
    """Return the mesh folder a workspace-relative path starts with, or None."""
    for folder in MESH_FOLDERS:
        if name.startswith(folder + "/"):
            return folder
    return None


def generated_geometry_of(root: Path, name: str) -> str | None:
    """Return the generated-geometry folder a workspace-relative path lies in, or None.

    Parameters
    ----------
    root : Path
        Resolved workspace root.
    name : str
        Workspace-relative path.

    Returns
    -------
    str or None
        ``inputs/geometry/<name>`` when the path lies in that folder and the folder holds
        :data:`GEOMETRY_SIDECAR`; None otherwise.
    """
    prefix = GEOMETRY_FOLDER + "/"
    if not name.startswith(prefix):
        return None
    parts = name[len(prefix) :].split("/")
    if len(parts) < 2 or not parts[0]:
        return None
    folder = f"{GEOMETRY_FOLDER}/{parts[0]}"
    return folder if (root / folder / GEOMETRY_SIDECAR).is_file() else None


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
        ``name`` itself when that file exists (a legacy path that names an existing file is
        honoured even when ``inputs/mesh`` holds a same-named file, with one warning that
        both exist); otherwise the path in the first folder (``inputs/mesh`` first) that
        holds the file; ``name`` unchanged when it names no mesh folder, lies in a generated
        geometry folder (:func:`generated_geometry_of`) or no folder holds it.
    """
    folder = mesh_folder_of(name)
    if folder is None or generated_geometry_of(root, name) is not None:
        return name
    leaf = name[len(folder) + 1 :]
    if (root / name).is_file():
        # The file the setup names exists: it is the one used, never a same-named file elsewhere.
        if folder != MESH_FOLDER:
            if (root / MESH_FOLDER / leaf).is_file():
                _note_both(folder, leaf)
            else:
                _note_legacy(folder)
        return name
    for candidate in MESH_FOLDERS:
        if (root / candidate / leaf).is_file():
            if candidate != MESH_FOLDER:
                _note_legacy(candidate)
            return f"{candidate}/{leaf}"
    return name


def mesh_candidates(root: Path, cell: str) -> list[str]:
    """Return the workspace-relative files a bare MESH name or stem can mean.

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
                if p.is_file() and not p.name.endswith(SIDECAR_SUFFIXES)
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
