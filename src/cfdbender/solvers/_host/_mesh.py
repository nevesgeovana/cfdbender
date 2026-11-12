"""Link a workspace mesh into a point folder; a mesh is never copied.

Every solver that stages a mesh (SU2, OpenFOAM, PyFR), for a local run or a
cluster job, places a relative symbolic link in the point folder, so the link
still resolves after the whole workspace moves to another machine. A Windows
host without the symbolic-link privilege gets a hard link instead: the same file
on disk, never a second copy. Where neither link can be made (another volume),
staging refuses.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_REFUSAL = (
    "cannot link the mesh {source} into the point folder: {error}. Meshes are always "
    "linked, never copied: keep the workspace on one volume whose filesystem supports "
    "symbolic or hard links (on Windows, Developer Mode allows symbolic links)"
)


def _symlink(target: Path, relative: str) -> None:
    target.symlink_to(relative)


def _hardlink(target: Path, source: Path) -> None:
    target.hardlink_to(source)


def link_mesh(source: Path, target: Path, digest: str | None) -> None:
    """Link target to source (relative symbolic link, else hard link) and verify it.

    Parameters
    ----------
    source : Path
        Mesh file inside the workspace.
    target : Path
        Link to create in the point folder, which must already exist.
    digest : str or None
        Planned SHA-256 of the mesh; a mismatch removes the link and refuses.

    Raises
    ------
    ValueError
        Neither link can be created, or the mesh changed after planning.
    """
    relative = os.path.relpath(source, target.parent)
    try:
        _symlink(target, relative)
    except (OSError, NotImplementedError, ValueError):
        try:
            _hardlink(target, source)
        except (OSError, NotImplementedError) as error:
            raise ValueError(_REFUSAL.format(source=source.name, error=error)) from error
    # Read through the link without allocating a multi-gigabyte buffer.
    with target.open("rb") as stream:
        measured = hashlib.file_digest(stream, "sha256").hexdigest()
    if measured != digest:
        target.unlink()
        raise ValueError("mesh changed after planning; plan again")
