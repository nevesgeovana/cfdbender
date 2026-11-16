"""What every cfd-geom and cfd-mesh verb shares: where it writes, its record, its exit code.

A mesh a verb writes lands in the workspace's ``inputs/mesh/`` when it is named without a folder,
and beside it a sidecar ``<file name>.provenance.json`` records how it was made: the command,
the source files with their SHA-256, the options and the digest of the mesh itself.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from cfdbender._digest import file_sha256
from cfdbender._errors import GeometryError
from cfdbender._textio import write_json
from cfdbender.core.mesh import MESH_FOLDER, PROVENANCE_SUFFIX

__all__ = [
    "PROVENANCE_SUFFIX",
    "mesh_target",
    "print_checks",
    "provenance_path",
    "write_provenance",
]

#: The schema of the provenance sidecar.
PROVENANCE_SCHEMA = "cfdbender-mesh-provenance/1"


def print_checks(work: Callable[[], Any]) -> int:
    """Run one verb, print the record it returns as JSON and return the exit code.

    A refusal of this package, a ``ValueError`` or an ``OSError`` is printed to the error stream
    and gives exit code 2; nothing else is caught, so a defect keeps its traceback.

    Parameters
    ----------
    work : callable
        The verb's work, called with no argument; it returns a JSON-serializable record.

    Returns
    -------
    int
        0 when the work returned, 2 when it was refused.
    """
    try:
        report = work()
    except (OSError, GeometryError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0


def mesh_target(workspace: str | Path, out: str | Path) -> Path:
    """Return where a verb writes the mesh named ``out``, creating its folder.

    Parameters
    ----------
    workspace : str or Path
        The workspace root.
    out : str or Path
        The output named on the command line. A bare file name lands in the workspace's
        ``inputs/mesh/``; a relative path with a folder is read from the workspace root and an
        absolute path is used as written.

    Returns
    -------
    Path
        The output path.
    """
    named = Path(out)
    if named.is_absolute():
        target = named
    elif named.parent == Path("."):
        target = Path(workspace) / MESH_FOLDER / named
    else:
        target = Path(workspace) / named
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def provenance_path(mesh: str | Path) -> Path:
    """Return the provenance sidecar of ``mesh``: ``<file name>.provenance.json`` beside it.

    Parameters
    ----------
    mesh : str or Path
        The mesh file.

    Returns
    -------
    Path
        The sidecar path.
    """
    path = Path(mesh)
    return path.with_name(path.name + PROVENANCE_SUFFIX)


def _shown(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _version() -> str:
    try:
        return metadata.version("cfdbender")
    except metadata.PackageNotFoundError:
        return "unknown"


def write_provenance(
    mesh: str | Path,
    *,
    command: str,
    sources: Sequence[str | Path],
    options: Mapping[str, Any],
    workspace: str | Path,
    outputs: Mapping[str, Any] | None = None,
) -> Path:
    """Write the sidecar that records how ``mesh`` was made, and return its path.

    Parameters
    ----------
    mesh : str or Path
        The mesh the verb wrote; it must exist.
    command : str
        The verb, without the program name.
    sources : sequence of str or Path
        The files the mesh was made from; each is recorded with its SHA-256.
    options : mapping
        The options the verb ran with, as JSON values.
    workspace : str or Path
        The workspace root; paths inside it are recorded relative to it.
    outputs : mapping, optional
        Further values the verb reports (counts, the files written beside the mesh).

    Returns
    -------
    Path
        The sidecar path, :func:`provenance_path` of ``mesh``.
    """
    root, path = Path(workspace), Path(mesh)
    record: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "command": f"cfd-mesh {command}",
        "mesh": _shown(path, root),
        "sha256": file_sha256(path),
        "sources": [
            {"path": _shown(Path(s), root), "sha256": file_sha256(Path(s))} for s in sources
        ],
        "options": dict(options),
        "outputs": dict(outputs or {}),
        "cfdbender": _version(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    sidecar = provenance_path(path)
    if options.get("overwrite") is True:
        write_json(sidecar, record, default=str)
    else:
        payload = json.dumps(record, indent=2, default=str, allow_nan=False) + "\n"
        try:
            with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
        except FileExistsError:
            raise ValueError(f"{sidecar}: existing provenance; refusing overwrite") from None
    return sidecar
