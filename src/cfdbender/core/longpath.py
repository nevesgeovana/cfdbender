"""Long paths on Windows: file operations under sims/ beyond the 260-character limit.

A deep solver tree (an OpenFOAM case with processor and time folders) can pass the
260-character limit of a Windows path, and an ordinary file operation then fails. The
extended-length spelling (a prefix of two backslashes, a question mark and a backslash on
the absolute path) lifts the limit. The helpers
here spell the path for the file operation only: names, relative paths and records are
always derived from the path as given. Elsewhere they return the path unchanged.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

_PREFIX = "\\\\?\\"


def extended_length(path: str | os.PathLike[str], *, windows: bool | None = None) -> str:
    """Return ``path`` in the spelling a file operation accepts at any length.

    Parameters
    ----------
    path : str or path-like
        The path to spell.
    windows : bool, optional
        Whether to spell for Windows; defaults to whether this process runs on Windows.

    Returns
    -------
    str
        On Windows the absolute path behind the extended prefix (the UNC form for a
        network path); an already extended path is unchanged. Elsewhere the path as given.
    """
    text = os.fspath(path)
    if windows is None:
        windows = os.name == "nt"
    if not windows or not text or text.startswith(_PREFIX):
        return text
    absolute = text.replace("/", "\\")
    if not (absolute.startswith("\\\\") or (len(absolute) > 2 and absolute[1:3] == ":\\")):
        absolute = os.path.abspath(absolute)
    if absolute.startswith("\\\\"):
        return _PREFIX + "UNC\\" + absolute[2:]
    return _PREFIX + absolute


def long_path(path: str | os.PathLike[str]) -> Path:
    """Return ``path`` as a path any file operation accepts at any length."""
    return Path(extended_length(path))


def plain(path: str | os.PathLike[str]) -> Path:
    """Return a path without the extended prefix, the spelling names and records use."""
    text = os.fspath(path)
    if text.startswith(_PREFIX + "UNC\\"):
        return Path("\\\\" + text[len(_PREFIX) + 4 :])
    if text.startswith(_PREFIX):
        return Path(text[len(_PREFIX) :])
    return Path(text)


def iter_files(folder: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield every file under ``folder``, links not followed, at any depth.

    The paths are in their plain spelling. A symbolic link or a junction is not entered
    and a link to a file is not yielded, as ``Path.rglob`` plus ``is_symlink`` would do.
    """
    base = plain(extended_length(folder))
    for current, folders, names in os.walk(extended_length(folder), followlinks=False):
        here = plain(current)
        folders[:] = [
            name
            for name in folders
            if not (Path(current, name).is_symlink() or os.path.isjunction(Path(current, name)))
        ]
        for name in names:
            if not Path(current, name).is_symlink():
                yield base / here.relative_to(base) / name


def file_size(path: str | os.PathLike[str]) -> int:
    """Return the size of one file, at any path length."""
    return long_path(path).stat().st_size


def rmtree(path: str | os.PathLike[str]) -> None:
    """Remove a folder tree, at any path length."""
    shutil.rmtree(long_path(path))


def copytree(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    """Copy a folder tree, at any path length."""
    shutil.copytree(long_path(source), long_path(target))


def copy2(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    """Copy one file and its metadata, at any path length."""
    shutil.copy2(long_path(source), long_path(target))


def replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    """Rename a file or folder over ``target``, at any path length."""
    os.replace(long_path(source), long_path(target))
