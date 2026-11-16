"""The level folder's staging, publish and release.

A level is written into a folder of the run's own, renamed to the level's
name only once complete and audited; an existing level is set aside and put
back on any failure, and the run's folder is removed at the end.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh._refine import _obj

#: The prefix of the run's own folder, inside which a level is written before it is renamed to
#: its own name; a unique name the run creates and alone removes,
#: ``<level start>.partial-<random>``.
STAGING_SUFFIX = ".partial"
#: How many characters of the level's name the run's own folder keeps. The level folder inside
#: it and every file in that carry the whole name, so the folder repeating it a third time was
#: what broke a deep user folder at Windows' 260 characters.
STAGING_STEM_CHARS = 8


def staging_prefix(level: str) -> str:
    """Return the name prefix of the run's own folder for the level called ``level``.

    Parameters
    ----------
    level : str
        The level folder's name.

    Returns
    -------
    str
        The first ``STAGING_STEM_CHARS`` characters of the name, ``.partial-``.
    """
    return f"{level[:STAGING_STEM_CHARS]}{STAGING_SUFFIX}-"


#: The name, inside the run's own staging folder, an existing level is moved to while
#: ``overwrite`` replaces it, until the new level is in place: an error before
#: then puts it back.
PREVIOUS_SUFFIX = ".previous"


def _refuse(where: str, reason: str) -> InputArtifactError:
    return InputArtifactError(f"{where}: {reason}. Nothing was written.", kind=_obj.KIND)


def _release(owned: Path, folder: Path) -> None:
    """Remove the run's own folder, never with an old level still inside it.

    When a publish failed and the old level is still set aside in the run's
    folder, it is put back first; when that also fails, the run's folder is
    kept, holding the old level, and a warning names it.
    """
    previous = owned / f"{folder.name}{PREVIOUS_SUFFIX}"
    if previous.exists() and not folder.exists():
        try:
            previous.rename(folder)
        except OSError:
            warnings.warn(
                f"{folder}: the level was not replaced and the previous level could not be "
                f"put back; it is kept in {previous}. Move it back to {folder}.",
                GeometryWarning,
                stacklevel=3,
            )
            return
    shutil.rmtree(owned, ignore_errors=True)
    if owned.exists():
        warnings.warn(
            f"{owned}: the run's staging folder could not be removed and is left behind; "
            "delete it.",
            GeometryWarning,
            stacklevel=3,
        )


def _publish(staging: Path, folder: Path, previous: Path, overwrite: bool) -> None:
    """Rename the complete, audited staging folder to the level.

    An existing level is moved aside to ``previous``, a path inside the run's
    own staging folder that the caller removes once the new level is in
    place; if the rename fails, the old level is put back. A path that is not
    a folder is never replaced. Without ``overwrite``, a level that appeared
    after :func:`_place` checked is refused as there, and the caller removes
    only the run's own folder.
    """
    if not overwrite and (folder.exists() or folder.is_symlink()):
        raise _refuse(
            str(folder),
            "the level appeared while the run wrote it, and was kept; give overwrite=True "
            "(--overwrite) to replace it",
        )
    if folder.exists() and not folder.is_dir():
        # The race only: _place refuses a file at the level path before any work.
        raise _refuse(str(folder), "a file took the level folder's name while the run wrote it")
    if folder.exists():
        folder.rename(previous)
    try:
        staging.rename(folder)
    except BaseException:
        if previous.exists() and not folder.exists():
            previous.rename(folder)
        raise


def _place(source: Path, stem: str, out_dir: str | Path | None, overwrite: bool) -> Path:
    """Return the level folder, refusing an existing one unless ``overwrite``.

    The level folder is a folder directly inside ``out_dir``, whatever gave
    its name, and never the source's folder, one holding it, or one inside
    it, so neither ``overwrite`` nor the write reaches anything else.
    """
    # Resolved first: a bare file name's parent is ".", whose parent is "." again.
    root = Path(out_dir) if out_dir is not None else source.resolve().parent.parent
    folder = root / stem
    level = folder.resolve()
    # F-731: a source that is itself a symlink has two homes, the folder the link sits
    # in and the folder of the file it points at; the guard covers both.
    homes = {source.parent.resolve(), source.resolve().parent}
    if Path(stem).name != stem or level.parent != root.resolve():
        raise _refuse(
            str(folder),
            f"the level folder is not a folder directly inside out_dir {root}; give the level "
            'one folder name as [refine] tag = "fine" in the refinement file '
            "config (CLI: --config)",
        )
    if level in homes:
        raise _refuse(
            str(folder),
            "the level folder is the source's folder; choose another out_dir (CLI: --out)",
        )
    if any(home.is_relative_to(level) for home in homes):
        raise _refuse(
            str(folder),
            "the level folder holds the source's folder; choose another out_dir (CLI: --out)",
        )
    if any(level.is_relative_to(home) for home in homes):
        raise _refuse(
            str(folder),
            "the level folder lies inside the source's folder; choose an out_dir "
            "(CLI: --out) outside the source's folder",
        )
    if folder.exists() and not folder.is_dir():
        raise _refuse(
            str(folder),
            "the level folder's name is held by a file, which is not a level to replace; "
            "move the file or choose another tag or out_dir (CLI: --out)",
        )
    if folder.exists() and not overwrite:
        raise _refuse(
            str(folder), "the level exists; give overwrite=True (--overwrite) to replace it"
        )
    return folder
