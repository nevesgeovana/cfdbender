"""The earlier post layout: files of cfdbender 0.5 left in place, named once, read as a fallback.

A post writes only the current tree. A file of the earlier layout already present in
``post/<matrix>/`` (``polars/P<POL>.csv``, the retired sweep table) is left untouched,
named once per kind in one
warning (through :func:`cfdbender.core.refusals.warn_or_refuse`, so ``--warning-refusals``
makes it a refusal) and listed in ``products.json`` under ``migrated`` with the file that
replaces it. :func:`polar_table` reads the current file and falls back to the earlier one
with the same warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cfdbender.core.refusals import warn_or_refuse
from cfdbender.core.settings import WorkspaceError

UPGRADE = "See docs/results.md#the-earlier-layout."
_POLAR = re.compile(r"^P(?P<pol>[^-]+)[.]csv$")
#: The campaign sweep table of 0.5, retired (removed in 0.6.0).
RETIRED_SWEEP = "campaign_sweep.csv"  # retired name, refused by name
SWEEP = (
    "campaign_sweep.csv is retired (removed in 0.6.0): each polar's TOTAL group file "
    "P<POL>-TOTAL-ACOEF.csv holds every point of the polar, and joining several polars is "
    "a concatenation of those files."
)


def refuse_retired(product: str) -> None:
    """Refuse a product path that names the retired campaign sweep table, by name.

    Raises
    ------
    WorkspaceError
        If ``product`` is a retired table.
    """
    if Path(product).name == RETIRED_SWEEP:
        raise WorkspaceError(f"{product}: {SWEEP} {UPGRADE}")


@dataclass(frozen=True)
class Retired:
    """One earlier-layout file: its kind, its path, the path replacing it and why."""

    kind: str
    old: str
    new: str
    sentence: str


def _polar_sentence() -> str:
    return (
        "polars/P<POL>.csv is the 0.5 polar table; the polar now has a file set: the "
        "families in FAM-DIM, FAM-ACOEF and FAM-RCOEF-P<POL>.csv and each group in "
        "P<POL>-<GROUP>-<KIND>.csv."
    )


def retired_products(folder: Path) -> list[Retired]:
    """Return every earlier-layout file under ``post/<matrix>/``, sorted by path."""
    found: list[Retired] = []
    if (folder / RETIRED_SWEEP).is_file():
        found.append(Retired("sweep", RETIRED_SWEEP, "polars/P<POL>-TOTAL-ACOEF.csv", SWEEP))
    polars = folder / "polars"
    if polars.is_dir():
        for path in sorted(polars.iterdir()):
            match = _POLAR.match(path.name)
            if path.is_file() and match:
                found.append(
                    Retired(
                        "polar_table",
                        f"polars/{path.name}",
                        f"polars/FAM-ACOEF-P{match['pol']}.csv",
                        _polar_sentence(),
                    )
                )
    return found


def warn_retired(folder: Path) -> dict[str, str]:
    """Name the earlier-layout files of one matrix, one warning per kind.

    Returns
    -------
    dict of str to str
        Each old path with the path that replaces it, relative to ``post/<matrix>/``.

    Raises
    ------
    WorkspaceError
        Under warning_refusals, when such a file is present.
    """
    found = retired_products(folder)
    by_kind: dict[str, list[Retired]] = {}
    for retired in found:
        by_kind.setdefault(retired.kind, []).append(retired)
    for items in by_kind.values():
        names = ", ".join(item.old for item in items)
        warn_or_refuse(f"post/{folder.name}: {names} left in place. {items[0].sentence} {UPGRADE}")
    return {retired.old: retired.new for retired in found}


def polar_table(folder: Path, pol: str) -> Path:
    """Return the FAM-ACOEF file of polar ``pol``, or the earlier P<POL>.csv with a warning.

    Raises
    ------
    FileNotFoundError
        If neither exists.
    """
    current = folder / "polars" / f"FAM-ACOEF-P{pol}.csv"
    if current.is_file():
        return current
    earlier = folder / "polars" / f"P{pol}.csv"
    if earlier.is_file():
        warn_or_refuse(
            f"post/{folder.name}: reading polars/{earlier.name}, the 0.5 polar table, because "
            f"polars/{current.name} is absent; run cfd-matrix post to write it. {UPGRADE}"
        )
        return earlier
    raise FileNotFoundError(f"post/{folder.name}: no polar table for P{pol}")
