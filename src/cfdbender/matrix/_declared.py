"""The declared products of one matrix: which points declare one, and their writing.

This module imports nothing numerical, so that a collect of a solver or a point that
declares no product runs without numpy; the products themselves are written by
:mod:`cfdbender.matrix._point_products`, imported only when a point declares one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import load_point
from cfdbender.core.contract import product_support
from cfdbender.core.plan import InputSession, Point, owned_path, workspace_path
from cfdbender.core.pproc import PRODUCT_TABLES, read_post


def matrix_products(
    root: Path,
    matrix: str,
    rows: Sequence[Mapping[str, object]],
    planned: Mapping[str, Point],
    adapter: object,
    solver: str,
    session: InputSession,
) -> tuple[tuple[Path, ...], dict[Path, dict[str, object]], tuple[tuple[str, str], ...]]:
    """Write the declared products of every point of one matrix that has a result.

    Returns the written paths, what ``products.json`` states about each, and every product
    skipped with its reason. A point whose inputs cannot be read is skipped with the reason.
    """
    support = product_support(adapter)
    from cfdbender.matrix.exports import CONTEXT, _cell, _context
    from cfdbender.matrix.exports import _write as write

    folder = workspace_path(root, f"post/{Path(matrix).stem}")
    written: list[Path] = []
    details: dict[Path, dict[str, object]] = {}
    skipped: list[tuple[str, str]] = []
    for row in rows:
        result = cast(Mapping[str, object] | None, row.get("result"))
        point_id = str(row["point_id"])
        if result is None or point_id not in planned:
            continue
        pol, name = point_id.split("/", 1)
        references = cast(Mapping[str, object], result.get("references") or {})
        cells, _ = _context(cast(Mapping[str, object], row["condition"]), references)
        cells["VREF"] = cells.get("VINF")
        lead = {column: _cell(cells.get(column)) for column in CONTEXT}
        moment = cast(Sequence[float], references.get("moment_origin") or (0.0, 0.0, 0.0))
        try:
            point = load_point(
                root, planned[point_id], solver=solver, session=session, post_may_change=True
            )
            if not any(table in point.post for table in PRODUCT_TABLES):
                continue
            if not support:
                declared = read_post(point.post, supported=(), warn=lambda key, text: None)
                skipped.extend((f"products of {point_id}", reason) for reason in declared.skipped)
                continue
            from cfdbender.matrix._point_products import point_products

            found = point_products(
                root,
                pol,
                name,
                point,
                adapter,
                lead=lead,
                moment_point=moment,
                cell=_cell,
                target=lambda relative: owned_path(
                    root, f"{folder.relative_to(root).as_posix()}/{relative}"
                ),
            )
        except (OSError, ValueError) as error:
            skipped.append((f"products of {point_id}", str(error)))
            continue
        for path, facts in found.files.items():
            text = cast(str, facts.pop("text"))
            try:
                written.append(write(path, text))
            except (OSError, ValueError) as error:
                skipped.append((path.relative_to(root).as_posix(), str(error)))
                continue
            details[path] = facts
        skipped += found.skipped
    return tuple(written), details, tuple(skipped)


__all__ = ["matrix_products"]
