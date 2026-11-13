"""Name every product of a post stage and the reason any was skipped.

``post/<matrix>/products.json`` lists each product written (the campaign summary,
the sweep table and one polar table per simulation) and each product skipped
with its reason, and ``complete`` is true only when nothing was skipped.
``post/<matrix>/post.log`` says the same in text. Both are derived from the
summary rows alone, are written on every post and never block it: a point that
has no readable result is a named skip, not a refusal.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from cfdbender.core.plan import workspace_path


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _skip_reason(row: Mapping[str, object]) -> str:
    error = row.get("error")
    if isinstance(error, str) and error:
        return f"status {row['status']}: {error}"
    if row["status"] not in ("COMPLETED", "SUBMITTED"):
        return f"status {row['status']}: the point has not run"
    return f"status {row['status']}: no readable result"


def _write(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".collect-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_products(
    root: Path,
    matrix: str,
    solver: str,
    rows: Sequence[Mapping[str, object]],
    summary: Path,
    tables: Sequence[Path],
) -> tuple[Path, Path]:
    """Write products.json and post.log of one matrix and return their paths.

    Parameters
    ----------
    root : Path
        Workspace root.
    matrix : str
        Recorded matrix path, inputs/matrices/<name>.fs.
    solver : str
        The workspace's solver.
    rows : sequence of mapping
        The campaign summary's point rows.
    summary : Path
        The written campaign_summary.json.
    tables : sequence of Path
        The written CSV tables.

    Returns
    -------
    tuple of Path
        products.json and post.log.
    """
    products = [
        {"name": "campaign_summary", "kind": "summary", "path": _relative(root, summary)},
        *(
            {
                "name": path.stem if path.parent.name == "polars" else "campaign_sweep",
                "kind": "table",
                "path": _relative(root, path),
            }
            for path in tables
        ),
    ]
    skipped = [
        {
            "product": "result",
            "point": str(row["point_id"]),
            "reason": _skip_reason(row),
        }
        for row in rows
        if row.get("result") is None
    ]
    data = {
        "schema_version": 1,
        "matrix": matrix,
        "solver": solver,
        "complete": not skipped,
        "products": products,
        "skipped": skipped,
    }
    folder = workspace_path(root, f"post/{Path(matrix).stem}")
    lines = [f"post {matrix} ({solver})"]
    lines += [f"wrote {p['path']}" for p in products]
    lines += [f"skipped {s['point']}: {s['reason']}" for s in skipped]
    lines.append(f"{len(products)} product(s) written, {len(skipped)} skipped")
    new_line = chr(10)
    _write(folder / "products.json", json.dumps(data, indent=2, sort_keys=True) + new_line)
    _write(folder / "post.log", new_line.join(lines) + new_line)
    return folder / "products.json", folder / "post.log"
