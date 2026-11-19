"""Name every product of a post stage and the reason any was skipped.

``post/<matrix>/products.json`` lists each product written (the campaign summary and
every table, each with its kind and, for a polar file, its polar and group) and each
product skipped with its reason; ``complete`` is true only when every point has a result.
Schema version 2 adds the statement of :mod:`cfdbender.matrix._manifest_schema` (tree,
kinds with units and frames, run block, frames, families, groups, polars, migrated).
``post/<matrix>/post.log`` says the same in text. Both are derived from the summary rows
alone and are written on every post. A point without a readable result warns and is
skipped, or refuses under warning_refusals. A group with no member remains a named skip
of a product that does not apply, without a warning or refusal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from cfdbender.core.plan import workspace_path
from cfdbender.core.refusals import warn_or_refuse
from cfdbender.matrix._family_polar import PolarFiles
from cfdbender.matrix._manifest_schema import SECTIONS_DIR, manifest_statement
from cfdbender.matrix._rotor_table import SUFFIX as ROTOR_SUFFIX
from cfdbender.matrix._rotor_table import absent_columns

_FAMILY_FILE = re.compile(r"^FAM-(?P<kind>DIM|ACOEF|RCOEF)-P(?P<pol>.+)[.]csv$")
_GROUP_FILE = re.compile(r"^P(?P<pol>[^-]+)-(?P<group>.+)-(?P<kind>DIM|ACOEF|RCOEF)[.]csv$")


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _skip_reason(row: Mapping[str, object]) -> str:
    error = row.get("error")
    if isinstance(error, str) and error:
        return f"status {row['status']}: {error}"
    if row["status"] not in ("COMPLETED", "SUBMITTED"):
        return f"status {row['status']}: the point has not run"
    return f"status {row['status']}: no readable result"


def _manifest_text(data: dict[str, object]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + chr(10)


def _digest(data: dict[str, object]) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _collision(path: Path) -> ValueError:
    return ValueError(
        f"{path}: unrecorded or edited output; refusing overwrite; move it aside and collect again"
    )


def _check_metadata(folder: Path, matrix: str, solver: str) -> tuple[bool, bool]:
    """Check both old metadata files before publishing any replacement."""
    manifest, log = folder / "products.json", folder / "post.log"
    for path in (manifest, log):
        if path.is_symlink():
            raise _collision(path)
    exists = manifest.exists(), log.exists()
    if not exists[0]:
        if exists[1]:
            raise _collision(log)
        return exists
    try:
        original = manifest.read_bytes()
        data = json.loads(original)
        if not isinstance(data, dict) or _manifest_text(data).encode() != original:
            raise ValueError("not a generated manifest")
        recorded = data.pop("content_sha256", None)
        if (
            recorded != _digest(data)
            or data.get("schema_version") != 2
            or data.get("matrix") != matrix
            or data.get("solver") != solver
        ):
            raise ValueError("manifest identity or digest differs")
    except (OSError, ValueError, TypeError):
        raise _collision(manifest) from None
    if exists[1]:
        try:
            matches = hashlib.sha256(log.read_bytes()).hexdigest() == data.get("post_log_sha256")
        except OSError:
            matches = False
        if not matches:
            raise _collision(log)
    return exists


def _write(target: Path, text: str, *, replace_existing: bool) -> None:
    data = text.encode("utf-8")
    if replace_existing and target.read_bytes() == data:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".collect-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace_existing:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _entry(root: Path, path: Path) -> dict[str, object]:
    """Return one product's entry: name, kind, path and, for a polar file, polar and group."""
    entry: dict[str, object] = {"name": path.stem, "kind": "table", "path": _relative(root, path)}
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if found := _FAMILY_FILE.match(path.name):
        entry.update(kind=f"FAM-{found['kind']}", polar=found["pol"])
    elif found := _GROUP_FILE.match(path.name):
        entry.update(kind=found["kind"], polar=found["pol"], group=found["group"])
    elif path.parent.name == "polars" and path.name.endswith(ROTOR_SUFFIX):
        pol, rotor = path.name.removesuffix(ROTOR_SUFFIX).removeprefix("P").split("-", 1)
        entry.update(kind="rotor", polar=pol, rotor=rotor, absent_columns=absent_columns(path))
    elif path.name == "campaign_summary.json":
        entry.update(name="campaign_summary", kind="summary")
    elif path.parent.name == "polars" and path.suffix == ".txt":
        entry.update(kind="polar", polar=path.stem.removeprefix("P"))
    elif SECTIONS_DIR in path.parts[-3:-1]:
        entry.update(kind="sections")
    return entry


def write_products(
    root: Path,
    matrix: str,
    solver: str,
    rows: Sequence[Mapping[str, object]],
    summary: Path,
    tables: Sequence[Path],
    polar: PolarFiles | None = None,
    migrated: Mapping[str, str] | None = None,
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
    polar : PolarFiles, optional
        What the polar files state: their kinds, frames, families and groups.
    migrated : mapping, optional
        Each earlier-layout file left in place, with the file that replaces it.

    Returns
    -------
    tuple of Path
        products.json and post.log.
    """
    folder = workspace_path(root, f"post/{Path(matrix).stem}")
    replace_manifest, replace_log = _check_metadata(folder, matrix, solver)
    products = [_entry(root, path) for path in (summary, *tables)]
    skipped: list[dict[str, object]] = [
        {
            "product": "result",
            "point": str(row["point_id"]),
            "reason": _skip_reason(row),
        }
        for row in rows
        if row.get("result") is None
    ]
    # A point without a result makes the post incomplete; a group that has no member is a
    # named skip of a product that does not apply, and leaves it complete.
    complete = not skipped
    for item in skipped:
        warn_or_refuse(f"product result skipped for {item['point']}: {item['reason']}")
    if polar is not None:
        skipped += [
            {"product": f"polars/P{pol}-{group}", "reason": reason}
            for pol in polar.run_types
            for group, reason in polar.skipped
        ]
        skipped += [
            {"product": product, "reason": reason} for product, reason in polar.sections_skipped
        ]
    data: dict[str, object] = {
        "matrix": matrix,
        "solver": solver,
        "complete": complete,
        "products": products,
        "skipped": skipped,
        "schema_version": 2,
        "migrated": dict(migrated or {}),
    }
    if polar is not None:
        data.update(manifest_statement(polar, migrated=migrated))
    lines = [f"post {matrix} ({solver})"]
    lines += [f"wrote {p['path']}" for p in products]
    lines += [f"skipped {s.get('point', s['product'])}: {s['reason']}" for s in skipped]
    lines += [f"left {old} in place; {new} replaces it" for old, new in (migrated or {}).items()]
    lines.append(f"{len(products)} product(s) written, {len(skipped)} skipped")
    new_line = chr(10)
    log_text = new_line.join(lines) + new_line
    data["post_log_sha256"] = hashlib.sha256(log_text.encode()).hexdigest()
    data["content_sha256"] = _digest(data)
    _write(folder / "products.json", _manifest_text(data), replace_existing=replace_manifest)
    _write(folder / "post.log", log_text, replace_existing=replace_log)
    return folder / "products.json", folder / "post.log"
