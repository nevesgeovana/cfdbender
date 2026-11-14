"""Storage verbs: space-in-use, free-space, delete-sims and sync.

Each call is recorded in storage_management.json. The verbs that change files preview
and change nothing unless ``--apply`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import tomllib
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cfdbender.core.longpath import copy2, file_size, iter_files, long_path, rmtree
from cfdbender.core.longpath import replace as long_replace
from cfdbender.core.plan import owned_path, workspace_path
from cfdbender.core.registry import (
    RegistryError,
    RunRecord,
    RunRegistry,
    encode_record,
)
from cfdbender.core.settings import WorkspaceError
from cfdbender.matrix._console import WORKSPACE_HELP, add_runs_option
from cfdbender.matrix._journal import STORAGE, append_journal, now
from cfdbender.matrix.selection import split_ids


def _size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file_size(p) for p in iter_files(path))


def _human(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{count} B"
        value /= 1024
    return f"{count} B"


def _top(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from error
    if number < 1:
        raise argparse.ArgumentTypeError("--top must be at least 1")
    return number


def run_space(args: argparse.Namespace) -> int:
    """Print the sizes on disk by top-level folder, by sim and by extension."""
    root = Path(args.workspace).resolve()
    by_folder = {p.name: _size(p) for p in root.iterdir() if p.is_dir() and p.name != "logs"}
    sims = root / "sims"
    by_sim = {p.name: _size(p) for p in sims.iterdir()} if sims.is_dir() else {}
    by_ext: dict[str, int] = {}
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            ext = path.suffix.lower() or "(none)"
            by_ext[ext] = by_ext.get(ext, 0) + path.stat().st_size
    for title, table in (("by folder", by_folder), ("by sim", by_sim), ("by extension", by_ext)):
        print(title)
        for name, size in sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top]:
            print(f"  {_human(size):>10}  {name}")
    append_journal(
        root, STORAGE, {"at": now(), "command": "space-in-use", "detail": f"top {args.top}"}
    )
    return 0


def _recipe(root: Path, name: str) -> dict[str, Any]:
    if not name.replace("_", "").replace("-", "").isalnum():
        raise WorkspaceError(f"{name!r}: a recipe id is a file stem in inputs/management/")
    path = root / "inputs/management" / f"{name}.toml"
    if not path.is_file():
        raise WorkspaceError(f"no recipe inputs/management/{name}.toml")
    try:
        recipe = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WorkspaceError(f"{path}: invalid recipe: {error}") from error
    if unknown := set(recipe) - {"delete_extensions"}:
        raise WorkspaceError(f"recipe {name}: unsupported step {sorted(unknown)[0]!r}")
    return recipe


def _own_files(base: Path) -> list[Path]:
    """Return the regular files under base, never entering a linked folder or junction.

    Path.rglob enters a Windows junction (it is not a symlink), so a junction under
    sims/ would put files outside the workspace in reach of a delete.
    """
    if not base.is_dir() or base.is_symlink() or os.path.isjunction(base):
        return []
    found: list[Path] = []
    stack = [base]
    while stack:
        folder = stack.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink() or os.path.isjunction(path):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                elif entry.is_file(follow_symlinks=False):
                    found.append(path)
    return sorted(found)


def run_free_space(args: argparse.Namespace) -> int:
    """Keep an applied management operation under the run writer lock."""
    registry = RunRegistry(Path(args.workspace).resolve(), args.runs)
    with registry.transaction() if args.apply else nullcontext():
        return _run_free_space(args, registry)


def _run_free_space(args: argparse.Namespace, registry: RunRegistry) -> int:
    """Run a storage recipe: delete named extensions under sims/ that no record names."""
    root = Path(args.workspace).resolve()
    if args.recipe is None:
        if not args.list_paths:
            raise WorkspaceError("name a recipe of inputs/management/, or add --list to see them")
        found = sorted(p.stem for p in (root / "inputs" / "management").glob("*.toml"))
        print(", ".join(found) if found else "no recipe in inputs/management/ (write <name>.toml)")
        return 0
    recipe = _recipe(root, args.recipe)
    kept = {
        workspace_path(root, name)
        for record in registry.read()
        for name in (*record.point.outputs, *record.outputs_sha256)
    }
    freed = 0
    for step in recipe.get("delete_extensions", []):
        extensions = {str(e).lower() for e in step.get("extensions", [])}
        if not extensions or any(not e.startswith(".") for e in extensions):
            raise WorkspaceError('delete_extensions needs extensions = [".ext", ...]')
        for path in _own_files(root / "sims"):
            if path.suffix.lower() not in extensions:
                continue
            protected = path.resolve() in kept
            if args.list_paths:
                print(
                    f"  {'keep' if protected else 'delete'} {path.relative_to(root).as_posix()} "
                    f"({_human(path.stat().st_size)})"
                )
            if not protected:
                freed += path.stat().st_size
                if args.apply:
                    path.unlink()
    verb = "freed" if args.apply else "would free"
    print(f"{verb} {_human(freed)}")
    if not args.apply:
        print("preview only; add --apply to delete")
    else:
        append_journal(
            root,
            STORAGE,
            {"at": now(), "command": "free-space", "detail": f"{args.recipe}: {freed} bytes"},
        )
    return 0


def _sim_folder(root: Path, sim: str) -> Path:
    """Return sims/sim_<sim>, refusing an id that is not one plain folder name of sims/.

    A separator or ``..`` would step out of sims/, and Windows drops trailing dots and
    spaces and folds case, so ``1001.`` names sim_1001; the resolved name must be the
    name asked for. A link or junction is refused, never followed.
    """
    name = f"sim_{sim}"
    if any(c in sim for c in '/\\:*?"<>|') or sim.strip(". ") != sim or not sim:
        raise RegistryError(f"--sims {sim!r}: not a simulation id")
    folder = owned_path(root, f"sims/{name}")
    if not folder.exists():
        return folder
    if folder.is_symlink() or os.path.isjunction(folder):
        raise RegistryError(f"--sims {sim}: sims/{name} is a link; refusing to follow it")
    resolved = folder.resolve()
    if resolved.parent != (root / "sims").resolve() or resolved.name != name:
        raise RegistryError(f"--sims {sim}: sims/{name} is not that simulation's folder")
    return folder


def _refuse_shared_sims(root: Path, runs: str, sims: Sequence[str]) -> None:
    """Refuse a sim that a manifest other than ``runs`` still records.

    The sim folder is shared by every manifest; deleting it would leave the other
    manifest's records without their files.
    """
    for path in sorted(root.glob("*.json")):
        if path.name == runs or path.name.endswith(".lock.json"):
            continue
        try:
            others = RunRegistry(root, path.name).read()
        except (RegistryError, ValueError, OSError):
            continue  # not a run manifest
        for sim in sims:
            if any(r.point.pol == sim for r in others):
                raise RegistryError(
                    f"--sims {sim}: {path.name} still records points of that simulation; "
                    f"delete them there first (--runs {path.name}) or choose another sim"
                )


def run_delete_sims(args: argparse.Namespace) -> int:
    """Keep an applied management operation under the run writer lock."""
    registry = RunRegistry(Path(args.workspace).resolve(), args.runs)
    with registry.transaction() if args.apply else nullcontext():
        return _run_delete_sims(args, registry)


def _run_delete_sims(args: argparse.Namespace, registry: RunRegistry) -> int:
    """Delete named sims: their folders and records, keeping a note in the manifest history."""
    root = Path(args.workspace).resolve()
    records = registry.read()
    sims = split_ids(args.sims)
    for sim in sims:
        _sim_folder(root, sim)
        if all(r.point.pol != sim for r in records) and not (root / "sims" / f"sim_{sim}").is_dir():
            raise RegistryError(f"--sims {sim}: no record or folder of that simulation")
    doomed = [r for r in records if r.point.pol in sims]
    for live in ("SUBMITTED", "RUNNING"):
        if (blocked := [r.point_id for r in doomed if r.status == live]) and not args.force:
            raise RegistryError(f"{blocked[0]} is still {live}; --force deletes it anyway")
    _refuse_shared_sims(root, args.runs, sims)
    matrices = sorted({r.matrix for r in doomed})
    products = [m for m in matrices if (root / "post" / Path(m).stem).is_dir()]
    if products and args.matrix_products is None:
        raise RegistryError(
            f"the products of {', '.join(Path(m).stem for m in products)} hold the deleted "
            "points; choose --matrix-products points-only or regenerate"
        )
    for sim in sims:
        folder = root / "sims" / f"sim_{sim}"
        print(
            f"{'deleting' if args.apply else 'would delete'} sims/sim_{sim} "
            f"({_human(_size(folder)) if folder.exists() else 'no folder'})"
        )
    for record in doomed:
        print(f"  record {record.point_id} ({record.status})")
    if not args.apply:
        print("preview only; add --apply to delete")
        return 0
    stamp = now()
    registry.replace_records(
        {},
        drop=tuple(r.point_id for r in doomed),
        history_entries=tuple(
            {"deleted_at": stamp, "record": encode_record(r), "reason": "delete-sims"}
            for r in doomed
        ),
    )
    for sim in sims:
        folder = root / "sims" / f"sim_{sim}"
        if folder.exists():
            rmtree(folder)
    append_journal(
        root,
        STORAGE,
        {
            "at": stamp,
            "command": "delete-sims",
            "detail": f"sims {','.join(sims)}; points {len(doomed)}; "
            f"products {args.matrix_products or 'none'}",
        },
    )
    if args.matrix_products == "regenerate":
        from cfdbender.matrix.collection import collect_workspace

        for matrix in products:
            if RunRegistry(root, args.runs).read():
                try:
                    collect_workspace(root, Path(matrix).stem, runs=args.runs, archive=args.archive)
                except RegistryError:
                    print(f"{Path(matrix).stem}: no recorded points left to collect")
    elif products:
        print(
            "matrix products that held the deleted points are stale: "
            + ", ".join(Path(m).stem for m in products)
        )
    return 0


def _sync_sources(root: Path, name: str | None) -> dict[str, Path]:
    path = root / "inputs/sync-workspaces.toml"
    if not path.is_file():
        raise WorkspaceError('sync needs inputs/sync-workspaces.toml: [workspaces] name = "path"')
    try:
        table = tomllib.loads(path.read_text(encoding="utf-8")).get("workspaces", {})
    except tomllib.TOMLDecodeError as error:
        raise WorkspaceError(f"{path}: invalid TOML: {error}") from error
    found = {str(k): Path(str(v)).resolve() for k, v in table.items()}
    if name is not None:
        if name not in found:
            raise WorkspaceError(f"--from {name}: not a workspace of sync-workspaces.toml")
        found = {name: found[name]}
    if not found:
        raise WorkspaceError("sync-workspaces.toml names no workspace")
    return found


def _copy_files(
    base: Path,
    other: Path,
    top: str,
    args: argparse.Namespace,
    saved: list[str],
    kept: frozenset[str] = frozenset(),
) -> int:
    """Copy the other workspace's files under ``top``; ``kept`` are main's own point folders.

    A file of a point whose record main keeps is never replaced, whatever ``--overwrite``
    says: the record holds the digests of main's files, and the two travel together.
    """
    count = 0
    source = owned_path(other, top)
    if not source.is_dir():
        return 0
    for path in _own_files(source):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(other)
        if not args.include_archives and "archive" in relative.parts:
            continue
        target = owned_path(base, relative.as_posix())
        name = relative.as_posix()
        if any(name.startswith(folder + "/") for folder in kept):
            print(f"  conflict {name}: main's copy kept, with the record of its point")
            continue
        if target.exists():
            if (
                hashlib.sha256(target.read_bytes()).digest()
                == hashlib.sha256(path.read_bytes()).digest()
            ):
                continue
            if not args.overwrite:
                print(f"  conflict {name}: main's copy kept")
                continue
            if args.apply:
                archive = owned_path(base, "archive")
                archive.mkdir(exist_ok=True)
                backup = (
                    Path(
                        tempfile.mkdtemp(
                            prefix=f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-sync-",
                            dir=archive,
                        )
                    )
                    / relative
                )
                backup.parent.mkdir(parents=True, exist_ok=True)
                copy2(target, backup)
                saved.append(relative.as_posix())
        print(f"  {'copy' if args.apply else 'would copy'} {relative.as_posix()}")
        count += 1
        if args.apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Replace the name, never write into the file it names: a staged mesh is a
            # symbolic or hard link to inputs/, and a write would change that input.
            with tempfile.NamedTemporaryFile(
                dir=long_path(target.parent), prefix=".sync-", suffix=".tmp", delete=False
            ) as stream:
                fresh = Path(stream.name)
            try:
                copy2(path, fresh)
                long_replace(fresh, target)
            finally:
                fresh.unlink(missing_ok=True)
    return count


def run_sync(args: argparse.Namespace) -> int:
    """Keep an applied management operation under the run writer lock."""
    registry = RunRegistry(Path(args.workspace).resolve(), args.runs)
    with registry.transaction() if args.apply else nullcontext():
        return _run_sync(args, registry)


def _run_sync(args: argparse.Namespace, registry: RunRegistry) -> int:
    """Bring records and results from the workspaces named in inputs/sync-workspaces.toml."""
    root = Path(args.workspace).resolve()
    if args.prefer_other and args.level != "runs" and not args.overwrite:
        raise RegistryError(
            "--prefer-other takes the other workspace's records, and at this level their files "
            "too only with --overwrite; add --overwrite, or use level runs to bring records alone"
        )
    main = registry
    mine = {r.point_id: r for r in main.read()}
    for name, other in _sync_sources(root, args.source).items():
        print(f"from {name}: {other}")
        theirs = RunRegistry(other, args.runs).read()
        changes: dict[str, RunRecord] = {}
        for record in theirs:
            held = mine.get(record.point_id)
            if held is None:
                changes[record.point_id] = record
                print(f"  record {record.point_id}: {'adding' if args.apply else 'would add'}")
            elif encode_record(held) != encode_record(record):
                if held.status == "SUBMITTED" or args.prefer_other:
                    changes[record.point_id] = record
                    print(
                        f"  record {record.point_id}: "
                        f"{'taking' if args.apply else 'would take'} the other's"
                    )
                else:
                    print(f"  conflict {record.point_id}: main's record kept")
        saved: list[str] = []
        copied = 0
        if args.level == "all" and not args.overwrite:
            # A new or automatically accepted SUBMITTED record needs its own files too.
            # Refuse before copying anything when the file policy would keep other bytes.
            for record in changes.values():
                for path in _own_files(owned_path(other, record.point.directory)):
                    relative = path.relative_to(other).as_posix()
                    target = owned_path(root, relative)
                    if target.exists() and target.read_bytes() != path.read_bytes():
                        raise RegistryError(
                            f"{relative}: taking record {record.point_id} requires replacing "
                            "its conflicting files; add --overwrite"
                        )
        if args.level in ("post", "all"):
            copied += _copy_files(root, other, "post", args, saved)
        if args.level == "all":
            own = frozenset(r.point.directory for r in mine.values() if r.point_id not in changes)
            copied += _copy_files(root, other, "sims", args, saved, own)
        if args.restore and args.apply:
            print("  --restore: run `cfd-matrix rebuild --apply` for sim folders no record carries")
        if args.apply and changes:
            main.replace_records(changes)
            mine.update(changes)
        if args.apply:
            append_journal(
                root,
                STORAGE,
                {
                    "at": now(),
                    "command": "sync",
                    "detail": f"{name}: {args.level}; records {len(changes)}; files {copied}",
                    "event": "apply" if args.apply else "preview",
                },
            )
        print(f"  {len(changes)} record(s), {copied} file(s)")
    if not args.apply:
        print("preview only; add --apply to change files")
    return 0


def register(commands: Any) -> dict[str, Any]:
    """Add the storage verbs to the cfd-matrix subparsers and return their handlers."""
    apply_help = "change files; without it the command previews and changes nothing"
    workspace_help = WORKSPACE_HELP
    space = commands.add_parser(
        "space-in-use",
        help="report disk space by folder, by sim and by extension",
        allow_abbrev=False,
    )
    space.add_argument("--workspace", default=".", help=workspace_help)
    space.add_argument("--top", type=_top, default=15, help="rows shown per grouping (default: 15)")
    free = commands.add_parser(
        "free-space",
        help="run a storage recipe inputs/management/<recipe>.toml (preview unless --apply)",
        allow_abbrev=False,
    )
    free.add_argument(
        "recipe",
        nargs="?",
        help="the recipe id: a file stem in inputs/management/ (with --list alone: list them)",
    )
    free.add_argument("--workspace", default=".", help=workspace_help)
    free.add_argument("--apply", action="store_true", help=apply_help)
    free.add_argument(
        "--list",
        dest="list_paths",
        action="store_true",
        help="print every path a step touches, with its size and what happens to it",
    )
    add_runs_option(free)
    delete = commands.add_parser(
        "delete-sims",
        help="delete simulations, their folders and their records (preview unless --apply)",
        allow_abbrev=False,
    )
    delete.add_argument(
        "--sims", nargs="+", required=True, metavar="POL", help="the simulations to delete"
    )
    delete.add_argument("--workspace", default=".", help=workspace_help)
    delete.add_argument(
        "--matrix-products",
        dest="matrix_products",
        choices=("points-only", "regenerate"),
        default=None,
        help="what happens to a matrix product that also holds the deleted points",
    )
    delete.add_argument(
        "--force",
        action="store_true",
        help="delete a sim whatever the status of its records; "
        "without it a SUBMITTED point refuses",
    )
    delete.add_argument("--apply", action="store_true", help=apply_help)
    delete.add_argument(
        "--archive", action="store_true", help="keep replaced matrix products when regenerating"
    )
    add_runs_option(delete)
    sync = commands.add_parser(
        "sync",
        help="bring records and results from the workspaces of inputs/sync-workspaces.toml",
        allow_abbrev=False,
    )
    sync.add_argument(
        "level",
        choices=("runs", "post", "all"),
        help="what to bring; each level adds to the one before",
    )
    sync.add_argument("--workspace", default=".", help=workspace_help)
    sync.add_argument(
        "--from",
        dest="source",
        help="one workspace name from sync-workspaces.toml (default: every one)",
    )
    sync.add_argument("--apply", action="store_true", help=apply_help)
    sync.add_argument(
        "--prefer-other",
        dest="prefer_other",
        action="store_true",
        help="on a record conflict, take the other record (levels post and all need --overwrite)",
    )
    sync.add_argument(
        "--overwrite",
        action="store_true",
        help="on a file conflict, archive main's copy and take the other's",
    )
    sync.add_argument(
        "--restore",
        action="store_true",
        help="with --apply, remind to rebuild the records of sim folders no record carries",
    )
    sync.add_argument(
        "--include-archives",
        dest="include_archives",
        action="store_true",
        help="also bring the files under folders named archive",
    )
    add_runs_option(sync)
    return {
        "space-in-use": run_space,
        "free-space": run_free_space,
        "delete-sims": run_delete_sims,
        "sync": run_sync,
    }
