"""Verbs that change the records: mark-failed, mark-converged, restore, rebuild, archive.

Each previews and changes nothing unless ``--apply`` is given. A change to the run
manifest copies it to archive/<stamp>/ first; nothing is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from cfdbender.core.adapter import SolverAdapter, load_point
from cfdbender.core.executables import build_mismatch
from cfdbender.core.longpath import copy2, copytree, iter_files, long_path, rmtree
from cfdbender.core.longpath import replace as long_replace
from cfdbender.core.plan import owned_path, workspace_path
from cfdbender.core.registry import (
    RegistryError,
    RunRecord,
    RunRegistry,
    decode_record,
    encode_record,
)
from cfdbender.core.settings import WorkspaceError, read_init
from cfdbender.matrix._console import WORKSPACE_HELP, add_runs_option
from cfdbender.matrix._journal import MARKS, STORAGE, append_journal, now, read_journal
from cfdbender.matrix.selection import chosen, split_ids
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.workspace.planning import plan_workspace


def _digests(root: Path, record: RunRecord) -> dict[str, str]:
    found = {}
    for name in record.point.outputs:
        path = workspace_path(root, name)
        if path.is_file():
            found[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def _marked_failed(root: Path, point_id: str) -> bool:
    verdicts = [
        e.get("verdict") for e in read_journal(root, MARKS) if e.get("point_id") == point_id
    ]
    return bool(verdicts) and verdicts[-1] == "failed"


def _mark(args: argparse.Namespace, verdict: str) -> int:
    registry = RunRegistry(Path(args.workspace).resolve(), args.runs)
    with registry.transaction() if args.apply else nullcontext():
        return _mark_locked(args, verdict, registry)


def _mark_locked(args: argparse.Namespace, verdict: str, registry: RunRegistry) -> int:
    root = Path(args.workspace).resolve()
    records = registry.read()
    keep = chosen((r.point for r in records), split_ids(args.sims), split_ids(args.points))
    stamp = now()
    changes: dict[str, RunRecord] = {}
    refused: list[str] = []
    for record in (r for r in records if r.point_id in keep):
        if verdict == "failed" and record.status == "RUNNING":
            refused.append(
                f"{record.point_id}: RUNNING; stop the run first, then mark the point "
                "(a live run would seal it over the mark)"
            )
            continue
        if verdict == "converged" and record.status != "COMPLETED":
            refused.append(f"{record.point_id}: {record.status}, not COMPLETED")
            continue
        if verdict == "converged" and _marked_failed(root, record.point_id):
            refused.append(f"{record.point_id}: marked failed")
            continue
        outputs = _digests(root, record)
        if verdict == "failed":
            reason = args.reason or "marked failed"
            updated = replace(
                record,
                status="COMPLETED",
                started_at=record.started_at or stamp,
                completed_at=record.completed_at or stamp,
                classification="failed",
                error=f"FAILED_MARKED: {reason}",
                outputs_sha256=MappingProxyType(outputs),
            )
        else:
            updated = replace(
                record,
                classification="converged",
                error=None,
                outputs_sha256=MappingProxyType(outputs),
            )
        try:
            decode_record(encode_record(updated), root)
        except RegistryError as error:
            refused.append(f"{record.point_id}: {error}")
            continue
        changes[record.point_id] = updated
    if refused:
        for line in refused:
            print(f"refused {line}", file=sys.stderr)
        print("nothing marked", file=sys.stderr)
        return 2
    for point_id in changes:
        before = next(r for r in records if r.point_id == point_id)
        print(
            f"{'marking' if args.apply else 'would mark'} {point_id} {verdict} "
            f"(was {before.status} {before.classification or '-'})"
        )
    if not args.apply:
        print("preview only; add --apply to change the records")
        return 0
    folder = registry.replace_records(changes)
    for point_id in changes:
        before = next(r for r in records if r.point_id == point_id)
        append_journal(
            root,
            MARKS,
            {
                "at": stamp,
                "point_id": point_id,
                "verdict": verdict,
                "reason": args.reason,
                "from_status": before.status,
                "from_classification": before.classification,
            },
        )
    print(f"marked {len(changes)} point(s) {verdict}; the manifest before is in {folder}")
    return 0


def run_mark_failed(args: argparse.Namespace) -> int:
    """Mark points FAILED with a reason, whatever status they had."""
    return _mark(args, "failed")


def run_mark_converged(args: argparse.Namespace) -> int:
    """Record the verdict that completed points converged, with its reason."""
    return _mark(args, "converged")


def _archives_with(root: Path, relative: str) -> list[Path]:
    base = owned_path(root, "archive")
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / relative).exists())


def run_restore(args: argparse.Namespace) -> int:
    """Restore the manifest, the storage journal or a matrix's products from archive/<stamp>."""
    root = Path(args.workspace).resolve()
    if args.kind in ("products", "plan") and args.matrix is None:
        stems = sorted(
            {
                d.name
                for a in _archives_with(root, "post")
                for d in (a / "post").iterdir()
                if d.is_dir()
            }
        )
        if len(stems) != 1:
            raise WorkspaceError(
                f"restore {args.kind}: name the matrix with --matrix"
                + (f" ({', '.join(stems)})" if stems else "; no archive holds one")
            )
        stem = stems[0]
    else:
        stem = Path(args.matrix).name.removesuffix(".fs") if args.matrix else ""
        # The stem becomes a folder of post/: an empty one or a step would name post/
        # itself or the workspace root, and the restore would replace all of it.
        if args.kind in ("products", "plan") and (not stem or stem.strip(". ") != stem):
            raise WorkspaceError(
                f"restore {args.kind}: --matrix {args.matrix!r}: not a matrix name"
            )
    relative = {
        "runs": "runs.json",
        "storage": STORAGE,
        "products": f"post/{stem}",
        "plan": f"post/{stem}/plan.json",
    }[args.kind]
    candidates = _archives_with(root, relative)
    if args.stamp:
        candidates = [c for c in candidates if c.name == args.stamp]
    if not candidates:
        raise WorkspaceError(
            f"restore {args.kind}: no archive/<stamp>/{relative} to restore"
            + (f" for stamp {args.stamp}" if args.stamp else "")
        )
    source = owned_path(root, (candidates[-1] / relative).relative_to(root).as_posix())
    target = owned_path(root, relative)
    if source.is_dir():
        # Inspect links themselves before copytree can follow them. iter_files intentionally
        # omits linked entries, so it cannot validate a tree that copytree will copy.
        for current, folders, files in os.walk(long_path(source), followlinks=False):
            for name in (*folders, *files):
                item = Path(current, name)
                if item.is_symlink() or os.path.isjunction(item):
                    raise WorkspaceError(f"{item}: linked restore source; refusing to follow it")
    print(
        f"{'restoring' if args.apply else 'would restore'} {args.kind} "
        f"from {source.relative_to(root).as_posix()}"
    )
    if not args.apply:
        print("preview only; add --apply to restore")
        return 0
    if args.kind == "runs":
        folder = RunRegistry(root).restore_from(source)
        print(f"restored; the manifest before is in {folder.relative_to(root).as_posix()}")
        return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    saved = owned_path(root, f"archive/{stamp}-restore")
    suffix = 1
    while saved.exists():
        saved = root / "archive" / f"{stamp}-restore-{suffix}"
        suffix += 1
    if target.exists():
        destination = saved / relative
        destination.parent.mkdir(parents=True)
        long_replace(target, destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        copytree(source, target)
    else:
        copy2(source, target)
    print(f"restored {relative}")
    return 0


def _only_matrix(root: Path, name: str | None) -> str:
    if name:
        return Path(name).name.removesuffix(".fs")
    stems = sorted({p.stem for home in (root / "inputs/matrices", root) for p in home.glob("*.fs")})
    if len(stems) != 1:
        raise WorkspaceError(
            "rebuild: name the matrix with --matrix"
            + (
                f" ({', '.join(stems)})"
                if stems
                else "; inputs/matrices and the workspace root hold none"
            )
        )
    return stems[0]


def run_rebuild(args: argparse.Namespace) -> int:
    """Rebuild under the destination manifest's writer lock."""
    registry = RunRegistry(Path(args.workspace).resolve(), args.out or args.runs)
    with registry.transaction() if args.apply else nullcontext():
        return _rebuild_locked(args, registry)


def _rebuild_locked(args: argparse.Namespace, registry: RunRegistry) -> int:
    """Rebuild records from the sim folders on disk, by the matrix's plan."""
    root = Path(args.workspace).resolve()
    config = read_init(root, available_solvers=available_solvers())
    adapter = cast(SolverAdapter, load_solver(config.solver))
    plan = plan_workspace(root, _only_matrix(root, args.matrix))
    recorded = {r.point_id for r in RunRegistry(root, args.runs).read()}
    on_disk = {p.name[4:] for p in (root / "sims").glob("sim_*") if p.is_dir()}
    sims = set(split_ids(args.sims))
    if sims - on_disk:
        raise WorkspaceError(f"--sims {sorted(sims - on_disk)[0]}: no sims/sim_<id> folder on disk")
    changes: dict[str, RunRecord] = {}
    for point in plan.points:
        folder = root / point.directory
        selected = (
            point.pol in sims
            if sims
            else point.pol in on_disk and (args.all_sims or point.point_id not in recorded)
        )
        if not selected or not folder.is_dir():
            continue
        times = [long_path(p).stat().st_mtime for p in [folder, *iter_files(folder)]]
        first = datetime.fromtimestamp(min(times), UTC).isoformat()
        last = datetime.fromtimestamp(max(times), UTC).isoformat()
        base = RunRecord(point, plan.matrix, plan.solver, "COMPLETED", first)
        found = _digests(root, base)
        error: str | None = None
        version: str | None = None
        try:
            collected = adapter.collect(load_point(root, point, solver=config.solver))
            classification, version = collected.classification, collected.solver_version
            if mismatch := build_mismatch(point.solver_build, version):
                classification, error = "failed", f"FAILED_EXECUTION: {mismatch}"
        except (OSError, ValueError) as failure:
            classification, error = "failed", f"rebuilt: cannot read the result: {failure}"
        missing = [name for name in point.outputs if name not in found]
        if classification != "failed" and missing:
            classification, error = "failed", f"rebuilt: declared output missing: {missing[0]}"
        changes[point.point_id] = replace(
            base,
            started_at=first,
            completed_at=last,
            classification=classification,
            outputs_sha256=MappingProxyType(found),
            error=error,
            solver_version=version,
        )
    for point_id, record in changes.items():
        print(
            f"{'rebuilding' if args.apply else 'would rebuild'} {point_id}: "
            f"{record.classification}" + (f" ({record.error})" if record.error else "")
        )
    if not changes:
        print("no sim folder without a record to rebuild")
    if not args.apply:
        print("preview only; add --apply to write the records")
        return 0
    if changes:
        saved = registry.replace_records(changes, archive_first=registry.path.exists())
        print(
            f"wrote {len(changes)} record(s) to {registry.path.name}"
            + (f"; manifest before in {saved}" if saved else "")
        )
    return 0


def archive_sim(root: Path, sim_id: str, runs: str = "runs.json") -> Path:
    """Archive the sim and its records under one writer lock."""
    registry = RunRegistry(root, runs)
    with registry.transaction():
        return _archive_sim_locked(root, sim_id, registry)


def _archive_sim_locked(root: Path, sim_id: str, registry: RunRegistry) -> Path:
    """Move one recorded sim's folder into archive/<stamp> and record it in the manifest."""
    ids = tuple(r.point_id for r in registry.read() if r.point.pol == sim_id)
    if not ids:
        raise WorkspaceError(f"sim {sim_id!r} is not recorded in {registry.path.name}; refusing")
    target = registry.archive(ids)
    leftover = owned_path(root, f"sims/sim_{sim_id}")
    if leftover.exists():
        destination = target / "sims" / leftover.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        for item in sorted(leftover.iterdir()):
            moved = destination / item.name
            if moved.exists() and item.is_dir():
                shutil.copytree(long_path(item), long_path(moved), dirs_exist_ok=True)
                rmtree(item)
            else:
                destination.mkdir(parents=True, exist_ok=True)
                shutil.move(str(long_path(item)), str(long_path(moved)))
        leftover.rmdir()
    return target


def _sims_option(parser: argparse.ArgumentParser, *, required: bool, help_text: str) -> None:
    parser.add_argument("--sims", nargs="+", required=required, metavar="POL", help=help_text)


def register(commands: Any) -> dict[str, Any]:
    """Add the record verbs to the cfd-matrix subparsers and return their handlers."""
    apply_help = "change files; without it the command previews and changes nothing"
    workspace_help = WORKSPACE_HELP
    for name, help_text, reason_required in (
        ("mark-failed", "mark points FAILED with a reason, whatever they ended in", False),
        ("mark-converged", "record the verdict that completed points converged", True),
    ):
        mark = commands.add_parser(
            name, help=help_text + " (preview unless --apply)", allow_abbrev=False
        )
        _sims_option(mark, required=True, help_text="the simulations to mark, by POL")
        mark.add_argument("--reason", required=reason_required, help="why, recorded as given")
        mark.add_argument(
            "--points", nargs="+", metavar="NAME", help="only these points of the sims"
        )
        mark.add_argument("--workspace", default=".", help=workspace_help)
        add_runs_option(mark)
        mark.add_argument("--apply", action="store_true", help=apply_help)
    restore = commands.add_parser(
        "restore",
        help="restore archived inputs or results from archive/<stamp> (preview unless --apply)",
        allow_abbrev=False,
    )
    restore.add_argument(
        "kind", choices=("runs", "storage", "products", "plan"), help="the file to restore"
    )
    restore.add_argument("--workspace", default=".", help=workspace_help)
    restore.add_argument("--stamp", help="the archive stamp to restore from (default: the latest)")
    restore.add_argument("--matrix", help="for products and plan: the matrix whose file to restore")
    restore.add_argument("--apply", action="store_true", help=apply_help)
    rebuild = commands.add_parser(
        "rebuild",
        help="rebuild run records from the sim folders on disk (preview unless --apply)",
        allow_abbrev=False,
    )
    rebuild.add_argument("--workspace", default=".", help=workspace_help)
    add_runs_option(rebuild)
    rebuild.add_argument(
        "--out",
        metavar="NAME",
        help="write the rebuilt records to this manifest in the workspace root",
    )
    rebuild.add_argument(
        "--all-sims",
        dest="all_sims",
        action="store_true",
        help="rebuild every sim folder on disk, recorded or not",
    )
    _sims_option(rebuild, required=False, help_text="rebuild these simulations")
    rebuild.add_argument("--matrix", help="the matrix the simulations ran from")
    rebuild.add_argument("--apply", action="store_true", help=apply_help)
    return {
        "mark-failed": run_mark_failed,
        "mark-converged": run_mark_converged,
        "restore": run_restore,
        "rebuild": run_rebuild,
    }
