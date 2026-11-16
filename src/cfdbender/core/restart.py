"""Resolve recorded restart sources and copy solution bytes into new point folders."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.plan import workspace_path
from cfdbender.core.registry import RunRegistry


def source_folder(point: AdapterPoint) -> Path:
    """Return this point's recorded source folder, refusing active or absent sources."""
    name = point.point.restart_from
    if name is None:
        raise ValueError("restart source is not stated")
    if point.point.restart_directory is not None:
        return workspace_path(point.root, point.point.restart_directory)
    for record in RunRegistry(point.root).read():
        if record.point_id == name:
            if record.solver != point.solver:
                raise ValueError(f"RESTART {name}: recorded solver differs from {point.solver}")
            if record.status != "COMPLETED":
                raise ValueError(f"RESTART {name}: source is {record.status}; wait for completion")
            return workspace_path(point.root, record.point.directory)
    raise ValueError(f"RESTART {name}: no recorded point; run the source point first")


def require_solution(point: AdapterPoint, files: Mapping[str, Path]) -> Mapping[str, Path]:
    """Refuse an absent solution with its recorded point identity."""
    if not files or any(not path.is_file() or path.stat().st_size == 0 for path in files.values()):
        raise ValueError(
            f"RESTART {point.point.restart_from}: solution is missing; keep its solution files"
        )
    source = source_folder(point).resolve()
    for name, path in files.items():
        if not path.resolve().is_relative_to(source):
            raise ValueError(
                f"RESTART {point.point.restart_from}: solution escapes its point folder"
            )
        expected = point.point.restart_sha256.get(name)
        if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"RESTART {point.point.restart_from}: solution changed before staging")
    return files


def copy_solution(folder: Path, files: Mapping[str, Path]) -> None:
    """Copy solution files exclusively, preserving the source bytes and refusing escapes."""
    for name, source in files.items():
        target = workspace_path(folder, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())


def check_restart(point: AdapterPoint, adapter: object) -> Mapping[str, Path]:
    """Check an adapter's selected solution before any record or archive is written."""
    if point.point.restart_from is not None:
        reader = getattr(adapter, "restart_files", None)
        if reader is None:
            raise ValueError(f"{point.solver} declares no restart file reader")
        files = cast(Mapping[str, Path], reader(point))
        require_solution(point, files)
        validator = getattr(adapter, "validate_restart", None)
        if validator is not None:
            validator(point)
        return files
    return {}
