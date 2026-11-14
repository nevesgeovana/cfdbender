"""The workspace journals: marks.json (verdicts) and storage_management.json (storage calls)."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from cfdbender.core.settings import WorkspaceError

MARKS = "marks.json"
STORAGE = "storage_management.json"


def now() -> str:
    """Return the UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def read_journal(root: Path, name: str) -> list[dict[str, object]]:
    """Return a journal's entries, oldest first; an absent journal has none."""
    path = root / name
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        entries = document["entries"]
        if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
            raise ValueError("entries must be a list of objects")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise WorkspaceError(f"{path}: cannot read journal: {error}") from error
    return entries


def verdicts(root: Path) -> dict[str, str]:
    """Return the last verdict (failed or converged) marked on each point."""
    return {
        str(entry["point_id"]): str(entry.get("verdict"))
        for entry in read_journal(root, MARKS)
        if "point_id" in entry
    }


def append_journal(root: Path, name: str, entry: dict[str, object]) -> None:
    """Append one entry to a journal, replacing the file atomically."""
    entries = [*read_journal(root, name), entry]
    data = json.dumps({"schema_version": 1, "entries": entries}, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=root, prefix=".journal-", suffix=".tmp", delete=False, encoding="utf-8"
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data + chr(10))
    os.replace(temporary, root / name)
