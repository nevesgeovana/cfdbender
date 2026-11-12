"""Read the authored requirement register using pyflightstream's table convention."""

import re
from pathlib import Path

REGISTER = Path("docs/unpublished/requirements/register.md")
ROW_COLUMNS = (
    "uid",
    "parent",
    "hid",
    "level",
    "status",
    "verification_method",
    "statement",
    "rationale",
    "derives_from",
    "code",
    "test",
)


def requirement_rows(text: str, path: str = "<text>") -> list[dict[str, str]]:
    """Parse requirement tables and reject invalid or incomplete rows."""
    rows: list[dict[str, str]] = []
    columns: tuple[str, ...] = ()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip().startswith("|"):
            columns = ()
            continue
        cells = [cell.strip().replace("&#124;", "|") for cell in line.strip().strip("|").split("|")]
        if cells[0].lower() == "uid":
            columns = tuple(cell.lower().replace(" ", "_") for cell in cells)
            if columns != ROW_COLUMNS:
                raise ValueError(f"{path}:{number}: unknown columns")
        elif columns and not all(re.fullmatch(":?-+:?", cell) for cell in cells):
            if len(cells) != len(columns):
                raise ValueError(f"{path}:{number}: malformed row")
            row = dict(zip(columns, cells, strict=True))
            if not re.fullmatch(r"N?FR-[0-9]+", row["uid"]):
                raise ValueError(f"{path}:{number}: invalid UID")
            if row["status"] not in {"implemented", "pending", "deferred", "retired"}:
                raise ValueError(f"{path}:{number}: invalid status")
            if (
                row["level"] not in {"L1", "L2"}
                or not row["statement"]
                or not row["verification_method"]
            ):
                raise ValueError(f"{path}:{number}: incomplete requirement")
            rows.append(row)
    return rows


def read_requirements(root: Path) -> list[dict[str, str]]:
    """Read the register and reject duplicate identities and invalid parents."""
    path = root / REGISTER
    rows = requirement_rows(path.read_text(encoding="utf-8"), str(path))
    by_id = {row["uid"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate requirement UID")
    for row in rows:
        if row["level"] == "L2" and (
            row["parent"] not in by_id or by_id[row["parent"]]["level"] != "L1"
        ):
            raise ValueError(f"invalid parent for {row['uid']}")
    return rows
