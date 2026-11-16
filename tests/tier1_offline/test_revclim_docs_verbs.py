"""RC-3: no page names a moved verb as a cfd-matrix command.

The thirteen verbs live under cfd-workspace. A page that lists them after ``cfd-matrix``
sends the reader to a command that refuses.
"""

from __future__ import annotations

import re
from pathlib import Path

from cfdbender.matrix.cli import MOVED

ROOT = Path(__file__).resolve().parents[2]
#: The page that states the refusal itself, so it quotes the old form on purpose.
QUOTES_OLD_FORM = {"cli.md"}
VERBS = "|".join(MOVED)
#: A program name, or a verb in code font (alone, or right after a program name).
TOKEN = re.compile(rf"(cfd-matrix|cfd-workspace)(?: ({VERBS})\b)?|`({VERBS})`")


def _wrong(line: str) -> bool:
    program = None
    for match in TOKEN.finditer(line):
        if match.group(1):
            program = match.group(1)
            if match.group(2) and program == "cfd-matrix":
                return True
        elif program == "cfd-matrix":
            return True
    return False


def test_rc3_the_detector_sees_the_old_forms_and_not_the_new_ones():
    assert _wrong("| Follow | `cfd-matrix status`, `show`, `history` |")
    assert _wrong("run `cfd-matrix show 1001`")
    assert not _wrong("| Follow | `cfd-matrix status`, `cfd-workspace show`, `history` |")
    assert not _wrong("run `cfd-workspace show 1001`")


def test_rc3_no_page_lists_a_moved_verb_under_cfd_matrix():
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    found = [
        f"{page.name}:{number}: {line.strip()}"
        for page in pages
        if page.name not in QUOTES_OLD_FORM
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1)
        if _wrong(line)
    ]
    assert not found, "\n".join(found)
