"""The CFD++ page says plainly that the adapter is incomplete, and the tables list it so."""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_the_page_states_its_status_in_one_sentence():
    text = (DOCS / "cfdpp-workspaces.md").read_text(encoding="utf-8")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    found = [
        s for s in sentences if "incomplete" in s and "not yet verified against the solver" in s
    ]
    assert len(found) == 1


def test_the_page_lists_what_works_offline_and_what_does_not_exist():
    text = (DOCS / "cfdpp-workspaces.md").read_text(encoding="utf-8")
    assert "Works offline" in text and "Does not exist yet" in text


def test_the_solver_tables_list_cfdpp_with_its_status():
    text = (DOCS / "solvers.md").read_text(encoding="utf-8")
    assert "[CFD++](cfdpp-workspaces.md) (incomplete)" in text
    assert "CFD++ is incomplete and not yet verified against the solver" in text
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    assert "incomplete and not yet verified against the solver" in index
