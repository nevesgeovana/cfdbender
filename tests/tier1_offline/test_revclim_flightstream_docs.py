"""RC-6: the FlightStream page says which program takes the moved verbs."""

from __future__ import annotations

from pathlib import Path

from cfdbender.matrix.cli import MOVED

PAGE = Path(__file__).resolve().parents[2] / "docs" / "flightstream-workspaces.md"


def test_rc6_the_flightstream_page_names_every_moved_verb():
    text = " ".join(PAGE.read_text(encoding="utf-8").split())
    missing = [verb for verb in MOVED if f"`{verb}`" not in text]
    assert not missing
    assert "`cfd-workspace` hands them to `pyfs-matrix`" in text
