"""Explain the replacement-file blind spot at both discovery points."""

from pathlib import Path

from cfdbender.matrix.cli import NO_READ_HELP


def test_identity_help_and_layout_explain_replaced_files():
    layout = Path(__file__).resolve().parents[2] / "docs/workspace-layout.md"
    for text in (NO_READ_HELP, layout.read_text(encoding="utf-8")):
        assert "replaced by a different file" in text
        assert "same size and modification time" in text
