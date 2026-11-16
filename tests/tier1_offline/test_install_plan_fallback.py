"""Installation instructions include the local fallback of cluster selection."""

from pathlib import Path


def test_install_describes_the_plan_local_fallback():
    root = Path(__file__).resolve().parents[2]
    text = " ".join((root / "docs/install.md").read_text(encoding="utf-8").split())
    assert "falls back to local execution" in text
    assert "so it does not look" not in text
