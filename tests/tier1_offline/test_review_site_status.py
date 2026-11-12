"""The site must describe the existing workspace command accurately."""

from pathlib import Path


def test_site_describes_mock_initialization():
    docs = Path(__file__).resolve().parents[2] / "docs"
    for name in ("index.md", "getting-started.md"):
        page = (docs / name).read_text(encoding="utf-8")
        assert "workspace command yet" not in page
        assert "workspace commands are not" not in page
        assert "mock" in page and "init" in page
