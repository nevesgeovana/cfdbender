"""P0600: publish consistent release coordinates and assembled release notes."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_p0600_release_and_kit_coordinates():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.6.1"
    for name in (
        "src/cfdbender/__init__.py",
        "CITATION.cff",
        "README.md",
        "tools/kit/FIRST-PROMPT.md",
        "tools/kit/README-FIRST.md",
        "guide/latex-sources/shared/info.tex",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "0.6.1" in text and "0.5.1" not in text, name
    assert "date-released: 2026-11-18" in (ROOT / "CITATION.cff").read_text(encoding="utf-8")


def test_p0600_changelog_is_assembled_for_the_release():
    notes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [0.6.0] - 2026-11-16" in notes
    release = notes.split("## [0.6.0] - 2026-11-16", 1)[1].split("## [0.5.2]", 1)[0]
    assert "## [0.5.3]" not in notes
    assert not [p for p in (ROOT / "changelog.d").glob("*.md") if p.name != "README.md"]
    for section in ("Added", "Changed", "Fixed", "Removed", "Migration"):
        assert release.count(f"### {section}\n") == 1
    for term in (
        "cluster:",
        "hpc:",
        "cfd-geom",
        "cfd-mesh airfoil-2d",
        "--warning-refusals",
        "RESTART",
        "force_groups",
        "P<POL>.txt",
        "FAM-ACOEF",
        "schema version 2",
        "cfdbender.geometry.blade_sections",
        "HPC plan receipts",
        "pip install",
    ):
        assert term in release, term


def test_p0600_all_guide_pdfs_state_the_release_version():
    pdfs = sorted((ROOT / "guide/pdf").glob("*.pdf"))
    assert len(pdfs) == 4
    for path in pdfs:
        assert path.stat().st_size > 20_000, path.name
        keywords = re.search(rb"/Keywords\s*\((.*?)\)", path.read_bytes())
        assert keywords is not None, path.name
        raw = re.sub(
            rb"\\(\d{3})", lambda octal: bytes([int(octal.group(1), 8)]), keywords.group(1)
        )
        text = raw.decode("utf-16") if raw.startswith(b"\xfe\xff") else raw.decode("latin-1")
        assert text == "cfdbender 0.6.1", path.name


def test_p0600_grid_crossing_advice_is_preserved():
    notes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release = notes.split("## [0.6.0]", 1)[1].split("## [0.5.2]", 1)[0]
    assert "trimming relationships" in release


def test_p0600_export_diagnostics_do_not_promise_a_face_for_every_check():
    notes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release = notes.split("## [0.6.0]", 1)[1].split("## [0.5.2]", 1)[0]
    assert "available location" in release
    assert "limit, component,\n  face and corrective action" not in release


def test_p0600_correspondence_describes_the_trim_fallback():
    notes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release = notes.split("## [0.6.0]", 1)[1].split("## [0.5.2]", 1)[0]
    assert "bounding-box fallback" in release
