"""P0600: publish consistent release coordinates and assembled release notes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
