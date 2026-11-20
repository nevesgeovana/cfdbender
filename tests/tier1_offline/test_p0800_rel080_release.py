"""P0800: release coordinates and notes of 0.8.0 (the current version)."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADING = "## [0.8.0] - 2026-11-20"


def _release() -> str:
    notes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert HEADING in notes
    assert notes.index("## [Unreleased]") < notes.index(HEADING) < notes.index("## [0.7.0]")
    between = notes.split("## [Unreleased]", 1)[1].split(HEADING, 1)[0]
    assert between.strip() == ""
    return notes.split(HEADING, 1)[1].split("## [0.7.0]", 1)[0]


def test_p0800_version_coordinates():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.8.0"
    init = (ROOT / "src/cfdbender/__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "0.8.0"' in init
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"(?m)^version: 0\.8\.0$", citation)
    assert "date-released: 2026-11-20" in citation
    for name in (
        "README.md",
        "tools/kit/FIRST-PROMPT.md",
        "tools/kit/README-FIRST.md",
        "guide/latex-sources/shared/info.tex",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "0.8.0" in text and "0.7.0" not in text, name


def test_p0800_all_guide_pdfs_state_the_release_version():
    pdfs = sorted((ROOT / "guide/pdf").glob("*.pdf"))
    assert len(pdfs) == 4
    for path in pdfs:
        keywords = re.search(rb"/Keywords\s*\((.*?)\)", path.read_bytes())
        assert keywords is not None, path.name
        raw = re.sub(
            rb"\\(\d{3})", lambda octal: bytes([int(octal.group(1), 8)]), keywords.group(1)
        )
        text = raw.decode("utf-16") if raw.startswith(b"\xfe\xff") else raw.decode("latin-1")
        assert text == "cfdbender 0.8.0", path.name


def test_p0800_changelog_sections_and_fragments():
    release = _release()
    for section in ("Added", "Changed", "Fixed", "Migration"):
        assert release.count(f"### {section}\n") == 1, section
    assert "### Removed" not in release
    assert not [p for p in (ROOT / "changelog.d").glob("*.md") if p.name != "README.md"]
    assert chr(0x2014) not in release and chr(0x2013) not in release


def test_p0800_changelog_names_the_capabilities():
    release = _release()
    for term in (
        "--continue",
        "RESTART_EVERY_REV",
        "[[volume_sections]]",
        "--overwrite",
        "--batch-per-polar",
        "cfd-matrix status",
        "`--json`",
        "every wall marker",
        "MARKER_MONITORING",
        "--new-pproc",
        "run_mrf.sh",
    ):
        assert term in release, term


def test_p0800_migration_tells_a_0_7_0_user_what_changes():
    migration = _release().split("### Migration\n", 1)[1]
    assert "status --json" in migration
    assert "MARKER_MONITORING" in migration
    assert "families_general" in migration
    assert "[[volume_sections]]" in migration
    assert "--overwrite" in migration
