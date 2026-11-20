"""P0700: release coordinates and notes of 0.7.0."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADING = "## [0.7.0] - 2026-11-19"


def _release() -> str:
    notes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert HEADING in notes
    assert "## [0.6.1]" not in notes
    assert notes.index("## [Unreleased]") < notes.index(HEADING) < notes.index("## [0.6.0]")
    return notes.split(HEADING, 1)[1].split("## [0.6.0]", 1)[0]


def test_p0700_changelog_sections_and_fragments():
    release = _release()
    for section in ("Added", "Changed", "Fixed", "Migration"):
        assert release.count(f"### {section}\n") == 1, section
    assert "### Removed" not in release
    assert not [p for p in (ROOT / "changelog.d").glob("*.md") if p.name != "README.md"]
    assert chr(0x2014) not in release and chr(0x2013) not in release


def test_p0700_changelog_names_the_capabilities():
    release = _release()
    for term in (
        "`NUMERICS`",
        "cfdbender.migrate_setup_numerics",
        "`WORKFLOW`",
        "`ADVANCE_RATIO`",
        "`cfdbender.core.unsteady_window`",
        "`[volume_section]`",
        "`cfdbender.post.farfield`",
        "`hpc.<name>: <value>`",
        "--batch N`",
        "`cfd-workspace setups",
        "incomplete and not yet verified against the solver",
    ):
        assert term in release, term


def test_p0700_changelog_keeps_the_folded_planning_items():
    release = _release()
    assert "`cfd-matrix plan`" in release
    assert "streams files above 8 MiB" in release
    assert "input changed while planning; plan again" in release
    assert "`--rehash-inputs`" in release
    assert ".input-digests.cache" in release
    assert "OpenFOAM" in release and "PyFR" in release
    assert "`--no-read-large-inputs`" in release
    assert "CFDBENDER_NO_READ_LARGE_INPUTS" in release
    assert "large_input_check" in release
    assert "identity:size=...;mtime_ns=..." in release
    assert "never verified" in release
    assert "`/.input-digests.cache`" in release
    assert "FlightStream workspace" in release


def test_p0700_migration_tells_a_0_6_0_user_what_changes():
    release = _release()
    migration = release.split("### Migration\n", 1)[1]
    assert "new input layout is optional" in migration
    assert "migrate_setup_numerics" in migration
    assert "`blade1`" in migration
    assert "`CT` is negative" in migration
    assert "pressure plus friction" in migration
    assert "last step only" in migration
