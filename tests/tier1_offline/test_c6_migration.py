"""The next release explains the commands and imports replacing removed interfaces."""

from pathlib import Path


def test_release_migration_is_in_the_changelog():
    root = Path(__file__).resolve().parents[2]
    current = (root / "CHANGELOG.md").read_text(encoding="utf-8").split("## [0.5.1]")[0]
    migration = current.split("### Migration\n", 1)[-1]
    assert "### Migration\n" in current
    for replacement in (
        "cfd-workspace show",
        "cfd-workspace sync",
        "campaign_sweep.csv",
        "P<POL>-TOTAL-ACOEF.csv",
        "schema version 2",
        "cfdbender.geometry.blade_sections",
        'until="S10"',
        "pip install",
    ):
        assert replacement in migration, replacement
