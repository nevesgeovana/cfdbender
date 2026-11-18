"""P0610: release coordinates and notes of 0.6.1."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_p0610_version_coordinates():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.6.1"
    init = (ROOT / "src/cfdbender/__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "0.6.1"' in init
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"(?m)^version: 0\.6\.1$", citation)
    assert "date-released: 2026-11-18" in citation


def test_p0610_changelog_names_the_plan_fix():
    notes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [0.6.1] - 2026-11-18" in notes
    assert notes.index("## [Unreleased]") < notes.index("## [0.6.1]") < notes.index("## [0.6.0]")
    release = notes.split("## [0.6.1] - 2026-11-18", 1)[1].split("## [0.6.0]", 1)[0]
    assert "cfd-matrix plan" in release
    assert "once" in release and "streams" in release
    assert "input changed while planning; plan again" in release


def test_p0610_changelog_names_the_run_fix_and_the_flag():
    notes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release = notes.split("## [0.6.1] - 2026-11-18", 1)[1].split("## [0.6.0]", 1)[0]
    assert "`run`" in release and "`collect`" in release and "once" in release
    assert "`--rehash-inputs`" in release
    assert ".input-digests.cache" in release
    assert "OpenFOAM" in release and "PyFR" in release
    assert "Known limit: OpenFOAM and PyFR still read" not in release
    assert "`--no-read-large-inputs`" in release
    assert "CFDBENDER_NO_READ_LARGE_INPUTS" in release
    assert "large_input_check" in release
    assert "identity:size=...;mtime_ns=..." in release
    assert "never verified" in release
    assert "`/.input-digests.cache`" in release
    assert "FlightStream workspace" in release
    assert chr(0x2014) not in release and chr(0x2013) not in release
