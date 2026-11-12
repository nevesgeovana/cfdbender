"""Check the stable release condition."""

from pathlib import Path


def test_stable_release_condition():
    register = (
        Path(__file__).resolve().parents[2] / "docs/unpublished/requirements/register.md"
    ).read_text()
    row = next(line for line in register.splitlines() if line.startswith("| NFR-7 |"))
    assert "at least three" in row
    assert "pyflightstream imports cfdbender" in row
    assert not any(name in row for name in ("SU2", "FlightStream", "Fluent", "OpenFOAM"))
