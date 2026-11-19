"""Repository guard: no file under tests/fixtures/cfdpp that the builder did not produce.

The builder writes every fixture into a temporary folder at test time, so the committed
folder holds the builder and nothing else. One named exemption exists: ``toy_schema/``,
a hand-written TOML database of invented sections and keywords for the toy solver, which
states no fact of the real solver and is read by the schema tests as data. Anything else
(a grid file, an input, a log, a harvested file) would be a vendor-looking file that the
builder did not write and fails here.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cfdpp"

BUILDER = "build.py"
TOY_SCHEMA = "toy_schema"
TOY_SCHEMA_SUFFIX = ".toml"


def _committed() -> list[Path]:
    return sorted(
        path.relative_to(FIXTURES)
        for path in FIXTURES.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def test_no_fixture_file_is_outside_the_builder_and_the_toy_schema() -> None:
    foreign = [
        path.as_posix()
        for path in _committed()
        if path.as_posix() != BUILDER and path.parts[0] != TOY_SCHEMA
    ]
    assert foreign == []


def test_the_toy_schema_exemption_holds_only_toml_data() -> None:
    toy = [path for path in _committed() if path.parts[0] == TOY_SCHEMA]
    assert toy, "the toy schema is missing"
    assert [path.as_posix() for path in toy if path.suffix != TOY_SCHEMA_SUFFIX] == []
    assert all(len(path.parts) == 2 for path in toy), "the toy schema holds no subfolder"
