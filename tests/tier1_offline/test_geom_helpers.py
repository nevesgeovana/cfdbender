"""The shared helpers of the geometry layers: refusals, digests, text writing, lengths."""

from __future__ import annotations

import hashlib
import io

import pytest

from cfdbender._digest import file_sha256, text_sha256
from cfdbender._errors import (
    CheckFailedError,
    GeometryError,
    InputArtifactError,
    InputValueError,
)
from cfdbender._lengths import METRES_PER_UNIT, scale
from cfdbender._textio import csv_writer, write_text


def test_geom_helpers_refusals_keep_standard_bases():
    assert issubclass(InputValueError, ValueError)
    assert issubclass(InputValueError, GeometryError)
    assert issubclass(InputArtifactError, RuntimeError)
    error = InputArtifactError("missing", kind="mesh", available=("a",), where={"nodes": [1]})
    assert (error.kind, error.available, error.where) == ("mesh", ("a",), {"nodes": [1]})
    assert InputArtifactError("x").where == {}


def test_geom_helpers_check_failed_carries_its_check():
    check = {"id": "closure.gap", "note": "gap too large", "status": "failed"}
    error = CheckFailedError(check)
    assert isinstance(error, InputValueError)
    assert str(error) == "closure.gap: gap too large"
    assert error.check == check and error.check is not check


def test_geom_helpers_digests_hash_bytes_and_utf8(tmp_path):
    target = tmp_path / "a.txt"
    target.write_bytes(b"a\r\nb")
    assert file_sha256(target) == hashlib.sha256(b"a\r\nb").hexdigest()
    assert text_sha256("é") == hashlib.sha256("é".encode()).hexdigest()


def test_geom_helpers_write_text_is_lf_and_replaces(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    assert write_text(target, "a\nb\n") == 4
    assert target.read_bytes() == b"a\nb\n"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_geom_helpers_write_text_failure_keeps_previous(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(UnicodeEncodeError):
        write_text(target, "é", encoding="ascii")
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_geom_helpers_csv_rows_end_with_lf():
    stream = io.StringIO(newline="")
    csv_writer(stream).writerow(["a", 1])
    assert stream.getvalue() == "a,1\n"


def test_geom_helpers_length_scale_is_exact():
    assert scale("METER", "MILLIMETER") == 1000.0
    assert scale("INCH", "MILLIMETER") == 25.4
    assert scale("METER", "OTHER") is None
    assert set(METRES_PER_UNIT) >= {"METER", "INCH", "FEET"}


def test_geom_sources_cite_no_unpublished_document():
    """RG-4: a public module states its limits without the ids of an unpublished hand-off."""
    from pathlib import Path

    import cfdbender

    package = Path(cfdbender.__file__).parent
    cited = [
        f"{path.relative_to(package).as_posix()}:{number}"
        for path in sorted(package.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "hand-off" in line
    ]
    assert cited == []
