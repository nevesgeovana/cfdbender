"""New result table names cannot replace files without matching product provenance."""

import hashlib
import json

import pytest

from cfdbender.matrix.exports import _write


def test_unrecorded_table_is_preserved(tmp_path):
    target = tmp_path / "post/study/polars/P1001.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user data\n")
    with pytest.raises(ValueError, match="overwrite"):
        _write(target, "generated data\n")
    assert target.read_bytes() == b"user data\n"


def test_edited_product_is_preserved(tmp_path):
    target = tmp_path / "post/study/polars/FAM-DIM-P1001.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user edits\n")
    manifest = target.parent.parent / "products.json"
    manifest.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "path": target.relative_to(tmp_path).as_posix(),
                        "sha256": hashlib.sha256(b"original\n").hexdigest(),
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="overwrite"):
        _write(target, "new generated data\n")
    assert target.read_bytes() == b"user edits\n"


def test_recorded_unchanged_product_can_be_regenerated(tmp_path):
    target = tmp_path / "post/study/polars/P1001.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original\n")
    manifest = target.parent.parent / "products.json"
    manifest.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "path": target.relative_to(tmp_path).as_posix(),
                        "sha256": hashlib.sha256(b"original\n").hexdigest(),
                    }
                ]
            }
        )
    )
    _write(target, "updated\n")
    assert target.read_bytes() == b"updated\n"
