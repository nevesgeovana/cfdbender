"""Post metadata must not replace unrelated or edited user files."""

import json

import pytest

from cfdbender.matrix.collection import _write_summary
from cfdbender.matrix.products import write_products


def _collect(root, rows=None):
    rows = [] if rows is None else rows
    summary = _write_summary(root, "inputs/matrices/study.fs", "mock", rows)
    return write_products(root, "inputs/matrices/study.fs", "mock", rows, summary, [])


@pytest.mark.parametrize("name", ["products.json", "post.log", "campaign_summary.json"])
@pytest.mark.parametrize("generated_first", [False, True])
def test_metadata_collision_preserves_user_bytes(tmp_path, name, generated_first):
    if generated_first:
        _collect(tmp_path)
    target = tmp_path / "post/study" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"user metadata\n")
    with pytest.raises(ValueError, match="overwrite"):
        _collect(tmp_path)
    assert target.read_bytes() == b"user metadata\n"


def test_unmodified_metadata_can_be_regenerated(tmp_path):
    manifest, log = _collect(tmp_path)
    original = manifest.read_bytes()
    _collect(tmp_path, [{"result": {}, "point_id": "updated"}])
    assert manifest.read_bytes() != original
    assert log.read_text().startswith("post inputs/matrices/study.fs (mock)")


@pytest.mark.parametrize("edit", ["value", "whitespace"])
def test_valid_json_edits_are_preserved(tmp_path, edit):
    manifest, _ = _collect(tmp_path)
    if edit == "value":
        data = json.loads(manifest.read_bytes())
        data["notes"] = "user notes"
        edited = json.dumps(data).encode()
    else:
        edited = b" " + manifest.read_bytes()
    manifest.write_bytes(edited)
    with pytest.raises(ValueError, match="overwrite"):
        _collect(tmp_path)
    assert manifest.read_bytes() == edited
