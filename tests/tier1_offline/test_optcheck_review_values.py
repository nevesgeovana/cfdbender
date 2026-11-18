"""Malformed provenance cannot satisfy an identity comparison or enter a receipt."""

import pytest

from cfdbender.core import plan
from cfdbender.matrix.receipt import _valid_points


@pytest.mark.parametrize(
    "bad",
    [
        "identity:",
        "identity:size=-1;mtime_ns=1",
        "identity:size=1;mtime_ns=-1",
        "identity:size=1;mtime_ns=1;extra=2",
        "identityish",
        "identity:size=" + "9" * 5000 + ";mtime_ns=1",
    ],
)
def test_malformed_values_never_agree_or_validate(tmp_path, bad):
    mesh = tmp_path / "mesh"
    mesh.write_bytes(b"x")
    current = plan.identity_value(mesh.stat())
    assert not plan.recorded_values_agree(tmp_path, "mesh", bad, current)
    assert not plan.recorded_values_agree(tmp_path, "mesh", bad, bad)
    assert not plan.recorded_value_valid(bad)
    assert not _valid_points([{"point_id": "p", "inputs_sha256": {"mesh": bad}}])
