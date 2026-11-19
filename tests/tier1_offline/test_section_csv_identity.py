"""Section identity cells containing CSV punctuation retain their column boundaries."""

import csv

import pytest
from test_wiring_friction import _plate, _sections, _surface

from cfdbender.post.sections import write_section_series


@pytest.mark.parametrize("label", ["wing,one", 'wing"one', "wing\none", "aile\u00e9"])
def test_section_identity_round_trips(tmp_path, label):
    result = _sections(_surface(*_plate()))
    result.family_label = label
    paths = write_section_series(
        [(result, {"FAMILY": label, "ROTOR": label})], tmp_path, "DP", "wing"
    )
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all(
            row["FAMILY"] == label and row["ROTOR"] == label and None not in row for row in rows
        )
