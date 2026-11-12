"""P0000: each saved coefficient states its own frame alongside its axis."""

import json

import pytest
from test_p0000_item3d_execute import FAKE, case, execution_api


@pytest.mark.parametrize("frame", ["wind", "body"])
def test_p0000_item3d_coefficient_frame_in_serialized_result(tmp_path, frame):
    expected = json.loads(FAKE.with_name("expected.json").read_text())
    point = case(tmp_path)
    adapter = execution_api()
    adapter.execute(point, check_ram=lambda: None)
    result = adapter.collect(point).to_dict()
    for name, column in expected["coefficients"].items():
        if column["axis_system"] == frame:
            assert result["coefficients"][name] == column, (
                "each coefficient must identify its frame"
            )
