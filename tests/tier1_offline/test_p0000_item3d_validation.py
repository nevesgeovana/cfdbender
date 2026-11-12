"""P0000: physically invalid neutral values refuse before creating run outputs."""

import pytest
from test_p0000_item3d_config import plan, workspace


@pytest.mark.parametrize(
    "name,value,problem",
    [
        ("mach", -0.1, "MACH_NUMBER"),
        ("freestream_temperature", 0, "FREESTREAM_TEMPERATURE"),
        ("freestream_pressure", -1, "FREESTREAM_PRESSURE"),
        ("freestream_density", 0, "FREESTREAM_DENSITY"),
        ("reynolds", -10, "REYNOLDS_NUMBER"),
        ("reynolds_length", 0, "REYNOLDS_LENGTH"),
    ],
)
def test_p0000_item3d_refuses_invalid_neutral_physics(tmp_path, name, value, problem):
    workspace(tmp_path, condition="ALPHA:2")
    path = tmp_path / "inputs/setups/s001.toml"
    lines = [line for line in path.read_text().splitlines() if not line.startswith(name + " =")]
    path.write_text(chr(10).join([*lines, f"{name} = {value}"]))
    with pytest.raises(ValueError, match=problem):
        plan(tmp_path)
    assert not (tmp_path / "sims").exists()


def test_p0000_item3d_unknown_option_names_key(tmp_path):
    workspace(tmp_path, extra="[solver]" + chr(10) + "threadz = 2")
    with pytest.raises(ValueError, match="threadz"):
        plan(tmp_path)
