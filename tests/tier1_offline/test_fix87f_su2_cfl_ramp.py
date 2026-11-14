"""The SU2 adapter writes a CFL ramp by default; any CFL key of the user replaces it whole."""

from __future__ import annotations

import test_p0000_item3d_config as su2_case


def _lines(root, extra=""):
    su2_case.workspace(root)
    setup = root / "inputs/setups/s001.toml"
    kept = [
        line
        for line in setup.read_text(encoding="utf-8").splitlines()
        if "CFL" not in line and not line.strip().startswith("cfl")
    ]
    setup.write_text("\n".join([*kept, extra, ""]), encoding="utf-8")
    return su2_case.api().translate(su2_case.prepared(root)).splitlines()


def test_default_ramp_written_when_no_cfl_key(tmp_path):
    lines = _lines(tmp_path)
    assert "CFL_NUMBER= 5" in lines
    assert "CFL_ADAPT= YES" in lines
    assert "CFL_ADAPT_PARAM= ( 0.5, 1.2, 5, 25 )" in lines


def test_cfl_number_of_the_user_wins_without_a_ramp(tmp_path):
    lines = _lines(tmp_path, "CFL_NUMBER = 10")
    assert "CFL_NUMBER= 10" in lines
    assert not any(line.startswith("CFL_ADAPT") for line in lines)


def test_neutral_cfl_key_wins_without_a_ramp(tmp_path):
    lines = _lines(tmp_path, "cfl = 12")
    assert "CFL_NUMBER= 12" in lines
    assert not any(line.startswith("CFL_ADAPT") for line in lines)


def test_user_adapt_param_replaces_the_whole_default(tmp_path):
    lines = _lines(tmp_path, "CFL_ADAPT = true\nCFL_ADAPT_PARAM = [0.1, 1.5, 1, 100]")
    assert "CFL_ADAPT_PARAM= ( 0.1, 1.5, 1, 100 )" in lines
    assert "CFL_ADAPT= YES" in lines
    assert not any(line.startswith("CFL_NUMBER") for line in lines)


def _value(lines, key):
    for line in lines:
        if line.startswith(key + "="):
            return float(line.split("=", 1)[1])
    raise AssertionError(key)


def test_gas_constants_come_from_core(tmp_path):
    from cfdbender.core.flight import GAMMA, GAS_CONSTANT, speed_of_sound

    lines = _lines(tmp_path)
    assert "GAS_CONSTANT= 287.05287" in lines
    assert "GAMMA_VALUE= 1.4" in lines
    t = (
        _value(lines, "FREESTREAM_TEMPERATURE")
        if any(line.startswith("FREESTREAM_TEMPERATURE=") for line in lines)
        else 288.15
    )
    speed = (_value(lines, "GAMMA_VALUE") * _value(lines, "GAS_CONSTANT") * t) ** 0.5
    assert abs(speed - speed_of_sound(t)) / speed_of_sound(t) < 1e-12
    assert (GAMMA, GAS_CONSTANT) == (1.4, 287.05287)


def test_native_gas_constant_wins(tmp_path):
    lines = _lines(tmp_path, "GAS_CONSTANT = 287.058")
    assert "GAS_CONSTANT= 287.058" in lines
