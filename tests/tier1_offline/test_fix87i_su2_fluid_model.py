"""FIX87I: SU2 applies cfdbender's gas constants, which STANDARD_AIR would override."""

from __future__ import annotations

import warnings

import pytest
from test_fix87f_su2_cfl_ramp import _lines


def test_ideal_gas_written_with_the_core_constants(tmp_path):
    lines = _lines(tmp_path)
    assert "FLUID_MODEL= IDEAL_GAS" in lines
    assert "GAS_CONSTANT= 287.05287" in lines
    assert "GAMMA_VALUE= 1.4" in lines


def test_native_gas_constant_alone_still_takes_effect(tmp_path):
    lines = _lines(tmp_path, "GAS_CONSTANT = 287.0")
    assert "FLUID_MODEL= IDEAL_GAS" in lines
    assert "GAS_CONSTANT= 287" in lines


def test_native_fluid_model_wins_and_warns(tmp_path):
    with pytest.warns(UserWarning, match="FLUID_MODEL"):
        lines = _lines(tmp_path, 'FLUID_MODEL = "STANDARD_AIR"')
    assert "FLUID_MODEL= STANDARD_AIR" in lines
    assert not any(line.startswith(("GAS_CONSTANT=", "GAMMA_VALUE=")) for line in lines)


def test_native_ideal_gas_with_the_core_constants_is_silent(tmp_path):
    extra = 'FLUID_MODEL = "IDEAL_GAS"\nGAS_CONSTANT = 287.05287\nGAMMA_VALUE = 1.4'
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines = _lines(tmp_path, extra)
    assert "FLUID_MODEL= IDEAL_GAS" in lines
    assert "GAS_CONSTANT= 287.05287" in lines


def test_native_ideal_gas_without_constants_warns(tmp_path):
    with pytest.warns(UserWarning, match="GAS_CONSTANT"):
        lines = _lines(tmp_path, 'FLUID_MODEL = "IDEAL_GAS"')
    assert not any(line.startswith(("GAS_CONSTANT=", "GAMMA_VALUE=")) for line in lines)
