"""FIX87I: the SU2 residual recorded per iteration is the CONV_FIELD residual, by name."""

from __future__ import annotations

import pytest

from cfdbender.solvers.su2._history import read, residual_field

CFG = "REF_AREA= 1\nREF_LENGTH= 1\nREF_ORIGIN_MOMENT_X= 0\nREF_ORIGIN_MOMENT_Y= 0\n"
CFG += "REF_ORIGIN_MOMENT_Z= 0\nITER= 2\n"
HISTORY = "Inner_Iter,rms[Rho],rms[RhoU],rms[RhoE],rms[nu],CL,CD\n"
HISTORY += "0,-3.0,-2.0,-1.0,-4.0,0.3,0.02\n1,-10.0,-6.0,-5.76,-8.0,0.3,0.02\n"


def _folder(tmp_path, extra=""):
    (tmp_path / "su2.cfg").write_text(CFG + extra, encoding="ascii")
    (tmp_path / "history.csv").write_text(HISTORY, encoding="ascii")
    return tmp_path


def test_default_conv_field_is_rms_density(tmp_path):
    folder = _folder(tmp_path)
    result, _, _ = read(folder)
    assert result.history[-1].residual == pytest.approx(1e-10)
    assert residual_field(folder) == "rms[Rho]"


def test_stated_conv_field_selects_its_column(tmp_path):
    folder = _folder(tmp_path, "CONV_FIELD= RMS_ENERGY\n")
    result, _, _ = read(folder)
    assert result.history[-1].residual == pytest.approx(10**-5.76)
    assert residual_field(folder) == "rms[RhoE]"


def test_several_conv_fields_take_the_largest_of_them(tmp_path):
    folder = _folder(tmp_path, "CONV_FIELD= ( RMS_DENSITY, RMS_NU_TILDE )\n")
    result, _, _ = read(folder)
    assert result.history[-1].residual == pytest.approx(1e-8)
    assert residual_field(folder) == "rms[Rho], rms[nu]"


def test_a_coefficient_conv_field_falls_back_to_the_largest_rms(tmp_path):
    folder = _folder(tmp_path, "CONV_FIELD= DRAG\n")
    result, _, _ = read(folder)
    assert result.history[-1].residual == pytest.approx(10**-5.76)
    assert residual_field(folder) == "largest rms"


def test_the_result_names_the_residual(tmp_path):
    from cfdbender.solvers.su2 import _SU2Result

    folder = _folder(tmp_path)
    result, coefficients, refs = read(folder)
    data = _SU2Result(
        coefficients,
        refs,
        result.history,
        "iteration_limit",
        "8.0.0",
        "executable",
        residual_field=residual_field(folder),
    ).to_dict()
    assert data["residual_field"] == "rms[Rho]"
