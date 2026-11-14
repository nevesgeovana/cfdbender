"""OpenFOAM converges robustly by default: under-relaxation, or an opt-in LTS route."""

from __future__ import annotations

import pytest
import test_p0000_openfoam_adapter as foam


def _files(root, top="", **kw):
    foam.case(root, **kw)
    setup = root / "inputs/setups/s001.toml"
    text = setup.read_text(encoding="utf-8")
    setup.write_text(text.replace("[solver]", top + "[solver]", 1), encoding="utf-8")
    point = foam.point(foam.plan_workspace(root, "study"))
    return foam.split(foam.api().translate(point))


def test_default_relaxation_written_when_absent(tmp_path):
    files = _files(tmp_path)
    solution = files["system/fvSolution"]
    assert "relaxationFactors { fields { p 0.3; } equations { U 0.7; nuTilda 0.7; } }" in solution
    assert "residualControl { p 1e-05; U 1e-05; nuTilda 1e-05; }" in solution
    assert "steadyState" in files["system/fvSchemes"]
    assert "application simpleFoam;" in files["system/controlDict"]


def test_user_relaxation_wins_entirely(tmp_path):
    top = 'relaxationFactors = "fields { p 0.2; } equations { U 0.5; }"\n'
    solution = _files(tmp_path, top)["system/fvSolution"]
    assert "relaxationFactors { fields { p 0.2; } equations { U 0.5; } }" in solution
    assert "0.7" not in solution and "nuTilda 0.7" not in solution


def test_lts_dictionaries_use_valid_keys(tmp_path):
    files = _files(tmp_path, 'time_integration = "lts"\n')
    assert "application pimpleFoam;" in files["system/controlDict"]
    assert "ddtSchemes { default localEuler; }" in files["system/fvSchemes"]
    assert "bounded" not in files["system/fvSchemes"]
    solution = files["system/fvSolution"]
    for line in (
        "PIMPLE",
        "maxCo 5;",
        "rDeltaTSmoothingCoeff 0.1;",
        "rDeltaTDampingCoeff 0.2;",
        "nOuterCorrectors 1;",
        '"p.*"',
    ):
        assert line in solution
    assert "relaxationFactors" not in solution and "SIMPLE" not in solution
    assert "maxDeltaT" not in solution
    assert "pimpleFoam > log.pimpleFoam" in files["Allrun"]


def test_lts_parameters_and_native_pimple(tmp_path):
    top = 'time_integration = "lts"\nlts_max_co = 10\nlts_max_delta_t = 0.5\n'
    solution = _files(tmp_path, top)["system/fvSolution"]
    assert "maxCo 10;" in solution and "maxDeltaT 0.5;" in solution
    top = 'time_integration = "lts"\nPIMPLE = "nOuterCorrectors 2;\\nnCorrectors 1;"\n'
    solution = _files(tmp_path / "n", top)["system/fvSolution"]
    assert "nOuterCorrectors 2;" in solution and "maxCo" not in solution


@pytest.mark.parametrize("top", ["lts_max_co = 5\n", 'PIMPLE = "x;"\n'])
def test_lts_keys_need_the_lts_route(tmp_path, top):
    with pytest.raises(ValueError, match='time_integration = "lts"'):
        _files(tmp_path, top)


def test_bad_time_integration_is_refused(tmp_path):
    with pytest.raises(ValueError, match="steady"):
        _files(tmp_path, 'time_integration = "unsteady"\n')


def test_lts_log_is_read_by_the_residual_rule():
    from cfdbender.solvers.openfoam import _history

    line = "smoothSolver: Solving for {}, Initial residual = 1e-7, Final residual = 1e-9"
    log = "\n".join(["Time = 1", *(line.format(f) for f in ("Ux", "Uy", "p", "nuTilda"))])
    assert set(_history._residuals(log)[1]) == {"Ux", "Uy", "p", "nuTilda"}
