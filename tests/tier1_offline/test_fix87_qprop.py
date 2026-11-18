"""Walk findings U-01 and U-02 and the QPROP section settings kept in init.toml."""

from __future__ import annotations

import tomllib
import warnings

import pytest
import qprop_workspace as qw

from cfdbender.core import progress
from cfdbender.core.settings import read_init
from cfdbender.solvers.qprop._config import options
from cfdbender.workspace.cli import main as workspace_cli


def test_q8_init_writes_both_section_settings_visibly(tmp_path):
    assert workspace_cli(["init", "--solver", "qprop", "--workspace", str(tmp_path)]) == 0
    text = (tmp_path / "init.toml").read_text(encoding="utf-8")
    assert "section_timeout_s = 60" in text and 'on_unconverged_section = "warn"' in text
    table = tomllib.loads(text)["solver"]
    assert table["name"] == "qprop" and table["section_timeout_s"] == 60
    declared = read_init(tmp_path, available_solvers=("qprop",))
    assert declared.solver == "qprop"
    merged = options({}, declared.solver_options)
    assert merged["section_timeout_s"] == 60 and merged["on_unconverged_section"] == "warn"


def test_q8_defaults_apply_when_init_states_nothing():
    merged = options({})
    assert merged["section_timeout_s"] == 60 and merged["on_unconverged_section"] == "warn"


def test_q8_a_setup_value_overrides_init():
    merged = options(
        {"solver": {"section_timeout_s": 5, "on_unconverged_section": "fail"}},
        {
            "section_timeout_s": 60,
            "on_unconverged_section": "warn",
        },
    )
    assert merged["section_timeout_s"] == 5 and merged["on_unconverged_section"] == "fail"


def test_q8_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="on_unconverged_section"):
        options({"solver": {"on_unconverged_section": "ignore"}})


def _run(tmp_path, monkeypatch, partial, policy):
    root = qw.workspace(tmp_path / "w", options=f'on_unconverged_section = "{policy}"\n')
    monkeypatch.setenv("CFDBENDER_QPROP_FAKE_XFOIL", f"sec_02={partial}")
    lines: list[str] = []
    with progress.sink(lines.append), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        point = qw.prepared(root)
        outcome = None
        try:
            qw.adapter().execute(point, check_ram=lambda: None)
        except ValueError as error:
            outcome = error
    return lines, [str(w.message) for w in caught], outcome


def test_q8_warn_drops_the_angles_runs_on_the_rest_and_names_section_and_angles(
    tmp_path, monkeypatch
):
    lines, warned, outcome = _run(tmp_path, monkeypatch, "partial", "warn")
    assert outcome is None
    hit = [w for w in warned if "section 2/" in w and "did not converge at alpha" in w]
    assert hit and "dropped" in hit[0], warned


def test_q8_fail_fails_the_point_naming_section_and_angles(tmp_path, monkeypatch):
    _, _, outcome = _run(tmp_path, monkeypatch, "partial", "fail")
    assert isinstance(outcome, ValueError)
    assert "section 2/" in str(outcome) and "alpha" in str(outcome)


def test_u02_one_progress_line_per_section_start_and_end(tmp_path, monkeypatch):
    lines, _, outcome = _run(tmp_path, monkeypatch, "partial", "warn")
    assert outcome is None
    started = [x for x in lines if x.endswith("started")]
    ended = [x for x in lines if " in " in x and x.rstrip().endswith(" s")]
    assert len(started) == 5 and len(ended) == 5, lines
    assert "section 2/5" in started[1]
    assert "did not converge" in next(x for x in ended if "section 2/5" in x)
    assert "XFOIL converged" in next(x for x in ended if "section 1/5" in x)


def test_u01_a_run_takes_the_blade_from_geometry_alone(tmp_path):
    root = qw.workspace(tmp_path / "w", setup=qw.setup_text(blade=None))
    matrix = root / "inputs/matrices/study.fs"
    text = matrix.read_text(encoding="ascii").replace(
        "POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES",
        "POL|RUN|SETUP|REF|PPROC|MESH|FLIGHT_CONDITION|SWEEP_VALUES",
    )
    text = text.replace("1001|1|001|001|001|", "1001|1|001|001|001|blade.obj|")
    matrix.write_text(text, encoding="ascii")
    (root / "inputs/mesh").mkdir(exist_ok=True)
    (root / "inputs/geometry/blade.obj").rename(root / "inputs/mesh/blade.obj")
    point = qw.prepared(root)
    qw.adapter().execute(point, check_ram=lambda: None)
    result = qw.adapter().collect(point)
    assert result.classification == "converged"


def test_collected_efficiency_is_the_value_qprop_prints():
    from pathlib import Path

    from cfdbender.solvers.qprop import _output

    text = (Path(__file__).resolve().parents[1] / "data/qprop_stdout_real.txt").read_text()
    row = _output.parse_table(text)
    got = _output.coefficients(row, rho=1.2, radius=0.2)
    assert got["EFFICIENCY"] == 0.6975, "effprop as printed, not J CT / CP"
    assert got["EFFICIENCY_OVERALL"] == 0.0018
    assert got["CT_QPROP"] == 0.07907 and got["CP_QPROP"] == 0.02887
    assert (got["T"], got["Q"], got["P"]) == (158.8, 11.78, 9309.0)
