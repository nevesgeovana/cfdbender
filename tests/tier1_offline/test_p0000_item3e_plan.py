"""P0000-ITEM3E-PLAN: descriptor-only planning and grouped walltime rules.

Executed usage example for integration:
profile = read_hpc_profile(example, solver_versions={"8.6.0"})
submission = plan_submission([point], profile)
print(submission.descriptor_text, submission.command)
"""

import importlib
import socket
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from cfdbender.hpc import HpcProfile


def plan(points, profile=None, **kwargs):
    module = importlib.import_module("cfdbender.hpc")
    function = getattr(module, "plan_submission", lambda *a, **k: None)
    return function(points, profile or fixture_profile(), **kwargs)


def fixture_profile(**kwargs):
    profile = HpcProfile(
        "SU2",
        "yaml",
        "submit.yaml",
        {
            "ApplicationId": "{application_id}",
            "job_name": "{job_name}",
            "walltime": "{walltime}",
            "ncpus": "{ncpus}",
            "version": "{solver_version_alias}",
            "working_directory": "{workdir}",
            "config_file": "{config_path}",
            "solver_type": "{solver_type}",
            "cluster": "testcluster",
        },
        ("submit-tool", "{descriptor_path}"),
        Path("h000.toml"),
        builds={"8.6.0": "8.6"},
        max_walltime_s=7200,
    )
    return replace(profile, **kwargs)


def point(row="1001", name="p1", **kwargs):
    data = dict(
        row_id=row,
        point=name,
        solver_version="8.6.0",
        ncpus=8,
        workdir="/study/sims/sim_1001/datapoints/DP-p1",
        config_path="/study/config.cfg",
        solver_type="RANS",
        walltime="1h",
    )
    data.update(kwargs)
    return data


def test_p0000_item3e_single_point_plan_has_text_path_command_and_clock(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("planning attempted a side effect")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    result = plan([point()])
    assert getattr(result, "walltime_s", None) == 3600
    assert result.walltime == "1h"
    assert result.descriptor_path.as_posix() == "/study/sims/sim_1001/datapoints/DP-p1/submit.yaml"
    assert result.command == ["submit-tool", result.descriptor_path.as_posix()]
    assert result.descriptor_text.startswith('ApplicationId: "SU2"\njob_name: "1001_p1"\n')
    assert 'ncpus: 8\nversion: "8.6"\n' in result.descriptor_text
    assert result.warnings == ()


def test_p0000_item3e_group_sums_cells_and_caps_with_warning():
    result = plan(
        [point(), point(name="p2"), point("1002")], workdir="/study/batch/job", job_name="batch_1"
    )
    assert getattr(result, "walltime_s", None) == 7200
    assert result.walltime == "120m"
    assert any("larger --batch n" in warning for warning in result.warnings)
    assert result.descriptor_path.as_posix() == "/study/batch/job/submit.yaml"
    assert 'job_name: "batch_1"' in result.descriptor_text


def test_p0000_item3e_group_sum_is_exact_without_rounding():
    result = plan([point(walltime="31s"), point(name="p2", walltime="31s")])
    assert getattr(result, "walltime_s", None) == 62
    assert result.walltime == "62s"


def test_p0000_item3e_cell_above_maximum_refuses_row():
    with pytest.raises(ValueError, match="WALLTIME 3h of row 1001.*max_walltime"):
        plan([point(walltime="3h")])


@pytest.mark.parametrize(("walltime", "expected"), [("BEST", 180), ("best", 180)])
def test_p0000_item3e_best_rounds_factor_plus_margin_to_minute(walltime, expected):
    result = plan([point(walltime=walltime, estimate_s=100, margin_s=20)], grouped=True)
    assert getattr(result, "walltime_s", None) == expected


def test_p0000_item3e_best_is_grouped_only():
    with pytest.raises(ValueError, match="1001.*BEST.*grouped"):
        plan([point(walltime="BEST", estimate_s=100)])


def test_p0000_item3e_mixed_best_and_cells_prices_each_row_once():
    result = plan(
        [
            point(walltime="2h"),
            point(name="p2", walltime="2h"),
            point("1002", walltime="BEST", estimate_s=3600, margin_s=1200),
            point("1002", name="p2", walltime="BEST", estimate_s=3600, margin_s=1200),
        ],
        fixture_profile(max_walltime_s=None),
    )
    assert getattr(result, "walltime_s", None) == 24600


@pytest.mark.parametrize("maximum", [7200, None])
def test_p0000_item3e_unknown_best_names_point_and_uses_maximum_or_refuses(maximum):
    if maximum is None:
        with pytest.raises(ValueError, match="1001/p1.*no max_walltime"):
            plan([point(walltime="BEST")], fixture_profile(max_walltime_s=None), grouped=True)
    else:
        result = plan([point(walltime="BEST")], grouped=True)
        assert getattr(result, "walltime_s", None) == maximum
        assert "1001/p1" in result.warnings[0]


@pytest.mark.parametrize("maximum", [7200, None])
def test_p0000_item3e_unstated_row_does_not_add_zero(maximum):
    rows = [point(), point("1002", walltime=None)]
    if maximum is None:
        with pytest.raises(ValueError, match="1002.*no WALLTIME"):
            plan(rows, fixture_profile(max_walltime_s=None))
    else:
        result = plan(rows)
        assert getattr(result, "walltime_s", None) == maximum
        assert any("1002" in warning and "max_walltime" in warning for warning in result.warnings)


def test_p0000_item3e_local_group_uses_largest_timeout_per_point():
    profile = fixture_profile(fields={"job": "{job_name}"}, max_walltime_s=None)
    result = plan(
        [point(walltime=None, timeout_s=30), point("1002", walltime=None, timeout_s=90)],
        profile,
        local=True,
    )
    assert getattr(result, "local_time_limit_s", None) == 180
    assert result.walltime_s is None


def test_p0000_item3e_local_group_falls_back_to_maximum():
    result = plan([point(walltime=None)], grouped=True, local=True)
    assert getattr(result, "local_time_limit_s", None) == 7200
    assert result.walltime_s == 7200
    assert any("no WALLTIME and no timeout_s" in warning for warning in result.warnings)


def test_p0000_item3e_unbounded_local_group_refuses_before_render():
    with pytest.raises(ValueError, match="WALLTIME.*timeout_s.*max_walltime"):
        plan([point(walltime=None)], fixture_profile(max_walltime_s=None), grouped=True, local=True)


@pytest.mark.parametrize("cell", ["240", "nanm", "infh", "0s", "-3h", "2hrs"])
def test_p0000_item3e_bad_row_clock_names_row_and_walltime(cell):
    with pytest.raises(ValueError, match="1001.*WALLTIME"):
        plan([point(walltime=cell)])


@pytest.mark.parametrize(("key", "value"), [("ncpus", 16), ("solver_version", "8.3.0")])
def test_p0000_item3e_group_never_mixes_resources_or_versions(key, value):
    with pytest.raises(ValueError, match=key):
        plan([point(), point("1002", **{key: value})])


def test_p0000_item3e_empty_group_refuses():
    with pytest.raises(ValueError, match="no points"):
        plan([])


def test_p0000_item3e_seconds_arithmetic_and_application_override():
    result = plan(
        [point()],
        fixture_profile(walltime_arithmetic="seconds", application_ids={"8.6.0": "SU2-test"}),
    )
    assert getattr(result, "walltime", None) == 3600
    assert 'ApplicationId: "SU2-test"' in result.descriptor_text
    assert "walltime: 3600" in result.descriptor_text


def test_p0000_item3e_documented_su2_fixture_runs_offline():
    module = importlib.import_module("cfdbender.hpc")
    example = Path(__file__).with_name("test_p0000_item3e_su2_profile.toml")
    profile = module.read_hpc_profile(example, solver_versions={"8.6.0"})
    result = plan([point()], profile)
    assert result is not None
    assert getattr(result, "command", None) == ["submit-tool", result.descriptor_path.as_posix()]
    for key in (
        "ApplicationId",
        "job_name",
        "walltime",
        "ncpus",
        "version",
        "working_directory",
        "config_file",
        "solver_type",
        "cluster",
    ):
        assert key + ": " in result.descriptor_text
