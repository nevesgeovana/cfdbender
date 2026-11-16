"""P060: an interrupted local point can continue from its saved solution safely."""

from dataclasses import replace

import pytest
from test_p060_gaps_restart import call, prepared, solution

from cfdbender.core.registry import RunRegistry


@pytest.mark.parametrize("solver", ["su2", "openfoam", "pyfr"])
def test_p060_restart_continues_an_interrupted_local_point_when_host_is_free(tmp_path, solver):
    source, argv = prepared(tmp_path, solver)
    solution(tmp_path, source, solver)
    registry = RunRegistry(tmp_path)
    interrupted = replace(
        source,
        status="RUNNING",
        completed_at=None,
        classification=None,
        error=None,
        outputs_sha256={},
    )
    registry.replace_records({source.point_id: interrupted}, archive_first=False)
    assert call([*argv, "--restart"]) == 0
    assert registry.read()[0].status == "COMPLETED"
    assert getattr(registry.read()[0].point, "restart_from", None) == source.point_id


def test_p060_interrupted_restart_refuses_busy_host_before_archiving(tmp_path, monkeypatch):
    from cfdbender.matrix import cli

    source, argv = prepared(tmp_path, "pyfr")
    solution(tmp_path, source, "pyfr")
    registry = RunRegistry(tmp_path)
    interrupted = replace(
        source,
        status="RUNNING",
        completed_at=None,
        classification=None,
        error=None,
        outputs_sha256={},
    )
    registry.replace_records({source.point_id: interrupted}, archive_first=False)
    before = registry.path.read_bytes()

    def busy(*args):
        raise ValueError("PyFR is already running")

    monkeypatch.setattr(cli, "judge_environment", busy)
    assert call([*argv, "--restart"]) == 2
    assert registry.path.read_bytes() == before
    assert not registry.history()
