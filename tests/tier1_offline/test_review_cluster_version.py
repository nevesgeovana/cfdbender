"""Cluster collection enforces the same solver identity as local execution."""

import pytest
from test_p0000_item3f_plan import cli, prepared

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.collection import collect_workspace


@pytest.mark.parametrize("build", [None, "mock-2"])
def test_cluster_collection_records_solver_version(tmp_path, build):
    from test_p0000_item3f_collect import outputs

    profile = prepared(tmp_path)
    if build:
        matrix = tmp_path / "inputs/matrices/study.fs"
        matrix.write_text("POL|RUN|SETUP|REF|PPROC|SOLVER_BUILD\n1001|1|001|001|001|mock-2\n")
        profile.write_text(profile.read_text() + '\n"mock-2" = "offline-build-2"\n')
    assert cli(tmp_path, "run") == 0
    outputs(tmp_path, RunRegistry(tmp_path).read()[0].point)
    collect_workspace(tmp_path, "study")
    assert RunRegistry(tmp_path).read()[0].solver_version == "mock-1"
    record = RunRegistry(tmp_path).read()[0]
    assert record.classification == ("failed" if build else "converged")
    if build:
        assert "SOLVER_BUILD" in record.error
