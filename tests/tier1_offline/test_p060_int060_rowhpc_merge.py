"""P060: per-row HPC choices retain shared refusal and job-end-file behavior."""

import json

import pytest
from test_p0000_item3f_plan import cli, prepared, profile
from test_p053_rowcl_cluster import matrix

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main


@pytest.mark.parametrize("source", ["global", "workspace"])
def test_p060_rowhpc_unused_cluster_obeys_shared_refusals(tmp_path, capsys, source):
    prepared(tmp_path, fields={"queue": "default"})
    matrix(tmp_path, ["hpc: h001 / cluster: unused"])
    options = []
    if source == "global":
        options = ["--warning-refusals"]
    else:
        init = tmp_path / "init.toml"
        init.write_text("warning_refusals = true\n" + init.read_text())
    assert main([*options, "plan", "study", "--workspace", str(tmp_path)]) == 2
    assert "cluster value ignored" in capsys.readouterr().err
    assert not (tmp_path / "post/study/plan.json").exists()


def test_p060_rowhpc_profiles_keep_job_end_files_and_registry_migration(tmp_path):
    first = prepared(tmp_path, fields={"queue": "{cluster}"})
    second = profile(tmp_path, "beta", fields={"queue": "{cluster}"})
    for path, marker in ((first, "alpha.done"), (second, "beta.done")):
        path.write_text(f'job_end_files = ["{marker}"]\n' + path.read_text())
    matrix(tmp_path, ["hpc: h001 / cluster: one", "hpc: beta / cluster: two"])
    assert cli(tmp_path) == 0
    assert cli(tmp_path, "run") == 0
    registry = RunRegistry(tmp_path)
    records = registry.read()
    for record, marker, selected, cluster in zip(
        records, ("alpha.done", "beta.done"), ("h001", "beta"), ("one", "two"), strict=True
    ):
        assert record.submission["job_end_files"] == (f"{record.point.directory}/{marker}",)
        assert record.submission["profile"] == f"inputs/hpc/{selected}.toml"
        assert record.submission["cluster"] == cluster
    document = json.loads(registry.path.read_text())
    assert document["schema_version"] == 2
