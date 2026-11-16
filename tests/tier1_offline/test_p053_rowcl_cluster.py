"""Plan two matrix rows with cluster: alpha and cluster: beta in one profile."""

import pytest
from test_p0000_item3f_plan import cli, prepared

from cfdbender.core.registry import RunRegistry
from cfdbender.hpc import plan_submission, resolve_hpc_profile


def matrix(root, values):
    path = root / "inputs/matrices/study.fs"
    path.write_text(
        "POL|RUN|SET|REF|PPROC|VAR_NAMES_VALUES\n"
        + "\n".join(f"{1001 + i}|1|001|001|001|{value}" for i, value in enumerate(values))
        + "\n"
    )


@pytest.mark.parametrize("format_name", ["json", "yaml", "toml", "text"])
def test_p053_rowcl_two_rows_and_record(tmp_path, capsys, format_name):
    profile = prepared(tmp_path, fields={"queue": "{cluster}"})
    profile.write_text(profile.read_text().replace('format = "json"', f'format = "{format_name}"'))
    matrix(tmp_path, ["ClUsTeR: alpha", "cluster: beta"])
    assert cli(tmp_path) == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "beta" in out
    assert cli(tmp_path, "run") == 0
    records = RunRegistry(tmp_path).read()
    assert [r.submission["cluster"] for r in records] == ["alpha", "beta"]
    for r in records:
        assert "cluster" not in (tmp_path / r.submission["config_path"]).read_text().lower()


def test_p053_rowcl_missing_value_refused(tmp_path, capsys):
    prepared(tmp_path, fields={"queue": "{cluster}"})
    matrix(tmp_path, ["-"])
    assert cli(tmp_path) == 2
    err = capsys.readouterr().err
    assert "1001" in err and "cluster: <name>" in err


def test_p053_rowcl_unused_value_warns_and_can_refuse(tmp_path, capsys):
    prepared(tmp_path, fields={"queue": "default"})
    matrix(tmp_path, ["cluster: alpha"])
    assert cli(tmp_path) == 0
    assert "cluster" in capsys.readouterr().err.lower()
    assert cli(tmp_path, "plan", "--warning-refusals") == 2


def test_p053_rowcl_group_and_command(tmp_path):
    prepared(tmp_path, fields={"queue": "{cluster}"}, extra=("{cluster}",))
    profile = resolve_hpc_profile(tmp_path / "inputs", "h001", solver_versions={"mock-1"})
    points = [
        {"row_id": "1001", "point": str(i), "workdir": str(tmp_path), "cluster": "alpha"}
        for i in range(2)
    ]
    job = plan_submission(points, profile)
    assert '"queue": "alpha"' in job.descriptor_text
    assert job.command[-1] == "alpha"
    points[1]["cluster"] = "beta"
    with pytest.raises(ValueError, match="cluster"):
        plan_submission(points, profile)


def test_p053_rowcl_local_ignores_cluster(tmp_path, capsys):
    prepared(tmp_path, fields={"queue": "{cluster}"})
    matrix(tmp_path, ["cluster: alpha"])
    assert cli(tmp_path, "plan", "--local") == 0
    output = capsys.readouterr()
    assert "would submit:" not in output.out
    assert "cluster" not in output.err.lower()


def test_rcl_1_quoted_cluster_is_a_string_value(tmp_path, capsys):
    prepared(tmp_path, fields={"cluster": "{cluster}"})
    matrix(tmp_path, ['CLUSTER: "no: queue"'])
    assert cli(tmp_path) == 0
    out = capsys.readouterr().out
    assert '"cluster": "no: queue"' in out


def test_rcl_2_literal_braces_do_not_require_cluster(tmp_path, capsys):
    prepared(tmp_path, fields={"note": "{{cluster}}"})
    matrix(tmp_path, ["-"])
    assert cli(tmp_path) == 0
    assert '"note": "{cluster}"' in capsys.readouterr().out
    matrix(tmp_path, ["cluster: alpha"])
    assert cli(tmp_path, "plan", "--warning-refusals") == 2
    assert "ignored" in capsys.readouterr().err
