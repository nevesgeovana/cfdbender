"""A row's hpc.<name> entries replace one value of the selected HPC profile."""

import json

import pytest
from test_p0000_item3f_plan import cli, prepared
from test_p053_rowcl_cluster import matrix

from cfdbender.core.registry import RunRegistry

FIELDS = {
    "application": "{application_id}",
    "mode": "solver",
    "queue": "{cluster}",
    "name": "{job_name}",
    "cpus": "{ncpus}",
}


def descriptors(capsys):
    """Return the JSON descriptors a plan printed, in point order."""
    text = capsys.readouterr().out
    found, decoder, index = [], json.JSONDecoder(), 0
    while (index := text.find("{", index)) != -1:
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            index += 1
            continue
        found.append(value)
        index = end
    return found


def setup(tmp_path, cells, fields=None):
    prepared(tmp_path, fields=fields or FIELDS)
    matrix(tmp_path, cells)


def test_submit_override_replaces_exactly_one_value(tmp_path, capsys):
    setup(tmp_path, ["cluster: one", "hpc: h001 / cluster: one / hpc.mode: custom"])
    assert cli(tmp_path) == 0
    plain, changed = descriptors(capsys)
    assert plain["mode"] == "solver" and changed["mode"] == "custom"
    assert {k: v for k, v in changed.items() if k != "mode" and k != "name"} == {
        k: v for k, v in plain.items() if k != "mode" and k != "name"
    }


def test_submit_override_cluster_and_application_id_substitutions(tmp_path, capsys):
    setup(tmp_path, ["cluster: one / hpc.cluster: site-b / hpc.application_id: SU2_CFD"])
    assert cli(tmp_path) == 0
    (job,) = descriptors(capsys)
    assert job["queue"] == "site-b" and job["application"] == "SU2_CFD"


def test_submit_override_unknown_name_refused_with_accepted_list(tmp_path, capsys):
    setup(tmp_path, ["cluster: one / hpc.priority: high"])
    assert cli(tmp_path) == 2
    error = capsys.readouterr().err
    assert "hpc.priority is not declared by the HPC profile h001.toml" in error
    assert "it accepts application, application_id, cluster, cpus, mode, name, queue" in error


@pytest.mark.parametrize("name,column", [("ncpus", "NCPUS"), ("walltime", "WALLTIME")])
def test_submit_override_of_a_column_names_the_column(tmp_path, capsys, name, column):
    setup(tmp_path, [f"cluster: one / hpc.{name}: 4"])
    assert cli(tmp_path) == 2
    assert f"is the {column} column" in capsys.readouterr().err


def test_submit_override_without_a_profile_refused(tmp_path, capsys):
    setup(tmp_path, ["hpc.mode: custom"])
    (tmp_path / "inputs/hpc/h001.toml").rename(tmp_path / "inputs/h001.bak")
    assert cli(tmp_path) == 2
    assert "no profile is selected" in capsys.readouterr().err


def test_submit_override_with_local_warns_once_and_ignores(tmp_path, capsys):
    setup(tmp_path, ["hpc.mode: custom", "hpc.mode: other"])
    assert cli(tmp_path, "plan", "--local") == 0
    seen = capsys.readouterr()
    assert seen.err.count("--local ignores the hpc.<name> entries of row(s) 1001, 1002") == 1
    assert '"mode"' not in seen.out


def test_submit_override_script_application_checked(tmp_path, capsys):
    setup(tmp_path, ['cluster: one / hpc.application_id: "tools/missing.py"'])
    assert cli(tmp_path) == 2
    assert "is not a file of the workspace" in capsys.readouterr().err
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/compute_polars.py").write_text("print(1)\n")
    matrix(tmp_path, ['cluster: one / hpc.application_id: "tools/compute_polars.py"'])
    assert cli(tmp_path) == 0
    (job,) = descriptors(capsys)
    assert job["application"] == (tmp_path / "tools/compute_polars.py").resolve().as_posix()
    matrix(tmp_path, ['cluster: one / hpc.application_id: "../outside.sh"'])
    assert cli(tmp_path) == 2


def test_submit_override_is_recorded_with_its_origin(tmp_path, capsys):
    setup(tmp_path, ["cluster: one / hpc.mode: custom", "cluster: one"])
    assert cli(tmp_path) == 0
    receipt = json.loads((tmp_path / "post/study/plan.json").read_text())
    assert receipt["points"][0]["hpc_overrides"] == {
        "origin": "matrix row",
        "values": {"mode": "custom"},
    }
    assert "hpc_overrides" not in receipt["points"][1]
    assert cli(tmp_path, "run") == 0
    records = RunRegistry(tmp_path).read()
    assert records[0].submission["hpc_overrides"]["origin"] == "matrix row"
    assert records[0].submission["hpc_overrides"]["values"] == {"mode": "custom"}
    assert "hpc_overrides" not in records[1].submission
    assert cli(tmp_path, "collect") != 2


def test_submit_override_never_reaches_the_solver_setup(tmp_path, capsys):
    setup(tmp_path, ["cluster: one / hpc.mode: custom"])
    assert cli(tmp_path, "run") == 0
    (record,) = RunRegistry(tmp_path).read()
    text = (tmp_path / record.submission["config_path"]).read_text().lower()
    assert "custom" not in text and "hpc" not in text


def test_submit_without_overrides_plan_is_unchanged(tmp_path, capsys):
    setup(tmp_path, ["cluster: one", "hpc: h001 / cluster: two"])
    assert cli(tmp_path) == 0
    first, second = descriptors(capsys)
    receipt = json.loads((tmp_path / "post/study/plan.json").read_text())
    assert all("hpc_overrides" not in entry for entry in receipt["points"])
    assert first["application"] == "offline" and first["mode"] == "solver"
    assert second["queue"] == "two" and first["queue"] == "one"
