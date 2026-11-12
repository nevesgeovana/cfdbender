"""P0000: cfd-matrix plan study --hpc h001 prints a job without writing it."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.matrix.cli import main

FAKE = Path(__file__).resolve().parents[1] / "fixtures/item3f_fake_submit.py"


def cli(root, command="plan", *options):
    try:
        return main([command, "study", "--workspace", str(root), *options])
    except SystemExit as error:
        return error.code


def tree(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        if p.is_file()
        else "directory"
        for p in root.rglob("*")
    }


def profile(root, name="h001", *, extra=(), fields=None):
    path = root / f"inputs/hpc/{name}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [Path(sys.executable).as_posix(), FAKE.as_posix(), "{descriptor_path}", *extra]
    values = fields or {
        "application": "{application_id}",
        "name": "{job_name}",
        "working_directory": "{workdir}",
        "config": "{config_path}",
        "version": "{solver_version_alias}",
        "cpus": "{ncpus}",
    }
    path.write_text(
        chr(10).join(
            [
                'application_id = "offline"',
                "[descriptor]",
                'format = "json"',
                'name = "submit.json"',
                "[descriptor.fields]",
                *(f"{key} = {json.dumps(value)}" for key, value in values.items()),
                "[submit]",
                f"command = {json.dumps(command)}",
                "[builds]",
                '"mock-1" = "offline-build"',
            ]
        ),
        encoding="utf-8",
    )
    return path


def prepared(root, **kwargs):
    workspace(root)
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(
        'solver_version = "mock-1"' + chr(10) + "ncpus = 2" + chr(10) + setup.read_text(),
        encoding="utf-8",
    )
    return profile(root, **kwargs)


@pytest.mark.parametrize("command", ["plan", "run", "collect"])
def test_p0000_item3f_help_has_hpc_and_local(command, capsys):
    with pytest.raises(SystemExit) as caught:
        main([command, "--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "--hpc NAME" in output
    assert "--local" in output


@pytest.mark.parametrize("options", [[], ["--hpc", "h001"], ["--hpc", "h001.toml"]])
def test_p0000_item3f_plan_descriptor_and_argv_without_writes(tmp_path, capsys, options):
    prepared(tmp_path)
    before = tree(tmp_path)
    assert cli(tmp_path, "plan", *options) == 0
    output = capsys.readouterr().out
    assert '"application": "offline"' in output
    assert '"version": "offline-build"' in output
    assert '"cpus": 2' in output
    assert "submit.json" in output and "mock.in" in output
    assert FAKE.as_posix() in output
    assert "would submit:" in output
    assert before == tree(tmp_path), "plan must not write even an empty folder"


@pytest.mark.parametrize("command", ["plan", "run", "collect"])
@pytest.mark.parametrize("options", [[], ["--hpc", "absent"]])
def test_p0000_item3f_selector_refuses_naming_available(tmp_path, command, options, capsys):
    prepared(tmp_path)
    profile(tmp_path, "h002")
    before = tree(tmp_path)
    assert cli(tmp_path, command, *options) == 2
    error = capsys.readouterr().err
    assert "h001" in error and "h002" in error
    assert before == tree(tmp_path)


def test_p0000_item3f_selection_is_per_call(tmp_path, capsys):
    prepared(tmp_path)
    profile(tmp_path, "h002", fields={"selected": "second"})
    assert cli(tmp_path, "plan", "--hpc", "h002") == 0
    assert '"selected": "second"' in capsys.readouterr().out
    assert cli(tmp_path, "plan") == 2
    assert "h001" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["plan", "run", "collect"])
def test_p0000_item3f_bad_profile_refuses_before_writes(tmp_path, command, capsys):
    path = prepared(tmp_path)
    path.write_text('application_id = "offline"' + chr(10) + "[descriptor.fields]" + chr(10))
    before = tree(tmp_path)
    assert cli(tmp_path, command, "--hpc", "h001") == 2
    assert "HPC profile" in capsys.readouterr().err
    assert before == tree(tmp_path)


def test_p0000_item3f_local_plan_keeps_adapter_plan(tmp_path, capsys):
    prepared(tmp_path)
    before = tree(tmp_path)
    assert cli(tmp_path, "plan", "--local") == 0
    output = capsys.readouterr().out
    assert "would submit:" not in output
    assert "would: write mock ASCII input" in output
    assert before == tree(tmp_path)


def test_p0000_item3f_missing_value_refuses_instead_of_inheriting(tmp_path, capsys):
    prepared(tmp_path)
    matrix = tmp_path / "inputs/matrices/study.fs"
    matrix.write_text(
        "POL|RUN|SET|REF|PPROC" + chr(10) + "1001|1|001|001|001" + chr(10) + "1002|1|002|001|001"
    )
    (tmp_path / "inputs/setups/s002.toml").write_text(
        'solver_version = "mock-1"'
        + chr(10)
        + "[mock]"
        + chr(10)
        + 'outcome = "converged"'
        + chr(10)
        + "iterations = 4"
    )
    before = tree(tmp_path)
    assert cli(tmp_path, "plan", "--hpc", "h001") == 2
    assert "ncpus" in capsys.readouterr().err
    assert before == tree(tmp_path)
