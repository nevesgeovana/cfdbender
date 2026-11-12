"""P0000-ITEM3E-RENDER: profile-owned descriptor fields and submit argv."""

import importlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


def call(name, *args, **kwargs):
    module = importlib.import_module("cfdbender.hpc")
    return getattr(module, name, lambda *a, **k: None)(*args, **kwargs)


def profile(fmt="yaml", **updates):
    data = dict(
        application_id="SU2",
        descriptor_format=fmt,
        descriptor_name="submit.yaml",
        fields={
            "job": "{job_name}",
            "ncpus": "{ncpus}",
            "version": "{solver_version}",
            "enabled": True,
            "ratio": "{ratio}",
            "config": "{config_path}",
        },
        submit=("submit-tool", "--cpus={ncpus}", "{descriptor_path}"),
        path=Path("h001.toml"),
        builds={"8.6.0": "8.6"},
        application_ids={"8.6.0": "SU2-test"},
    )
    data.update(updates)
    return SimpleNamespace(**data)


VALUES = dict(
    job_name='job "one"',
    ncpus=8,
    solver_version="8.6.0",
    ratio=0.5,
    config_path="/case with spaces/config.cfg",
)


@pytest.mark.parametrize("fmt", ["yaml", "json", "toml", "text"])
def test_p0000_item3e_descriptor_scalar_types_and_escaping(fmt):
    text = call("render_descriptor", profile(fmt), VALUES)
    assert text is not None
    if fmt == "json":
        data = json.loads(text)
    elif fmt == "toml":
        data = tomllib.loads(text)
    elif fmt == "yaml":
        data = {
            line.split(": ", 1)[0]: json.loads(line.split(": ", 1)[1]) for line in text.splitlines()
        }
    else:
        assert 'job: job "one"\n' in text
        return
    assert data == dict(
        job='job "one"',
        ncpus=8,
        version="8.6.0",
        enabled=True,
        ratio=0.5,
        config="/case with spaces/config.cfg",
    )


def test_p0000_item3e_submit_argv_is_profile_owned_with_path_spaces():
    command = call("build_submit_command", profile(), VALUES, Path("/job with spaces/submit.yaml"))
    assert command == ["submit-tool", "--cpus=8", "/job with spaces/submit.yaml"]


@pytest.mark.parametrize("template", ["{queue}", "{ncpus:{queue}}", "{ncpus.missing}", "{broken"])
def test_p0000_item3e_submit_unknown_or_invalid_placeholder_names_problem(template):
    with pytest.raises(ValueError, match="queue|missing|brace|broken"):
        call(
            "build_submit_command",
            profile(submit=("submit-tool", template)),
            VALUES,
            Path("s.yaml"),
        )


@pytest.mark.parametrize("name", ["walltime", "config_path", "job_name"])
@pytest.mark.parametrize("missing", [None, ""])
def test_p0000_item3e_descriptor_missing_values_are_refused_by_name(name, missing):
    with pytest.raises(ValueError, match=name):
        call("render_descriptor", profile(fields={"field": "{" + name + "}"}), {name: missing})


def test_p0000_item3e_unmapped_solver_version_names_version_and_builds():
    with pytest.raises(ValueError, match="builds.*8.3.0"):
        call(
            "render_descriptor",
            profile(fields={"version": "{solver_version_alias}"}),
            {"solver_version": "8.3.0"},
        )


def test_p0000_item3e_translation_tables_are_independent():
    text = call(
        "render_descriptor",
        profile(fields={"app": "{application_id}", "version": "{solver_version_alias}"}),
        {"solver_version": "8.6.0"},
    )
    assert text == 'app: "SU2-test"\nversion: "8.6"\n'
    with pytest.raises(ValueError, match="builds.*8.6.0"):
        call(
            "build_submit_command",
            profile(builds={}, submit=("submit-tool", "{solver_version_alias}")),
            {"solver_version": "8.6.0"},
            Path("s.yaml"),
        )


def test_p0000_item3e_descriptor_unknown_name_and_bad_format():
    with pytest.raises(ValueError, match="queue"):
        call("render_descriptor", profile(fields={"field": "{queue}"}), VALUES)
    with pytest.raises(ValueError, match="xml"):
        call("render_descriptor", profile("xml"), VALUES)
