"""P0000-ITEM3E-PROFILES: portable HPC profile requirements.

Usage: read_hpc_profile(path, solver_versions={"8.6.0"}).
The test-only no-op declaration lets the new feature fail on values on the
0.0.0 base, rather than fail collection on an absent module or public name.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def call(name, *args, **kwargs):
    try:
        module = importlib.import_module("cfdbender.hpc")
    except ModuleNotFoundError as error:
        if error.name != "cfdbender.hpc":
            raise
        return None
    return getattr(module, name)(*args, **kwargs)


PROFILE = """application_id = "SU2"
max_walltime = "48:00:01"
[descriptor]
format = "yaml"
name = "submit.yaml"
[descriptor.fields]
application = "{application_id}"
version = "{solver_version_alias}"
[submit]
command = ["submit-tool", "{descriptor_path}"]
[builds]
"8.6.0" = "8.6"
[application_ids]
"8.6.0" = "SU2-test"
"""


def write(tmp_path, text=PROFILE, name="h001"):
    path = tmp_path / "hpc" / f"{name}.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def read(path):
    return call("read_hpc_profile", path, solver_versions={"8.6.0", "8.3.0"})


def test_p0000_item3e_profile_values_and_translation(tmp_path):
    profile = read(write(tmp_path))
    assert getattr(profile, "max_walltime_s", None) == 172801.0
    assert profile.builds == {"8.6.0": "8.6"}
    assert profile.application_id_for("8.6.0") == "SU2-test"
    assert profile.application_id_for("8.3.0") == "SU2"
    assert profile.submit == ("submit-tool", "{descriptor_path}")


@pytest.mark.parametrize("clock", ['"0:00:00"', '"48h"', '"01:60:00"', "123", '"-1:00:00"'])
def test_p0000_item3e_invalid_maximum_names_key(tmp_path, clock):
    with pytest.raises(ValueError, match="max_walltime"):
        read(write(tmp_path, PROFILE.replace('"48:00:01"', clock)))


def test_p0000_item3e_defaults_warn_and_are_ignored(tmp_path):
    with pytest.warns(UserWarning, match="defaults.*ignored"):
        profile = read(write(tmp_path, PROFILE + "\n[defaults]\nncpus = 99\n"))
    assert "ncpus" not in profile.fields


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('application_id = "SU2"', 'application_id = ""', "application_id"),
        ('format = "yaml"', 'format = "xml"', "xml"),
        ('name = "submit.yaml"', 'name = "../other.yaml"', "plain file name"),
        ('name = "submit.yaml"', 'name = "SUBMIT.yaml."', "plain file name"),
        ('name = "submit.yaml"', 'name = "NUL"', "plain file name"),
        (
            'command = ["submit-tool", "{descriptor_path}"]',
            'command = "submit-tool"',
            "submit command",
        ),
        ('"8.6.0" = "8.6"', '"8.6" = "8.6"', "builds.*ONE registered solver version"),
        ('"8.6.0" = "SU2-test"', '"8.6" = "SU2-test"', "application_ids.*ONE registered"),
        ('"8.6.0" = "8.6"', '"8.6.0" = ""', "cannot be empty"),
        ('max_walltime = "48:00:01"', "misspelled = true", "misspelled"),
        ('format = "yaml"', 'max_walltime = "48:00:01"', "top level"),
        ("[submit]", "[submit]\nmisspelled = true", "misspelled"),
    ],
)
def test_p0000_item3e_profile_refusals(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        read(write(tmp_path, PROFILE.replace(old, new)))


@pytest.mark.parametrize("table", ["descriptor", "submit"])
@pytest.mark.parametrize("value", ['""', "false", "[]", "0"])
def test_p0000_item3e_values_cannot_stand_for_tables(tmp_path, table, value):
    text = f'{table} = {value}\napplication_id = "SU2"\n'
    if table != "descriptor":
        text += '[descriptor.fields]\njob = "{job_name}"\n'
    if table != "submit":
        text += '[submit]\ncommand = ["submit-tool", "{descriptor_path}"]\n'
    with pytest.raises(ValueError, match=f"{table} entry that is not a table"):
        read(write(tmp_path, text))


def test_p0000_item3e_profile_selection(tmp_path):
    assert call("resolve_hpc_profile", tmp_path) is None
    first = write(tmp_path)
    selected = call("resolve_hpc_profile", tmp_path, solver_versions={"8.6.0"})
    assert getattr(selected, "path", None) == first
    second = write(tmp_path, name="h002")
    with pytest.raises(ValueError, match="2 profiles"):
        call("resolve_hpc_profile", tmp_path)
    chosen = call("resolve_hpc_profile", tmp_path, "h002.toml", solver_versions={"8.6.0"})
    assert chosen.path == second
    with pytest.raises(ValueError, match="missing.*available: h001, h002"):
        call("resolve_hpc_profile", tmp_path, "missing")


def test_p0000_item3e_invalid_toml_names_file(tmp_path):
    path = write(tmp_path, "[broken")
    with pytest.raises(ValueError, match="h001.toml"):
        read(path)


def test_p0000_item3e_missing_file_names_file(tmp_path):
    with pytest.raises(ValueError, match="absent.toml"):
        read(Path(tmp_path) / "absent.toml")
