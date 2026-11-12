"""Profile field keys must remain one key in structured descriptors."""

import importlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("fmt", ["yaml", "toml"])
def test_p0000_review5_descriptor_escapes_field_keys(fmt):
    try:
        render = getattr(importlib.import_module("cfdbender.hpc"), "render_descriptor", None)
    except ModuleNotFoundError:
        render = None
    profile = SimpleNamespace(
        descriptor_format=fmt,
        fields={"job\ninjected: true": "{value}"},
        submit=(),
        path=Path("profile.toml"),
        builds={},
        application_ids={},
        application_id="test",
    )
    text = render(profile, {"value": 'a"\nb'}) if render else ""
    key = json.dumps("job\ninjected: true")
    assert text.startswith(key + (": " if fmt == "yaml" else " = "))
    if fmt == "toml":
        assert tomllib.loads(text) == {"job\ninjected: true": 'a"\nb'}
    else:
        assert json.loads(text[len(key) + 2 :]) == 'a"\nb'
