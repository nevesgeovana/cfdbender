"""The offline wheel includes the geometry schema declared as package data."""

import importlib.util
from pathlib import Path

import pytest


def test_kit_stages_geometry_schema_before_build(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("kit_schema_builder", root / "tools/build_kit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class BuildReachedError(Exception):
        pass

    def inspect_build(argv, **kwargs):
        source = Path(argv[-1])
        relative = "src/cfdbender/geometry/schemas/component.schema.json"
        assert (source / relative).is_file(), "offline wheel loses its geometry schema"
        assert (source / relative).read_bytes() == (root / relative).read_bytes()
        raise BuildReachedError

    monkeypatch.setattr(module, "run", inspect_build)
    with pytest.raises(BuildReachedError):
        module.build(tmp_path)
