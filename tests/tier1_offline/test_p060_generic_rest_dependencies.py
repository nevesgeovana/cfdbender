"""A normal installation includes the spatial index used by surface remeshing."""

import re
import tomllib
from pathlib import Path


def test_p060_generic_rest_projection_index_is_a_runtime_dependency():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    required = {re.split(r"[<>=!~ ;\[]", item, maxsplit=1)[0] for item in project["dependencies"]}
    assert "rtree" in required, "trimesh surface projection needs its spatial index at runtime"
