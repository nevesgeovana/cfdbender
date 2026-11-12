"""P0000-CIFIX2: CI installs the generator version used for the reference mesh."""

import tomllib
from pathlib import Path


def test_p0000_cifix2_dev_pins_mesh_generator():
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    dependencies = tomllib.loads(project.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]["dev"]
    assert "gmsh==4.15.2" in dependencies
