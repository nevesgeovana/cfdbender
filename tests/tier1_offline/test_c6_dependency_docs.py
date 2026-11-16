"""Runtime dependency documentation covers every direct dependency and geometry imports."""

import re
import tomllib
from pathlib import Path


def test_every_runtime_dependency_has_a_licence_row():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    text = (root / "docs/dependency-licences.md").read_text(encoding="utf-8").lower()
    for requirement in project["project"]["dependencies"]:
        name = re.split(r"[<>=!~]", requirement)[0].lower()
        assert f"| {name} |" in text, name


def test_architecture_describes_geometry_dependencies():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/architecture.md").read_text(encoding="utf-8")
    assert "Blade slicing on the standard library alone" not in text
    assert "numpy, scipy and pydantic" in text
    assert "cfdbender.mesh._gmsh" in text
