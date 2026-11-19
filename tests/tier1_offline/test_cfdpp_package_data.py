"""The wheel carries the data files the adapters read as package resources."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOLVERS = ROOT / "src" / "cfdbender" / "solvers"


def _package_data() -> dict[str, list[str]]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    return project["tool"]["setuptools"]["package-data"]


def test_cfdpp_database_files_are_package_data() -> None:
    data = SOLVERS / "cfdpp" / "_schema" / "data"
    assert list(data.glob("*.toml")), "the database folder holds no TOML file"
    assert "data/*.toml" in _package_data()["cfdbender.solvers.cfdpp._schema"]
    assert not list(data.glob("*.yaml")), "a YAML file would not ship and needs a dependency"


def test_every_solver_setup_templates_folder_is_package_data() -> None:
    folders = sorted(SOLVERS.glob("*/setup_templates"))
    assert folders, "no solver ships setup templates"
    assert "setup_templates/*.toml" in _package_data()["*"]
    for folder in folders:
        assert (folder.parent / "__init__.py").is_file(), f"{folder.parent.name} is not a package"
        assert list(folder.glob("*.toml")), f"{folder} holds no template"
