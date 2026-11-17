"""docs/cli-geometry-mesh.md lists every command and option of cfd-geom and cfd-mesh."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "cli-geometry-mesh.md"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "build_geom_mesh_cli_page", ROOT / "tools/build_geom_mesh_cli_page.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docs_cli_geom_mesh_the_page_is_what_the_parsers_give():
    tool = _tool()
    current = PAGE.read_text(encoding="utf-8")
    assert tool.render(current) == current, "run python tools/build_geom_mesh_cli_page.py"


def test_docs_cli_geom_mesh_every_verb_has_a_section():
    tool = _tool()
    text = PAGE.read_text(encoding="utf-8")
    for _, prog, names in tool.sections():
        for name in names:
            assert f"### `{prog} {name}`" in text


def test_docs_cli_geom_mesh_both_programs_are_installed_and_the_page_is_in_the_nav():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert scripts["cfd-geom"] == "cfdbender.geometry.cli:main"
    assert scripts["cfd-mesh"] == "cfdbender.mesh.cli:main"
    assert "cli-geometry-mesh.md" in (ROOT / "properdocs.yml").read_text(encoding="utf-8")


def test_docs_cli_geom_mesh_a_tuple_metavar_is_written_once():
    text = PAGE.read_text(encoding="utf-8")
    assert "`--span-axis X Y Z`" in text
    assert "X Y Z X Y Z" not in text
