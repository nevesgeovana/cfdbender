"""P0600: find the runtime licence table from the documentation navigation."""

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_p0600_dependency_licences_are_discoverable():
    pages = [p for p in (ROOT / "docs").glob("*.md") if re.search("depend|licen", p.stem)]
    assert pages, "a dependency licence page must be discoverable by filename"
    nav = (ROOT / "properdocs.yml").read_text(encoding="utf-8")
    text = "\n".join(p.read_text(encoding="utf-8") for p in pages).lower()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for requirement in project["dependencies"]:
        name = re.split(r"[<>=!~]", requirement)[0].lower()
        assert re.search(rf"\| {name} \| [^|]+\|", text), name
    assert "gpl-2.0-or-later" in text and "linking exception" in text
    assert any(p.name in nav for p in pages)


def test_p0600_numeric_dependencies_and_single_gmsh_import_boundary():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {re.split(r"[<>=!~]", item)[0] for item in project["dependencies"]}
    assert {"numpy", "scipy", "trimesh", "gmsh"} <= names
    sites = set()
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Import) and any(a.name == "gmsh" for a in node.names)) or (
                isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "gmsh"
            ):
                sites.add(path.relative_to(ROOT).as_posix())
    assert sites == {"src/cfdbender/mesh/_gmsh.py"}
