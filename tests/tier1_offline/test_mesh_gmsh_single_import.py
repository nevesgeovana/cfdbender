"""gmsh is imported from one module of the package, and its licence is stated in the docs."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "cfdbender"
HOME = SOURCE / "mesh" / "_gmsh.py"


def _gmsh_imports(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "gmsh" for a in node.names):
            lines.append(node.lineno)
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "gmsh":
            lines.append(node.lineno)
    return lines


def test_mesh_gmsh_single_import_only_the_kernel_module_imports_gmsh():
    assert HOME.is_file(), "the one gmsh module is cfdbender/mesh/_gmsh.py"
    assert _gmsh_imports(HOME), "the kernel module imports gmsh"
    others = {
        path.relative_to(SOURCE).as_posix(): lines
        for path in SOURCE.rglob("*.py")
        if path != HOME and (lines := _gmsh_imports(path))
    }
    assert others == {}


def test_mesh_gmsh_single_import_gmsh_is_a_base_dependency_with_its_licence_stated():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(item.startswith("gmsh") for item in project["dependencies"])
    install = (ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    assert "GNU General Public License" in install
    assert "cfdbender/mesh/_gmsh.py" in install
