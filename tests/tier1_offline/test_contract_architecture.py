"""Adapter contract, import direction and mesh-folder clauses, by static analysis.

The checks read the source of the library with the ast module: no adapter imports
an application layer (C-04) and no adapter spells a legacy mesh folder (C-13).
Each has a planted-source control showing the scan can go red.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src/cfdbender"
FORBIDDEN = ("cfdbender.matrix", "cfdbender.workspace", "cfdbender.hpc")
LEGACY_FOLDERS = ("inputs/geometry", "inputs/geometries", "inputs/meshes")


def adapter_files() -> list[Path]:
    return sorted((SRC / "solvers").rglob("*.py"))


def imported(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def offending_imports(source: str) -> list[str]:
    return sorted(
        name
        for name in imported(source)
        if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN)
    )


def code_strings(source: str) -> list[str]:
    """Return string constants that are not docstrings."""
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_c04_no_adapter_imports_matrix_workspace_or_hpc():
    """C-04: no module under solvers/ imports cfdbender.matrix, workspace or hpc."""
    found = {
        path.relative_to(SRC).as_posix(): bad
        for path in adapter_files()
        if (bad := offending_imports(path.read_text(encoding="utf-8")))
    }
    assert not found, found
    assert adapter_files(), "the scan found no adapter file"


@pytest.mark.parametrize(
    "line",
    [
        "import cfdbender.matrix",
        "from cfdbender.workspace import planning",
        "from cfdbender import hpc",
        "from cfdbender.hpc._submit import run",
    ],
)
def test_c04_the_scan_detects_a_planted_import(line):
    """C-04: control, a planted forbidden import is reported ."""
    assert offending_imports(line + "\n")


def test_c13_no_adapter_spells_a_legacy_mesh_folder():
    """C-13: no code string under solvers/ names inputs/geometry, geometries or meshes."""
    found = {
        path.relative_to(SRC).as_posix(): sorted(
            {
                text
                for text in code_strings(path.read_text(encoding="utf-8"))
                if any(f in text for f in LEGACY_FOLDERS)
            }
        )
        for path in adapter_files()
    }
    found = {name: texts for name, texts in found.items() if texts}
    assert not found, found


def test_c13_the_scan_detects_a_planted_folder_literal():
    """C-13: control, a code string spelling a legacy folder is reported, a docstring is not."""
    planted = 'x = "inputs/geometries/a.obj"\n'
    assert any(f in text for text in code_strings(planted) for f in LEGACY_FOLDERS)
    documented = '"""inputs/geometries/a.obj"""\n'
    assert not code_strings(documented)
