"""P0000-ITEM3E-ARCHITECTURE: HPC cannot import a solver adapter."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "source",
    [
        "import cfdbender.solvers.mock",
        "from ..solvers import su2",
        'import importlib\nimportlib.import_module("cfdbender.solvers.su2")',
    ],
)
def test_p0000_item3e_hpc_solver_import_is_refused(tmp_path, source):
    checker = Path(__file__).resolve().parents[2] / "tools/check_architecture.py"
    spec = importlib.util.spec_from_file_location("architecture", checker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "cfdbender"
    (root / "core").mkdir(parents=True)
    (root / "core/__init__.py").write_text('"""Core."""', encoding="utf-8")
    (root / "hpc").mkdir()
    (root / "hpc/mutant.py").write_text(source, encoding="utf-8")
    assert any("hpc must not import solvers" in line for line in module.check(root))
