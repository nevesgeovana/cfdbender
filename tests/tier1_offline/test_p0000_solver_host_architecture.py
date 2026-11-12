"""P0000: planted sibling solver imports fail the static architecture guard."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "owner,sibling",
    [
        ("xfoil", "su2"),
        ("su2", "openfoam"),
        ("openfoam", "xfoil"),
        ("pyfr", "openfoam"),
        ("qprop", "xfoil"),
        ("_host", "su2"),
    ],
)
@pytest.mark.parametrize(
    "source",
    [
        "import cfdbender.solvers.{sibling}._config",
        "from cfdbender.solvers import {sibling}",
        "from ..{sibling}._config import number",
        "from .. import {sibling}",
        "def deferred():\n    from cfdbender.solvers.{sibling} import _execute",
        'import importlib as il\nil.import_module("cfdbender.solvers.{sibling}._config")',
        'from importlib import import_module as load\nload("..{sibling}", __package__)',
        '__import__("cfdbender.solvers.{sibling}")',
    ],
)
def test_p0000_solver_host_sibling_import_refused(tmp_path, owner, sibling, source):
    package = tmp_path / "cfdbender"
    (package / "core").mkdir(parents=True)
    (package / "core/__init__.py").write_text('"""Core."""')
    target = package / "solvers" / owner / "mutant.py"
    target.parent.mkdir(parents=True)
    target.write_text(source.format(sibling=sibling))
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_architecture.py"), "--root", str(package)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "must not import sibling solver" in result.stdout
    assert f"solvers/{owner}/mutant.py:" in result.stdout


@pytest.mark.parametrize("owner", ["su2", "xfoil", "openfoam", "pyfr", "qprop", "_host"])
def test_p0000_solver_host_legal_imports(tmp_path, owner):
    package = tmp_path / "cfdbender"
    (package / "core").mkdir(parents=True)
    (package / "core/__init__.py").write_text('"""Core."""')
    target = package / "solvers" / owner / "__init__.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        f"from cfdbender.solvers.{owner} import local\n"
        "from cfdbender.solvers._host import _ram\n"
        "from .._host import process_lease\n"
        "from . import local\n"
        "from cfdbender.solvers import available_solvers, load_solver\n"
        "from .. import available_solvers\n"
        "# import cfdbender.solvers.other\n"
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_architecture.py"), "--root", str(package)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
