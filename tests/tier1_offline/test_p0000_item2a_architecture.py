"""Check static core boundaries with executable planted mutants."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tools" / "check_architecture.py"


def _check(package):
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(package)],
        capture_output=True,
        text=True,
        check=False,
    )


def _tree(tmp_path, source, *, location="core/mutant.py"):
    package = tmp_path / "cfdbender"
    (package / "core").mkdir(parents=True)
    (package / "core/__init__.py").write_text('"""Core."""', encoding="utf-8")
    target = package / location
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return package


def test_p0000_item2a_live_core_never_imports_solvers():
    result = _check(ROOT / "src/cfdbender")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "architecture: ok" in result.stdout


@pytest.mark.parametrize(
    "source",
    [
        "import cfdbender.solvers.mock",
        "import cfdbender.solvers as adapters",
        "from cfdbender.solvers import mock",
        "from cfdbender import solvers",
        "from ..solvers import mock",
        "from .. import solvers",
        "def deferred():" + chr(10) + "    import cfdbender.solvers.mock",
        "import importlib" + chr(10) + 'importlib.import_module("cfdbender.solvers.mock")',
        "from importlib import import_module as load" + chr(10) + 'load("cfdbender.solvers.mock")',
        '__import__("cfdbender.solvers.mock")',
    ],
)
def test_p0000_item2a_core_solver_import_mutant_is_caught(tmp_path, source):
    package = _tree(tmp_path, source)
    result = _check(package)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "core/mutant.py:" in result.stdout.replace(chr(92), "/")
    assert "core must not import solvers" in result.stdout


@pytest.mark.parametrize(
    "source",
    [
        '"""import cfdbender.solvers.mock"""',
        "# from cfdbender.solvers import mock",
        "from cfdbender.core import settings",
        "from . import settings",
    ],
)
def test_p0000_item2a_core_comments_and_legal_imports_are_allowed(tmp_path, source):
    result = _check(_tree(tmp_path, source))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("location", ["workspace/cli.py", "core/mutant.py"])
def test_p0000_item2a_upstream_import_outside_facade_is_caught(tmp_path, location):
    package = _tree(tmp_path, "import pyflightstream.workspace", location=location)
    result = _check(package)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "use cfdbender._pyfs" in result.stdout


def test_p0000_item2a_upstream_facade_and_adapter_composition_are_allowed(tmp_path):
    package = _tree(tmp_path, "from pyflightstream import workspace", location="_pyfs.py")
    (package / "workspace.py").write_text("from cfdbender.solvers import mock", encoding="utf-8")
    result = _check(package)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["missing", "empty", "syntax"])
def test_p0000_item2a_architecture_refuses_unmeasurable_tree(tmp_path, kind):
    if kind == "syntax":
        package = _tree(tmp_path, "def broken(")
    else:
        package = tmp_path / "cfdbender"
        if kind == "empty":
            package.mkdir()
    result = _check(package)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "architecture refused" in result.stderr


@pytest.mark.parametrize(
    "source",
    [
        'import importlib\nimportlib.import_module("..solvers.mock", package="cfdbender.core")',
        'import importlib\nimportlib.import_module("..solvers.mock", package=__package__)',
    ],
)
def test_review2_relative_dynamic_solver_import_is_caught(tmp_path, source):
    result = _check(_tree(tmp_path, source))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "core must not import solvers" in result.stdout
