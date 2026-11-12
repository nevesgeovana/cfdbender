"""P0000-FTS: planted core and HPC imports cannot reach the upstream facade."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("layer", ["core", "hpc"])
@pytest.mark.parametrize(
    "source",
    [
        "import cfdbender._pyfs",
        "from cfdbender import _pyfs",
        "from .._pyfs import workspace_main",
        "from .. import _pyfs",
        "def deferred():\n    import cfdbender._pyfs",
        'from importlib import import_module as load\nload("cfdbender._pyfs")',
    ],
)
def test_p0000_fts_core_hpc_cannot_import_facade(tmp_path, layer, source):
    package = tmp_path / "cfdbender"
    (package / "core").mkdir(parents=True)
    (package / "core/__init__.py").write_text('"""Core."""')
    target = package / layer / "mutant.py"
    target.parent.mkdir(exist_ok=True)
    target.write_text(source)
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_architecture.py"), "--root", str(package)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{layer} must not import cfdbender._pyfs" in result.stdout
