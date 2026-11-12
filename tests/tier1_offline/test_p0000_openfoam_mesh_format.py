"""The mesh builder satisfies the required formatter gate."""

import subprocess
import sys
from pathlib import Path


def test_p0000_openfoam_mesh_formatter():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            str(root / "tools/kit/openfoam_naca0012_mesh.py"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
