"""Verify optional FlightStream imports type-check with both platform targets."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("platform", [None, "linux"])
def test_rfrun_002_source_mypy(platform):
    pytest.importorskip("mypy", reason="install the dev extra to run mypy")
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "mypy", str(root / "src")]
    if platform is not None:
        command.extend(["--platform", platform])
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
