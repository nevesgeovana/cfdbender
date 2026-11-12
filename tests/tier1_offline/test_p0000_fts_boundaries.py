"""P0000-FTS: the optional dependency refuses before touching a workspace."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFUSAL = "install the FlightStream support: pip install cfdbender[flightstream]"


@pytest.mark.parametrize("kind", ["init", "workspace", "matrix", "loader"])
def test_p0000_fts_missing_extra_refuses_honestly(tmp_path, kind):
    (tmp_path / "init.toml").write_text('schema_version = 1\nsolver = "flightstream"\n')
    program = """
import sys
sys.modules["pyflightstream"] = None
from cfdbender.workspace.cli import main as workspace
from cfdbender.matrix.cli import main as matrix
from cfdbender.solvers import load_solver
kind, root = sys.argv[1:]
if kind == "loader":
    try:
        load_solver("flightstream")
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
elif kind == "init":
    raise SystemExit(workspace(["init", "--solver", "flightstream", root + "/new"]))
else:
    main = workspace if kind == "workspace" else matrix
    raise SystemExit(main(["--help", "--workspace", root]))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, kind, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert REFUSAL in result.stderr
    assert not (tmp_path / "new").exists()
