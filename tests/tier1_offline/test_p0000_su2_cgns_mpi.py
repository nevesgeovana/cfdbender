"""P0000 HPC-2: [solver] mpi_launcher=['mpirun', '-np', '192']."""

import json
import sys
from pathlib import Path

import pytest
from test_p0000_item3d_config import api, prepared, workspace

from cfdbender.solvers.su2._config import options
from cfdbender.solvers.su2._execute import _argv

FAKE = Path(__file__).resolve().parents[1] / "fixtures/su2/fake_su2.py"


@pytest.mark.parametrize("launcher", [[], ["mpirun", "-np", "192"], ["srun"]])
def test_p0000_su2_cgns_mpi_argv(tmp_path, launcher):
    try:
        opts = options({"solver": {"mpi_launcher": launcher, "executable": str(FAKE)}})
    except ValueError as error:
        pytest.fail(f"P0000-HPC-2: MPI setup refused: {error}")
    assert _argv(opts, tmp_path) == [*launcher, sys.executable, str(FAKE), "su2.cfg"]


@pytest.mark.parametrize("launcher", ["mpirun", [1], [""], ["bad\nvalue"], None])
def test_p0000_su2_cgns_mpi_validation(launcher):
    with pytest.raises(ValueError, match="mpi_launcher.*list.*strings"):
        options({"solver": {"mpi_launcher": launcher}})


def test_p0000_su2_cgns_mpi_offline_launch(tmp_path):
    launcher = [sys.executable, str(FAKE.with_name("fake_mpi.py")), "-np", "192"]
    workspace(
        tmp_path,
        extra='[solver]\nroute="executable"\nexecutable='
        + json.dumps(str(FAKE))
        + "\nmpi_launcher="
        + json.dumps(launcher),
    )
    point = prepared(tmp_path)
    result = api().execute(point, check_ram=lambda: None)
    assert api().classify(result) == "converged"
    argv = json.loads((tmp_path / point.point.directory / "mpi-argv.json").read_text())
    assert argv == ["-np", "192", sys.executable, str(FAKE), "su2.cfg"]
