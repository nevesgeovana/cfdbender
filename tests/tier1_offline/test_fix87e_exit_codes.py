"""A refused or interrupted run never records exit code 0; PyFR budget sampling."""

import json
import sys

import pytest

from cfdbender.solvers.openfoam._execute import _launch as openfoam_launch
from cfdbender.solvers.pyfr._execute import _launch as pyfr_launch
from cfdbender.solvers.qprop._execute import _run_owned as qprop_run

TOUCH = [sys.executable, "-c", "import pathlib; pathlib.Path('done').write_text('x')"]


def refuse_after_done(folder):
    def check():
        if (folder / "done").exists():
            raise ValueError("RAM below the floor")

    return check


def test_pyfr_ram_refusal_in_second_command_records_failure(tmp_path):
    check = refuse_after_done(tmp_path)
    with pytest.raises(ValueError, match="RAM"):
        pyfr_launch([[sys.executable, "-c", "pass"], TOUCH], tmp_path, 30, check)
    meta = json.loads((tmp_path / "execution.json").read_bytes())
    assert meta["exit_code"] != 0
    assert meta["failed_command"] == 1


def test_pyfr_successful_sequence_records_no_failed_command(tmp_path):
    ok = [sys.executable, "-c", "pass"]
    pyfr_launch([ok, ok], tmp_path, 30, lambda: None)
    meta = json.loads((tmp_path / "execution.json").read_bytes())
    assert meta["exit_code"] == 0 and meta["failed_command"] is None


def test_openfoam_refusal_after_a_clean_exit_is_not_recorded_as_success(tmp_path):
    opts = {"threads": 1, "processes": 1, "mpi_launcher": None}
    with pytest.raises(ValueError, match="RAM"):
        openfoam_launch(TOUCH, tmp_path, opts, 30, refuse_after_done(tmp_path))
    assert json.loads((tmp_path / "execution.json").read_bytes())["exit_code"] != 0


def test_qprop_refusal_after_a_clean_exit_is_not_recorded_as_success(tmp_path):
    with pytest.raises(ValueError, match="RAM"):
        qprop_run(tmp_path, TOUCH, 30, {"ram_estimate_gib": 0}, refuse_after_done(tmp_path))
    assert json.loads((tmp_path / "execution.json").read_bytes())["exit_code"] != 0
