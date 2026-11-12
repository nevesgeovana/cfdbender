"""A submitted point folder holds every input its solver reads, and collect reads the job.

Each test runs ``cfd-matrix run`` with a fake submit client, checks the point
folder against the inputs a local run stages (the mesh as a link, never a copy),
then plays the cluster job with the solver's offline stand-in in that folder and
proves ``cfd-matrix collect`` completes the point from what the job wrote.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import qprop_workspace
import test_p0000_item3d_config as su2_case
import test_p0000_openfoam_adapter as openfoam_case
import test_p0000_pyfr_adapter as pyfr_case
from test_p0000_item3f_plan import profile

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIELDS = {"name": "{job_name}", "working_directory": "{workdir}", "config": "{config_path}"}


def submit(root):
    matrix = root / "inputs/matrices/study.fs"
    header, *rows = matrix.read_text(encoding="ascii").strip().splitlines()
    lines = [header + "|SOLVER_BUILD", *(row + "|mock-1" for row in rows)]
    matrix.write_text("\n".join(lines) + "\n", encoding="ascii")
    profile(root, fields=FIELDS)
    assert main(["run", "study", "--workspace", str(root)]) == 0
    records = RunRegistry(root).read()
    assert [r.status for r in records] == ["SUBMITTED"]
    data = records[0].submission
    descriptor = json.loads((root / data["descriptor"]).read_bytes())
    folder = Path(descriptor["working_directory"])
    assert folder == root / records[0].point.directory
    assert Path(descriptor["config"]) == root / data["config_path"]
    return folder, Path(descriptor["config"])


def job(argv, folder, env=None):
    """Play the cluster job: the solver runs in the point folder, channels to the logs."""
    with (
        (folder / "stdout.log").open("xb") as stdout,
        (folder / "stderr.log").open("xb") as stderr,
    ):
        subprocess.run(
            argv, cwd=folder, stdout=stdout, stderr=stderr, env=env, check=True, timeout=60
        )


def collected(root):
    code = main(["collect", "study", "--workspace", str(root)])
    summary = json.loads((root / "post/study/campaign_summary.json").read_bytes())
    assert code == 0, summary["points"][0]["error"]
    record = RunRegistry(root).read()[0]
    assert record.status == "COMPLETED"
    return record


def outstanding(root):
    assert main(["collect", "study", "--workspace", str(root)]) == 3
    assert RunRegistry(root).read()[0].status == "SUBMITTED"


def test_su2_point_folder_holds_cfg_and_linked_mesh(tmp_path):
    su2_case.workspace(tmp_path)
    folder, config = submit(tmp_path)
    assert config.name == "su2.cfg", "the descriptor names the file SU2 reads"
    assert not (folder / "solver.in").exists()
    assert "MESH_FILENAME= mesh.su2" in config.read_text()
    assert (folder / "mesh.su2").samefile(tmp_path / "inputs/meshes/tiny.su2")
    outstanding(tmp_path)
    job([sys.executable, str(FIXTURES / "su2/fake_su2.py"), config.name], folder)
    record = collected(tmp_path)
    assert record.classification == "converged"
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    assert summary["points"][0]["result"]["route"] == "cluster"
    assert summary["points"][0]["result"]["coefficients"]["CL"]["value"] == 0.3


def test_openfoam_point_folder_holds_case_tree_and_linked_mesh(tmp_path):
    openfoam_case.case(tmp_path, sweep="4")
    folder, config = submit(tmp_path)
    assert config.name == "Allrun"
    assert all((folder / name).is_file() for name in openfoam_case.CASE_FILES)
    assert (folder / "mesh.msh").samefile(tmp_path / "inputs/meshes/tiny.msh")
    outstanding(tmp_path)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "FAKE_OPENFOAM_MODE": "converged"}
    job([sys.executable, str(FIXTURES / "openfoam/fake_openfoam.py")], folder, env)
    assert collected(tmp_path).classification == "converged"


def test_pyfr_point_folder_holds_case_and_linked_mesh(tmp_path):
    pyfr_case.case(tmp_path)
    folder, config = submit(tmp_path)
    assert config.name == "case.ini"
    assert "[solver-elements-quad]" in config.read_text()
    assert (folder / "mesh.msh").samefile(tmp_path / "inputs/meshes/tiny.msh")
    outstanding(tmp_path)
    standin = str(tmp_path / "standin.py")
    job([sys.executable, standin, "import", "mesh.msh", "mesh.pyfrm"], folder)
    (folder / "stdout.log").unlink()
    (folder / "stderr.log").unlink()
    job([sys.executable, standin, "run", "mesh.pyfrm", "case.ini"], folder)
    assert collected(tmp_path).classification == "converged"


def test_qprop_point_folder_holds_prop_flight_file_and_sections(tmp_path):
    qprop_workspace.workspace(tmp_path)
    folder, config = submit(tmp_path)
    assert config.name == "propeller.prop"
    assert (folder / "qcon.def").is_file()
    rpm = json.loads((folder / "sections.json").read_bytes())["rpm"]
    outstanding(tmp_path)
    job(
        [sys.executable, str(qprop_workspace.FAKE_QPROP), config.name, "none", "20", str(rpm)],
        folder,
    )
    assert collected(tmp_path).classification == "converged"


def test_staging_failure_is_recorded_and_nothing_is_submitted(tmp_path, monkeypatch):
    su2_case.workspace(tmp_path)
    from cfdbender.solvers._host import _mesh

    def unavailable(*args, **kwargs):
        raise OSError("links unavailable")

    monkeypatch.setattr(_mesh, "_symlink", unavailable)
    monkeypatch.setattr(_mesh, "_hardlink", unavailable)
    with pytest.raises(AssertionError):
        submit(tmp_path)
    record = RunRegistry(tmp_path).read()[0]
    assert record.status == "COMPLETED" and record.classification == "failed"
    assert record.error.startswith("FAILED_STAGING") and record.submission is None
    assert not list(tmp_path.rglob("fake-submit-argv.json"))


@pytest.mark.parametrize("name", ["su2", "openfoam", "pyfr", "qprop", "xfoil", "mock"])
def test_every_library_solver_names_its_staged_config(name):
    from cfdbender.solvers import load_solver

    adapter = load_solver(name)
    assert callable(adapter.stage) and adapter.STAGED_CONFIG
