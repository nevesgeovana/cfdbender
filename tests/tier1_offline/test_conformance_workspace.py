"""One conformance suite over every library adapter, offline, with fake executables.

The same assertions run for su2, openfoam, xfoil, pyfr, qprop and mock: the matrix
header is accepted, the flight condition is read, the GEOMETRY file resolves, the
s/r/p ids and the DP folder names follow the contract, runs.json records the plan,
a resumed run skips what is complete, collect writes the common CSV and the HPC
descriptor renders. No real solver runs: each adapter executes a stand-in script.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest
from test_matrix_columns_all_solvers import build
from test_p0000_item3f_plan import FAKE as SUBMIT_FAKE

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.matrix.exports import CONTEXT
from cfdbender.matrix.receipt import receipt_path, require_receipt, write_receipt
from cfdbender.matrix.submission import select_profile, submission_plans
from cfdbender.workspace.planning import plan_workspace

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SOLVERS = ["su2", "openfoam", "xfoil", "pyfr", "qprop", "mock"]
STANDIN = (
    "import pathlib,sys\n"
    "if 'import' in sys.argv: pathlib.Path('mesh.pyfrm').write_text('mesh')\n"
    "else: pathlib.Path('forces.csv').write_text('t,px,py,vx,vy\\n'"
    " + '0,0.1,0.3,0,0\\n1,0.1,0.3,0,0\\n2,0.1,0.3,0,0\\n3,0.1,0.3,0,0\\n')\n"
)


def executables(root: Path, solver: str) -> str:
    """Return the setup table that points the adapter at its offline stand-in."""
    if solver == "mock":
        return ""
    if solver == "pyfr":
        script = root / "standin.py"
        script.write_text(STANDIN, encoding="ascii")
        return f"\n[solver]\nexecutable = {json.dumps(script.as_posix())}\n"
    fake = {
        "su2": FIXTURES / "su2/fake_su2.py",
        "openfoam": FIXTURES / "openfoam/fake_openfoam.py",
        "xfoil": FIXTURES / "xfoil/fake_xfoil.py",
        "qprop": FIXTURES / "qprop/fake_qprop.py",
    }[solver]
    text = f"\n[solver]\nexecutable = {json.dumps(fake.as_posix())}\n"
    if solver == "qprop":
        text += f"xfoil_executable = {json.dumps((FIXTURES / 'qprop/fake_xfoil.py').as_posix())}\n"
    return text


@pytest.fixture(params=SOLVERS)
def case(request, tmp_path):
    solver = request.param
    root = build(tmp_path, solver)
    setup = root / "inputs/setups/s001.toml"
    setup.write_bytes(setup.read_bytes() + executables(root, solver).encode("ascii"))
    matrix = root / "inputs/matrices/full.fs"
    if solver == "xfoil":
        # The stand-in script accepts only the incompressible keystroke file.
        matrix.write_text(matrix.read_text(encoding="ascii").replace("MACH:0.1", "MACH:0"))
    if solver == "qprop":
        # QPROP executes only a blade named in its setup under its own folder, while
        # the planner resolves the GEOMETRY column in either: a gap for the layout lane.
        setup.write_bytes(b'blade = "inputs/geometry/blade.obj"' + bytes([10]) + setup.read_bytes())
        (root / "inputs/geometry").mkdir()
        (root / "inputs/geometries/blade.obj").rename(root / "inputs/geometry/blade.obj")
    return solver, root


def data_row(root):
    return (root / "inputs/matrices/full.fs").read_text(encoding="ascii").splitlines()[2].split("|")


def local(root, *options):
    return main(["run", "full", "--workspace", str(root), "--local", *options])


def test_the_matrix_header_is_accepted_and_the_flight_condition_read(case):
    solver, root = case
    plan = plan_workspace(root, "full")
    assert plan.solver == solver and plan.points
    point = plan.points[0]
    stated = dict(
        item.split(":", 1) for item in data_row(root)[6].replace(" ", "").split(",") if ":" in item
    )
    for key, value in stated.items():
        if value != "sweep":
            assert point.condition[key] == float(value), f"{solver}: {key}"
    assert (point.aircraft, point.configuration, point.description) == (
        "Model",
        "clean",
        "Full header row",
    )


def test_geometry_resolves_and_the_ids_and_folder_names_follow_the_contract(case):
    solver, root = case
    point = plan_workspace(root, "full").points[0]
    named = data_row(root)[8].strip()
    if named == "-":
        assert point.geometry is None
    else:
        assert point.geometry is not None and point.geometry.endswith(named)
        assert (root / point.geometry).is_file() and point.geometry in point.inputs_sha256
    assert (point.setup_id, point.reference_id, point.post_id) == ("s001", "r001", "p001")
    assert point.pol == "7001" and point.point_id.startswith("7001/DP-")
    assert re.fullmatch(r"DP-[A-Za-z0-9+_.-]+", point.point_id.split("/", 1)[1])
    assert point.directory == f"sims/sim_{point.pol}/datapoints/{point.point_id.split('/')[1]}"


def test_plan_writes_the_receipt_and_the_registry_records_planned_points(case):
    _, root = case
    plan = plan_workspace(root, "full")
    path = write_receipt(plan)
    assert path == receipt_path(root, "full") and path.is_file()
    require_receipt(plan)
    RunRegistry(root).record(plan)
    assert json.loads((root / "runs.json").read_bytes())
    records = RunRegistry(root).read()
    assert sorted(r.point_id for r in records) == sorted(p.point_id for p in plan.points)
    assert {r.status for r in records} == {"PLANNED"}


def test_run_completes_resume_skips_and_collect_writes_the_common_csv(case):
    solver, root = case
    plan = plan_workspace(root, "full")
    write_receipt(plan)
    assert local(root) == 0, solver
    records = RunRegistry(root).read()
    assert [r.status for r in records] == ["COMPLETED"] * len(plan.points)
    assert all(r.classification in ("converged", "iteration_limit") for r in records)
    stamps = {r.point_id: r.completed_at for r in records}
    assert local(root, "--resume") == 0
    again = RunRegistry(root).read()
    assert {r.point_id: r.completed_at for r in again} == stamps
    assert main(["collect", "full", "--workspace", str(root)]) == 0
    sweep = root / "post/full/campaign_sweep.csv"
    with sweep.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0])[:2] == ["POL", "point"] and list(rows[0])[2:18] == list(CONTEXT)
    assert len(rows) == len(plan.points)
    for row in rows:
        assert row["status"] == "COMPLETED" and row["classification"] != "NA"
        assert row["solver_version"] != "NA" and row["SREF"] != "NA"
    text = sweep.read_text(encoding="utf-8")
    assert '"' not in text and ",," not in text
    products = json.loads((root / "post/full/products.json").read_bytes())
    assert products["complete"] is True and (root / "post/full/post.log").is_file()


def test_the_hpc_descriptor_renders_for_every_point(case):
    solver, root = case
    path = root / "inputs/hpc/h001.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [Path(sys.executable).as_posix(), SUBMIT_FAKE.as_posix(), "{descriptor_path}"]
    path.write_text(
        chr(10).join(
            [
                'application_id = "offline"',
                "[descriptor]",
                'format = "json"',
                'name = "submit.json"',
                "[descriptor.fields]",
                'name = "{job_name}"',
                'working_directory = "{workdir}"',
                'walltime = "{walltime}"',
                "[submit]",
                f"command = {json.dumps(command)}",
                "[builds]",
                '"1.0" = "offline-build"',
            ]
        ),
        encoding="utf-8",
    )
    plan = plan_workspace(root, "full")
    profile = select_profile(root, "h001")
    assert profile is not None
    jobs = submission_plans(plan, profile)
    assert len(jobs) == len(plan.points)
    for job, point in zip(jobs, plan.points, strict=True):
        data = json.loads(job.descriptor_text)
        assert Path(data["working_directory"]) == root / point.directory, solver
        assert data["name"] and data["walltime"]


def prepared(root, solver):
    from cfdbender.core.adapter import load_point
    from cfdbender.solvers import load_solver

    point = plan_workspace(root, "full").points[0]
    return load_solver(solver), load_point(root, point, solver=solver)


@pytest.mark.parametrize(
    "rule", ["deterministic", "results", "no_overwrite", "execution_isolation"]
)
def test_the_shared_contract_assertions_hold_for_every_adapter(case, rule):
    from cfdbender import testing

    solver, root = case
    adapter, point = prepared(root, solver)
    getattr(testing, f"assert_{rule}")(adapter, point)
