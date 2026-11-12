"""An SU2 workspace planned from a matrix written like a pyflightstream matrix."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import PlanError
from cfdbender.core.workspace import init_workspace
from cfdbender.matrix.cli import main
from cfdbender.matrix.submission import select_profile, submission_plans
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.workspace.planning import plan_workspace

HEADER = (
    "POL | HIDDEN | RUN | AIRCRAFT | CONFIGURATION | DESCRIPTION | FLIGHT_CONDITION "
    "| SWEEP_VALUES | GEOMETRY | REF | SET | PPROC | SYMMETRY | SYMMETRY_LOADS | NCPUS "
    "| WALLTIME | SOLVER_BUILD | WORKFLOW | VAR_NAMES_VALUES"
)
ROW = (
    " 5001 | 0 | 1 | WB | cruise | Polar near design "
    "| MACH:0.85, REmi:5, TK:310.928, ALPHA:sweep, BETA:0 | 2.5,2.75 | wb.su2 "
    "| 001 | 001 | 001 | - | - | 192 | 4h | 8.3.0 | - | iterations: 25 "
)
PROFILE = """application_id = "SU2"
max_walltime = "96:00:00"

[descriptor]
format = "yaml"
name = "submit.yaml"

[descriptor.fields]
ApplicationId = "{application_id}"
job_name = "{job_name}"
walltime = "{walltime}"
ncpus = "{ncpus}"
version = "{solver_version_alias}"
working_directory = "{workdir}"
config_file = "{config_path}"
solver_type = "{solver_type}"
cluster = "example-cluster"

[submit]
command = ["submit-tool", "{descriptor_path}"]

[builds]
"8.3.0" = "8.3.0"
"""


def workspace(root: Path, *, row: str = ROW, setup: str = "", profile: bool = False) -> Path:
    files = {
        "inputs/setups/s001.toml": "\n".join(
            ["iterations = 40", 'SOLVER = "RANS"', 'MARKER_FAR = ["farfield"]', setup]
        ),
        "inputs/references/r001.toml": "area = 1.0\nlength = 0.5\nmoment_origin = [0.0, 0.0, 0.0]",
        "inputs/pproc/p001.toml": "history = true",
        "inputs/geometries/wb.su2": "NDIME= 3",
        "inputs/matrices/dpw.fs": "\n".join([HEADER, "-" * 80, row]),
    }
    if profile:
        files["inputs/hpc/h001.toml"] = PROFILE
    init_workspace(root, solver="su2", available_solvers=available_solvers())
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text + "\n", encoding="ascii")
    return root


def deck(root: Path, index: int = 0) -> dict[str, str]:
    plan = plan_workspace(root, "dpw")
    point = load_point(root, plan.points[index], solver="su2")
    text = load_solver("su2").translate(point)
    return dict(line.split("= ", 1) for line in text.splitlines())


def test_su2_init_creates_only_the_folders_it_reads(tmp_path):
    init_workspace(tmp_path, solver="su2", available_solvers=available_solvers())
    assert sorted(p.name for p in (tmp_path / "inputs").iterdir() if p.is_dir()) == [
        "geometries",
        "hpc",
        "matrices",
        "pproc",
        "references",
        "setups",
    ]


def test_su2_row_flight_condition_geometry_and_variables_reach_the_deck(tmp_path):
    values = deck(workspace(tmp_path))
    assert values["MESH_FILENAME"] == "mesh.su2"
    assert values["ITER"] == "25", "VAR_NAMES_VALUES overrides the setup key"
    assert float(values["AOA"]) == 2.5
    assert float(values["SIDESLIP_ANGLE"]) == 0.0
    assert float(values["MACH_NUMBER"]) == pytest.approx(0.85)
    assert float(values["FREESTREAM_TEMPERATURE"]) == pytest.approx(310.928)
    assert float(values["REYNOLDS_NUMBER"]) == pytest.approx(5e6)
    assert float(values["REYNOLDS_LENGTH"]) == 0.5
    density, pressure = float(values["FREESTREAM_DENSITY"]), float(values["FREESTREAM_PRESSURE"])
    assert math.isclose(pressure, density * 287.05287 * 310.928, rel_tol=1e-9)
    plan = plan_workspace(tmp_path, "dpw")
    assert "inputs/geometries/wb.su2" in plan.points[0].inputs_sha256


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("mach = 0.8", "MACH_NUMBER"),
        ('mesh = "inputs/geometries/other.su2"', "GEOMETRY"),
        ("reynolds = 1e6", "REYNOLDS_NUMBER"),
    ],
)
def test_su2_setup_and_matrix_must_agree(tmp_path, setup, message):
    workspace(tmp_path, setup=setup)
    with pytest.raises(PlanError, match=message):
        plan_workspace(tmp_path, "dpw")


def test_su2_setup_value_that_agrees_is_accepted(tmp_path):
    values = deck(workspace(tmp_path, setup="freestream_temperature = 310.928"))
    assert float(values["FREESTREAM_TEMPERATURE"]) == pytest.approx(310.928)


def test_su2_bare_mach_row_keeps_the_setup_flow_state(tmp_path):
    row = ROW.replace("MACH:0.85, REmi:5, TK:310.928", "MACH:0.3")
    values = deck(
        workspace(tmp_path, row=row, setup="reynolds = 1e6\nfreestream_temperature = 288")
    )
    assert (values["MACH_NUMBER"], values["REYNOLDS_NUMBER"]) == ("0.3", "1000000")
    assert "FREESTREAM_PRESSURE" not in values


def test_su2_hpc_descriptor_from_row_columns(tmp_path):
    root = workspace(tmp_path, profile=True)
    plan = plan_workspace(root, "dpw")
    profile = select_profile(root, None)
    assert profile is not None
    job = submission_plans(plan, profile)[0]
    text = job.descriptor_text
    assert 'walltime: "4h"' in text
    assert "ncpus: 192" in text
    assert 'version: "8.3.0"' in text
    assert 'solver_type: "SU2"' in text
    assert 'cluster: "example-cluster"' in text
    assert job.command[0] == "submit-tool"


def test_su2_plan_listing_carries_the_row_text(tmp_path, capsys):
    root = workspace(tmp_path, profile=True)
    assert main(["plan", "dpw", "--workspace", str(root)]) == 0
    out = capsys.readouterr().out
    assert "AIRCRAFT=WB CONFIGURATION=cruise DESCRIPTION=Polar near design" in out
    assert "GEOMETRY=inputs/geometries/wb.su2 NCPUS=192 SOLVER_BUILD=8.3.0" in out
    assert "variable iterations = 25" in out
    assert out.count("descriptor: ") == 2
