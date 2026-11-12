"""Matrix columns read with the names and meanings of a pyflightstream matrix."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cfdbender.core.flight import FlightConditionError, resolve_flow
from cfdbender.core.matrix import MatrixError, read_matrix
from cfdbender.core.plan import apply_variables

HEADER = (
    "POL | HIDDEN | RUN | AIRCRAFT | CONFIGURATION | DESCRIPTION | FLIGHT_CONDITION "
    "| SWEEP_VALUES | GEOMETRY | REF | SET | PPROC | SYMMETRY | SYMMETRY_LOADS | NCPUS "
    "| WALLTIME | FS_BUILD | WORKFLOW | VAR_NAMES_VALUES"
)
ROW = (
    " 5001 |   1    |  1  | WingBody | clean, flaps up | Polar at cruise "
    "| MACH:0.85, REmi:5, TK:310.928, alpha:sweep, BETA:0 | 2.5,2.75 | wb.su2 "
    "| 001 | 002 | 003 | - | - | 192 | 4h | 8.3.0 | - "
    '| ITER: 2000 / CFL_NUMBER: 5.0 / MARKER_FAR: ["farfield"] / solver.threads: 2 '
)


def write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "study.fs"
    path.write_text("\n".join(["# example", *lines, ""]), encoding="utf-8")
    return path


def test_full_pyflightstream_header_is_read_with_padded_cells(tmp_path):
    (row,) = read_matrix(write(tmp_path, HEADER, "-" * 40, ROW))
    assert (row.pol, row.setup_id, row.reference_id, row.post_id) == (
        "5001",
        "s002",
        "r001",
        "p003",
    )
    assert row.hidden is True and row.run == 1, "HIDDEN never removes a row; RUN decides"
    assert (row.aircraft, row.configuration, row.description) == (
        "WingBody",
        "clean, flaps up",
        "Polar at cruise",
    )
    assert dict(row.condition) == {"MACH": 0.85, "REmi": 5.0, "TK": 310.928, "BETA": 0.0}
    assert (row.sweep_key, row.sweep_values) == ("ALPHA", (2.5, 2.75))
    assert (row.geometry, row.ncpus, row.walltime, row.solver_build) == (
        "wb.su2",
        192,
        "4h",
        "8.3.0",
    )
    assert row.workflow == "steady", "a - in WORKFLOW states nothing"
    assert dict(row.variables) == {
        "ITER": 2000,
        "CFL_NUMBER": 5.0,
        "MARKER_FAR": ["farfield"],
        "solver.threads": 2,
    }


def test_solver_build_column_and_old_layout(tmp_path):
    header = "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES|SOLVER_BUILD"
    (row,) = read_matrix(write(tmp_path, header, "1|1|001|001|001|alpha:sweep|2.5|8.3.0"))
    assert row.solver_build == "8.3.0"
    old = "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES"
    (row,) = read_matrix(write(tmp_path, old, " 5001 |1  | 001 | 001 | 001 | alpha:sweep | 2.50"))
    assert (row.sweep_key, row.ncpus, row.geometry, dict(row.variables)) == (
        "ALPHA",
        None,
        None,
        {},
    )


@pytest.mark.parametrize(
    ("header", "row", "message"),
    [
        ("POL|RUN|SET|REF|PPROC|FLIGHT_CONDITON", "1|1|1|1|1|-", "FLIGHT_CONDITION"),
        ("POL|RUN|SET|REF|PPROC|FS_BUILD|SOLVER_BUILD", "1|1|1|1|1|-|-", "one column"),
        ("POL|RUN|SET|REF|PPROC|SYMMETRY", "1|1|1|1|1|MIRROR", "SYMMETRY"),
        ("POL|RUN|SET|REF|PPROC|NCPUS", "1|1|1|1|1|0", "NCPUS"),
        ("POL|RUN|SET|REF|PPROC|HIDDEN", "1|1|1|1|1|2", "HIDDEN"),
        ("POL|RUN|SET|REF|PPROC|VAR_NAMES_VALUES", "1|1|1|1|1|ITER: 1 / ITER: 2", "twice"),
        ("POL|RUN|SET|REF|PPROC|VAR_NAMES_VALUES", "1|1|1|1|1|NCPUS: 8", "NCPUS column"),
        ("POL|RUN|SET|REF|PPROC|VAR_NAMES_VALUES", "1|1|1|1|1|ITER 4", "KEY: value"),
        ("POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION", "1|1|1|1|1|MACH:0.2, mach:0.3", "repeats"),
    ],
)
def test_column_refusals_name_the_cell(tmp_path, header, row, message):
    with pytest.raises(MatrixError, match=message):
        read_matrix(write(tmp_path, header, row))


def test_variables_override_setup_keys_and_tables():
    setup = {"ITER": 10, "solver": {"route": "executable", "threads": 1}}
    result = apply_variables(setup, {"ITER": 20, "solver.threads": 4})
    assert result == {"ITER": 20, "solver": {"route": "executable", "threads": 4}}
    assert setup["solver"] == {"route": "executable", "threads": 1}, "the setup is not modified"


def test_flow_state_matches_the_pyflightstream_examples():
    tunnel = resolve_flow({"MACH": 0.20, "REmi": 5.5}, reference_length=1.0, label="P1")
    assert tunnel.velocity == pytest.approx(68.0588, abs=1e-4)
    assert tunnel.density == pytest.approx(1.446, abs=1e-3)
    assert tunnel.pressure == pytest.approx(119.6e3, rel=1e-3)
    assert tunnel.reynolds == pytest.approx(5.5e6)
    assert tunnel.density_source == "solved-from-reynolds"
    air = resolve_flow(
        {"TASmps": 68.08, "ALTFT": 10000, "dISA": 5}, reference_length=1.0, label="P1"
    )
    assert air.mach == pytest.approx(0.2054, abs=1e-4)
    assert air.density_source == "atmosphere"
    sea = resolve_flow({"MACH": 0.3}, reference_length=None, label="P1")
    assert (sea.temperature, sea.pressure, sea.reynolds) == (288.15, 101325.0, None)
    assert sea.density == pytest.approx(1.225, abs=1e-3)
    pinned = resolve_flow({"MACH": 0.85, "TK": 310.928, "PPA": 2e5}, reference_length=2, label="")
    assert pinned.density == pytest.approx(2e5 / (287.05287 * 310.928))
    assert math.isclose(pinned.mach, 0.85)


@pytest.mark.parametrize(
    ("condition", "length", "message"),
    [
        ({"REmi": 5.0}, 1.0, "exactly one of MACH or TASmps"),
        ({"MACH": 0.2, "TASmps": 60.0}, 1.0, "exactly one"),
        ({"MACH": 0.2, "REmi": 5.0}, None, "reference length"),
        ({"MACH": 0.2, "REmi": 5.0, "RHOkgm3": 1.2}, 1.0, "state one"),
        ({"MACH": 0.2, "ALTFT": 70000}, 1.0, "outside"),
        ({"MACH": 0.0, "REmi": 5.0}, 1.0, "positive speed"),
    ],
)
def test_flow_state_refusals(condition, length, message):
    with pytest.raises(FlightConditionError, match=message):
        resolve_flow(condition, reference_length=length, label="POL 1")


def test_registry_keeps_the_row_columns(tmp_path):
    from cfdbender.core.plan import RowRequirements, plan_matrix
    from cfdbender.core.registry import RunRegistry

    files = {
        "init.toml": 'schema_version = 1\nsolver = "mock"\n',
        "inputs/setups/s001.toml": "",
        "inputs/references/r001.toml": "",
        "inputs/pproc/p001.toml": "",
        "inputs/geometries/wing.dat": "mesh",
        "inputs/matrices/study.fs": "\n".join(
            [
                "POL|RUN|HIDDEN|DESCRIPTION|GEOMETRY|NCPUS|SOLVER_BUILD|VAR_NAMES_VALUES|SET|REF|PPROC",
                "7|1|1|cruise point|wing|8|2.0|ITER: 3|001|001|001",
            ]
        ),
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")

    def hook(row, inputs):
        inputs.read_bytes(inputs.geometry)
        assert inputs.setup == {"ITER": 3}
        return RowRequirements(("solve",), ("sims/sim_{pol}/datapoints/{point}/out.txt",))

    plan = plan_matrix(tmp_path, "study", validators={"mock": hook})
    (point,) = plan.points
    assert (point.geometry, point.ncpus, point.solver_build, point.hidden) == (
        "inputs/geometries/wing.dat",
        8,
        "2.0",
        True,
    )
    assert point.description == "cruise point"
    registry = RunRegistry(tmp_path)
    registry.record(plan)
    (record,) = registry.read()
    assert record.point == point
    stored = json.loads((tmp_path / "runs.json").read_text())
    assert "geometry" in json.dumps(stored)

    def ignores(row, inputs):
        return RowRequirements(("solve",), ("sims/sim_{pol}/datapoints/{point}/out.txt",))

    from cfdbender.core.plan import PlanError

    with pytest.raises(PlanError, match="does not read the GEOMETRY column"):
        plan_matrix(tmp_path, "study", validators={"mock": ignores})
