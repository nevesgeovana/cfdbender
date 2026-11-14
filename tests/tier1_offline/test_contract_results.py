"""Adapter contract, results clauses: references, history, tables, units.

Six adapters with offline stand-ins; each test docstring names its clause of
docs/adapter-contract.md.
"""

from __future__ import annotations

import csv
import io
import math

import pytest
from test_conformance_workspace import case, prepared  # noqa: F401

from cfdbender.core.results import Coefficient, Iteration, NeutralResult, References

REFS = References(1.0, 1.0, (0.0, 0.0, 0.0), "body", ("aft", "right", "up"))


def run_and_collect(solver, root):
    adapter, point = prepared(root, solver)
    adapter.execute(point, check_ram=lambda: None)
    return adapter, point, adapter.collect(point)


def test_c39_fixed_references_are_the_ones_the_page_states(case):  # noqa: F811
    """C-39: xfoil, pyfr and qprop state their fixed references; others read them."""
    solver, root = case
    _, point, result = run_and_collect(solver, root)
    refs = result.references
    if solver == "xfoil":
        assert (refs.area, refs.length) == (1.0, 1.0)
        assert tuple(refs.moment_origin) == (0.25, 0.0, 0.0)
    elif solver == "pyfr":
        assert refs.length == 1.0
    elif solver == "qprop":
        assert refs.length > 0 and refs.area == pytest.approx(math.pi * (refs.length / 2) ** 2)
        assert tuple(refs.moment_origin) == (0.0, 0.0, 0.0)
    elif solver == "openfoam":
        assert (refs.area, refs.length) == (0.1, 1.0)
        assert tuple(refs.moment_origin) == (0.0, 0.0, 0.0)
    elif solver == "su2":
        assert (refs.area, refs.length) == (1.0, 1.0)


def test_c40_every_adapter_history_counts_from_one_with_a_linear_residual(case):  # noqa: F811
    """C-40: iterations are positive integers in order and residuals finite and non-negative."""
    solver, root = case
    _, _, result = run_and_collect(solver, root)
    assert result.history, solver
    numbers = [item.iteration for item in result.history]
    assert numbers[0] == 1 and numbers == sorted(numbers), (solver, numbers)
    assert all(math.isfinite(i.residual) and i.residual >= 0 for i in result.history)


@pytest.mark.parametrize(
    "kwargs",
    [{"iteration": 0, "residual": 0.1}, {"iteration": 1, "residual": -1.0}],
)
def test_c40_the_iteration_type_refuses_zero_based_and_negative_residuals(kwargs):
    """C-40: control, the type that every reader fills refuses what the clause forbids."""
    with pytest.raises(ValueError):
        Iteration(**kwargs)


def test_c42_a_result_that_is_not_failed_must_carry_a_coefficient():
    """C-42: failed may be empty; converged and iteration_limit may not."""
    history = (Iteration(1, 0.1, True),)
    NeutralResult({}, REFS, history, "failed", "unknown")
    for name in ("converged", "iteration_limit"):
        with pytest.raises(ValueError, match="coefficient"):
            NeutralResult({}, REFS, history, name, "unknown")
    column = {"CL": Coefficient(0.1, "z", "force")}
    NeutralResult(column, REFS, history, "converged", "unknown")


def reverse_columns(text: str, solver: str) -> str:
    """Return the table with its columns in the opposite order, header included."""
    if solver == "openfoam":
        lines = []
        for line in text.splitlines():
            if line.startswith("# Time"):
                line = "# " + " ".join(reversed(line[1:].split()))
            elif line.strip() and not line.lstrip().startswith("#"):
                line = " ".join(reversed(line.split()))
            lines.append(line)
        return "\n".join(lines) + "\n"
    if solver == "xfoil":
        lines, out, seen = text.splitlines(), [], False
        for line in lines:
            tokens = line.split()
            if seen and tokens and not set(line.strip()) <= {"-", " "}:
                line = "  ".join(reversed(tokens))
            elif tokens and tokens[0] == "alpha":
                seen, line = True, "  ".join(reversed(tokens))
            out.append(line)
        return "\n".join(out) + "\n"
    rows = list(csv.reader(io.StringIO(text)))
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow(list(reversed(row)))
    return buffer.getvalue()


TABLES = {
    "su2": "history.csv",
    "pyfr": "forces.csv",
    "xfoil": "polar.txt",
    "mock": "history.csv",
    "openfoam": "postProcessing/forceCoeffs/0/coefficient.dat",
}


@pytest.mark.parametrize("solver_name", sorted(TABLES))
def test_c41_tables_are_read_by_column_name_not_position(solver_name, tmp_path):
    """C-41: reversing the column order of the output table leaves the result unchanged."""
    from test_conformance_workspace import build, executables

    from cfdbender.core.adapter import load_point
    from cfdbender.solvers import load_solver
    from cfdbender.workspace.planning import plan_workspace

    root = build(tmp_path, solver_name)
    setup = root / "inputs/setups/s001.toml"
    setup.write_bytes(setup.read_bytes() + executables(root, solver_name).encode("ascii"))
    if solver_name == "xfoil":
        matrix = root / "inputs/matrices/full.fs"
        matrix.write_text(matrix.read_text(encoding="ascii").replace("MACH:0.1", "MACH:0"))
    adapter = load_solver(solver_name)
    point = load_point(root, plan_workspace(root, "full").points[0], solver=solver_name)
    adapter.execute(point, check_ram=lambda: None)
    before = adapter.collect(point)
    table = root / point.point.directory / TABLES[solver_name]
    original = table.read_text(encoding="ascii")
    reordered = reverse_columns(original, solver_name)
    assert reordered != original
    table.write_text(reordered, encoding="ascii")
    after = adapter.collect(point)
    assert after.coefficients == before.coefficients, solver_name
    assert after.classification == before.classification, solver_name
