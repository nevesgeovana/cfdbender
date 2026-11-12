"""Reusable pytest assertions for independently prepared adapter conformance cases.

Import these helpers from adapter tests. Prepare a fresh AdapterPoint fixture per
helper; assert_classifications takes three independently prepared cases keyed by
converged, iteration_limit and failed. assert_validation takes a deliberately
invalid MatrixRow/RowInputs pair and the problem its refusal must name. Cases
belong to the adapter's tests: these helpers import no solver and launch none.
They are ordinary assertions and do not require pytest at runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint, SolverAdapter, load_point
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import RowInputs, RunPlan
from cfdbender.core.registry import RunRecord, RunRegistry
from cfdbender.core.results import NeutralResult
from cfdbender.core.run import execute_plan


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _refuses(action: Callable[[], object], rule: str, problem: str = "") -> None:
    try:
        action()
    except (ValueError, OSError) as error:
        assert problem in str(error), f"{rule}: refusal must name {problem!r}: {error}"
    else:
        raise AssertionError(f"{rule}: expected a refusal")


def assert_reads_init(adapter: SolverAdapter, point: AdapterPoint) -> None:
    """Assert translation refuses an invalid workspace solver declaration.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    point : AdapterPoint
        Fresh valid case. Its init.toml is temporarily changed and restored.
    """
    path = point.root / "init.toml"
    original = path.read_bytes()
    try:
        path.write_bytes(b"solver = []")
        _refuses(lambda: adapter.translate(point), "reads_init", "init.toml")
    finally:
        path.write_bytes(original)


def assert_validation(
    adapter: SolverAdapter, row: MatrixRow, workspace: RowInputs, problem: str
) -> None:
    """Assert a bad row is refused with a message naming its actual problem.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    row : MatrixRow
        Prepared invalid row.
    workspace : RowInputs
        That row's tracked invalid input files.
    problem : str
        Required problem name, supplied independently from the requirement.
    """
    before = _snapshot(workspace.root)
    _refuses(lambda: adapter.validate_row(row, workspace), "validation", problem)
    assert _snapshot(workspace.root) == before, "validation: hook must not write"


def assert_deterministic(adapter: SolverAdapter, point: AdapterPoint) -> None:
    """Assert repeated translation is nonempty identical ASCII without writes.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    point : AdapterPoint
        Fresh valid case.
    """
    before = _snapshot(point.root)
    first = adapter.translate(point)
    assert first and first.isascii(), "deterministic: require nonempty ASCII input"
    assert first == adapter.translate(point), "deterministic: translation changed"
    assert before == _snapshot(point.root), "deterministic: translation must not write"


def assert_classifications(adapter: SolverAdapter, cases: Mapping[str, AdapterPoint]) -> None:
    """Assert execution and collection reach all three prepared numerical outcomes.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    cases : Mapping of str to AdapterPoint
        Fresh cases keyed by converged, iteration_limit and failed.
    """
    assert set(cases) == {"converged", "iteration_limit", "failed"}, "classifications: cases"
    for expected, point in cases.items():
        checks: list[None] = []

        def check_ram(checks: list[None] = checks) -> None:
            checks.append(None)

        result = adapter.execute(point, check_ram=check_ram)
        assert len(checks) >= 2, "classifications: check RAM before and during execution"
        assert adapter.classify(result) == expected, f"classifications: expected {expected}"
        collected = adapter.collect(point)
        assert collected.classification == expected, f"classifications: collect expected {expected}"
        assert collected.history == result.history, "classifications: saved history differs"


def assert_exit_evidence(adapter: SolverAdapter, point: AdapterPoint) -> None:
    """Assert zero exit without history fails and nonzero exit cannot converge.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    point : AdapterPoint
        Fresh prepared converged case.
    """
    result = adapter.execute(point, check_ram=lambda: None)
    assert adapter.classify(replace(result, exit_code=0, history=())) == "failed", (
        "exit_evidence: successful exit alone is not convergence"
    )
    assert adapter.classify(replace(result, exit_code=1)) == "failed", (
        "exit_evidence: nonzero process exit cannot converge"
    )


def assert_results(adapter: SolverAdapter, point: AdapterPoint) -> None:
    """Assert collected columns carry explicit axes, references, history and version.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    point : AdapterPoint
        Fresh prepared converged case.
    """
    adapter.execute(point, check_ram=lambda: None)
    result = adapter.collect(point)
    _assert_neutral(result, "results")


def _assert_neutral(result: NeutralResult, rule: str) -> None:
    refs = result.references
    assert refs.axis_system and len(refs.axis_directions) == 3 and all(refs.axis_directions), (
        f"{rule}: axis system and positive directions required"
    )
    assert math.isfinite(refs.area) and refs.area > 0, f"{rule}: reference area"
    assert math.isfinite(refs.length) and refs.length > 0, f"{rule}: reference length"
    assert len(refs.moment_origin) == 3 and all(map(math.isfinite, refs.moment_origin)), (
        f"{rule}: moment origin"
    )
    assert result.coefficients, f"{rule}: no coefficient columns"
    for name, column in result.coefficients.items():
        assert name and column.axis in ("x", "y", "z"), f"{rule}: coefficient axis missing"
        assert column.kind in ("force", "moment"), f"{rule}: coefficient kind missing"
        assert math.isfinite(column.value), f"{rule}: coefficient value"
    assert result.history and result.solver_version, f"{rule}: history and solver version"


def assert_no_overwrite(adapter: SolverAdapter, point: AdapterPoint) -> None:
    """Assert a second execution refuses and preserves every first-run byte.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    point : AdapterPoint
        Fresh valid case, executed exactly once by this assertion.
    """
    adapter.execute(point, check_ram=lambda: None)
    before = _snapshot(point.root)
    _refuses(lambda: adapter.execute(point, check_ram=lambda: None), "no_overwrite")
    assert _snapshot(point.root) == before, "no_overwrite: first run changed"


def assert_execution_isolation(adapter: SolverAdapter, point: AdapterPoint) -> None:
    """Assert execution changes only files in the fresh point directory.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    point : AdapterPoint
        Fresh valid case with no point output directory.
    """
    before = _snapshot(point.root)
    adapter.execute(point, check_ram=lambda: None)
    after = _snapshot(point.root)
    prefix = point.point.directory.rstrip("/") + "/"
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    outside = sorted(name for name in changed if not name.startswith(prefix))
    assert not outside, f"execution_isolation: changed files outside point: {outside}"


# Runner injection lets an adapter exercise the same assertions through its
# own composition layer, and lets the conformance tests falsify each rule.
Runner = Callable[..., tuple[RunRecord, ...]]


def assert_run_record(
    adapter: SolverAdapter,
    plan: RunPlan,
    *,
    runner: Runner = execute_plan,
) -> None:
    """Assert execution updates records, keeping input digests and timestamp order.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    plan : RunPlan
        Fresh valid converged plan prepared independently by the adapter's tests.
    runner : callable, optional
        Plan runner accepting adapter and check_ram, defaulting to execute_plan.
    """
    records = runner(plan, adapter, check_ram=lambda: None)
    assert len(records) == len(plan.points) > 0, "run_record: one record per point"
    for record, point in zip(records, plan.points, strict=True):
        assert record.status == "COMPLETED", "run_record: execution did not update status"
        assert record.point == point, "run_record: proposal or input digests changed"
        assert record.matrix == plan.matrix and record.solver == plan.solver, "run_record: identity"
        assert record.classification == "converged", "run_record: numerical classification"
        assert record.started_at and record.completed_at, "run_record: execution timestamps absent"
        stamps = [
            datetime.fromisoformat(value)
            for value in (
                record.created_at,
                record.started_at,
                record.completed_at,
            )
        ]
        assert all(stamp.utcoffset() is not None for stamp in stamps), "run_record: timezone"
        assert stamps == sorted(stamps), "run_record: timestamp order"
        assert set(record.outputs_sha256) == set(point.outputs), "run_record: output digests"
    assert RunRegistry(plan.root).read() == records, "run_record: update was not persisted"


def assert_finished_no_overwrite(
    adapter: SolverAdapter,
    plan: RunPlan,
    *,
    runner: Runner = execute_plan,
) -> None:
    """Assert a finished-point second run refuses before changing any first-run bytes.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    plan : RunPlan
        Fresh valid plan, executed once by this assertion.
    runner : callable, optional
        Plan runner accepting adapter and check_ram, defaulting to execute_plan.
    """
    runner(plan, adapter, check_ram=lambda: None)
    before = _snapshot(plan.root)
    _refuses(lambda: runner(plan, adapter, check_ram=lambda: None), "finished_no_overwrite")
    assert _snapshot(plan.root) == before, "finished_no_overwrite: first-run evidence changed"


def assert_collected_results(
    adapter: SolverAdapter,
    plan: RunPlan,
    *,
    runner: Runner = execute_plan,
) -> None:
    """Assert recorded points collect neutral results with explicit axes and references.

    Parameters
    ----------
    adapter : SolverAdapter
        Adapter under test.
    plan : RunPlan
        Fresh valid converged plan, executed once by this assertion.
    runner : callable, optional
        Plan runner accepting adapter and check_ram, defaulting to execute_plan.
    """
    records = runner(plan, adapter, check_ram=lambda: None)
    assert len(records) == len(plan.points) > 0, "collected_results: one record per point"
    before = _snapshot(plan.root)
    for record in records:
        point = load_point(plan.root, record.point, solver=plan.solver)
        result = adapter.collect(point)
        _assert_neutral(result, "collected_results")
        assert result.classification == record.classification, "collected_results: classification"
    assert before == _snapshot(plan.root), "collected_results: reading results must not write"
