"""Residual file, force file and log readers, and the classification they feed."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from cfdpp_support import builder

from cfdbender.core.results import ExecutionResult, Iteration, References
from cfdbender.solvers.cfdpp import _forces, _history

B = builder()
REFS = References(
    B.KNOWN["area"], B.KNOWN["length"], B.KNOWN["origin"], "geometry", ("a", "b", "c")
)


def _series(steps: list[_history.StepForces]) -> dict[str, list[float]]:
    walls = tuple(sorted(steps[0].selectors))
    return _history.lift_drag_series(steps, walls, B.KNOWN["dynamic_pressure"], REFS, 0.0, 0.0, 3)


# --- info0 ---------------------------------------------------------------


def test_info0_reads_eight_columns_per_step() -> None:
    steps = _history.parse_info0(B.info0_converged_text())
    assert len(steps) == 121
    assert steps[0].step == 1 and steps[-1].step == 121
    assert steps[0].residual_avg == 1.0
    assert steps[0].residual_max == pytest.approx(10.0)
    assert steps[0].eigen_max == 100.0


def test_info0_skips_comments_and_blank_lines() -> None:
    text = "# header\n\n" + B.info0_text([1.0, 0.5])
    assert [s.step for s in _history.parse_info0(text)] == [1, 2]


def test_info0_refuses_a_wrong_column_count_naming_the_line() -> None:
    text = B.info0_text([1.0, 0.5]) + "3 0.0 1.0 0.1\n"
    with pytest.raises(_history.HistoryError, match="line 3.*8 columns"):
        _history.parse_info0(text)


@pytest.mark.parametrize("steps", [[2, 1], [1, 1], [0, 1]])
def test_info0_refuses_a_step_column_that_is_not_increasing_from_one(steps: list[int]) -> None:
    rows = [f"{s} 0.0 1.0 {1.0 / s if s else 1.0} 1.0 1e15 5.0 100.0" for s in steps]
    with pytest.raises(_history.HistoryError, match="step"):
        _history.parse_info0("\n".join(rows) + "\n")


def test_info0_refuses_a_non_finite_residual() -> None:
    with pytest.raises(_history.HistoryError, match="line 3.*finite"):
        _history.parse_info0(B.info0_nan_text())


def test_info0_refuses_an_empty_file() -> None:
    with pytest.raises(_history.HistoryError, match="no step"):
        _history.parse_info0("# only a comment\n")


# --- info1 ---------------------------------------------------------------


def test_info1_reads_steps_selectors_and_both_tables() -> None:
    steps = _history.parse_info1(B.info1_known_text(1, 3))
    assert [s.step for s in steps] == [1, 2, 3]
    assert sorted(steps[0].selectors) == [1, 2]
    wing = steps[0].selectors[1]
    assert wing.faces == 2
    assert wing.dimensional["force_z"].total == pytest.approx(450.0)
    assert wing.dimensional["force_x"].inviscid == pytest.approx(6.0)
    assert wing.dimensional["force_x"].viscous == pytest.approx(2.0)
    assert wing.nondimensional["force_z"].total == pytest.approx(0.45)
    assert wing.areas == (1.0, 0.0, 0.0, 1.0)
    assert wing.area_moments == (0.0, 0.0, 0.0)


def test_info1_refuses_reference_scales_other_than_one() -> None:
    text = B.info1_text(
        {1: B.known_selectors()}, scales=(1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0)
    )
    with pytest.raises(_history.HistoryError, match="scales"):
        _history.parse_info1(text)


def test_info1_refuses_a_missing_row() -> None:
    kept = [
        line for line in B.info1_known_text(1, 1).splitlines() if not line.startswith("moment_z")
    ]
    text = chr(10).join(kept) + chr(10)
    with pytest.raises(_history.HistoryError, match="moment_z"):
        _history.parse_info1(text)


def test_info1_refuses_text_that_holds_no_step() -> None:
    with pytest.raises(_history.HistoryError, match="no step"):
        _history.parse_info1("scales " + "1 " * 10 + "\n")


def test_step_forces_refuses_an_unknown_selector() -> None:
    step = _history.parse_info1(B.info1_known_text(1, 1))[0]
    with pytest.raises(_history.HistoryError, match="selector 9"):
        _history.step_forces(step, (9,))


def test_force_history_cl_cd_are_the_known_values_at_every_step() -> None:
    steps = _history.parse_info1(B.info1_known_text(1, 3))
    series = _series(steps)
    assert series["CL"] == pytest.approx([0.5] * 3)
    assert series["CD"] == pytest.approx([0.01] * 3)


def test_last_step_mismatch_between_the_two_files_fails() -> None:
    info0 = _history.parse_info0(B.info0_text([1.0, 0.5, 0.25]))
    info1 = _history.parse_info1(B.info1_known_text(1, 2))
    with pytest.raises(_history.HistoryError, match="last step"):
        _history.check_last_step(info0, info1)
    _history.check_last_step(info0, _history.parse_info1(B.info1_known_text(1, 3)))


# --- R and C -------------------------------------------------------------


def _first_settled_step(
    cl: list[float], cd: list[float], window: int = 250, eps: float = 1e-6
) -> int | None:
    """One-based step where both series have a mean relative change below eps over the window."""

    def mean_change(values: list[float], end: int) -> float:
        span = range(end - window + 1, end + 1)
        return sum(abs(values[i] - values[i - 1]) / abs(values[i]) for i in span) / window

    for end in range(window, len(cl)):
        if mean_change(cl, end) < eps and mean_change(cd, end) < eps:
            return end + 1
    return None


def test_r_holds_at_the_first_step_with_the_requested_drop() -> None:
    steps = _history.parse_info0(B.info0_converged_text())
    assert _history.r_step(steps, 9.55) == 97  # one decade per ten steps: 9.6 orders at step 97
    assert _history.r_step(steps, 12.5) is None


def test_r_measures_the_drop_from_the_run_maximum() -> None:
    steps = _history.parse_info0(B.info0_text([1.0, 100.0, 50.0, 1.0, 0.5]))
    assert _history.r_step(steps, 2.0) == 4  # 100 / 1 = two orders, not 1 / 1


def test_r_does_not_hold_on_a_plateau() -> None:
    assert _history.r_step(_history.parse_info0(B.info0_limit_text()), 3.0) is None


def test_c_holds_at_the_right_step_once_residual_has_dropped_three_orders() -> None:
    forces = _history.parse_info1(B.info1_cauchy_text(400))
    series = _series(forces)
    steps = _history.parse_info0(B.info0_slow_text(400))
    step = _history.c_step(steps, series["CL"], series["CD"])
    assert step == _first_settled_step(series["CL"], series["CD"])
    assert step is not None and step > 151  # the residual is three orders down from step 151


def test_c_does_not_hold_without_the_three_order_drop() -> None:
    forces = _history.parse_info1(B.info1_cauchy_text(400))
    series = _series(forces)
    plateau = _history.parse_info0(B.info0_text([10.0 ** (-0.1 * min(n, 20)) for n in range(400)]))
    assert _history.c_step(plateau, series["CL"], series["CD"]) is None


def test_c_does_not_hold_while_the_forces_still_move() -> None:
    steps = _history.parse_info0(B.info0_slow_text(400))
    moving = [0.5 + 0.001 * n for n in range(400)]
    assert _history.c_step(steps, moving, moving) is None


def test_c_refuses_series_of_the_wrong_length() -> None:
    steps = _history.parse_info0(B.info0_slow_text(400))
    with pytest.raises(_history.HistoryError, match="length"):
        _history.c_step(steps, [0.5] * 10, [0.5] * 10)


# --- log -----------------------------------------------------------------


def test_scan_log_reads_the_version_and_finds_no_failure() -> None:
    facts = _history.scan_log(B.log_text("22.5.U3"))
    assert facts.version == "22.5.U3"
    assert facts.failures == ()
    assert facts.redo_events == 0


def test_scan_log_without_a_banner_version_is_unknown() -> None:
    assert _history.scan_log(B.log_text(version=None)).version == "unknown"


def test_scan_log_counts_step_redo_events() -> None:
    assert _history.scan_log(B.log_text(redo_events=3)).redo_events == 3


@pytest.mark.parametrize("kind", B.LOG_FAIL_KINDS)
def test_scan_log_names_each_failure_fragment(kind: str) -> None:
    assert _history.scan_log(B.log_text(fail=kind)).failures == (kind,)


def test_every_failure_fragment_is_covered_by_a_fixture() -> None:
    assert set(_history.LOG_FAILURE_FRAGMENTS) == set(B.LOG_FAIL_KINDS)


# --- evidence and classification ----------------------------------------


def _evidence(
    info0: str, *, info1: str | None = None, log: str | None = None, **kw: object
) -> _history.Evidence:
    forces = _history.parse_info1(info1) if info1 else None
    series = _series(forces) if forces else None
    options = {"exit_code": 0, "iteration_limit": 1000, "convergence_orders": 9.55}
    options.update(kw)
    return _history.build_evidence(info0, info1, log or B.log_text(), series=series, **options)  # type: ignore[arg-type]


def test_converged_by_the_residual_drop() -> None:
    evidence = _evidence(B.info0_converged_text())
    assert evidence.history[-1].converged is True
    assert evidence.criterion == "R"
    assert evidence.drop_orders == pytest.approx(12.0)
    assert evidence.solver_version == "22.5.U3"
    assert _history.classify(evidence) == "converged"


def test_converged_by_the_force_cauchy_test_when_the_residual_stalls_three_orders_down() -> None:
    evidence = _evidence(
        B.info0_slow_text(400),
        info1=B.info1_cauchy_text(400),
        convergence_orders=20.0,
        iteration_limit=1000,
    )
    assert evidence.criterion == "C"
    assert evidence.history[-1].converged is True
    assert _history.classify(evidence) == "converged"


def test_plateau_below_the_target_reaches_the_iteration_limit() -> None:
    evidence = _evidence(B.info0_limit_text(), iteration_limit=40)
    assert evidence.history[-1].converged is False
    assert _history.classify(evidence) == "iteration_limit"


def test_early_stop_without_evidence_is_failed() -> None:
    evidence = _evidence(B.info0_limit_text(), iteration_limit=1000)
    assert _history.classify(evidence) == "failed"


def test_iteration_limit_counts_from_the_first_step_of_the_run() -> None:
    # a restart run: steps 101..140, limit 40 steps of this run
    evidence = _evidence(B.info0_text([0.5] * 40, first_step=101), iteration_limit=40)
    assert _history.classify(evidence) == "iteration_limit"


def test_a_failure_fragment_in_the_log_fails_a_run_that_looks_converged() -> None:
    evidence = _evidence(B.info0_converged_text(), log=B.log_text(fail="negative_volume"))
    assert evidence.evidence_failed is True
    assert evidence.failures == ("negative_volume",)
    assert evidence.history[-1].converged is False
    assert _history.classify(evidence) == "failed"


def test_a_non_zero_exit_code_fails_the_run() -> None:
    assert _history.classify(_evidence(B.info0_converged_text(), exit_code=1)) == "failed"


def test_non_finite_residual_gives_failed_evidence_with_the_problem_stated() -> None:
    evidence = _evidence(B.info0_nan_text())
    assert evidence.history == ()
    assert evidence.evidence_failed is True
    assert "finite" in evidence.problem
    assert _history.classify(evidence) == "failed"


def test_force_file_stale_against_the_residual_file_fails() -> None:
    evidence = _evidence(B.info0_text([1.0, 0.5, 0.25, 0.1]), info1=B.info1_known_text(1, 2))
    assert evidence.evidence_failed is True
    assert "last step" in evidence.problem


def test_classify_with_no_history_is_failed() -> None:
    bare = _history.Evidence(0, (), 10, "unknown")
    assert _history.classify(bare) == "failed"


def test_classify_reads_only_the_execution_result_fields() -> None:
    result = ExecutionResult(0, (Iteration(1, 1.0), Iteration(2, 0.1, True)), 10, "22.5.U3")
    assert _history.classify(_history.Evidence(*_fields(result))) == "converged"


def _fields(result: ExecutionResult) -> tuple[int, tuple[Iteration, ...], int, str]:
    return result.exit_code, result.history, result.iteration_limit, result.solver_version


# --- reading a folder ----------------------------------------------------


def test_read_evidence_reads_the_three_files_of_a_point_folder(tmp_path: Path) -> None:
    (tmp_path / _history.INFO0_NAME).write_text(B.info0_converged_text(), encoding="ascii")
    (tmp_path / _history.INFO1_NAME).write_text(B.info1_known_text(1, 121), encoding="ascii")
    (tmp_path / _history.LOG_NAME).write_text(B.log_text(), encoding="ascii")
    evidence = _history.read_evidence(
        tmp_path, exit_code=0, iteration_limit=1000, convergence_orders=9.55
    )
    assert _history.classify(evidence) == "converged"


def test_read_evidence_without_a_residual_file_is_failed(tmp_path: Path) -> None:
    evidence = _history.read_evidence(
        tmp_path, exit_code=0, iteration_limit=10, convergence_orders=3.0
    )
    assert evidence.history == () and "mcfd.info0" in evidence.problem
    assert _history.classify(evidence) == "failed"


def test_force_history_reads_the_total_forces_per_step_from_a_folder(tmp_path: Path) -> None:
    (tmp_path / _history.INFO1_NAME).write_text(B.info1_known_text(1, 2), encoding="ascii")
    steps = _history.read_info1(tmp_path)
    total = _history.step_forces(steps[-1], (1, 2))
    assert total == _forces.Forces((10.0, 0.0, 500.0), (0.0, -145.0, 0.0))
    assert math.isfinite(total.force[0])
