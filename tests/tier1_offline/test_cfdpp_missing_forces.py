"""A successful residual history cannot substitute for the required force output."""

from pathlib import Path

from cfdpp_support import builder

from cfdbender.solvers.cfdpp import _history


def test_missing_force_file_fails_even_when_residuals_converge(tmp_path: Path) -> None:
    data = builder()
    (tmp_path / _history.INFO0_NAME).write_text(data.info0_converged_text(), encoding="ascii")
    (tmp_path / _history.LOG_NAME).write_text(data.log_text(), encoding="ascii")
    evidence = _history.read_evidence(
        tmp_path, exit_code=0, iteration_limit=1000, convergence_orders=6.0
    )
    assert _history.classify(evidence) == "failed"
    assert "mcfd.info1" in evidence.problem
