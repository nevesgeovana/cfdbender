"""P2D-1: the per-point convergence rule of a 2D polar, R or C, for SU2 and OpenFOAM.

A point is converged when either (R) the residual reaches its target, or (C) CL and CD
both pass a Cauchy test (mean relative change below 1e-6 over the last 250 iterations)
on a row where the residual lies at least 3 orders below its first value. SU2 reads
rms[Rho] of history.csv (target -12, a log10 value); OpenFOAM reads the first initial
residual of every field per iteration of the solver log (target 1e-6) and the forces of
forceCoeffs. The synthetic cases are the five of the rule's self-test (R, C, drifting
forces, a 2-order drop, a frozen history) and their OpenFOAM equivalents; the recorded
cases are reduced copies of real solver outputs.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cfdbender.core import convergence as rule

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/polar2d"


def su2_rows(rms_end: float, drift: float) -> list[dict[str, float]]:
    rows = []
    for n in range(1000):
        settle = 1.0 + drift * n + 0.5 * 0.97**n
        rows.append(
            {
                "Inner_Iter": float(n),
                "rms[Rho]": -4 + (rms_end + 4) * n / 999,
                "CL": settle,
                "CD": 0.01 * settle,
            }
        )
    return rows


FROZEN = [
    {"Inner_Iter": float(n), "rms[Rho]": -4.6 if n == 0 else -4.86, "CL": 4.19, "CD": 1.84}
    for n in range(1000)
]


@pytest.mark.parametrize(
    ("rows", "wanted"),
    [
        (su2_rows(-12.5, 1e-4), "R"),
        (su2_rows(-9.0, 0.0), "C"),
        (su2_rows(-8.0, 1e-4), None),
        (su2_rows(-6.0, 0.0), None),
        (FROZEN, None),
    ],
    ids=["target", "cauchy", "drifting", "two-orders", "frozen"],
)
def test_p0600_su2_self_test_cases(rows, wanted):
    verdict = rule.judge_su2(rows)
    assert verdict.criterion == wanted
    assert verdict.converged is (wanted is not None)


def test_p0600_su2_header_with_quotes_and_blanks(tmp_path):
    path = tmp_path / "history.csv"
    lines = ['"Inner_Iter",  "rms[Rho]"  ,"CL","CD"']
    lines += [
        f"{int(r['Inner_Iter'])},{r['rms[Rho]']},{r['CL']},{r['CD']}" for r in su2_rows(-12.5, 1e-4)
    ]
    path.write_text(chr(10).join(lines) + chr(10), encoding="ascii")
    assert rule.judge_su2(rule.read_su2_history(path)).criterion == "R"


def test_p0600_su2_recorded_1221_meets_c_before_r():
    # Reduced copy of a real JST point (alpha 10): C first holds at 611, R at 712.
    verdict = rule.judge_su2(rule.read_su2_history(FIXTURES / "su2-1221-history.csv"))
    assert verdict.criterion == "C"
    assert verdict.first_iteration_cauchy == 611
    assert verdict.first_iteration_rms == 712
    assert verdict.iteration == 611
    assert verdict.drop_orders_last == pytest.approx(7.837, abs=1e-3)


def test_p0600_su2_recorded_1802_meets_c_without_r():
    # Reduced copy of a real point at alpha 15, cut before rms[Rho] reaches -12.
    verdict = rule.judge_su2(rule.read_su2_history(FIXTURES / "su2-1802-history.csv"))
    assert verdict.criterion == "C"
    assert verdict.first_iteration_cauchy == 653
    assert verdict.first_iteration_rms is None


def test_p0600_drop_is_measured_from_the_first_row():
    rows = su2_rows(-9.0, 0.0)
    verdict = rule.judge_su2(rows)
    assert verdict.drop_orders_last == pytest.approx(5.0, abs=1e-9)
    # A larger required drop than the run gives turns the C verdict off.
    assert rule.judge_su2(rows, min_drop=6.0).criterion is None


def test_p0600_empty_history_is_not_converged():
    verdict = rule.judge_su2([])
    assert verdict.criterion is None and verdict.rows == 0


def openfoam_point(root: Path, residual_end: float | None, drift: float) -> Path:
    point = root
    (point / "postProcessing/forceCoeffs/0").mkdir(parents=True)
    log, coefficients = [], ["# Time\tCd\tCd(f)\tCl\tCmPitch"]
    for n in range(1, 1001):
        if residual_end is None:
            residual = 1.0 if n == 1 else 0.5
            lift = drag = 0.3
        else:
            residual = 10 ** (math.log10(residual_end) * (n - 1) / 999)
            lift = 1.0 + drift * n + 0.5 * 0.97**n
            drag = 0.01 * lift
        log += [
            f"Time = {n}",
            f"smoothSolver:  Solving for Ux, Initial residual = {residual}, Final residual = 0",
            f"GAMG:  Solving for p, Initial residual = {residual}, Final residual = 0",
            f"GAMG:  Solving for p, Initial residual = {residual * 1e-9}, Final residual = 0",
        ]
        coefficients.append(f"{n}\t{drag}\t0\t{lift}\t0")
    (point / "log.simpleFoam").write_text(chr(10).join(log) + chr(10), encoding="ascii")
    (point / "postProcessing/forceCoeffs/0/coefficient.dat").write_text(
        chr(10).join(coefficients) + chr(10), encoding="ascii"
    )
    return point


@pytest.mark.parametrize(
    ("residual_end", "drift", "wanted"),
    [(1e-7, 1e-4, "R"), (1e-5, 0.0, "C"), (1e-2, 0.0, None), (None, 0.0, None)],
    ids=["target", "cauchy", "shallow", "frozen"],
)
def test_p0600_openfoam_self_test_cases(tmp_path, residual_end, drift, wanted):
    rows = rule.read_openfoam(openfoam_point(tmp_path, residual_end, drift))
    assert rule.judge_openfoam(rows).criterion == wanted


def test_p0600_openfoam_uses_the_first_initial_residual_of_each_field(tmp_path):
    rows = rule.read_openfoam(openfoam_point(tmp_path, 1e-7, 1e-4))
    # The second p solve of an iteration (a corrector, 1e-9 times smaller) is not the one
    # residualControl tests: the first one is kept.
    assert rows[0].residuals == {"Ux": 1.0, "p": 1.0}
    assert rows[-1].iteration == 1000


def recorded_2007(root: Path) -> Path:
    """Lay out the reduced copy of a real OpenFOAM point as its datapoint folder."""
    (root / "postProcessing/forceCoeffs/0").mkdir(parents=True)
    (root / "log.simpleFoam").write_bytes((FIXTURES / "openfoam-2007-log.simpleFoam").read_bytes())
    (root / "postProcessing/forceCoeffs/0/coefficient.dat").write_bytes(
        (FIXTURES / "openfoam-2007-coefficient.txt").read_bytes()
    )
    return root


def test_p0600_openfoam_recorded_2007_is_not_converged(tmp_path):
    # Reduced copy of a real alpha 0 point run to 20000 iterations: R never reached.
    folder = recorded_2007(tmp_path)
    verdict = rule.judge_openfoam(rule.read_openfoam(folder))
    assert verdict.criterion is None
    assert verdict.first_iteration_rms is None
    assert verdict.drop_orders_last == pytest.approx(4.84, abs=5e-3)


def test_p0600_judge_point_reads_each_solver_folder(tmp_path):
    folder = tmp_path / "su2"
    folder.mkdir()
    (folder / "history.csv").write_bytes((FIXTURES / "su2-1221-history.csv").read_bytes())
    assert rule.judge_point(folder, "su2").criterion == "C"
    assert rule.judge_point(recorded_2007(tmp_path / "foam"), "openfoam").criterion is None
    with pytest.raises(ValueError, match="history.csv"):
        rule.judge_point(tmp_path / "missing", "su2")
    with pytest.raises(ValueError, match="no convergence rule"):
        rule.judge_point(folder, "xfoil")


def test_p0600_verdict_text_names_the_measured_state():
    text = rule.judge_su2(su2_rows(-8.0, 1e-4)).describe()
    assert "R not reached" in text and "C not met" in text and "orders" in text
    assert rule.judge_su2(su2_rows(-12.5, 1e-4)).describe().startswith("R at iteration")


def test_review_nonfinite_residual_cannot_reach_target(tmp_path):
    rows = [{"rms[Rho]": float("-inf"), "CL": 1.0, "CD": 0.1}]
    assert rule.judge_su2(rows).criterion is None
    (tmp_path / "postProcessing/forceCoeffs/0").mkdir(parents=True)
    (tmp_path / "log.simpleFoam").write_text(
        "Time = 1\nSolving for Ux, Initial residual = 1e-8, Final residual = 0\n"
        "Solving for p, Initial residual = nan, Final residual = 0\n",
        encoding="ascii",
    )
    (tmp_path / "postProcessing/forceCoeffs/0/coefficient.dat").write_text(
        "# Time Cd Cl\n1 0.1 1.0\n", encoding="ascii"
    )
    assert rule.judge_openfoam(rule.read_openfoam(tmp_path)).criterion is None


def test_review_openfoam_residual_target_without_force_sample(tmp_path):
    (tmp_path / "postProcessing/forceCoeffs/0").mkdir(parents=True)
    (tmp_path / "log.simpleFoam").write_text(
        "Time = 1\nSolving for p, Initial residual = 1e-7, Final residual = 0\n"
        "Time = 2\nSolving for p, Initial residual = 1e-3, Final residual = 0\n",
        encoding="ascii",
    )
    (tmp_path / "postProcessing/forceCoeffs/0/coefficient.dat").write_text(
        "# Time Cd Cl\n2 0.1 1.0\n", encoding="ascii"
    )
    verdict = rule.judge_openfoam(rule.read_openfoam(tmp_path))
    assert verdict.criterion == "R"
    assert verdict.iteration == 1


def test_review_fractional_openfoam_times_remain_distinct(tmp_path):
    (tmp_path / "postProcessing/forceCoeffs/0").mkdir(parents=True)
    (tmp_path / "log.pimpleFoam").write_text(
        "Time = 0.1\nSolving for p, Initial residual = 0.1, Final residual = 0\n"
        "Solving for p, Initial residual = 1e-9, Final residual = 0\n"
        "Time = 0.2\nSolving for p, Initial residual = 1e-7, Final residual = 0\n",
        encoding="ascii",
    )
    (tmp_path / "postProcessing/forceCoeffs/0/coefficient.dat").write_text(
        "# Time Cd Cl\n0.1 0.1 1.0\n0.2 0.2 2.0\n", encoding="ascii"
    )
    rows = rule.read_openfoam(tmp_path)
    assert [row.iteration for row in rows] == [0.1, 0.2]
    assert rows[0].residuals["p"] == 0.1
    assert rule.judge_openfoam(rows).iteration == 0.2
