"""The per-point convergence rule of a polar, R or C, for SU2 and OpenFOAM.

A point is converged when EITHER

* (R) the residual reaches its target: SU2's ``rms[Rho]`` (a log10 value) at or below
  -12 on some row of ``history.csv``, or every solved field's initial residual at or
  below 1e-6 on one iteration of the OpenFOAM solver log; OR
* (C) CL and CD both pass a Cauchy test on the same row: with
  ``d_n = |f_n - f_(n-1)| / |f_n|`` (SU2's own Cauchy monitor), the mean of ``d`` over
  the last 250 iterations is below 1e-6, AND on that row the residual lies at least 3
  orders of magnitude below its first value, so that a frozen state cannot pass.

The criterion that holds first decides; R wins a tie. OpenFOAM's residual of a field
in one iteration is the FIRST "Solving for <field>, Initial residual" after the
"Time = n" line (the one residualControl tests; later pressure correctors are not
used), and its forces are the Cl and Cd columns of ``forceCoeffs``.

The rule is applied beside the adapter's own classification, which it never changes.
Caution: a solution frozen in a non-physical state, or a two-state limit cycle whose
forces repeat to seven digits, also passes C; the forces and the residual level are
therefore reported beside each verdict.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: SU2's R target, the log10 of rms[Rho].
RMS_TARGET = -12.0
#: OpenFOAM's R target, every field's initial residual.
RESIDUAL_TARGET = 1e-6
#: The Cauchy window, in iterations.
CAUCHY_ELEMS = 250
#: The Cauchy tolerance on the mean relative change of CL and of CD.
CAUCHY_EPS = 1e-6
#: The residual drop below the first value that C requires, in orders of magnitude.
MIN_DROP = 3.0
#: The solvers the rule reads.
SOLVERS = ("su2", "openfoam")

_RESIDUAL = re.compile(r"Solving for (\w+), Initial residual = ([^,\s]+)")


@dataclass(frozen=True)
class Verdict:
    """The rule's verdict on one point.

    ``criterion`` is ``"R"``, ``"C"`` or None (not converged); ``iteration`` is the
    iteration at which the deciding criterion first held. ``first_iteration_rms`` and
    ``first_iteration_cauchy`` are where R and C first held (None when never).
    ``drop_orders_last`` is the residual drop below its first value at the last row,
    in orders of magnitude (the smallest over the fields for OpenFOAM). ``last`` holds
    the last row's residual and forces.
    """

    criterion: str | None
    iteration: int | float | None = None
    first_iteration_rms: int | float | None = None
    first_iteration_cauchy: int | float | None = None
    drop_orders_last: float | None = None
    rows: int = 0
    last: Mapping[str, object] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """Return True when R or C holds."""
        return self.criterion is not None

    def describe(self) -> str:
        """Return the measured state in one line."""
        if not self.rows:
            return "empty history"
        drop = f"{self.drop_orders_last:.2f} orders below the first value at the last row"
        if self.criterion is not None:
            return f"{self.criterion} at iteration {self.iteration}, {drop}"
        state = ", ".join(f"{key} {value:.6g}" for key, value in _numbers(self.last).items())
        return f"R not reached, C not met after {self.rows} rows ({state}); {drop}"


def _numbers(values: Mapping[str, object]) -> dict[str, float]:
    found: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, int | float):
            found[key] = float(value)
        elif isinstance(value, Mapping):
            found.update({f"{key}[{k}]": float(v) for k, v in value.items()})
    return found


def cauchy_passes(values: Sequence[float], elems: int, eps: float) -> list[bool]:
    """Return, per row, whether the mean relative change over the last ``elems`` is below eps.

    Parameters
    ----------
    values : sequence of float
        One coefficient per row.
    elems : int
        The window, in rows.
    eps : float
        The tolerance on the mean relative change.

    Returns
    -------
    list of bool
        False for the first ``elems`` rows, whose window is not full.
    """
    change = [0.0] * len(values)
    for n in range(1, len(values)):
        change[n] = abs(values[n] - values[n - 1]) / abs(values[n]) if values[n] else math.inf
    passed = [False] * len(values)
    total = 0.0
    nonfinite = 0
    for n in range(1, len(values)):
        if math.isfinite(change[n]):
            total += change[n]
        else:
            nonfinite += 1
        if n > elems:
            if math.isfinite(change[n - elems]):
                total -= change[n - elems]
            else:
                nonfinite -= 1
        if n >= elems:
            passed[n] = nonfinite == 0 and total / elems < eps
    return passed


def read_su2_history(path: Path) -> list[dict[str, float]]:
    """Return the rows of an SU2 history.csv, column names stripped of quotes and blanks.

    Rows shorter than the header or holding a cell that is not a number are skipped.

    Raises
    ------
    ValueError
        The file cannot be read.
    """
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = [name.strip().strip('"').strip() for name in next(reader)]
            rows = []
            for cells in reader:
                if len(cells) < len(header):
                    continue
                try:
                    rows.append(
                        {name: float(cell) for name, cell in zip(header, cells, strict=False)}
                    )
                except ValueError:
                    continue
    except (OSError, StopIteration, UnicodeError) as error:
        raise ValueError(f"{path.name}: cannot read the SU2 history: {error}") from error
    return rows


def _decide(
    first_r: int | None, first_c: int | None, at: Sequence[int | float]
) -> tuple[str | None, int | float | None]:
    if first_r is None and first_c is None:
        return None, None
    if first_c is None or (first_r is not None and first_r <= first_c):
        return "R", at[first_r] if first_r is not None else None
    return "C", at[first_c]


def judge_su2(
    rows: Sequence[Mapping[str, float]],
    *,
    rms: float = RMS_TARGET,
    elems: int = CAUCHY_ELEMS,
    eps: float = CAUCHY_EPS,
    min_drop: float = MIN_DROP,
) -> Verdict:
    """Apply the rule to SU2 history rows (``Inner_Iter``, ``rms[Rho]``, ``CL``, ``CD``).

    Raises
    ------
    ValueError
        A row lacks rms[Rho], CL or CD.
    """
    if not rows:
        return Verdict(None)
    try:
        first = rows[0]["rms[Rho]"]
        drop = [
            first - row["rms[Rho]"]
            if math.isfinite(first) and math.isfinite(row["rms[Rho]"])
            else -math.inf
            for row in rows
        ]
        lift = cauchy_passes([row["CL"] for row in rows], elems, eps)
        drag = cauchy_passes([row["CD"] for row in rows], elems, eps)
    except KeyError as error:
        raise ValueError(f"the SU2 history has no {error.args[0]} column") from error
    at = [int(row.get("Inner_Iter", n)) for n, row in enumerate(rows)]
    first_r = next(
        (
            n
            for n, row in enumerate(rows)
            if math.isfinite(row["rms[Rho]"]) and row["rms[Rho]"] <= rms
        ),
        None,
    )
    first_c = next(
        (n for n in range(len(rows)) if lift[n] and drag[n] and drop[n] >= min_drop), None
    )
    criterion, iteration = _decide(first_r, first_c, at)
    last = rows[-1]
    return Verdict(
        criterion,
        iteration,
        None if first_r is None else at[first_r],
        None if first_c is None else at[first_c],
        round(drop[-1], 3),
        len(rows),
        {key: last[key] for key in ("rms[Rho]", "CL", "CD")},
    )


@dataclass(frozen=True)
class FoamRow:
    """One OpenFOAM iteration: the first initial residual of each field, CL and CD."""

    iteration: int | float
    residuals: Mapping[str, float]
    lift: float
    drag: float


def _solver_log(folder: Path) -> Path:
    for name in ("log.simpleFoam", "log.pimpleFoam"):
        if (folder / name).is_file():
            return folder / name
    raise ValueError(f"{folder.name}: no log.simpleFoam or log.pimpleFoam to judge")


def read_openfoam(folder: Path) -> list[FoamRow]:
    """Return OpenFOAM residual iterations, with NaN forces where no sample exists.

    Raises
    ------
    ValueError
        The solver log or forceCoeffs coefficient.dat is missing or unreadable.
    """
    residuals: dict[float, dict[str, float]] = {}
    current: float | None = None
    with _solver_log(folder).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("Time = "):
                try:
                    current = float(line.split("=", 1)[1])
                    if not math.isfinite(current):
                        current = None
                except ValueError:
                    current = None
                continue
            found = _RESIDUAL.search(line)
            if found and current is not None:
                try:
                    value = float(found[2])
                except ValueError:
                    value = math.nan
                residuals.setdefault(current, {}).setdefault(found[1], value)
    paths = sorted(folder.glob("postProcessing/forceCoeffs/*/coefficient.dat"))
    if not paths:
        raise ValueError(f"{folder.name}: no forceCoeffs coefficient.dat to judge")
    header: list[str] = []
    forces: dict[float, tuple[float, float]] = {}
    for line in paths[0].read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.lstrip().removeprefix("#").split()
        if line.lstrip().startswith("#"):
            if "Time" in tokens:
                if len(set(tokens)) != len(tokens) or not {"Time", "Cl", "Cd"} <= set(tokens):
                    raise ValueError("forceCoeffs needs unique Time, Cl and Cd headers")
                header = tokens
            continue
        if not header or not tokens:
            continue
        try:
            forces[float(tokens[header.index("Time")])] = (
                float(tokens[header.index("Cl")]),
                float(tokens[header.index("Cd")]),
            )
        except (ValueError, IndexError):
            continue
    return [
        FoamRow(int(n) if n.is_integer() else n, residuals[n], *forces.get(n, (math.nan, math.nan)))
        for n in sorted(residuals)
    ]


def judge_openfoam(
    rows: Sequence[FoamRow],
    *,
    target: float = RESIDUAL_TARGET,
    elems: int = CAUCHY_ELEMS,
    eps: float = CAUCHY_EPS,
    min_drop: float = MIN_DROP,
) -> Verdict:
    """Apply the rule to OpenFOAM iterations (initial residuals, CL and CD)."""
    if not rows:
        return Verdict(None)
    fields = sorted({name for row in rows for name in row.residuals})
    first = {
        name: next(row.residuals[name] for row in rows if name in row.residuals) for name in fields
    }

    def drop(row: FoamRow) -> float:
        if not fields or any(name not in row.residuals for name in fields):
            return -math.inf
        return min(
            math.log10(first[name] / row.residuals[name])
            if math.isfinite(first[name])
            and math.isfinite(row.residuals[name])
            and first[name] > 0
            and row.residuals[name] > 0
            else (
                math.inf
                if math.isfinite(first[name]) and first[name] > 0 and row.residuals[name] == 0
                else -math.inf
            )
            for name in fields
        )

    first_r = next(
        (
            n
            for n, row in enumerate(rows)
            if fields
            and all(
                name in row.residuals
                and math.isfinite(row.residuals[name])
                and 0 <= row.residuals[name] <= target
                for name in fields
            )
        ),
        None,
    )
    lift = cauchy_passes([row.lift for row in rows], elems, eps)
    drag = cauchy_passes([row.drag for row in rows], elems, eps)
    first_c = next(
        (n for n in range(len(rows)) if lift[n] and drag[n] and drop(rows[n]) >= min_drop),
        None,
    )
    at = [row.iteration for row in rows]
    criterion, iteration = _decide(first_r, first_c, at)
    last = rows[-1]
    return Verdict(
        criterion,
        iteration,
        None if first_r is None else at[first_r],
        None if first_c is None else at[first_c],
        round(drop(last), 3),
        len(rows),
        {"residual": dict(last.residuals), "CL": last.lift, "CD": last.drag},
    )


def judge_point(folder: Path, solver: str) -> Verdict:
    """Apply the rule, with the default parameters, to one point folder.

    Parameters
    ----------
    folder : Path
        The point's datapoint folder.
    solver : str
        ``su2`` or ``openfoam``.

    Raises
    ------
    ValueError
        The solver has no convergence rule, or the point's files cannot be read.
    """
    if solver == "su2":
        return judge_su2(read_su2_history(folder / "history.csv"))
    if solver == "openfoam":
        return judge_openfoam(read_openfoam(folder))
    raise ValueError(f"{solver}: no convergence rule for this solver")
