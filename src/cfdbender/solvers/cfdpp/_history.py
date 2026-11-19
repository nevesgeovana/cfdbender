"""Read the residual file, the force file and the log; build the evidence and classify.

EVERY FILE LAYOUT IN THIS MODULE IS A HYPOTHESIS. No file written by the solver
has been read yet: the layouts below are the documented ones where the
documentation states them and the adapter's own assumption where it does not, and
each assumption is a named constant in the "Layout hypotheses" block. The parsers
read those constants; nothing else in the module spells a layout.

Convergence follows the library's rule, applied to the two files:

* R: the residual has fallen by the target number of orders, measured from the
  largest residual of the run so far (the solver's own stop criterion is a drop
  relative to the run's maximum).
* C: lift and drag both pass the Cauchy test of ``core.convergence`` on a step at
  which the residual is at least three orders below its first value.

The last step is marked converged only when R or C holds there and the log shows
no failure; ``classify`` then depends on the execution result alone.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cfdbender.core.convergence import cauchy_passes
from cfdbender.core.results import Classification, ExecutionResult, Iteration, References
from cfdbender.solvers.cfdpp import _forces

# --- Layout hypotheses -----------------------------------------------------

INFO0_NAME = "mcfd.info0"
INFO1_NAME = "mcfd.info1"
LOG_NAME = "stdout.log"

# Residual file: text, one whitespace-separated line per step, '#' comments allowed.
INFO0_COLUMNS = (
    "step",
    "time",
    "time_step",
    "residual_avg",
    "residual_max",
    "cfl_global",
    "cfl_local",
    "eigen_max",
)
INFO0_FIRST_STEP_MIN = 1  # steps are numbered from one and strictly increase

# Force file: text. A scales line, then per step a step line, and per selector
# two tables and the area rows. Tokens are the assumed spelling.
INFO1_SCALES_KEY = "scales"
INFO1_SCALES_COUNT = 10
INFO1_STEP_KEY = "step"  # step <n> <tau> <time>
INFO1_SELECTOR_KEY = "selector"  # selector <n> faces <count>
INFO1_TABLE_KEY = "table"  # table nondimensional | dimensional
INFO1_TABLES = ("nondimensional", "dimensional")
INFO1_COLUMNS = ("total", "inviscid", "viscous")
INFO1_ROWS = (
    "energy_flux",
    "mass_flux",
    "force_x",
    "force_y",
    "force_z",
    "moment_x",
    "moment_y",
    "moment_z",
)
INFO1_AREAS_KEY = "areas"  # four numbers
INFO1_AREA_MOMENTS_KEY = "areamoments"  # three numbers

# Log: a version line, failure fragments (case-insensitive) and the redo marker.
LOG_VERSION = re.compile(r"^\s*version\s*[:=]\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)
LOG_FAILURE_FRAGMENTS = {
    "mesh_match": re.compile(r"face match", re.IGNORECASE),
    "mesh_match_parallel": re.compile(r"did not match with mesh", re.IGNORECASE),
    "negative_volume": re.compile(r"negative volume", re.IGNORECASE),
    "cfl_floor": re.compile(r"cfl.*below", re.IGNORECASE),
}
LOG_REDO = re.compile(r"redoing step", re.IGNORECASE)
UNKNOWN_VERSION = "unknown"

# Residual-drop and Cauchy convergence thresholds.
CAUCHY_WINDOW = 250
CAUCHY_EPS = 1e-6
CAUCHY_MIN_DROP_ORDERS = 3.0


class HistoryError(ValueError):
    """A result file that cannot be read as the layout the adapter expects."""


# --- Residual file ----------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One line of the residual file."""

    step: int
    time: float
    time_step: float
    residual_avg: float
    residual_max: float
    cfl_global: float
    cfl_local: float
    eigen_max: float


def parse_info0(text: str) -> list[Step]:
    """Parse the residual file: eight columns per step, steps strictly increasing."""
    steps: list[Step] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != len(INFO0_COLUMNS):
            raise HistoryError(
                f"{INFO0_NAME} line {number}: expected {len(INFO0_COLUMNS)} columns, "
                f"found {len(fields)}"
            )
        try:
            step = int(fields[0])
            values = [float(field) for field in fields[1:]]
        except ValueError as error:
            raise HistoryError(f"{INFO0_NAME} line {number}: not a number ({error})") from error
        if not all(math.isfinite(value) for value in values):
            raise HistoryError(f"{INFO0_NAME} line {number}: a value is not finite")
        if values[2] < 0 or values[3] < 0:
            raise HistoryError(f"{INFO0_NAME} line {number}: residuals must be non-negative")
        floor = steps[-1].step + 1 if steps else INFO0_FIRST_STEP_MIN
        if step < floor:
            raise HistoryError(
                f"{INFO0_NAME} line {number}: step {step} does not follow the step column "
                f"(needs at least {floor})"
            )
        steps.append(Step(step, *values))
    if not steps:
        raise HistoryError(f"{INFO0_NAME} holds no step")
    return steps


# --- Force file -------------------------------------------------------------


@dataclass(frozen=True)
class Triple:
    """One table row: the total, inviscid and viscous columns."""

    total: float
    inviscid: float
    viscous: float


Table = dict[str, Triple]


@dataclass(frozen=True)
class SelectorForces:
    """The two tables and the area rows of one boundary selector at one step."""

    faces: int
    nondimensional: Table
    dimensional: Table
    areas: tuple[float, ...]
    area_moments: tuple[float, ...]


@dataclass(frozen=True)
class StepForces:
    """The force records of one step."""

    step: int
    tau: float
    time: float
    selectors: dict[int, SelectorForces]


def _floats(fields: Sequence[str], count: int, where: str) -> list[float]:
    try:
        values = [float(field) for field in fields]
    except ValueError as error:
        raise HistoryError(f"{where}: not a number ({error})") from error
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise HistoryError(f"{where}: expected {count} finite numbers")
    return values


def _int(token: str, where: str) -> int:
    try:
        return int(token)
    except ValueError as error:
        raise HistoryError(f"{where}: {token!r} is not an integer") from error


@dataclass
class _Pending:
    """A selector block while its lines are being read."""

    selector: int
    faces: int
    tables: dict[str, Table]
    areas: tuple[float, ...] = ()
    area_moments: tuple[float, ...] = ()

    def finish(self) -> SelectorForces:
        for name in INFO1_TABLES:
            if name not in self.tables:
                raise HistoryError(f"{INFO1_NAME}: selector {self.selector} lacks the {name} table")
            for row in INFO1_ROWS:
                if row not in self.tables[name]:
                    raise HistoryError(
                        f"{INFO1_NAME}: selector {self.selector} {name} table lacks the row {row}"
                    )
        return SelectorForces(
            self.faces,
            self.tables[INFO1_TABLES[0]],
            self.tables[INFO1_TABLES[1]],
            self.areas,
            self.area_moments,
        )


def parse_info1(text: str) -> list[StepForces]:
    """Parse the force file; the dimensional table needs every reference scale equal to one."""
    scales: list[float] | None = None
    steps: list[StepForces] = []
    pending: _Pending | None = None
    table: Table | None = None

    def close() -> None:
        nonlocal pending
        if pending is not None:
            steps[-1].selectors[pending.selector] = pending.finish()
        pending = None

    for number, raw in enumerate(text.splitlines(), start=1):
        fields = raw.split()
        if not fields or raw.lstrip().startswith("#"):
            continue
        where = f"{INFO1_NAME} line {number}"
        key = fields[0]
        if key == INFO1_SCALES_KEY:
            scales = _floats(fields[1:], INFO1_SCALES_COUNT, where)
            if any(scale != 1.0 for scale in scales):
                raise HistoryError(f"{where}: the reference scales must all be one, found {scales}")
        elif key == INFO1_STEP_KEY:
            close()
            if len(fields) != 4:
                raise HistoryError(f"{where}: a step line needs the step, tau and time")
            tau, time = _floats(fields[2:], 2, where)
            steps.append(StepForces(_int(fields[1], where), tau, time, {}))
        elif key == INFO1_SELECTOR_KEY:
            close()
            if not steps or len(fields) != 4:
                raise HistoryError(
                    f"{where}: a selector line needs a step before it and four fields"
                )
            pending = _Pending(_int(fields[1], where), _int(fields[3], where), {})
        elif pending is None:
            raise HistoryError(f"{where}: {key!r} appears outside a selector block")
        elif key == INFO1_TABLE_KEY:
            name = fields[1] if len(fields) == 2 else ""
            if name not in INFO1_TABLES:
                raise HistoryError(f"{where}: unknown table {name!r}")
            table = pending.tables.setdefault(name, {})
        elif key == INFO1_COLUMNS[0] and fields[1:] == list(INFO1_COLUMNS[1:]):
            continue  # column header line
        elif key in INFO1_ROWS and table is not None:
            total, inviscid, viscous = _floats(fields[1:], 3, where)
            table[key] = Triple(total, inviscid, viscous)
        elif key == INFO1_AREAS_KEY:
            pending.areas = tuple(_floats(fields[1:], 4, where))
        elif key == INFO1_AREA_MOMENTS_KEY:
            pending.area_moments = tuple(_floats(fields[1:], 3, where))
        else:
            raise HistoryError(f"{where}: unrecognized line starting {key!r}")
    close()
    if scales is None:
        raise HistoryError(f"{INFO1_NAME}: the {INFO1_SCALES_KEY} line is missing")
    if not steps:
        raise HistoryError(f"{INFO1_NAME} holds no step")
    return steps


_PART_COLUMN = {"total": "total", "inviscid": "inviscid", "viscous": "viscous"}


def step_forces(step: StepForces, selectors: Sequence[int], part: str = "total") -> _forces.Forces:
    """Return the dimensional force and moment summed over the selectors, for one column."""
    if part not in _PART_COLUMN:
        raise HistoryError(f"unknown column {part!r}")
    force = [0.0, 0.0, 0.0]
    moment = [0.0, 0.0, 0.0]
    for selector in selectors:
        if selector not in step.selectors:
            raise HistoryError(f"selector {selector} is not in step {step.step} of {INFO1_NAME}")
        table = step.selectors[selector].dimensional
        for index, axis in enumerate("xyz"):
            force[index] += getattr(table[f"force_{axis}"], part)
            moment[index] += getattr(table[f"moment_{axis}"], part)
    return _forces.Forces((force[0], force[1], force[2]), (moment[0], moment[1], moment[2]))


def lift_drag_series(
    steps: Sequence[StepForces],
    selectors: Sequence[int],
    dynamic_pressure: float,
    references: References,
    alpha_deg: float,
    beta_deg: float,
    dimensions: int,
) -> dict[str, list[float]]:
    """Return ``CL`` and ``CD`` per step, from the total forces on the given selectors."""
    series: dict[str, list[float]] = {"CL": [], "CD": []}
    for step in steps:
        values = _forces.coefficients(step_forces(step, selectors), dynamic_pressure, references)
        cf = (values["CFx"].value, values["CFy"].value, values["CFz"].value)
        cm = (values["CMx"].value, values["CMy"].value, values["CMz"].value)
        wind = _forces.neutral(cf, cm, alpha_deg, beta_deg, dimensions)
        series["CL"].append(wind["CL"])
        series["CD"].append(wind["CD"])
    return series


def check_last_step(info0: Sequence[Step], info1: Sequence[StepForces]) -> None:
    """Refuse a force file whose last step differs from the residual file's."""
    if info0[-1].step != info1[-1].step:
        raise HistoryError(
            f"the last step of {INFO1_NAME} ({info1[-1].step}) differs from the last step of "
            f"{INFO0_NAME} ({info0[-1].step})"
        )


# --- Log --------------------------------------------------------------------


@dataclass(frozen=True)
class LogFacts:
    """What the log states: the solver version and the failure evidence."""

    version: str
    failures: tuple[str, ...]
    redo_events: int


def scan_log(text: str) -> LogFacts:
    """Return the version (or ``unknown``), the failure fragments found and the redo count."""
    match = LOG_VERSION.search(text)
    failures = tuple(
        name for name, pattern in LOG_FAILURE_FRAGMENTS.items() if pattern.search(text)
    )
    redo = sum(1 for line in text.splitlines() if LOG_REDO.search(line))
    return LogFacts(match.group(1) if match else UNKNOWN_VERSION, failures, redo)


# --- Convergence ------------------------------------------------------------


def drop_orders(steps: Sequence[Step]) -> float:
    """Return log10 of the run's largest residual over the last residual (inf if it is zero)."""
    peak = max(step.residual_avg for step in steps)
    last = steps[-1].residual_avg
    return math.inf if last <= 0 else (math.log10(peak) - math.log10(last))


def r_step(steps: Sequence[Step], orders: float) -> int | None:
    """Return the first step whose residual is ``orders`` below the run maximum so far."""
    peak = 0.0
    for step in steps:
        peak = max(peak, step.residual_avg)
        if step.residual_avg <= 0 or (math.log10(peak) - math.log10(step.residual_avg)) >= orders:
            return step.step
    return None


def c_step(steps: Sequence[Step], cl: Sequence[float], cd: Sequence[float]) -> int | None:
    """Return the first step where CL and CD both pass the Cauchy test and the residual is down."""
    return next(
        (step.step for step, holds in zip(steps, _c_passes(steps, cl, cd), strict=True) if holds),
        None,
    )


def _c_passes(steps: Sequence[Step], cl: Sequence[float], cd: Sequence[float]) -> list[bool]:
    if len(cl) != len(steps) or len(cd) != len(steps):
        raise HistoryError(
            f"the coefficient series (length {len(cl)}, {len(cd)}) do not match "
            f"the {len(steps)} steps"
        )
    lift = cauchy_passes(cl, CAUCHY_WINDOW, CAUCHY_EPS)
    drag = cauchy_passes(cd, CAUCHY_WINDOW, CAUCHY_EPS)
    first = steps[0].residual_avg
    result = []
    for index, step in enumerate(steps):
        down = step.residual_avg <= 0 or (
            first > 0
            and (math.log10(first) - math.log10(step.residual_avg)) >= CAUCHY_MIN_DROP_ORDERS
        )
        result.append(lift[index] and drag[index] and down)
    return result


# --- Evidence and classification -------------------------------------------


@dataclass(frozen=True)
class Evidence(ExecutionResult):
    """Execution result plus the evidence behind it; the extra fields are for reports."""

    criterion: str = "none"
    drop_orders: float = 0.0
    redo_events: int = 0
    failures: tuple[str, ...] = ()
    evidence_failed: bool = False
    problem: str = ""


def _failed(
    exit_code: int, iteration_limit: int, version: str, problem: str, log: LogFacts
) -> Evidence:
    return Evidence(
        exit_code,
        (),
        iteration_limit,
        version,
        failures=log.failures,
        redo_events=log.redo_events,
        evidence_failed=True,
        problem=problem,
    )


def build_evidence(
    info0_text: str,
    info1_text: str | None,
    log_text: str,
    *,
    exit_code: int,
    iteration_limit: int,
    convergence_orders: float,
    series: Mapping[str, Sequence[float]] | None = None,
) -> Evidence:
    """Build the evidence from the three file texts.

    ``series`` holds ``CL`` and ``CD`` per step (see ``lift_drag_series``); without
    it only R can hold. An unreadable or inconsistent file gives failed evidence
    with the problem stated, never an exception.
    """
    log = scan_log(log_text)
    try:
        steps = parse_info0(info0_text)
        if info1_text is not None:
            check_last_step(steps, parse_info1(info1_text))
        r = drop_orders(steps) >= convergence_orders
        c = _c_passes(steps, series["CL"], series["CD"])[-1] if series else False
    except HistoryError as error:
        return _failed(exit_code, iteration_limit, log.version, str(error), log)
    criterion = "+".join(name for name, hit in (("R", r), ("C", c)) if hit) or "none"
    last_converged = (r or c) and not log.failures
    history = tuple(
        Iteration(step.step, step.residual_avg, last_converged and index == len(steps) - 1)
        for index, step in enumerate(steps)
    )
    return Evidence(
        exit_code,
        history,
        iteration_limit,
        log.version,
        criterion=criterion,
        drop_orders=drop_orders(steps),
        redo_events=log.redo_events,
        failures=log.failures,
        evidence_failed=bool(log.failures),
    )


def classify(result: ExecutionResult) -> Classification:
    """Classify a run from its execution result alone.

    failed when the exit code is non-zero, there is no history or the evidence
    failed; converged when the last step is converged; iteration_limit when the
    last step reached the run's step budget; failed otherwise (stopped early
    without evidence).
    """
    if result.exit_code != 0 or not result.history or getattr(result, "evidence_failed", False):
        return "failed"
    if result.history[-1].converged:
        return "converged"
    if result.history[-1].iteration >= result.history[0].iteration - 1 + result.iteration_limit:
        return "iteration_limit"
    return "failed"


def read_evidence(
    folder: Path,
    *,
    exit_code: int,
    iteration_limit: int,
    convergence_orders: float,
    series: Callable[[list[StepForces]], Mapping[str, Sequence[float]]] | None = None,
) -> Evidence:
    """Read the required residual and force files and the log of a point folder."""
    log_path = folder / LOG_NAME
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    info0_path = folder / INFO0_NAME
    if not info0_path.is_file():
        return _failed(
            exit_code,
            iteration_limit,
            scan_log(log_text).version,
            f"{INFO0_NAME} is missing",
            scan_log(log_text),
        )
    info1_path = folder / INFO1_NAME
    if not info1_path.is_file():
        return _failed(
            exit_code,
            iteration_limit,
            scan_log(log_text).version,
            f"{INFO1_NAME} is missing",
            scan_log(log_text),
        )
    info1_text = (
        info1_path.read_text(encoding="utf-8", errors="replace") if info1_path.is_file() else None
    )
    forces_series = None
    if series is not None and info1_text is not None:
        try:
            forces_series = series(parse_info1(info1_text))
        except HistoryError as error:
            return _failed(
                exit_code,
                iteration_limit,
                scan_log(log_text).version,
                str(error),
                scan_log(log_text),
            )
    return build_evidence(
        info0_path.read_text(encoding="utf-8", errors="replace"),
        info1_text,
        log_text,
        exit_code=exit_code,
        iteration_limit=iteration_limit,
        convergence_orders=convergence_orders,
        series=forces_series,
    )


def read_info1(folder: Path) -> list[StepForces]:
    """Read the force file of a point folder."""
    path = folder / INFO1_NAME
    if not path.is_file():
        raise HistoryError(f"{INFO1_NAME} is missing")
    return parse_info1(path.read_text(encoding="utf-8", errors="replace"))
