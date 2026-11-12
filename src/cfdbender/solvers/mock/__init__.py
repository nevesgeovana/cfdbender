"""Deterministic offline adapter using the same ASCII/history path as a solver.

The s file requires [mock] outcome (converged, iteration_limit or failed) and
iterations >=2. The r file requires area, length, moment_origin, axis_system and
axis_directions. The p file requires history=true. Outputs are deliberately fake
coefficients, unsuitable for physical predictions. No executable is launched.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Mapping
from typing import Literal, cast

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import RowInputs, RowRequirements, workspace_path
from cfdbender.core.results import (
    Classification,
    Coefficient,
    ExecutionResult,
    Iteration,
    NeutralResult,
    References,
)

SOLVER_NAME = "mock"
STAGED_CONFIG = "mock.in"
JOB_OUTPUTS = ("history.csv", "execution.json")
_COEFFICIENTS = ("CX", "CY", "CZ", "Cl", "Cm", "Cn")
_HEADER = ("iteration", "residual", "converged", *_COEFFICIENTS)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _references(data: Mapping[str, object]) -> References:
    origin = data.get("moment_origin")
    directions = data.get("axis_directions")
    system = data.get("axis_system")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("moment_origin needs three coordinates")
    if (
        not isinstance(directions, list)
        or len(directions) != 3
        or any(not isinstance(value, str) for value in directions)
    ):
        raise ValueError("axis_directions needs three named directions")
    if not isinstance(system, str):
        raise ValueError("axis_system must be explicit")
    return References(
        _number(data.get("area"), "area"),
        _number(data.get("length"), "length"),
        cast(tuple[float, float, float], tuple(_number(v, "moment_origin") for v in origin)),
        system,
        cast(tuple[str, str, str], tuple(directions)),
    )


def _validate(
    setup: Mapping[str, object], reference: Mapping[str, object], post: Mapping[str, object]
) -> tuple[str, int, References]:
    table = setup.get("mock")
    if not isinstance(table, dict):
        raise ValueError("setup needs a [mock] table with outcome and iterations")
    outcome = table.get("outcome")
    iterations = table.get("iterations")
    if outcome not in ("converged", "iteration_limit", "failed"):
        raise ValueError("mock.outcome must be converged, iteration_limit or failed")
    if type(iterations) is not int or not 2 <= iterations <= 10000:
        raise ValueError("mock.iterations must be an integer between 2 and 10000")
    refs = _references(reference)
    if post.get("history") is not True:
        raise ValueError("pproc history must be true for classification")
    return str(outcome), iterations, refs


def validate_row(row: MatrixRow, workspace: RowInputs) -> RowRequirements:
    """Refuse invalid s/r/p fields and declare all mock output paths.

    Parameters
    ----------
    row : MatrixRow
        Row being planned.
    workspace : RowInputs
        Tracked s/r/p inputs; this hook only reads.

    Returns
    -------
    RowRequirements
        Deterministic actions and output templates per point.
    """
    _validate(workspace.setup, workspace.reference, workspace.post)
    if row.workflow != "steady":
        raise ValueError("mock WORKFLOW must be steady")
    if workspace.geometry is not None:
        # The mock reads nothing from the geometry; tracking it records its digest.
        workspace.read_bytes(workspace.geometry)
    return RowRequirements(
        actions=("write mock ASCII input", "run mock", "read mock history"),
        outputs=tuple(
            f"sims/sim_{{pol}}/datapoints/{{point}}/{name}"
            for name in ("mock.in", "history.csv", "execution.json")
        ),
    )


def translate(point: AdapterPoint) -> str:
    """Return deterministic ASCII containing every s/r/p field and point condition.

    Parameters
    ----------
    point : AdapterPoint
        Verified context bound to init.toml.
    """
    point.verify()
    _validate(point.setup, point.reference, point.post)
    return json.dumps(
        {
            "setup": dict(point.setup),
            "reference": dict(point.reference),
            "post": dict(point.post),
            "condition": dict(point.point.condition),
        },
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
    ) + chr(10)


def classify(result: ExecutionResult) -> Classification:
    """Distinguish solver convergence, an exhausted iteration budget and failure.

    Parameters
    ----------
    result : ExecutionResult
        Read-back evidence. A successful exit with no history is failed.
    """
    if result.exit_code != 0 or not result.history:
        return "failed"
    if result.history[-1].converged:
        return "converged"
    if result.history[-1].iteration >= result.iteration_limit:
        return "iteration_limit"
    return "failed"


def _read(point: AdapterPoint) -> tuple[ExecutionResult, dict[str, Coefficient]]:
    point.verify()
    folder = workspace_path(point.root, point.point.directory)
    metadata = json.loads((folder / "execution.json").read_bytes())
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"exit_code", "iteration_limit", "solver_version"}
        or type(metadata["exit_code"]) is not int
        or type(metadata["iteration_limit"]) is not int
    ):
        raise ValueError("execution.json needs exit_code, iteration_limit and solver_version")
    if metadata["iteration_limit"] < 2 or metadata["solver_version"] != "mock-1":
        raise ValueError("execution.json has invalid iteration_limit or solver_version")
    reader = csv.DictReader(io.StringIO((folder / "history.csv").read_text(encoding="ascii")))
    if tuple(reader.fieldnames or ()) != _HEADER:
        raise ValueError("history.csv has invalid columns")
    history: list[Iteration] = []
    coefficients: dict[str, Coefficient] = {}
    for row in reader:
        if row["converged"] not in ("0", "1"):
            raise ValueError("history.csv converged must be 0 or 1")
        item = Iteration(int(row["iteration"]), float(row["residual"]), row["converged"] == "1")
        if history and item.iteration <= history[-1].iteration:
            raise ValueError("history.csv iterations must increase")
        history.append(item)
        coefficients = {
            name: Coefficient(
                float(row[name]),
                cast(Literal["x", "y", "z"], "xyz"[i % 3]),
                "force" if i < 3 else "moment",
            )
            for i, name in enumerate(_COEFFICIENTS)
        }
    return ExecutionResult(
        metadata["exit_code"],
        tuple(history),
        metadata["iteration_limit"],
        metadata["solver_version"],
    ), coefficients


def stage(point: AdapterPoint) -> None:
    """Write mock.in into a new point folder without running the mock solver.

    Parameters
    ----------
    point : AdapterPoint
        Planned point whose folder does not exist yet.
    """
    text = translate(point)
    folder = workspace_path(point.root, point.point.directory)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "mock.in").write_bytes(text.encode("ascii"))


def execute(point: AdapterPoint, *, check_ram: Callable[[], None]) -> ExecutionResult:
    """Write ASCII input, simulate its run and read the saved history once.

    Parameters
    ----------
    point : AdapterPoint
        Planned mock point. Existing point directories refuse overwrite.
    check_ram : callable
        Floor guard, checked before writes and on every fake iteration.

    Returns
    -------
    ExecutionResult
        Parsed on-disk process and numerical evidence.
    """
    check_ram()
    stage(point)
    folder = workspace_path(point.root, point.point.directory)
    deck = json.loads((folder / "mock.in").read_bytes())
    outcome, limit, _ = _validate(deck["setup"], deck["reference"], deck["post"])
    count = 2 if outcome == "converged" else limit if outcome == "iteration_limit" else 1
    with (folder / "history.csv").open("x", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, lineterminator=chr(10))
        writer.writerow(_HEADER)
        for iteration in range(1, count + 1):
            check_ram()
            residual = 10.0 ** (-iteration)
            writer.writerow(
                (
                    iteration,
                    residual,
                    int(outcome == "converged" and iteration == count),
                    *(round(i * 0.01 + residual, 8) for i in range(6)),
                )
            )
    (folder / "execution.json").write_text(
        json.dumps(
            {
                "exit_code": int(outcome == "failed"),
                "iteration_limit": limit,
                "solver_version": "mock-1",
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    return _read(point)[0]


def collect(point: AdapterPoint) -> NeutralResult:
    """Read mock history with its explicit coefficient axes and r-file references.

    Parameters
    ----------
    point : AdapterPoint
        Recorded point with saved history and execution metadata.
    """
    result, coefficients = _read(point)
    return NeutralResult(
        coefficients,
        _references(point.reference),
        result.history,
        classify(result),
        result.solver_version,
    )
