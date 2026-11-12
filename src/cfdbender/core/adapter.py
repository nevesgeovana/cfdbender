"""Define an injected adapter contract without importing a solver.

Adapters expose these functions at their registered module boundary. Planning
calls validate_row(row, workspace), where workspace is the existing RowInputs
tracked reader. It must only read and validate. Every extra input is read through
that reader. translate returns deterministic ASCII from the prepared s/r/p inputs.
execute writes that input, runs the solver and reads history; it must call
check_ram before and during work and refuse existing output files. classify
requires numerical evidence; collect reads the outputs into NeutralResult.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import InputFiles, Point, RowInputs, RowRequirements, apply_variables
from cfdbender.core.results import Classification, ExecutionResult, NeutralResult
from cfdbender.core.settings import read_init


@dataclass(frozen=True)
class AdapterPoint:
    """A planned point and its workspace, with digest-checked s/r/p snapshots."""

    root: Path
    point: Point
    solver: str
    setup: Mapping[str, object]
    reference: Mapping[str, object]
    post: Mapping[str, object]

    @property
    def geometry(self) -> str | None:
        """The workspace-relative GEOMETRY file of the row, or None."""
        return self.point.geometry

    def verify(self) -> None:
        """Re-read init.toml and every planned input, refusing changed bytes."""
        config = read_init(self.root, available_solvers=(self.solver,))
        if config.solver != self.solver:
            raise ValueError("init.toml solver differs from the planned solver")
        files = InputFiles(self.root)
        for name, digest in self.point.inputs_sha256.items():
            if hashlib.sha256(files.read_bytes(name)).hexdigest() != digest:
                raise ValueError(f"input changed after planning: {name}; plan again")


def load_point(root: str | Path, point: Point, *, solver: str) -> AdapterPoint:
    """Read planned s/r/p snapshots, verifying all digests and the solver binding.

    Parameters
    ----------
    root : str or Path
        Workspace root.
    point : Point
        Point from the validated plan or registry.
    solver : str
        Expected init.toml solver, supplied by the composition layer.

    Returns
    -------
    AdapterPoint
        Prepared context. No solver is imported and no output is written.
    """
    target = Path(root).resolve()
    files = InputFiles(target)
    result = AdapterPoint(
        target,
        point,
        solver,
        apply_variables(files.read_toml(f"inputs/setups/{point.setup_id}.toml"), point.variables),
        files.read_toml(f"inputs/references/{point.reference_id}.toml"),
        files.read_toml(f"inputs/pproc/{point.post_id}.toml"),
    )
    result.verify()
    for name, digest in files.digests().items():
        if point.inputs_sha256.get(name) != digest:
            raise ValueError(f"untracked or changed input: {name}")
    return result


@runtime_checkable
class SolverAdapter(Protocol):
    """Structural module protocol shared by real and licence-free adapters."""

    def validate_row(self, row: MatrixRow, workspace: RowInputs) -> RowRequirements:
        """Validate s/r/p and describe actions and output claims, without writes.

        Parameters
        ----------
        row : MatrixRow
            Parsed row, including conditions and sweep values.
        workspace : RowInputs
            Parsed s/r/p with tracked readers for all additional inputs.

        Returns
        -------
        RowRequirements
            Actions and output templates. Raise ValueError naming invalid input.
        """
        ...

    def translate(self, point: AdapterPoint) -> str:
        """Return deterministic ASCII solver input from a verified point.

        Parameters
        ----------
        point : AdapterPoint
            Planned point with s/r/p data and workspace binding.
        """
        ...

    def execute(self, point: AdapterPoint, *, check_ram: Callable[[], None]) -> ExecutionResult:
        """Write input, execute once and read history, checking RAM throughout.

        Parameters
        ----------
        point : AdapterPoint
            Planned point with s/r/p data and workspace binding.
        check_ram : callable
            Raises on unavailable RAM or a threatened 0.5 GiB free floor.
        """
        ...

    def classify(self, result: ExecutionResult) -> Classification:
        """Classify numerical evidence, never inferring convergence from exit zero.

        Parameters
        ----------
        result : ExecutionResult
            Process outcome, history, iteration limit and version.
        """
        ...

    def collect(self, point: AdapterPoint) -> NeutralResult:
        """Read saved results with explicit coefficient axes and references.

        Parameters
        ----------
        point : AdapterPoint
            Recorded point and the workspace holding its outputs.
        """
        ...
