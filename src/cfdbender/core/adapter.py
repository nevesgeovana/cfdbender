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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from cfdbender.core.matrix import MatrixRow
from cfdbender.core.mesh import resolve_setup_paths
from cfdbender.core.plan import (
    InputFiles,
    InputSession,
    Point,
    RowInputs,
    RowRequirements,
    apply_variables,
    is_identity_value,
    is_matrix_file,
)
from cfdbender.core.results import Classification, ExecutionResult, NeutralResult
from cfdbender.core.settings import read_init, setup_defaults

CAPABILITIES = frozenset({"restart", "unsteady", "boundary_groups", "cluster_staging"})


def capabilities(adapter: object) -> frozenset[str]:
    """Read optional behaviours, refusing an invalid adapter declaration.

    Parameters
    ----------
    adapter : object
        Adapter module. A missing CAPABILITIES declaration means no support.

    Returns
    -------
    frozenset of str
        Validated capability names from the core's closed vocabulary.

    Raises
    ------
    ValueError
        If CAPABILITIES is not a frozenset of known names, a developer error.
    """
    declared: object = getattr(adapter, "CAPABILITIES", frozenset())
    label = getattr(adapter, "__name__", type(adapter).__name__)
    known = ", ".join(sorted(CAPABILITIES))
    if not isinstance(declared, frozenset) or any(not isinstance(x, str) for x in declared):
        raise ValueError(f"{label}: CAPABILITIES must be a frozenset of known names: {known}")
    if unknown := declared - CAPABILITIES:
        raise ValueError(
            f"{label}: unknown CAPABILITIES {', '.join(sorted(unknown))}; known names: {known}"
        )
    return cast(frozenset[str], declared)


@dataclass(frozen=True)
class AdapterPoint:
    """A planned point and its workspace, with digest-checked s/r/p snapshots."""

    root: Path
    point: Point
    solver: str
    setup: Mapping[str, object]
    reference: Mapping[str, object]
    post: Mapping[str, object]
    session: InputSession | None = field(default=None, compare=False, repr=False)

    @property
    def geometry(self) -> str | None:
        """The workspace-relative GEOMETRY file of the row, or None."""
        return self.point.geometry

    def inputs(self) -> InputFiles:
        """Return a tracking reader over this point's workspace, sharing its command's reads."""
        return self.session.files() if self.session is not None else InputFiles(self.root)

    def verify(self) -> None:
        """Re-read init.toml and every planned input, refusing changed bytes.

        A point loaded with an ``InputSession`` checks a large input against the command's
        shared read, which is read again when its size or timestamps changed.
        """
        config = read_init(self.root, available_solvers=(self.solver,))
        if config.solver != self.solver:
            raise ValueError("init.toml solver differs from the planned solver")
        files = self.inputs()
        for name, digest in self.point.inputs_sha256.items():
            if is_matrix_file(name):
                continue  # records of earlier plans carry it; a grown matrix is not stale
            if not is_identity_value(digest):
                files.track(name)
            if not files.agrees(name, digest):
                raise ValueError(f"input changed after planning: {name}; plan again")


def load_point(
    root: str | Path,
    point: Point,
    *,
    solver: str,
    session: InputSession | None = None,
) -> AdapterPoint:
    """Read planned s/r/p snapshots, verifying all digests and the solver binding.

    Parameters
    ----------
    root : str or Path
        Workspace root.
    point : Point
        Point from the validated plan or registry.
    solver : str
        Expected init.toml solver, supplied by the composition layer.
    session : InputSession, optional
        The reads of one command. Passing the same session for every point makes the
        command read and hash each large input once; every point is still checked against
        its planned digests. Without it each call reads its inputs afresh.

    Returns
    -------
    AdapterPoint
        Prepared context. No solver is imported and no output is written.
    """
    target = Path(root).resolve()
    shared = session if session is not None and session.root == target else None
    files = shared.files() if shared is not None else InputFiles(target)
    config = read_init(target, available_solvers=(solver,))
    result = AdapterPoint(
        target,
        point,
        solver,
        resolve_setup_paths(
            target,
            apply_variables(
                setup_defaults(
                    config.defaults,
                    files.read_toml(f"inputs/setups/{point.setup_id}.toml"),
                    omitted=point.defaults_omitted,
                ),
                point.variables,
            ),
        ),
        files.read_toml(f"inputs/references/{point.reference_id}.toml"),
        files.read_toml(f"inputs/pproc/{point.post_id}.toml"),
        shared,
    )
    result.verify()
    for name in files.digests():
        planned = point.inputs_sha256.get(name)
        if planned is None or not files.agrees(name, planned):
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
