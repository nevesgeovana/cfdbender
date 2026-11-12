"""Solver-neutral coefficients with explicit reference values and iteration evidence.

Classification describes numerical termination, separately from process exit.
Axis directions describe positive x, y and z in the stated axis system.
Reference area and length use square metres and metres; moment origin uses metres.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

Classification = Literal["converged", "iteration_limit", "failed"]


@dataclass(frozen=True)
class References:
    """Dimensional reference values and positive axis directions for coefficients."""

    area: float
    length: float
    moment_origin: tuple[float, float, float]
    axis_system: str
    axis_directions: tuple[str, str, str]

    def __post_init__(self) -> None:
        """Refuse absent axes and invalid dimensional references."""
        if not math.isfinite(self.area) or self.area <= 0:
            raise ValueError("reference area must be finite and positive")
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError("reference length must be finite and positive")
        if len(self.moment_origin) != 3 or not all(map(math.isfinite, self.moment_origin)):
            raise ValueError("moment_origin must contain three finite coordinates")
        if not self.axis_system.strip():
            raise ValueError("axis_system must be explicit")
        if len(self.axis_directions) != 3 or not all(s.strip() for s in self.axis_directions):
            raise ValueError("axis_directions must describe positive x, y and z")


@dataclass(frozen=True)
class Coefficient:
    """One dimensionless force or moment coefficient about an explicit axis."""

    value: float
    axis: Literal["x", "y", "z"]
    kind: Literal["force", "moment"]

    def __post_init__(self) -> None:
        """Refuse non-finite values and ambiguous axes."""
        if not math.isfinite(self.value):
            raise ValueError("coefficient value must be finite")
        if self.axis not in ("x", "y", "z") or self.kind not in ("force", "moment"):
            raise ValueError("coefficient axis and kind must be explicit")


@dataclass(frozen=True)
class Iteration:
    """One solver iteration, its residual and the solver's convergence evidence."""

    iteration: int
    residual: float
    converged: bool = False

    def __post_init__(self) -> None:
        """Refuse malformed iteration evidence."""
        if type(self.iteration) is not int or self.iteration < 1:
            raise ValueError("iteration must be a positive integer")
        if not math.isfinite(self.residual) or self.residual < 0:
            raise ValueError("residual must be finite and non-negative")
        if type(self.converged) is not bool:
            raise ValueError("converged must be a boolean")


@dataclass(frozen=True)
class ExecutionResult:
    """Process outcome and history; exit_code zero alone proves no convergence."""

    exit_code: int
    history: tuple[Iteration, ...]
    iteration_limit: int
    solver_version: str


@dataclass(frozen=True)
class NeutralResult:
    """Collected coefficients, references, iteration history and termination class."""

    coefficients: Mapping[str, Coefficient]
    references: References
    history: tuple[Iteration, ...]
    classification: Classification
    solver_version: str

    def __post_init__(self) -> None:
        """Refuse undocumented successful results."""
        if self.classification not in ("converged", "iteration_limit", "failed"):
            raise ValueError("unknown classification")
        if not self.solver_version.strip():
            raise ValueError("solver_version must be explicit")
        if self.classification != "failed" and not self.coefficients:
            raise ValueError("successful results must carry coefficient columns")
        if self.classification == "converged" and (
            not self.history or not self.history[-1].converged
        ):
            raise ValueError("converged requires iteration evidence")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready neutral data, retaining coefficient axes and references."""
        return {
            "coefficients": {key: asdict(value) for key, value in self.coefficients.items()},
            "references": asdict(self.references),
            "history": [asdict(value) for value in self.history],
            "classification": self.classification,
            "solver_version": self.solver_version,
        }
