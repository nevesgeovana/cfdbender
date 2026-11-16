"""Measure exported faces with a checked, coefficient-based area enclosure."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning

from cfdbender._errors import InputArtifactError
from cfdbender.geometry._geom_area_domain import NUMERICAL_RTOL
from cfdbender.geometry._geom_area_integral import bounded_area
from cfdbender.mesh._cad import CadKernel, Face


def measure_area(kernel: CadKernel, face: Face) -> tuple[float, float]:
    """Return area and absolute numerical uncertainty, refusing an unproved result."""
    definition = getattr(kernel, "_export_area_definition", None)
    if definition is None:
        raise InputArtifactError(
            f"face {face.tag}: measured area unverified, numerical relative limit "
            f"{NUMERICAL_RTOL}; the kernel provides no coefficient certificate. "
            "Use a supported face definition and rebuild.",
            kind="geometry",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            warnings.simplefilter("error", RuntimeWarning)
            value, error = bounded_area(definition(face), kernel.area(face))
    except (IntegrationWarning, RuntimeWarning) as warning:
        raise InputArtifactError(
            f"face {face.tag}: measured area unverified after {type(warning).__name__}: {warning}; "
            f"numerical relative limit {NUMERICAL_RTOL}. "
            "Simplify the surface or trims and rebuild.",
            kind="geometry",
        ) from warning
    except InputArtifactError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        StopIteration,
        FloatingPointError,
    ) as cause:
        raise InputArtifactError(
            f"face {face.tag}: measured area unverified ({cause}), numerical relative limit "
            f"{NUMERICAL_RTOL}. Check the surface and trim definitions and rebuild.",
            kind="geometry",
        ) from cause
    if (
        not np.isfinite([value, error]).all()
        or value <= 0
        or error < 0
        or error > NUMERICAL_RTOL * value
    ):
        raise InputArtifactError(
            f"face {face.tag}: measured area {value} m2, numerical error {error} m2, "
            f"limit <= {NUMERICAL_RTOL * value} m2. Simplify the surface or trims and rebuild.",
            kind="geometry",
        )
    return float(value), float(error)


def export_area(kernel: CadKernel, face: Face) -> float:
    """Return a face area only after its numerical uncertainty has been bounded."""
    return measure_area(kernel, face)[0]
