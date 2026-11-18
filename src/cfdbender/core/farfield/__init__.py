"""The far-field numeric core: quadrature, sample containers and conservation balances, on numpy.

Pipeline role: the solver-agnostic arithmetic of the far-field family. A facade over a numpy
core with no input or output: :mod:`.fields` (the survey and plane containers and the one
refusal), :mod:`.quadrature` (the plane, ring and strip rules and the harmonic layer, each
once) and :mod:`.balances` (the mass, momentum, torque, moment and crossflow-energy balances
of a survey lattice).

The equations are those of the definitions page, The far-field family, with the free-stream
density a parameter of each call.

UNITS. The core converts nothing. A ledger is the field's units times the square of the
length unit of ``tip_radius`` (the areas it integrates over are ``tip_radius**2`` times a
dimensionless weight), so it is in SI (N, W, kg/s, N m) only when the fields are in SI and
``tip_radius`` is in METRES.
"""

from __future__ import annotations

from cfdbender.core.farfield.fields import (
    FarfieldInputError,
    LatticeFields,
    PlaneFields,
    ProbeLattice,
    cylindrical_components,
    lattice_fields,
    plane_fields,
)

__all__ = [
    "FarfieldInputError",
    "LatticeFields",
    "PlaneFields",
    "ProbeLattice",
    "cylindrical_components",
    "lattice_fields",
    "plane_fields",
]
