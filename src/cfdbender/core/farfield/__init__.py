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

from cfdbender.core.farfield.balances import (
    CrossflowEnergy,
    InPlaneMoment,
    MassClosure,
    TransverseForce,
    axial_flux,
    axial_force,
    crossflow_integrand,
    crossflow_kinetic_energy,
    in_plane_moment,
    lateral_flux,
    mass_closure,
    mass_flux,
    shaft_torque,
    spurious_diagnostic,
    to_counts,
    transverse_flux,
    transverse_force,
)
from cfdbender.core.farfield.fields import (
    FarfieldInputError,
    LatticeFields,
    PlaneFields,
    ProbeLattice,
    cylindrical_components,
    lattice_fields,
    plane_fields,
)
from cfdbender.core.farfield.quadrature import (
    azimuthal_harmonics,
    grid_coverage,
    grid_integral,
    grid_weights,
    plane_integral,
    ring_sample_weights,
    sample_area_coverage,
    sample_coverage,
    strip_edges,
    symmetry_floor,
)

__all__ = [
    "CrossflowEnergy",
    "FarfieldInputError",
    "InPlaneMoment",
    "LatticeFields",
    "MassClosure",
    "PlaneFields",
    "ProbeLattice",
    "TransverseForce",
    "axial_flux",
    "axial_force",
    "azimuthal_harmonics",
    "crossflow_integrand",
    "crossflow_kinetic_energy",
    "cylindrical_components",
    "grid_coverage",
    "grid_integral",
    "grid_weights",
    "in_plane_moment",
    "lateral_flux",
    "lattice_fields",
    "mass_closure",
    "mass_flux",
    "plane_fields",
    "plane_integral",
    "ring_sample_weights",
    "sample_area_coverage",
    "sample_coverage",
    "shaft_torque",
    "spurious_diagnostic",
    "strip_edges",
    "symmetry_floor",
    "to_counts",
    "transverse_flux",
    "transverse_force",
]
