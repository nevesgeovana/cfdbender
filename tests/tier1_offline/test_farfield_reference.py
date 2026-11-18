"""Cross-check of the far-field core against recorded reference numbers.

``tests/data/farfield_reference.json`` holds seeded inputs and the outputs the reference
implementation produced for them (written by ``tools/record_farfield_reference.py``). Every
output is recomputed here from the stored inputs. The operations are the same, so the
comparison is exact, bit for bit, NaN with NaN. No output of the ported core is a
least-squares fit, so the 1e-12 relative arm is not needed; ``FIT_RTOL`` is the tolerance
any such output would use. The test reads only the JSON.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from farfield_support import SurveyLattice

from cfdbender.core.farfield import (
    axial_flux,
    axial_force,
    azimuthal_harmonics,
    crossflow_kinetic_energy,
    cylindrical_components,
    grid_coverage,
    grid_integral,
    grid_weights,
    in_plane_moment,
    lateral_flux,
    lattice_fields,
    mass_closure,
    mass_flux,
    plane_fields,
    plane_integral,
    ring_sample_weights,
    sample_area_coverage,
    sample_coverage,
    shaft_torque,
    spurious_diagnostic,
    strip_edges,
    symmetry_floor,
    to_counts,
    transverse_flux,
    transverse_force,
)

FIT_RTOL = 1e-12
DATA = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "farfield_reference.json").read_text(
        encoding="utf-8"
    )
)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "complex": [[float(c.real), float(c.imag)] for c in value.ravel()],
                "shape": list(value.shape),
            }
        return {"array": value.tolist(), "shape": list(value.shape)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _same(got: Any, expected: Any, path: str) -> None:
    """Exact equality of two JSON values, NaN equal to NaN."""
    if isinstance(expected, dict):
        assert isinstance(got, dict) and got.keys() == expected.keys(), path
        for key in expected:
            _same(got[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(got, list) and len(got) == len(expected), path
        for index, (g, e) in enumerate(zip(got, expected, strict=True)):
            _same(g, e, f"{path}[{index}]")
    elif isinstance(expected, float) and math.isnan(expected):
        assert isinstance(got, float) and math.isnan(got), path
    else:
        assert got == expected, f"{path}: {got!r} != {expected!r}"


def _check(outputs: dict[str, Any], recorded: dict[str, Any], case: str) -> None:
    assert outputs.keys() == recorded.keys(), case
    for name, expected in recorded.items():
        _same(_plain(outputs[name]), expected, f"{case}:{name}")


def _array(record: dict[str, Any]) -> np.ndarray:
    return np.array(record["array"], dtype=float).reshape(record["shape"])


@pytest.mark.parametrize("case", DATA["lattice_cases"], ids=lambda c: c["name"])
def test_lattice_balances_equal_the_recorded_reference_exactly(case):
    lattice = SurveyLattice(
        tip_radius=case["lattice"]["tip_radius"],
        stations=tuple(case["lattice"]["stations"]),
        ring_edges=tuple(case["lattice"]["ring_edges"]),
        n_psi=case["lattice"]["n_psi"],
        lateral_radius=case["lattice"]["lateral_radius"],
        lateral_stations=tuple(case["lattice"]["lateral_stations"]),
    )
    fields = {k: _array(v) for k, v in case["fields"].items()}
    lateral = case["lateral_fields"]
    lateral_in = None if lateral is None else {k: _array(v) for k, v in lateral.items()}
    rho, v_inf = case["rho"], case["v_inf"]
    ds = cylindrical_components(lattice_fields(lattice, fields=fields, lateral=lateral_in))
    outputs: dict[str, Any] = {
        "cylindrical.v_r": ds["v_r"],
        "cylindrical.v_theta": ds["v_theta"],
        "plane_integral.u": plane_integral(ds, ds["u"]),
        "plane_integral.p_prime": plane_integral(ds, ds["p_prime"]),
        "ring_sample_weights": ring_sample_weights(ds),
        "sample_coverage.u": sample_coverage(ds["u"]),
        "sample_area_coverage.u": sample_area_coverage(ds["u"], ds),
        "symmetry_floor.u": symmetry_floor(ds["u"], psi=ds.psi),
        "symmetry_floor.u.m3": symmetry_floor(ds["u"], psi=ds.psi, m_max=3, excluded_orders=[2]),
        "mass_flux": mass_flux(ds, rho),
        "mass_closure": mass_closure(ds, rho),
        "axial_flux": axial_flux(ds, rho, v_inf),
        "axial_force": axial_force(ds, rho, v_inf, inlet=-2.0, outlet=2.0),
        "axial_force.mid": axial_force(ds, rho, v_inf, inlet=0.5, outlet=1.0),
        "transverse_flux.w.quadrature": transverse_flux(
            ds, rho, component="w", method="quadrature"
        ),
        "transverse_flux.w.harmonic": transverse_flux(ds, rho, component="w", method="harmonic"),
        "transverse_flux.v.quadrature": transverse_flux(
            ds, rho, component="v", method="quadrature"
        ),
        "transverse_force.w": transverse_force(ds, rho, inlet=-2.0, outlet=2.0),
        "shaft_torque": shaft_torque(ds, rho),
        "in_plane_moment.y.quadrature": in_plane_moment(ds, rho, v_inf, inlet=-2.0, outlet=2.0),
        "in_plane_moment.y.harmonic": in_plane_moment(
            ds, rho, v_inf, inlet=-2.0, outlet=2.0, method="harmonic"
        ),
        "in_plane_moment.z.quadrature": in_plane_moment(
            ds, rho, v_inf, inlet=0.5, outlet=2.0, axis="z"
        ),
        "crossflow_kinetic_energy": crossflow_kinetic_energy(ds, rho),
    }
    if lateral_in is not None:
        flux = lateral_flux(ds, rho=rho, inlet=-2.0, outlet=2.0)
        outputs["lateral_flux.w"] = flux
        outputs["transverse_force.w.lateral"] = transverse_force(
            ds, rho, inlet=-2.0, outlet=2.0, lateral_flux=flux
        )
    _check(outputs, case["outputs"], case["name"])


@pytest.mark.parametrize("case", DATA["grid_cases"], ids=lambda c: c["name"])
def test_grid_quadrature_equals_the_recorded_reference_exactly(case):
    y, z = np.array(case["y"]), np.array(case["z"])
    field = np.array(case["field"])
    positions = np.array(case["positions"])
    plane = plane_fields(y, z=z, fields={"v": field, "w": field[::-1].copy()})
    outputs: dict[str, Any] = {
        "grid_weights": grid_weights(y, z=z),
        "grid_integral": grid_integral(y, z=z, integrand=field),
        "grid_integral.square": grid_integral(y, z=z, integrand=field**2),
        "grid_coverage.axes": grid_coverage(field, y=y, z=z),
        "grid_coverage.count": grid_coverage(field),
        "strip_edges.extended": strip_edges(positions),
        "strip_edges.station": strip_edges(positions, ends="station"),
        "plane_fields.v": plane.values["v"],
    }
    _check(outputs, case["outputs"], case["name"])


@pytest.mark.parametrize("case", DATA["harmonic_cases"], ids=lambda c: c["name"])
def test_azimuthal_harmonics_equal_the_recorded_reference_exactly(case):
    psi = np.array(case["psi"])
    values = np.array(case["values"])
    outputs: dict[str, Any] = {
        "azimuthal_harmonics.m3": azimuthal_harmonics(values, psi=psi, m_max=3),
        "azimuthal_harmonics.default": azimuthal_harmonics(values, psi=psi),
        "symmetry_floor": symmetry_floor(values[0, 0], psi=psi, m_max=4),
    }
    _check(outputs, case["outputs"], case["name"])


def test_counts_equal_the_recorded_reference_exactly():
    case = DATA["counts_case"]
    outputs: dict[str, Any] = {
        "to_counts": to_counts(1.7, rho_inf=1.2, v_inf=30.0, s_ref=0.8),
        "spurious_diagnostic": spurious_diagnostic(
            101.0, 100.0, rho_inf=1.2, v_inf=30.0, s_ref=1.0
        ),
    }
    _check(outputs, case["outputs"], case["name"])


def test_the_reference_record_covers_every_ported_balance():
    names = {k for case in DATA["lattice_cases"] for k in case["outputs"]}
    for balance in (
        "mass_flux",
        "axial_flux",
        "axial_force",
        "transverse_flux.w.quadrature",
        "transverse_force.w",
        "shaft_torque",
        "in_plane_moment.y.quadrature",
        "crossflow_kinetic_energy",
        "lateral_flux.w",
    ):
        assert balance in names
