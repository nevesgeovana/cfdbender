"""Record the far-field reference numbers of pyflightstream on seeded inputs.

Run once with the reference library on the path, for example
``PYTHONPATH=<pyflightstream>/src python tools/record_farfield_reference.py``. The script
calls the reference functions of the far-field core (quadrature, containers, balances) on
seeded synthetic fields and writes a JSON of the inputs and the outputs. The cfdbender test
``tests/tier1_offline/test_farfield_reference.py`` recomputes every output from the stored
inputs with ``cfdbender.core.farfield`` and compares: exactly where the operations are the
same, to 1e-12 relative where an output is a least-squares fit. The test reads only the
JSON, so it does not need the reference library. This script is a tool and lives outside
``src``; it is the one place that imports the reference library.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1
RHO = 1.2
V_INF = 30.0


def _plain(value: Any) -> Any:
    """Turn arrays, complex numbers and dataclasses into JSON values."""
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


def _lattice_case(
    name: str, n_psi: int, seed: int, holes: dict[str, tuple[int, int, int]], lateral: bool
) -> dict[str, Any]:
    from pyflightstream.farfield import (
        axial_flux,
        axial_force,
        crossflow_kinetic_energy,
        cylindrical_components,
        in_plane_moment,
        lateral_flux,
        lattice_fields,
        mass_closure,
        mass_flux,
        plane_integral,
        ring_sample_weights,
        sample_area_coverage,
        sample_coverage,
        shaft_torque,
        symmetry_floor,
        transverse_flux,
        transverse_force,
    )
    from pyflightstream.probes import build_lattice

    rng = np.random.default_rng(seed)
    stations = (-2.0, 0.5, 1.0, 2.0)
    lattice = build_lattice(
        tip_radius=1.5,
        stations=stations,
        n_r=5,
        r_inner=0.125,
        r_outer=2.25,
        n_psi=n_psi,
        lateral_radius=2.5 if lateral else None,
    )
    shape = (len(stations), lattice.n_r, lattice.n_psi)
    fields = {
        "u": V_INF + 2.0 * rng.standard_normal(shape),
        "v": 0.5 * rng.standard_normal(shape),
        "w": 0.5 * rng.standard_normal(shape),
        "p_prime": 10.0 * rng.standard_normal(shape),
    }
    for field_name, index in holes.items():
        fields[field_name][index] = np.nan
    lateral_fields_in = None
    if lateral:
        lat_shape = (len(lattice.lateral_stations), lattice.n_psi)
        lateral_fields_in = {
            "v": 0.5 * rng.standard_normal(lat_shape),
            "w": 0.5 * rng.standard_normal(lat_shape),
            "p_prime": 10.0 * rng.standard_normal(lat_shape),
        }
    base = lattice_fields(lattice, fields=fields, lateral=lateral_fields_in)
    ds = cylindrical_components(base)
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
        "mass_flux": mass_flux(ds, RHO),
        "mass_closure": mass_closure(ds, RHO),
        "axial_flux": axial_flux(ds, RHO, V_INF),
        "axial_force": axial_force(ds, RHO, V_INF, inlet=-2.0, outlet=2.0),
        "axial_force.mid": axial_force(ds, RHO, V_INF, inlet=0.5, outlet=1.0),
        "transverse_flux.w.quadrature": transverse_flux(
            ds, RHO, component="w", method="quadrature"
        ),
        "transverse_flux.w.harmonic": transverse_flux(ds, RHO, component="w", method="harmonic"),
        "transverse_flux.v.quadrature": transverse_flux(
            ds, RHO, component="v", method="quadrature"
        ),
        "transverse_force.w": transverse_force(ds, RHO, inlet=-2.0, outlet=2.0),
        "shaft_torque": shaft_torque(ds, RHO),
        "in_plane_moment.y.quadrature": in_plane_moment(ds, RHO, V_INF, inlet=-2.0, outlet=2.0),
        "in_plane_moment.y.harmonic": in_plane_moment(
            ds, RHO, V_INF, inlet=-2.0, outlet=2.0, method="harmonic"
        ),
        "in_plane_moment.z.quadrature": in_plane_moment(
            ds, RHO, V_INF, inlet=0.5, outlet=2.0, axis="z"
        ),
        "crossflow_kinetic_energy": crossflow_kinetic_energy(ds, RHO),
    }
    if lateral:
        flux = lateral_flux(ds, rho=RHO, inlet=-2.0, outlet=2.0)
        outputs["lateral_flux.w"] = flux
        outputs["transverse_force.w.lateral"] = transverse_force(
            ds, RHO, inlet=-2.0, outlet=2.0, lateral_flux=flux
        )
    return {
        "name": name,
        "lattice": {
            "tip_radius": float(lattice.tip_radius),
            "stations": [float(s) for s in lattice.stations],
            "ring_edges": [float(e) for e in lattice.ring_edges],
            "n_psi": int(lattice.n_psi),
            "lateral_radius": None
            if lattice.lateral_radius is None
            else float(lattice.lateral_radius),
            "lateral_stations": [float(s) for s in lattice.lateral_stations],
        },
        "fields": {k: _plain(v) for k, v in fields.items()},
        "lateral_fields": None
        if lateral_fields_in is None
        else {k: _plain(v) for k, v in lateral_fields_in.items()},
        "rho": RHO,
        "v_inf": V_INF,
        "outputs": {k: _plain(v) for k, v in outputs.items()},
    }


def _grid_case(seed: int, holed: bool) -> dict[str, Any]:
    from pyflightstream.farfield import (
        grid_coverage,
        grid_integral,
        grid_weights,
        plane_fields,
        strip_edges,
    )

    rng = np.random.default_rng(seed)
    y = np.linspace(-1.0, 1.0, 9)
    z = np.linspace(0.0, 2.0, 7)
    field = rng.standard_normal((9, 7))
    if holed:
        field[0, 0] = np.nan
        field[4, 3] = np.nan
    plane = plane_fields(y, z=z, fields={"v": field, "w": field[::-1].copy()})
    positions = np.sort(rng.uniform(-2.0, 2.0, 11))
    outputs = {
        "grid_weights": grid_weights(y, z=z),
        "grid_integral": grid_integral(y, z=z, integrand=field),
        "grid_integral.square": grid_integral(y, z=z, integrand=field**2),
        "grid_coverage.axes": grid_coverage(field, y=y, z=z),
        "grid_coverage.count": grid_coverage(field),
        "strip_edges.extended": strip_edges(positions),
        "strip_edges.station": strip_edges(positions, ends="station"),
        "plane_fields.v": plane.values["v"],
    }
    return {
        "name": "grid-holed" if holed else "grid",
        "y": y.tolist(),
        "z": z.tolist(),
        "field": field.tolist(),
        "positions": positions.tolist(),
        "outputs": {k: _plain(v) for k, v in outputs.items()},
    }


def _harmonic_case(seed: int, n: int, psi_0: float) -> dict[str, Any]:
    from pyflightstream.farfield import azimuthal_harmonics, symmetry_floor

    rng = np.random.default_rng(seed)
    psi = psi_0 + 2.0 * np.pi * np.arange(n) / n
    values = rng.standard_normal((3, 4, n))
    outputs = {
        "azimuthal_harmonics.m3": azimuthal_harmonics(values, psi=psi, m_max=3),
        "azimuthal_harmonics.default": azimuthal_harmonics(values, psi=psi),
        "symmetry_floor": symmetry_floor(values[0, 0], psi=psi, m_max=4),
    }
    return {
        "name": f"harmonics-{n}",
        "psi": psi.tolist(),
        "values": values.tolist(),
        "outputs": {k: _plain(v) for k, v in outputs.items()},
    }


def _counts_case() -> dict[str, Any]:
    from pyflightstream.farfield import spurious_diagnostic, to_counts

    outputs = {
        "to_counts": to_counts(1.7, rho_inf=RHO, v_inf=V_INF, s_ref=0.8),
        "spurious_diagnostic": spurious_diagnostic(
            101.0, 100.0, rho_inf=RHO, v_inf=V_INF, s_ref=1.0
        ),
    }
    return {"name": "counts", "outputs": {k: _plain(v) for k, v in outputs.items()}}


def record() -> dict[str, Any]:
    """Return the whole reference record."""
    return {
        "schema": SCHEMA,
        "lattice_cases": [
            _lattice_case("even", 16, 20261012, {}, True),
            _lattice_case("odd", 15, 20261013, {}, False),
            _lattice_case("holed", 16, 20261014, {"u": (1, 0, 0), "w": (2, 3, 5)}, True),
            _lattice_case(
                "holed-transverse", 16, 20261015, {"v": (1, 2, 3), "w": (1, 2, 3)}, False
            ),
        ],
        "grid_cases": [_grid_case(1, False), _grid_case(2, True)],
        "harmonic_cases": [
            _harmonic_case(11, 16, 0.0),
            _harmonic_case(12, 15, 0.3),
            _harmonic_case(13, 24, 0.0),
        ],
        "counts_case": _counts_case(),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the reference record and return 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/data/farfield_reference.json",
    )
    args = parser.parse_args(argv)
    data = record()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, indent=1, allow_nan=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
