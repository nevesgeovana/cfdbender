"""The containers the far-field core integrates: survey samples on a lattice or a plane.

Pipeline role: the frozen numpy containers that replaced the xarray dataset.
:class:`LatticeFields` carries the samples of a cylindrical
survey lattice, shaped ``(station, ring, azimuth)``, in the shaft-aligned
coordinates of the survey lattice (see :class:`ProbeLattice`); :class:`PlaneFields`
carries the samples of a rectangular far-field plane, shaped ``(y, z)``. Neither
reads or writes a file.

Field names: axial velocity ``u`` (m/s, +x downstream), transverse Cartesian
``v`` (+y) and ``w`` (+z), cylindrical ``v_r`` and ``v_theta`` (derived, see
:func:`cylindrical_components`), pressure perturbation ``p_prime`` = p - p_inf
(Pa). Radii and stations are nondimensionalized by the lattice tip radius.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol

import numpy as np

from cfdbender._errors import InputValueError

__all__ = [
    "FarfieldInputError",
    "LatticeFields",
    "PlaneFields",
    "cylindrical_components",
    "lattice_fields",
    "plane_fields",
]

_EMPTY: Mapping[str, np.ndarray] = MappingProxyType({})


class ProbeLattice(Protocol):
    """What :func:`lattice_fields` reads from a survey lattice (structural, no base class).

    Any object carrying these attributes is a lattice: the annular stations, the
    ring geometry, the uniform azimuths, the tip radius and the lateral cylinder.
    """

    @property
    def stations(self) -> Sequence[float]: ...

    @property
    def n_r(self) -> int: ...

    @property
    def n_psi(self) -> int: ...

    @property
    def lateral_stations(self) -> Sequence[float]: ...

    @property
    def ring_centers(self) -> np.ndarray: ...

    @property
    def ring_edges(self) -> Sequence[float]: ...

    @property
    def psi(self) -> np.ndarray: ...

    @property
    def tip_radius(self) -> float: ...

    @property
    def lateral_radius(self) -> float | None: ...


class FarfieldInputError(InputValueError):
    """A far-field reduction was given fields it cannot integrate.

    Survey fields missing a sample array the ledger needs, a harmonic request
    the azimuthal sampling cannot resolve, a flux or moment call whose arrays
    do not agree on shape or frame, a station the lattice does not carry.

    Also inherits from ``ValueError``.
    """


@dataclass(frozen=True, eq=False)
class LatticeFields:
    """Samples of a cylindrical survey lattice with its coordinates.

    Attributes
    ----------
    stations : numpy.ndarray
        Axial positions x/R of the annular planes, shape ``(n_stations,)``.
    ring_centers : numpy.ndarray
        Sampling radii r/R, shape ``(n_r,)``.
    ring_edges : numpy.ndarray
        Ring edges r/R, shape ``(n_r + 1,)``; the quadrature weights come
        from them.
    psi : numpy.ndarray
        Uniform azimuths in radians, shape ``(n_psi,)``.
    tip_radius : float
        Reference radius R, in THE LENGTH UNIT OF THE SAMPLES' POSITIONS. No
        unit conversion is performed by this container or by
        :func:`lattice_fields`: every area the balances integrate over is
        ``tip_radius`` squared times a dimensionless weight, so a ledger comes
        out in the field's units times that length unit squared (N for a
        momentum flux, W for an energy flux, kg/s for a mass flux, when the
        fields are in SI) and is SI only when R is given in meters. The post
        product path gives meters: it builds the lattice from the meter radius
        the artifact states and turns the exported coordinates into meters by
        the saved simulation's unit before it reads them. A caller of this
        library holding R in the simulation's unit (millimeters, say) owns the
        conversion: pass ``tip_radius`` in meters, with the fields in SI.
    values : mapping of str to numpy.ndarray
        Sampled fields, each shaped ``(n_stations, n_r, n_psi)``.
    lateral_radius : float or None
        Radius r/R of the lateral closure cylinder; None without one.
    lateral_stations : numpy.ndarray
        Axial positions x/R of the lateral rings, shape ``(n_lateral,)``.
    lateral_values : mapping of str to numpy.ndarray
        Lateral ring samples, each shaped ``(n_lateral, n_psi)``.
    """

    stations: np.ndarray
    ring_centers: np.ndarray
    ring_edges: np.ndarray
    psi: np.ndarray
    tip_radius: float
    values: Mapping[str, np.ndarray]
    lateral_radius: float | None = None
    lateral_stations: np.ndarray = field(default_factory=lambda: np.empty(0))
    lateral_values: Mapping[str, np.ndarray] = _EMPTY

    @property
    def shape(self) -> tuple[int, int, int]:
        """The sample shape ``(n_stations, n_r, n_psi)``."""
        return (len(self.stations), len(self.ring_centers), len(self.psi))

    def __getitem__(self, name: str) -> np.ndarray:
        """Return one sampled field, refusing a name the samples do not carry."""
        try:
            return self.values[name]
        except KeyError:
            raise FarfieldInputError(
                f"the survey samples carry no field {name!r}; they carry {sorted(self.values)}"
            ) from None

    def with_values(self, **arrays: np.ndarray) -> LatticeFields:
        """Return a copy with fields added or replaced (shapes checked)."""
        merged = dict(self.values)
        for name, array in arrays.items():
            merged[name] = _checked(name, array, self.shape)
        return replace(self, values=MappingProxyType(merged))

    def with_lateral_values(self, **arrays: np.ndarray) -> LatticeFields:
        """Return a copy with lateral-ring fields added or replaced."""
        shape = (len(self.lateral_stations), len(self.psi))
        merged = dict(self.lateral_values)
        for name, array in arrays.items():
            merged[name] = _checked(name, array, shape)
        return replace(self, lateral_values=MappingProxyType(merged))

    def station_index(self, station: float) -> int:
        """Return the index of a station x/R, allowing floating-point roundoff.

        An exact match wins; otherwise the one station within ``1e-12``
        (relative and absolute) is taken, so ``0.1 * 3`` finds the plane at
        ``0.3``. A nearby plane is never guessed.

        Raises
        ------
        FarfieldInputError
            If no plane, or more than one, lies at ``station`` within the
            roundoff tolerance; the message names the available stations.
        """
        found = np.flatnonzero(self.stations == station)
        if found.size == 0:
            found = np.flatnonzero(np.isclose(self.stations, station, rtol=1e-12, atol=1e-12))
        if found.size != 1:
            raise FarfieldInputError(
                f"the lattice has no single annular plane at x/R = {station!r} "
                f"({found.size} match within 1e-12); available stations are "
                f"{[float(s) for s in self.stations]}"
            )
        return int(found[0])

    def points(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Flatten the survey into the inputs of the point writers.

        The replacement of the retired ``post.dataset_to_points``: positions
        rebuilt from the survey coordinates (``x = station R``,
        ``y = r sin(psi) R``, ``z = r cos(psi) R``, the convention of
        the survey lattice) and every field flattened in the same
        station, ring, azimuth order, so the result drops straight into
        a point writer.

        Returns
        -------
        tuple
            ``(points, fields)``: positions ``(n, 3)`` in the length unit of
            ``tip_radius`` (meters on the post product path) and the
            flattened scalar fields.
        """
        tip = float(self.tip_radius)
        x, rr, pp = np.meshgrid(self.stations, self.ring_centers, self.psi, indexing="ij")
        points = np.column_stack(
            [
                (x * tip).ravel(),
                (rr * np.sin(pp) * tip).ravel(),
                (rr * np.cos(pp) * tip).ravel(),
            ]
        )
        return points, {name: np.asarray(array).ravel() for name, array in self.values.items()}


def _checked(name: str, values: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise FarfieldInputError(
            f"field {name!r} has shape {array.shape}, but this lattice samples "
            f"{shape} (stations, rings, azimuths)"
        )
    return array


def lattice_fields(
    lattice: ProbeLattice,
    *,
    fields: Mapping[str, np.ndarray],
    lateral: Mapping[str, np.ndarray] | None = None,
) -> LatticeFields:
    """Assemble the survey samples of a lattice, the replacement of ``lattice_dataset``.

    Parameters
    ----------
    lattice : ProbeLattice
        Lattice that placed the probes; supplies stations, ring centers, ring
        edges, azimuths, the tip radius and the lateral cylinder. Supply the
        tip radius in meters for SI balances; this function copies it
        without converting simulation length units.
    fields : mapping of str to numpy.ndarray
        Sampled fields of the annular planes, each shaped
        ``(n_stations, n_r, n_psi)`` in the lattice sampling order.
    lateral : mapping of str to numpy.ndarray, optional
        Sampled fields of the lateral rings, each ``(n_lateral, n_psi)``.

    Returns
    -------
    LatticeFields
        The frozen container every balance integrates.
    """
    shape = (len(lattice.stations), lattice.n_r, lattice.n_psi)
    values = {name: _checked(name, array, shape) for name, array in fields.items()}
    lateral_stations = np.asarray(lattice.lateral_stations, dtype=float)
    lateral_values: dict[str, np.ndarray] = {}
    if lateral:
        lateral_shape = (len(lateral_stations), lattice.n_psi)
        lateral_values = {
            name: _checked(name, array, lateral_shape) for name, array in lateral.items()
        }
    return LatticeFields(
        stations=np.asarray(lattice.stations),
        ring_centers=lattice.ring_centers,
        ring_edges=np.asarray(lattice.ring_edges),
        psi=lattice.psi,
        tip_radius=float(lattice.tip_radius),
        values=MappingProxyType(values),
        lateral_radius=lattice.lateral_radius,
        lateral_stations=lateral_stations,
        lateral_values=MappingProxyType(lateral_values),
    )


def cylindrical_components(fields: LatticeFields) -> LatticeFields:
    """Add the cylindrical velocity components ``v_r`` and ``v_theta``.

    Convention (fixed here and pinned by a test):
    ``y = r sin(psi)``, ``z = r cos(psi)``, so the radial unit vector is
    ``(0, sin psi, cos psi)`` and the azimuthal unit vector, in the direction
    of growing psi (+z toward +y), is ``(0, cos psi, -sin psi)``. A
    right-handed rotation about +x therefore has negative ``v_theta`` in this
    convention. The lateral rings gain the same two components where they
    carry ``v`` and ``w``.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples carrying ``v`` and ``w`` (m/s).

    Returns
    -------
    LatticeFields
        A copy with ``v_r`` and ``v_theta`` added (m/s).
    """
    sin_psi = np.sin(fields.psi)
    cos_psi = np.cos(fields.psi)
    out = fields.with_values(
        v_r=fields["v"] * sin_psi + fields["w"] * cos_psi,
        v_theta=fields["v"] * cos_psi - fields["w"] * sin_psi,
    )
    lateral = fields.lateral_values
    if "v" in lateral and "w" in lateral:
        out = out.with_lateral_values(
            v_r=lateral["v"] * sin_psi + lateral["w"] * cos_psi,
            v_theta=lateral["v"] * cos_psi - lateral["w"] * sin_psi,
        )
    return out


@dataclass(frozen=True, eq=False)
class PlaneFields:
    """Samples of a uniform rectangular far-field plane (Trefftz C, wing).

    Attributes
    ----------
    y, z : numpy.ndarray
        The grid axes in the plane, in m, uniform and strictly increasing,
        shapes ``(n_y,)`` and ``(n_z,)``; ``y`` to the right and ``z`` up in
        the wind axes of the point.
    values : mapping of str to numpy.ndarray
        Sampled fields, each shaped ``(n_y, n_z)``: ``u``, ``v``, ``w`` in wind
        axes (m/s) and ``p_prime`` (Pa) where the analysis needs it.
    """

    y: np.ndarray
    z: np.ndarray
    values: Mapping[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        """Return one sampled field, refusing a name the samples do not carry."""
        try:
            return self.values[name]
        except KeyError:
            raise FarfieldInputError(
                f"the plane samples carry no field {name!r}; they carry {sorted(self.values)}"
            ) from None


def _uniform_axis(name: str, axis: object) -> np.ndarray:
    array = np.asarray(axis, dtype=float)
    if array.ndim != 1 or array.size < 3:
        raise FarfieldInputError(f"the plane axis {name} needs at least three points")
    steps = np.diff(array)
    if not np.all(steps > 0) or not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
        raise FarfieldInputError(
            f"the plane axis {name} is not uniform and increasing; the trapezoid rule "
            "and the vorticity differences of Trefftz C are written for a uniform grid"
        )
    return array


def plane_fields(y: object, *, z: object, fields: Mapping[str, np.ndarray]) -> PlaneFields:
    """Assemble the samples of a far-field plane.

    Parameters
    ----------
    y, z : array_like
        Uniform grid axes in m, each with at least three points.
    fields : mapping of str to numpy.ndarray
        Sampled fields shaped ``(len(y), len(z))``.

    Returns
    -------
    PlaneFields
        The frozen container the Trefftz C integrals take.

    Raises
    ------
    FarfieldInputError
        If coordinates or field arrays do not form the declared rectangular plane.
    """
    y_axis = _uniform_axis("y", y)
    z_axis = _uniform_axis("z", z)
    shape = (len(y_axis), len(z_axis))
    values = {}
    for name, array in fields.items():
        data = np.asarray(array, dtype=float)
        if data.shape != shape:
            raise FarfieldInputError(
                f"field {name!r} has shape {data.shape}, but this plane samples {shape} (y, z)"
            )
        values[name] = data
    return PlaneFields(y=y_axis, z=z_axis, values=MappingProxyType(values))
