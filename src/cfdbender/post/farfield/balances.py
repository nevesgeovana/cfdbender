"""The conservation balances of a survey lattice.

Pipeline role: the mass, momentum, swirl and crossflow-kinetic-energy balances
of the far-field extraction, on the samples of a
:class:`~cfdbender.post.farfield.fields.LatticeFields`. Every integral goes
through :func:`~cfdbender.post.farfield.quadrature.plane_integral`; the
crossflow kinetic energy has one integrand, :func:`crossflow_integrand`, which
the probe-plane crossflow-energy drag calls too.

Solver asymmetry, by design:
the panel-method side is taken to run the purely kinematic, reversible balances
with the constant free-stream density; any cross-solver delta in a reversible
channel is then attributed to numerics, an assumption nothing measures.

Units: the ledgers are in the units of the field samples times
``tip_radius`` squared (forces and moments gain one further factor), with no
unit conversion in the balances. They are SI only when ``tip_radius`` is given
in meters and the fields are in SI; a radius held in the simulation's length
unit must be converted by the caller first, which is the same unit discipline
as the length unit of the survey's sample positions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import cast

import numpy as np

from cfdbender._errors import InputTypeError
from cfdbender.post.farfield.fields import FarfieldInputError, LatticeFields
from cfdbender.post.farfield.quadrature import (
    azimuthal_harmonics,
    delta_psi,
    half_spectrum,
    parseval_weights,
    plane_integral,
    ring_sample_weights,
)

__all__ = [
    "CrossflowEnergy",
    "InPlaneMoment",
    "MassClosure",
    "TransverseForce",
    "axial_flux",
    "axial_force",
    "crossflow_integrand",
    "crossflow_kinetic_energy",
    "in_plane_moment",
    "lateral_flux",
    "mass_closure",
    "mass_flux",
    "shaft_torque",
    "spurious_diagnostic",
    "to_counts",
    "transverse_flux",
    "transverse_force",
]


@dataclass(frozen=True)
class MassClosure:
    """Gate G1: the mass flow per station and its relative spread.

    Attributes
    ----------
    mdot : numpy.ndarray
        Mass flow through each annular plane, kg/s.
    relative_spread : float
        ``max |mdot - mean| / |mean|`` over the stations; NaN where any
        station is incomplete or where the mean is zero (see ``status``).
    absolute_spread : float
        ``max |mdot - mean|`` in kg/s; NaN where any station is incomplete.
    status : str
        ``"ok"`` when ``relative_spread`` is defined; ``"incomplete"`` where a
        station has a missing sample; ``"zero-mean-zero-spread"`` for a zero
        mean with every station equal (nothing flows anywhere); and
        ``"zero-mean-nonzero-spread"`` for a zero mean with unequal stations,
        where only ``absolute_spread`` carries the answer.
    """

    mdot: np.ndarray
    relative_spread: float
    absolute_spread: float = float("nan")
    status: str = "ok"


@dataclass(frozen=True)
class TransverseForce:
    """A transverse force with its lateral term reported separately (N)."""

    plane_term: float
    lateral_term: float
    total: float


@dataclass(frozen=True)
class InPlaneMoment:
    """An in-plane moment split into its loading and moment-arm terms (N m)."""

    loading_term: float
    moment_arm_term: float
    total: float


@dataclass(frozen=True)
class CrossflowEnergy:
    """The crossflow kinetic-energy flux per station, split by channel (W)."""

    total: np.ndarray
    swirl: np.ndarray
    induced: np.ndarray


def mass_flux(fields: LatticeFields, rho: float) -> np.ndarray:
    """Mass flow through each annular plane, ``integral(rho u dA)`` in kg/s.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples in the shaft-aligned axes of
        the survey lattice, with the units of
        :class:`~cfdbender.post.farfield.fields.LatticeFields`.
    rho : float
        Density in kg/m3.

    Returns
    -------
    np.ndarray
        Mass flow in kg/s for each station.
    """
    return plane_integral(fields, rho * fields["u"])


def mass_closure(fields: LatticeFields, rho: float) -> MassClosure:
    """Gate G1: station-to-station mass-flow closure diagnostic.

    Returns
    -------
    MassClosure
        ``mdot`` per station (kg/s) and the scalar ``relative_spread``
        (max deviation from the station mean over the mean); reported per
        run, judged against the case tolerance elsewhere.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples in the shaft-aligned axes of
        the survey lattice, with the units of
        :class:`~cfdbender.post.farfield.fields.LatticeFields`.
    rho : float
        Density in kg/m3.
    """
    mdot = mass_flux(fields, rho)
    mean = mdot.mean()
    absolute = float(np.abs(mdot - mean).max())
    if not np.isfinite(absolute):
        return MassClosure(
            mdot=mdot, relative_spread=float("nan"), absolute_spread=absolute, status="incomplete"
        )
    if mean == 0.0:
        status = "zero-mean-zero-spread" if absolute == 0.0 else "zero-mean-nonzero-spread"
        return MassClosure(
            mdot=mdot, relative_spread=float("nan"), absolute_spread=absolute, status=status
        )
    return MassClosure(
        mdot=mdot, relative_spread=absolute / float(abs(mean)), absolute_spread=absolute
    )


def axial_flux(fields: LatticeFields, rho: float, v_inf: float) -> np.ndarray:
    """Axial momentum-plus-pressure flux per station (N).

    ``integral(rho u (u - v_inf) + p_prime) dA`` with the axial perturbation
    ``u' = u - v_inf``.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples in the shaft-aligned axes of
        the survey lattice, with the units of
        :class:`~cfdbender.post.farfield.fields.LatticeFields`.
    rho : float
        Density in kg/m3.
    v_inf : float
        Free-stream speed in m/s.

    Returns
    -------
    np.ndarray
        Axial momentum-plus-pressure flux in N for each station.
    """
    u = fields["u"]
    return plane_integral(fields, rho * u * (u - v_inf) + fields["p_prime"])


def axial_force(
    fields: LatticeFields, rho: float, v_inf: float, *, inlet: float, outlet: float
) -> float:
    """Axial force from the control volume between two stations (N).

    Parameters
    ----------
    fields : LatticeFields
        Survey samples.
    rho : float
        Density (kg/m3); the free-stream value on the FlightStream side.
    v_inf : float
        Free-stream axial velocity (m/s).
    inlet, outlet : float
        Station coordinates x/R of the inlet and outlet planes, matched
        exactly; the lateral cylinder is assumed at free stream, which the
        lateral ring probes must verify.

    Returns
    -------
    float
        ``F_X`` = outlet flux minus inlet flux, +x downstream, so a propulsive
        disk yields positive axial force.
    """
    flux = axial_flux(fields, rho, v_inf)
    return float(flux[fields.station_index(outlet)] - flux[fields.station_index(inlet)])


def transverse_flux(
    fields: LatticeFields,
    rho: float,
    *,
    component: str = "w",
    method: str = "quadrature",
) -> np.ndarray:
    """Transverse momentum flux ``integral(rho u c dA)`` per station (N).

    Parameters
    ----------
    fields : LatticeFields
        Survey samples.
    rho : float
        Density (kg/m3).
    component : str
        Transverse velocity component: ``"w"`` (+z) for the F_Z / M_Y balance,
        ``"v"`` (+y) for F_Y / M_Z.
    method : str
        ``"quadrature"`` is the direct product integral; ``"harmonic"``
        evaluates the same number from the azimuthal spectra of ``u`` and the
        component (Parseval), the second, independent code path the
        definition demands.

    Returns
    -------
    np.ndarray
        Transverse momentum flux in N for each station.

    Raises
    ------
    FarfieldInputError
        If the integration method is unsupported.
    """
    if method == "quadrature":
        return plane_integral(fields, rho * fields["u"] * fields[component])
    if method != "harmonic":
        raise FarfieldInputError(f"unknown method {method!r}; use 'quadrature' or 'harmonic'")
    # The spectrum runs to the true half-band, Nyquist INCLUDED: a
    # Parseval sum needs every order the grid carries or it is not the same
    # number, and an omitted Nyquist order made the two documented independent
    # verifications disagree silently (quadrature pi, harmonic 0.0 at 16
    # azimuths). Orders 1..(n-1)/2 count twice, order 0 once, and for EVEN n
    # the Nyquist order n/2 is its own conjugate and counts ONCE.
    delta_psi(fields.psi)
    n_psi = len(fields.psi)
    half = n_psi // 2
    u_hat = half_spectrum(fields["u"], half)
    c_hat = half_spectrum(fields[component], half)
    weights = parseval_weights(n_psi, half)
    product_mean = (np.real(u_hat * np.conj(c_hat)) * weights).sum(axis=-1)
    ring_area = ring_sample_weights(fields) * n_psi
    scale = float(fields.tip_radius) ** 2
    return cast(np.ndarray, (rho * product_mean * ring_area).sum(axis=-1) * scale)


def _lateral_station(stations: np.ndarray, station: float) -> np.ndarray:
    """Return the indices of the lateral ring at ``station``: exact first, then within roundoff."""
    found = np.flatnonzero(stations == station)
    if found.size == 0:
        found = np.flatnonzero(np.isclose(stations, station, rtol=1e-12, atol=1e-12))
    return found


def lateral_flux(
    fields: LatticeFields,
    *,
    rho: float,
    inlet: float,
    outlet: float,
    component: str = "w",
) -> float:
    """Momentum-plus-pressure flux through the lateral cylinder between two stations (N).

    ``D_lat = integral over the cylinder of (rho v_r c + p' n_c) dA`` with
    ``c`` the transverse component (``w`` for F_Z, ``v`` for F_Y) and ``n_c``
    its share of the outward normal (``cos psi`` for ``w``, ``sin psi`` for
    ``v``): the azimuthal rectangle rule around each lateral ring and the
    trapezoid rule along x over the lateral stations from ``inlet`` to
    ``outlet``. NaN where a ring sample is missing. Stations match within
    ``1e-12`` as the annular planes do. The far-field family definition states
    the balance from the inlet station downstream and no convention for a
    reversed pair, so an ``outlet`` upstream of ``inlet`` is refused rather
    than given a sign; equal stations bound no cylinder and give ``0.0``.

    Raises
    ------
    FarfieldInputError
        If the lattice has no lateral cylinder, its samples carry no ``v_r``,
        ``p_prime`` or component, ``inlet`` or ``outlet`` is not a lateral
        station, or ``outlet`` lies upstream of ``inlet``.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples in the shaft-aligned axes of
        the survey lattice, with the units of
        :class:`~cfdbender.post.farfield.fields.LatticeFields`.
    rho : float
        Density in kg/m3.
    inlet : float
        Inlet station coordinate x/R.
    outlet : float
        Outlet station coordinate x/R.
    component : str
        Transverse velocity field name: v for +y or w for +z.

    Returns
    -------
    float
        Lateral-cylinder momentum-plus-pressure flux in N.
    """
    if fields.lateral_radius is None or len(fields.lateral_stations) == 0:
        raise FarfieldInputError("the lattice has no lateral cylinder to integrate over")
    lateral = fields.lateral_values
    for name in ("v_r", "p_prime", component):
        if name not in lateral:
            raise FarfieldInputError(f"the lateral rings carry no field {name!r}")
    stations = fields.lateral_stations
    start = _lateral_station(stations, inlet)
    stop = _lateral_station(stations, outlet)
    if start.size != 1 or stop.size != 1:
        raise FarfieldInputError(
            f"the lateral rings have no single station at x/R = {inlet!r} or {outlet!r} "
            f"(within 1e-12); they are at {[float(s) for s in stations]}"
        )
    i0, i1 = int(start[0]), int(stop[0])
    if i1 < i0:
        raise FarfieldInputError(
            f"the outlet x/R = {outlet!r} lies upstream of the inlet x/R = {inlet!r}; "
            "the lateral term is defined from the inlet downstream, so give the inlet first"
        )
    normal = np.cos(fields.psi) if component == "w" else np.sin(fields.psi)
    integrand = rho * lateral["v_r"] * lateral[component] + lateral["p_prime"] * normal
    tip = float(fields.tip_radius)
    ring = integrand.sum(axis=-1) * delta_psi(fields.psi) * float(fields.lateral_radius) * tip
    x = stations[i0 : i1 + 1] * tip
    segment = ring[i0 : i1 + 1]
    return float((0.5 * (segment[1:] + segment[:-1]) * np.diff(x)).sum())


def transverse_force(
    fields: LatticeFields,
    rho: float,
    *,
    inlet: float,
    outlet: float,
    component: str = "w",
    lateral_flux: float = 0.0,
) -> TransverseForce:
    """Transverse force with the lateral term reported separately (N).

    ``F_Z = integral_out(rho u w dA) - integral_in(rho u w dA) + Delta_lat``. The
    lateral-cylinder momentum and pressure flux is
    never folded silently into the plane term: it is a separate value of the
    result, zero only when zero incidence makes it so.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples.
    rho : float
        Density (kg/m3).
    inlet, outlet : float
        Station coordinates x/R of the bounding planes.
    component : str
        ``"w"`` for F_Z, ``"v"`` for F_Y.
    lateral_flux : float
        Lateral-surface momentum-plus-pressure flux (N), from
        :func:`lateral_flux`; the default 0.0 is legitimate only at zero
        incidence, where symmetry nulls it.

    Returns
    -------
    TransverseForce
        Plane, lateral and total force terms in N.
    """
    flux = transverse_flux(fields, rho, component=component)
    plane_term = float(flux[fields.station_index(outlet)] - flux[fields.station_index(inlet)])
    return TransverseForce(
        plane_term=plane_term,
        lateral_term=float(lateral_flux),
        total=plane_term + float(lateral_flux),
    )


def shaft_torque(fields: LatticeFields, rho: float) -> np.ndarray:
    """Swirl flux ``integral(rho (r v_theta) u dA)`` per station (N m).

    The order-0 ``v_theta`` content weighted by the dimensional moment arm
    ``r``; the sign follows the ``v_theta`` convention of
    :func:`~cfdbender.post.farfield.fields.cylindrical_components`.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples in the shaft-aligned axes of
        the survey lattice, with the units of
        :class:`~cfdbender.post.farfield.fields.LatticeFields`.
    rho : float
        Density in kg/m3.

    Returns
    -------
    np.ndarray
        Shaft torque flux in N m for each station.
    """
    arm = fields.ring_centers * float(fields.tip_radius)
    return plane_integral(fields, rho * arm[:, None] * fields["v_theta"] * fields["u"])


def in_plane_moment(
    fields: LatticeFields,
    rho: float,
    v_inf: float,
    *,
    inlet: float,
    outlet: float,
    axis: str = "y",
    method: str = "quadrature",
) -> InPlaneMoment:
    """In-plane moment about an axis through the disk center (N m).

    For ``axis="y"``: ``M_Y = integral(z g dA) -
    x_p integral(rho u w dA)`` evaluated at the outlet minus the same terms at
    the inlet, with ``g = rho u u' + p_prime`` and ``z = r cos(psi)``. The two
    contributions stay separate: the loading term is the order-1 cosine
    harmonic of the axial flux (the 1P loading asymmetry) and the moment-arm
    term carries the transverse flux. Both axes follow the right-handed cross
    product: ``axis="z"`` gives ``M_Z = x_p integral(rho u v dA) -
    integral(y g dA)``, with ``y = r sin(psi)``.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples.
    rho, v_inf : float
        Density (kg/m3) and free-stream axial velocity (m/s).
    inlet, outlet : float
        Station coordinates x/R of the bounding planes; the moment arm of each
        plane is its own station position.
    axis : str
        ``"y"`` or ``"z"``, the moment axis through the disk center.
    method : str
        ``"quadrature"`` weights the axial flux by the sampled lever arm
        directly; ``"harmonic"`` rebuilds the loading term from the order-1
        azimuthal coefficient. Two independent code paths.

    Returns
    -------
    InPlaneMoment
        Loading, moment-arm and total moment terms in N m. The total is
        plane-only: unlike :func:`transverse_force` it has no lateral-cylinder
        term, so it is valid where the lateral flux vanishes (zero incidence,
        or a lateral cylinder at free stream).

    Raises
    ------
    FarfieldInputError
        If axis or integration method is unsupported.
    """
    r = fields.ring_centers
    if axis == "y":
        lever = r[:, None] * np.cos(fields.psi)[None, :]
        component = "w"
    elif axis == "z":
        lever = -r[:, None] * np.sin(fields.psi)[None, :]
        component = "v"
    else:
        raise FarfieldInputError(f"axis must be 'y' or 'z', got {axis!r}")

    u = fields["u"]
    g = rho * u * (u - v_inf) + fields["p_prime"]
    tip = float(fields.tip_radius)
    if method == "quadrature":
        loading = plane_integral(fields, lever * tip * g)
    elif method == "harmonic":
        # sum_k cos(psi_k) g = N Re[g_hat_1]; sum_k sin(psi_k) g = -N Im[g_hat_1]
        g_hat_1 = azimuthal_harmonics(g, psi=fields.psi, m_max=1)[..., 1]
        projected = np.real(g_hat_1) if axis == "y" else np.imag(g_hat_1)
        ring_area = ring_sample_weights(fields) * len(fields.psi)
        loading = (projected * r * ring_area).sum(axis=-1) * tip**3
    else:
        raise FarfieldInputError(f"unknown method {method!r}; use 'quadrature' or 'harmonic'")

    arm_flux = transverse_flux(fields, rho, component=component)
    if axis == "y":
        arm = -fields.stations * tip * arm_flux
    else:
        arm = fields.stations * tip * arm_flux
    out, inn = fields.station_index(outlet), fields.station_index(inlet)
    loading_term = float(loading[out] - loading[inn])
    moment_arm_term = float(arm[out] - arm[inn])
    return InPlaneMoment(
        loading_term=loading_term,
        moment_arm_term=moment_arm_term,
        total=loading_term + moment_arm_term,
    )


def crossflow_integrand(
    rho: float, *, first: np.ndarray, second: np.ndarray, advecting: np.ndarray | float
) -> np.ndarray:
    """Return the one crossflow kinetic-energy integrand, ``(rho / 2) (a^2 + b^2) u_adv``.

    The balance passes the cylindrical components ``v_r``, ``v_theta`` and the
    sampled axial velocity ``u``; Trefftz C passes the Cartesian crossflow
    ``v``, ``w`` and the free-stream speed ``V``. The
    integrand is transverse components only: it contains no axial-deficit term
    by construction, a hard rule of the design note.

    Parameters
    ----------
    rho : float
        Density in kg/m3.
    first : np.ndarray
        First transverse velocity component, in m/s.
    second : np.ndarray
        Second transverse velocity component, in m/s.
    advecting : np.ndarray | float
        Advecting velocity in m/s, scalar or sampled with the components.

    Returns
    -------
    np.ndarray
        Crossflow kinetic-energy flux density at each sample.
    """
    return cast(np.ndarray, 0.5 * rho * (first**2 + second**2) * advecting)


def crossflow_kinetic_energy(fields: LatticeFields, rho: float) -> CrossflowEnergy:
    """Crossflow kinetic-energy flux split into swirl and induced content (W).

    ``E = 0.5 integral(rho (v_r^2 + v_theta^2) u dA)`` per station, with the
    axisymmetric (order-0) ``v_theta`` part reported as the stator-recoverable
    swirl channel and the remainder as the non-axisymmetric induced channel.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples in the shaft-aligned axes of
        the survey lattice, with the units of
        :class:`~cfdbender.post.farfield.fields.LatticeFields`.
    rho : float
        Density in kg/m3.

    Returns
    -------
    CrossflowEnergy
        Total, swirl and induced energy flux in W for each station.
    """
    v_theta_0 = fields["v_theta"].mean(axis=-1)
    u = fields["u"]
    total = plane_integral(
        fields, crossflow_integrand(rho, first=fields["v_r"], second=fields["v_theta"], advecting=u)
    )
    swirl = plane_integral(fields, 0.5 * rho * v_theta_0[..., None] ** 2 * u)
    return CrossflowEnergy(total=total, swirl=swirl, induced=total - swirl)


def _requires_reference_area[**CountParameters](
    function: Callable[CountParameters, float],
) -> Callable[CountParameters, float]:
    """Explain an omitted keyword without supplying a physical reference area."""

    @wraps(function)
    def checked(*args: CountParameters.args, **kwargs: CountParameters.kwargs) -> float:
        if "s_ref" not in kwargs:
            raise InputTypeError(
                "s_ref is required: supply the reference area in m2 used by the load "
                "coefficients; there is no safe default"
            )
        return function(*args, **kwargs)

    return checked


@_requires_reference_area
def to_counts(delta: float, *, rho_inf: float, v_inf: float, s_ref: float) -> float:
    """Express a force delta in drag counts.

    One count is ``1e-4`` of the dynamic-pressure force scale:
    ``counts = 2 delta / (rho_inf v_inf^2 s_ref) * 1e4``.

    Parameters
    ----------
    delta : float
        Force difference (N).
    rho_inf, v_inf : float
        Free-stream density (kg/m3) and speed (m/s).
    s_ref : float
        Explicit reference area (m2), the same area used by the load coefficients.

    Raises
    ------
    TypeError
        If s_ref is omitted: the reference area in m2 has no safe default.
    FarfieldInputError
        If density, speed or area is not finite and strictly positive.
        Dynamic-pressure counts are undefined in hover; retain dimensional loads there.

    Returns
    -------
    float
        Force delta expressed in drag counts.
    """
    for name, value in (("rho_inf", rho_inf), ("v_inf", v_inf), ("s_ref", s_ref)):
        if not np.isfinite(value) or value <= 0:
            raise FarfieldInputError(
                f"{name}={value!r} must be finite and strictly positive for drag counts; "
                "use dimensional loads when the dynamic-pressure scale vanishes"
            )
    return 2.0 * delta / (rho_inf * v_inf**2 * s_ref) * 1.0e4


@_requires_reference_area
def spurious_diagnostic(
    near_field: float,
    far_field: float,
    *,
    rho_inf: float,
    v_inf: float,
    s_ref: float,
) -> float:
    """Inviscid near-field minus far-field load, in counts (gate G4).

    The residual between the integrated inviscid near-field load and the far-field
    balance is treated as numerical, not physical (no report measures it); reported in
    counts so runs are
    comparable across cases and solvers.

    Parameters
    ----------
    near_field : float
        Integrated inviscid near-field force in N (total minus viscous projection).
    far_field : float
        Far-field force balance in N.
    rho_inf : float
        Free-stream density in kg/m3.
    v_inf : float
        Free-stream speed in m/s.
    s_ref : float
        Reference area in m2 used by the load coefficients.

    Raises
    ------
    TypeError
        If s_ref is omitted: the reference area in m2 has no safe default.

    Returns
    -------
    float
        Near-field minus far-field force expressed in drag counts.
    """
    return to_counts(near_field - far_field, rho_inf=rho_inf, v_inf=v_inf, s_ref=s_ref)
