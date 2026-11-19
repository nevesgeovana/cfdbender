"""The Trefftz-plane integrals: the discrete trace (A) and the probe plane (C), and rotor channels.

Pipeline role: the far-field lift, side force and induced drag of a configuration, from a
discrete trailing-vortex sheet in the Trefftz plane (variant A) or from the velocities
sampled on a plane of probes (variant C), and the far-field loss channels of a rotor. The
equations are those of the definitions page, The far-field family; every component here is
already in the axes the docstring names, turned by the caller, and nothing here reads or
writes a file.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from cfdbender.post.farfield.balances import (
    axial_force,
    crossflow_integrand,
    crossflow_kinetic_energy,
)
from cfdbender.post.farfield.fields import FarfieldInputError, LatticeFields, PlaneFields
from cfdbender.post.farfield.quadrature import (
    grid_coverage,
    grid_integral,
    grid_weights,
    plane_integral,
    sample_area_coverage,
    strip_edges,
)

__all__ = [
    "PlaneDrag",
    "RotorChannels",
    "TraceResult",
    "bernoulli_residual",
    "plane_induced_drag",
    "plane_lift",
    "plane_vorticity_drag",
    "rotor_channels",
    "trace_coefficients",
    "trace_forces",
]


def _induced(
    py: np.ndarray, pz: np.ndarray, fy: np.ndarray, fz: np.ndarray, d_gamma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Crossflow ``(v, w)`` at points ``p`` from trailing filaments at ``f``.

    A filament shedding ``dGamma`` has circulation ``-dGamma`` about +x.
    """
    ry = py[:, None] - fy[None, :]
    rz = pz[:, None] - fz[None, :]
    r2 = ry * ry + rz * rz
    silent = d_gamma == 0.0
    # A filament that sheds no circulation induces nothing, even where a point of
    # another sheet sits on it (0/0 otherwise); a nonzero one stays singular there.
    if np.any((r2 == 0.0) & ~silent[None, :]):
        raise FarfieldInputError(
            "a trace point lies on a shed filament, where the induced velocity is singular; "
            "place the strip edges away from the stations (the 'extended' rule does)"
        )
    safe = np.where(silent[None, :] & (r2 == 0.0), 1.0, r2)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(silent[None, :], 0.0, -d_gamma[None, :] / (2.0 * math.pi * safe))
    v = (factor * -rz).sum(axis=1)
    w = (factor * ry).sum(axis=1)
    return v, w


def _require_density(rho: float) -> None:
    """Refuse a density that is not finite and positive."""
    if not (math.isfinite(rho) and rho > 0.0):
        raise FarfieldInputError(
            f"the density must be finite and positive, got {rho!r}; a negative density "
            "would reverse the sign of the induced drag"
        )


@dataclass(frozen=True, eq=False)
class TraceResult:
    """The forces of a discrete trailing-vortex trace and its strips.

    Forces are those of the traced (mirrored, where asked) configuration, in N;
    the per-strip arrays run over every strip of the trace.

    Attributes
    ----------
    lift, side_force, induced_drag : float
        ``L``, ``Y`` and ``D_i`` of the whole trace, N.
    sheets : tuple of int
        The sheet labels, in order of first appearance.
    sheet_lift, sheet_side_force : numpy.ndarray
        Per sheet, N.
    sheet_drag_share : numpy.ndarray
        Per sheet, the drag of its strips under the normalwash of every
        sheet's filaments; the shares add up to ``induced_drag``.
    sheet_drag_isolated : numpy.ndarray
        Per sheet, the drag of its strips under its own filaments only.
    y, z, dy, dz, ds, gamma : numpy.ndarray
        Per strip: the point, the strip's projections and length (m), and the
        circulation (m^2/s).
    normalwash : numpy.ndarray
        Per strip, the Trefftz-plane velocity along the strip's upward normal,
        induced by every sheet (m/s).
    sheet : numpy.ndarray
        Per strip, its sheet label.
    original : numpy.ndarray
        Per strip, True where the strip is a station of the input (False on a
        mirror image).
    """

    lift: float
    side_force: float
    induced_drag: float
    sheets: tuple[int, ...]
    sheet_lift: np.ndarray
    sheet_side_force: np.ndarray
    sheet_drag_share: np.ndarray
    sheet_drag_isolated: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    ds: np.ndarray
    gamma: np.ndarray
    normalwash: np.ndarray
    sheet: np.ndarray
    original: np.ndarray

    @property
    def downwash(self) -> np.ndarray:
        """Per strip, the downwash at the lifting line, m/s, positive down."""
        return -0.5 * self.normalwash

    def scaled(self, k: float) -> TraceResult:
        """Return the trace of the circulation ``k Gamma``: forces times ``k`` or ``k^2``."""
        k2 = k * k
        return replace(
            self,
            lift=self.lift * k,
            side_force=self.side_force * k,
            induced_drag=self.induced_drag * k2,
            sheet_lift=self.sheet_lift * k,
            sheet_side_force=self.sheet_side_force * k,
            sheet_drag_share=self.sheet_drag_share * k2,
            sheet_drag_isolated=self.sheet_drag_isolated * k2,
            gamma=self.gamma * k,
            normalwash=self.normalwash * k,
        )


def _full_sheet(
    y: np.ndarray, z: np.ndarray, gamma: np.ndarray, mirror: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original = np.ones(len(y), dtype=bool)
    if not mirror:
        return y, z, gamma, original
    if not (np.all(y >= 0.0) or np.all(y <= 0.0)) or not np.any(y != 0.0):
        raise FarfieldInputError(
            "a mirrored trace needs every station on one side of the symmetry plane; "
            "these straddle it, and the export cannot say which half was meant"
        )
    order = np.argsort(y, kind="stable")
    y, z, gamma = y[order], z[order], gamma[order]
    if np.any(np.diff(y) <= 0.0):
        raise FarfieldInputError("a mirrored trace needs distinct station positions along y")
    # A station on the mirror plane is shared, not a second coincident strip.
    reflected = np.flatnonzero(y != 0.0)[::-1]
    if y[0] >= 0.0:
        return (
            np.concatenate([-y[reflected], y]),
            np.concatenate([z[reflected], z]),
            np.concatenate([gamma[reflected], gamma]),
            np.concatenate([np.zeros(len(reflected), dtype=bool), original]),
        )
    return (
        np.concatenate([y, -y[reflected]]),
        np.concatenate([z, z[reflected]]),
        np.concatenate([gamma, gamma[reflected]]),
        np.concatenate([original, np.zeros(len(reflected), dtype=bool)]),
    )


def trace_forces(
    y: np.ndarray,
    *,
    z: np.ndarray,
    gamma: np.ndarray,
    rho: float,
    v_inf: float,
    sheets: Sequence[int] | np.ndarray | None = None,
    ends: Literal["extended", "station"] = "extended",
    mirror: bool = False,
) -> TraceResult:
    """Lift, side force and induced drag of a discrete trace (Trefftz A).

    The trace is a list of points ``(y, z)`` in the Trefftz plane, in wind
    axes (m), each carrying the circulation of the strip around it, grouped in
    sheets and ordered along each sheet. Strip edges follow
    :func:`~cfdbender.post.farfield.quadrature.strip_edges`; a filament leaves
    every edge with the circulation shed there; the normalwash at each point is
    induced by every sheet's filaments; and

    ``L = rho V sum Gamma dy``, ``Y = -rho V sum Gamma dz``,
    ``D_i = -(rho / 2) sum Gamma v_n ds``.

    Parameters
    ----------
    y, z : numpy.ndarray
        The trace points, m, shape ``(n,)``.
    gamma : numpy.ndarray
        Circulation of each strip, m^2/s (positive for positive lift on a
        sheet running toward +y).
    rho, v_inf : float
        Density (kg/m3) and free-stream speed (m/s).
    sheets : sequence of int, optional
        Sheet label per point; one sheet when omitted. Each sheet is traced in
        the order its points are given.
    ends : {"extended", "station"}
        The strip-edge rule beyond the end points; the far field uses
        ``"extended"``.
    mirror : bool
        Mirror each sheet about the plane ``y = 0`` before the integral (a half
        model); the inner edge of the innermost strip then lies on
        the plane.

    Returns
    -------
    TraceResult
        The forces of the traced configuration and the per-strip arrays.

    Raises
    ------
    FarfieldInputError
        If trace samples, density, symmetry or integration settings are invalid.
    """
    _require_density(rho)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if not (y.shape == z.shape == gamma.shape) or y.ndim != 1:
        raise FarfieldInputError("the trace needs one y, z and Gamma per point")
    labels = np.zeros(len(y), dtype=int) if sheets is None else np.asarray(sheets, dtype=int)
    if labels.shape != y.shape:
        raise FarfieldInputError("the trace needs one sheet label per point")
    order: list[int] = []
    for label in labels.tolist():
        if label not in order:
            order.append(label)

    parts = []
    for label in order:
        pick = labels == label
        sy, sz, sg, original = _full_sheet(y[pick], z[pick], gamma[pick], mirror)
        if len(sy) < 2:
            raise FarfieldInputError(f"sheet {label} has fewer than two points")
        ey = strip_edges(sy, ends=ends)
        ez = strip_edges(sz, ends=ends)
        dy, dz = np.diff(ey), np.diff(ez)
        if np.any(np.hypot(np.diff(sy), np.diff(sz)) == 0.0):
            raise FarfieldInputError(
                f"sheet {label} has coincident trace points, so a strip has zero length"
            )
        d_gamma = np.diff(np.concatenate([[0.0], sg, [0.0]]))
        parts.append((label, sy, sz, sg, original, ey, ez, dy, dz, d_gamma))

    all_y = np.concatenate([p[1] for p in parts])
    all_z = np.concatenate([p[2] for p in parts])
    all_fy = np.concatenate([p[5] for p in parts])
    all_fz = np.concatenate([p[6] for p in parts])
    all_dg = np.concatenate([p[9] for p in parts])
    v_all, w_all = _induced(all_y, all_z, all_fy, all_fz, all_dg)

    sheet_lift, sheet_side, share, isolated = [], [], [], []
    strips: dict[str, list[np.ndarray]] = {
        key: [] for key in ("y", "z", "dy", "dz", "ds", "gamma", "vn", "sheet", "original")
    }
    start = 0
    for label, sy, sz, sg, original, ey, ez, dy, dz, d_gamma in parts:
        n = len(sy)
        ds = np.hypot(dy, dz)
        v, w = v_all[start : start + n], w_all[start : start + n]
        vn = (-dz * v + dy * w) / ds
        v_own, w_own = _induced(sy, sz, ey, ez, d_gamma)
        vn_own = (-dz * v_own + dy * w_own) / ds
        sheet_lift.append(rho * v_inf * float((sg * dy).sum()))
        sheet_side.append(-rho * v_inf * float((sg * dz).sum()))
        share.append(-0.5 * rho * float((sg * vn * ds).sum()))
        isolated.append(-0.5 * rho * float((sg * vn_own * ds).sum()))
        for key, value in (
            ("y", sy),
            ("z", sz),
            ("dy", dy),
            ("dz", dz),
            ("ds", ds),
            ("gamma", sg),
            ("vn", vn),
            ("sheet", np.full(n, label)),
            ("original", original),
        ):
            strips[key].append(value)
        start += n

    lift = np.asarray(sheet_lift)
    side = np.asarray(sheet_side)
    drag_share = np.asarray(share)
    return TraceResult(
        lift=float(lift.sum()),
        side_force=float(side.sum()),
        induced_drag=float(drag_share.sum()),
        sheets=tuple(order),
        sheet_lift=lift,
        sheet_side_force=side,
        sheet_drag_share=drag_share,
        sheet_drag_isolated=np.asarray(isolated),
        y=np.concatenate(strips["y"]),
        z=np.concatenate(strips["z"]),
        dy=np.concatenate(strips["dy"]),
        dz=np.concatenate(strips["dz"]),
        ds=np.concatenate(strips["ds"]),
        gamma=np.concatenate(strips["gamma"]),
        normalwash=np.concatenate(strips["vn"]),
        sheet=np.concatenate(strips["sheet"]),
        original=np.concatenate(strips["original"]),
    )


def trace_coefficients(
    lift: float,
    *,
    side_force: float,
    induced_drag: float,
    q: float,
    s_ref: float,
    b_ref: float,
) -> dict[str, float]:
    """Return the coefficients ``CL_T``, ``CY_T``, ``CDI_T``, ``E_T`` and ``AR`` of a result.

    ``CL_T = L / (q SREF)`` and likewise, ``AR = BREF^2 / SREF`` and
    ``E_T = CL_T^2 / (pi AR CDI_T)``, NaN where ``CDI_T`` is zero or not finite
    or a reference value is missing.

    Parameters
    ----------
    lift : float
        Lift force in N.
    side_force : float
        Side force in N.
    induced_drag : float
        Induced drag force in N.
    q : float
        Free-stream dynamic pressure in Pa.
    s_ref : float
        Reference area in m2 used by the load coefficients.
    b_ref : float
        Reference span in m.

    Returns
    -------
    dict[str, float]
        Trefftz coefficients and aspect ratio, with NaN for undefined values.
    """
    nan = float("nan")
    if not (q > 0.0 and s_ref > 0.0 and math.isfinite(q * s_ref)):
        return {"CL_T": nan, "CY_T": nan, "CDI_T": nan, "E_T": nan, "AR": nan}
    scale = q * s_ref
    cl, cy, cdi = lift / scale, side_force / scale, induced_drag / scale
    aspect = b_ref * b_ref / s_ref if b_ref > 0.0 and math.isfinite(b_ref) else nan
    e = nan
    if math.isfinite(cdi) and cdi != 0.0 and math.isfinite(aspect):
        e = cl * cl / (math.pi * aspect * cdi)
    return {"CL_T": cl, "CY_T": cy, "CDI_T": cdi, "E_T": e, "AR": aspect}


@dataclass(frozen=True)
class PlaneDrag:
    """The crossflow-energy induced drag of a probe plane (Trefftz C).

    Attributes
    ----------
    induced_drag : float
        ``D_i`` by the trapezoid rule on the whole grid, N.
    coarse_drag : float
        The same samples integrated on every other grid point, N; NaN where
        the grid's point counts do not allow it to span the same extent.
    quadrature_error : float
        ``induced_drag - coarse_drag``, N.
    coverage : float
        The fraction of the plane's AREA whose crossflow samples are finite,
        each sample weighted by the area its trapezoid weight integrates.
    """

    induced_drag: float
    coarse_drag: float
    quadrature_error: float
    coverage: float


def plane_induced_drag(plane: PlaneFields, *, rho: float, v_inf: float) -> PlaneDrag:
    """Induced drag from the crossflow kinetic energy on a plane.

    ``D_i = (rho / 2) integral (v^2 + w^2) dS``: the crossflow-energy
    integrand of the balances advected by ``V`` and divided by it, so the
    balance and Trefftz C share one integrand. NaN at ``V = 0``.

    Parameters
    ----------
    plane : PlaneFields
        Rectangular probe-plane samples in free-stream axes.
    rho : float
        Density in kg/m3.
    v_inf : float
        Free-stream speed in m/s.

    Returns
    -------
    PlaneDrag
        Fine and coarse induced drag in N, their difference and sample coverage.
    """
    _require_density(rho)
    v, w = plane["v"], plane["w"]
    integrand = crossflow_integrand(rho, first=v, second=w, advecting=v_inf)
    coverage = grid_coverage(integrand, y=plane.y, z=plane.z)
    if v_inf == 0.0:
        nan = float("nan")
        return PlaneDrag(nan, nan, nan, coverage)
    fine = grid_integral(plane.y, z=plane.z, integrand=integrand) / v_inf
    coarse = float("nan")
    if (len(plane.y) - 1) % 2 == 0 and (len(plane.z) - 1) % 2 == 0:
        coarse = grid_integral(plane.y[::2], z=plane.z[::2], integrand=integrand[::2, ::2]) / v_inf
    return PlaneDrag(fine, coarse, fine - coarse, coverage)


def _cell_log_integral(hy: float, hz: float) -> float:
    """``integral of ln r`` over a ``hy`` by ``hz`` cell centered on the origin."""
    a, b = 0.5 * hy, 0.5 * hz
    return 2.0 * (
        a * b * (math.log(a * a + b * b) - 3.0)
        + a * a * math.atan(b / a)
        + b * b * math.atan(a / b)
    )


def plane_vorticity_drag(plane: PlaneFields, *, rho: float, chunk: int = 512) -> float:
    """Induced drag from the streamwise vorticity on a plane (Maskell).

    ``xi = dw/dy - dv/dz`` by central differences on the grid, its stream
    function ``psi(P) = -(1 / (2 pi)) sum_Q xi_Q ln|P - Q| dS_Q`` (the cell's
    own logarithm integrated exactly), and ``D = (rho / 2) sum_P psi_P xi_P
    dS_P``. NaN where any sample is missing.

    Parameters
    ----------
    plane : PlaneFields
        Rectangular probe-plane samples in free-stream axes.
    rho : float
        Density in kg/m3.
    chunk : int
        Number of source cells processed together to bound temporary memory;
        a positive integer.

    Returns
    -------
    float
        Induced drag in N from the plane vorticity.

    Raises
    ------
    FarfieldInputError
        If the density is not finite and positive, or ``chunk`` is not a
        positive integer.
    """
    _require_density(rho)
    try:
        whole = float(chunk).is_integer() and chunk >= 1
    except (TypeError, ValueError):
        whole = False
    if isinstance(chunk, bool) or not whole:
        raise FarfieldInputError(f"chunk must be a positive integer, got {chunk!r}")
    v, w = plane["v"], plane["w"]
    dy = float(plane.y[1] - plane.y[0])
    dz = float(plane.z[1] - plane.z[0])
    xi = np.gradient(w, dy, axis=0) - np.gradient(v, dz, axis=1)
    if not np.all(np.isfinite(xi)):
        return float("nan")
    weights = grid_weights(plane.y, z=plane.z).ravel()
    yy, zz = np.meshgrid(plane.y, plane.z, indexing="ij")
    py, pz, source = yy.ravel(), zz.ravel(), (xi.ravel() * weights)
    self_term = _cell_log_integral(dy, dz) / (dy * dz)
    psi = np.empty(py.size)
    for first in range(0, py.size, chunk):
        last = min(first + chunk, py.size)
        ry = py[first:last, None] - py[None, :]
        rz = pz[first:last, None] - pz[None, :]
        r2 = ry * ry + rz * rz
        rows = np.arange(last - first)
        r2[rows, np.arange(first, last)] = 1.0
        log_r = 0.5 * np.log(r2)
        log_r[rows, np.arange(first, last)] = 0.0
        total = log_r @ source
        total += source[first:last] * self_term
        psi[first:last] = -total / (2.0 * math.pi)
    return float(0.5 * rho * (psi * xi.ravel() * weights).sum())


def plane_lift(plane: PlaneFields, *, rho: float, v_inf: float) -> float:
    """Lift from the first moment of the streamwise vorticity on a plane.

    ``L = rho V sum_P y_P xi_P dS_P`` with ``xi = dw/dy - dv/dz`` by central
    differences on the grid: the Kutta-Joukowski lift of the trailing vorticity
    the plane holds, which for a flat sheet of strip circulation ``Gamma`` is
    ``rho V integral Gamma dy`` and which roll-up does not change. It does not
    depend on where ``y`` is measured from once the plane holds the whole wake
    (its vorticity then sums to zero). NaN where any sample is missing.

    Parameters
    ----------
    plane : PlaneFields
        Rectangular probe-plane samples in free-stream axes.
    rho : float
        Density in kg/m3.
    v_inf : float
        Free-stream speed in m/s.

    Returns
    -------
    float
        Lift in N, positive along the plane's ``z``.
    """
    _require_density(rho)
    v, w = plane["v"], plane["w"]
    dy = float(plane.y[1] - plane.y[0])
    dz = float(plane.z[1] - plane.z[0])
    xi = np.gradient(w, dy, axis=0) - np.gradient(v, dz, axis=1)
    if not np.all(np.isfinite(xi)):
        return float("nan")
    yy = np.broadcast_to(np.asarray(plane.y, dtype=float)[:, None], xi.shape)
    return float(rho * v_inf * (yy * xi * grid_weights(plane.y, z=plane.z)).sum())


def bernoulli_residual(plane: PlaneFields, *, rho: float, v_inf: float) -> float:
    """Return the Betz-Maskell term of a plane, in N.

    With the total-pressure deficit
    ``p0_inf - p0 = -p' - (rho / 2) (u^2 + v^2 + w^2 - V^2)`` and the
    artificial axial velocity ``u* = sqrt(u^2 + 2 (p0_inf - p0) / rho)``,
    ``integral over p0 < p0_inf of [(p0_inf - p0) + (rho / 2) (u* - u)
    (u* + u - 2 V)] dS``. Zero by construction in a field that holds
    Bernoulli's relation; written as a diagnostic, never as a drag.

    Parameters
    ----------
    plane : PlaneFields
        Rectangular probe-plane samples in free-stream axes.
    rho : float
        Density in kg/m3.
    v_inf : float
        Free-stream speed in m/s.

    Returns
    -------
    float
        Integrated Betz-Maskell diagnostic in N over the total-pressure deficit region.
    """
    u, v, w, p_prime = plane["u"], plane["v"], plane["w"], plane["p_prime"]
    deficit = -p_prime - 0.5 * rho * (u * u + v * v + w * w - v_inf * v_inf)
    lossy = deficit > 0.0
    u_star = np.sqrt(u * u + 2.0 * np.where(lossy, deficit, 0.0) / rho)
    integrand = np.where(
        lossy, deficit + 0.5 * rho * (u_star - u) * (u_star + u - 2.0 * v_inf), 0.0
    )
    integrand = np.where(np.isfinite(deficit), integrand, np.nan)
    return grid_integral(plane.y, z=plane.z, integrand=integrand)


@dataclass(frozen=True)
class RotorChannels:
    """The far-field loss channels of a rotor at one lattice station (W).

    ``thrust`` is the axial force of the balance between the inlet and the
    station (N); ``eta_ff`` is NaN at ``V = 0``.
    """

    station: float
    p_axial: float
    p_swirl: float
    p_crossflow: float
    p_total: float
    thrust: float
    eta_ff: float
    coverage: float


def rotor_channels(
    fields: LatticeFields, *, rho: float, v_inf: float, station: float
) -> RotorChannels:
    """Axial, swirl and crossflow loss powers of a rotor at a station.

    ``P_AXIAL = (rho / 2) I[u'^2 u]``, ``P_SWIRL`` and ``P_CROSSFLOW`` the swirl
    and induced channels of the crossflow kinetic energy (one integrand),
    ``P_TOTAL = I[(p0 - p0_inf) u]`` and
    ``ETA_FF = T V / (T V + P_AXIAL + P_SWIRL + P_CROSSFLOW)`` with ``T`` the
    axial force between the first station and this one.

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
    station : float
        Outlet station coordinate x/R.

    Returns
    -------
    RotorChannels
        Rotor loss powers in W, axial force in N, efficiency and coverage.
    """
    index = fields.station_index(station)
    u = fields["u"]
    u_prime = u - v_inf
    p_axial = plane_integral(fields, 0.5 * rho * u_prime * u_prime * u)[index]
    energy = crossflow_kinetic_energy(fields, rho)
    head = fields["p_prime"] + 0.5 * rho * (
        u * u + fields["v"] * fields["v"] + fields["w"] * fields["w"] - v_inf * v_inf
    )
    p_total = plane_integral(fields, head * u)[index]
    thrust = axial_force(fields, rho, v_inf, inlet=float(fields.stations[0]), outlet=station)
    losses = float(p_axial) + float(energy.swirl[index]) + float(energy.induced[index])
    useful = thrust * v_inf
    denominator = useful + losses
    eta = float("nan") if v_inf == 0.0 or denominator == 0.0 else useful / denominator
    # Coverage is the area where EVERY channel the powers read is finite, not the axial one alone.
    sampled = np.stack([u[index], fields["v"][index], fields["w"][index], fields["p_prime"][index]])
    coverage = float(
        sample_area_coverage(np.where(np.isfinite(sampled).all(axis=0), 1.0, np.nan), fields)
    )
    return RotorChannels(
        station=float(station),
        p_axial=float(p_axial),
        p_swirl=float(energy.swirl[index]),
        p_crossflow=float(energy.induced[index]),
        p_total=float(p_total),
        thrust=float(thrust),
        eta_ff=eta,
        coverage=coverage,
    )
