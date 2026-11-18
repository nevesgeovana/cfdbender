"""The quadratures of the far field, each implemented once.

Pipeline role: the annular-plane quadrature of the survey lattice, its
coverage, the trapezoid rule of a rectangular far-field plane, and the strip
edges of a spanwise trace. Every balance and every Trefftz integral calls these
functions; none of them has a second copy elsewhere in the package.

A missing sample is never dropped: every sum here propagates NaN,
and the coverage functions say how much of a plane was sampled.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np

from cfdbender.core.farfield.fields import FarfieldInputError, LatticeFields

__all__ = [
    "azimuthal_harmonics",
    "grid_coverage",
    "grid_integral",
    "grid_weights",
    "plane_integral",
    "ring_sample_weights",
    "sample_area_coverage",
    "sample_coverage",
    "strip_edges",
    "symmetry_floor",
]


# THE AZIMUTH HELPERS, shared with the balances (one home each). They
# carry public names so that a sibling module imports them without reaching into
# an underscore-private name; they are outside __all__ and not part of the API.


def delta_psi(psi: np.ndarray) -> float:
    """Return the uniform azimuth step, refusing a grid that is not uniform."""
    psi = np.asarray(psi)
    steps = np.diff(np.concatenate([psi, [psi[0] + 2.0 * np.pi]]))
    if not np.allclose(steps, steps[0], rtol=1e-10, atol=1e-12):
        raise FarfieldInputError(
            "the azimuth grid is not uniform; uniform periodic sampling is what "
            "makes the rectangle rule spectrally accurate and the FFT direct "
            ", so nonuniform azimuths are refused, never averaged"
        )
    return float(steps[0])


def half_spectrum(values: np.ndarray, half: int) -> np.ndarray:
    """Azimuthal orders ``0..half`` with no anti-aliasing cap.

    The Parseval sum of the transverse flux needs every order the grid
    carries, including Nyquist. :func:`azimuthal_harmonics` cannot supply that
    and should not: its cap exists so that STORED harmonics are unaliased,
    which is a different question from reconstructing a product the grid
    already determines.
    """
    data = np.asarray(values)
    n = data.shape[-1]
    spectrum = np.fft.fft(data, axis=-1) / n
    return np.take(spectrum, np.arange(half + 1), axis=-1)


def parseval_weights(n: int, half: int) -> np.ndarray:
    """Multiplicity of each order ``0..half`` in the real-field Parseval sum.

    Order 0 appears once. Orders with a distinct conjugate partner appear
    twice. For EVEN ``n`` the Nyquist order ``n/2`` is its own conjugate and
    appears once; giving it 2 double counts it, which is the error opposite
    to the omitted-Nyquist error and just as silent.
    """
    weights = np.full(half + 1, 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[half] = 1.0
    return weights


def ring_sample_weights(fields: LatticeFields) -> np.ndarray:
    """Per-sample annular quadrature weights of each ring.

    Returns
    -------
    numpy.ndarray
        ``0.5 (r_{j+1/2}^2 - r_{j-1/2}^2) dpsi`` per ring, shape ``(n_r,)``, in
        units of the tip radius squared; multiplying an integrand sample and
        summing over rings and azimuths is the whole plane quadrature.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples in the shaft-aligned axes of
        the survey lattice, with the units of
        :class:`~cfdbender.core.farfield.fields.LatticeFields`.
    """
    edges = np.asarray(fields.ring_edges)
    return cast(np.ndarray, 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2) * delta_psi(fields.psi))


def plane_integral(fields: LatticeFields, integrand: np.ndarray) -> np.ndarray:
    """Integrate a sampled field over each annular plane.

    Implemented once and reused by every balance: the
    ring-edge rule in radius times the rectangle rule in azimuth.

    Parameters
    ----------
    fields : LatticeFields
        Survey samples supplying the weights and the tip radius.
    integrand : numpy.ndarray
        Any array whose two trailing axes are ``(n_r, n_psi)``.

    Returns
    -------
    numpy.ndarray
        The integral over the two trailing axes, the leading axes preserved,
        in integrand units times square meters; NaN where any sample is not
        finite.

    Raises
    ------
    FarfieldInputError
        If the trailing axes are not ``(n_r, n_psi)`` of the lattice.
    """
    scale = float(fields.tip_radius) ** 2
    # A missing sample propagates as NaN and is never dropped: a sum
    # that skipped it while its ring weight stayed in the geometry returned an
    # integral smaller in exact proportion to the missing data, a number that is
    # not wrong-looking and is indistinguishable from a real physical reduction
    # of flux, force, torque or energy. Use sample_coverage() to ask HOW much is
    # missing before deciding whether a plane is usable.
    weights = ring_sample_weights(fields)[:, None]
    data = np.asarray(integrand)
    if data.ndim < 2 or data.shape[-2:] != (weights.shape[0], len(fields.psi)):
        raise FarfieldInputError(
            f"the integrand's trailing axes are {data.shape[-2:]} but the lattice has "
            f"{weights.shape[0]} rings and {len(fields.psi)} azimuths; a partial plane would "
            "broadcast into a plausible reduced integral"
        )
    total = (data * weights).sum(axis=(-2, -1)) * scale
    # Infinity is a missing sample like NaN: the integral is NaN, never infinite.
    return np.where(np.isfinite(data).all(axis=(-2, -1)), total, np.nan)


def sample_coverage(integrand: np.ndarray) -> np.ndarray:
    """Fraction of ``(r, psi)`` samples that are finite, per leading index.

    The diagnostic half of the NaN rule: :func:`plane_integral` returns NaN when
    any sample is missing, which says THAT a plane is unusable and not how far
    off it is; this says how much of the plane was actually sampled.

    THIS IS A COUNT OF SAMPLES, every sample weighing the same. On an annular
    lattice the samples stand for very different areas, so a missing sample at
    the rim costs the integral far more than one at the hub and the count
    reports them alike: use :func:`sample_area_coverage` for the fraction of
    the plane that was sampled.

    Parameters
    ----------
    integrand : numpy.ndarray
        Any array whose two trailing axes are ``(n_r, n_psi)``.

    Returns
    -------
    numpy.ndarray
        Coverage in ``[0, 1]``; 1.0 means every sample is finite.
    """
    array = np.asarray(integrand)
    total = array.shape[-2] * array.shape[-1]
    return cast(np.ndarray, np.isfinite(array).sum(axis=(-2, -1)) / total)


def sample_area_coverage(integrand: np.ndarray, fields: LatticeFields) -> np.ndarray:
    """Fraction of the plane's AREA that was sampled, per leading index.

    Each sample stands for the annular cell its quadrature weight
    (:func:`ring_sample_weights`) integrates, so a missing sample at the rim,
    which stands for far more of the disc than one at the hub, costs what the
    integral loses by it. A count (:func:`sample_coverage`) would report the same
    0.99 for a hole at the hub and for a hole at the rim whose cost in the
    integral is many times larger.

    Parameters
    ----------
    integrand : numpy.ndarray
        Any array whose two trailing axes are ``(n_r, n_psi)``.
    fields : LatticeFields
        The survey whose ring edges give each ring's represented area. Its ring
        count must be the integrand's ``n_r``.

    Returns
    -------
    numpy.ndarray
        Area coverage in ``[0, 1]``; 1.0 means every sample is finite.

    Raises
    ------
    FarfieldInputError
        If ``fields`` has another number of rings than the integrand.
    """
    array = np.asarray(integrand)
    weights = ring_sample_weights(fields)
    if len(weights) != array.shape[-2]:
        raise FarfieldInputError(
            f"the lattice has {len(weights)} rings and the integrand {array.shape[-2]}, so the "
            "area each sample stands for is not defined"
        )
    represented = (np.isfinite(array) * weights[:, None]).sum(axis=(-2, -1))
    # A plane with no hole covers all of its area exactly, whatever the rounding of the sum.
    complete = sample_coverage(array) == 1.0
    return np.where(complete, 1.0, represented / (float(weights.sum()) * array.shape[-1]))


def grid_weights(y: np.ndarray, *, z: np.ndarray) -> np.ndarray:
    """Trapezoid weights of a uniform rectangular grid, in m^2.

    Corners carry a quarter of a cell, edges a half, the inside a whole cell.

    Parameters
    ----------
    y, z : numpy.ndarray
        Uniform grid axes, m.

    Returns
    -------
    numpy.ndarray
        Weights shaped ``(len(y), len(z))``.
    """
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    wy = np.full(len(y), dy)
    wy[[0, -1]] = 0.5 * dy
    wz = np.full(len(z), dz)
    wz[[0, -1]] = 0.5 * dz
    return cast(np.ndarray, wy[:, None] * wz[None, :])


def grid_integral(y: np.ndarray, *, z: np.ndarray, integrand: np.ndarray) -> float:
    """Integrate a sampled field over a uniform rectangular grid by the trapezoid rule.

    A missing or non-finite sample makes the integral NaN.

    Parameters
    ----------
    y : np.ndarray
        Ordered spanwise grid coordinates in m.
    z : np.ndarray
        Ordered vertical grid coordinates in m.
    integrand : np.ndarray
        Sampled scalar integrand, with shape (len(y), len(z)) for a grid.

    Returns
    -------
    float
        Trapezoidal integral over the rectangular grid.

    Raises
    ------
    FarfieldInputError
        If the integrand is not shaped ``(len(y), len(z))``.
    """
    data = np.asarray(integrand)
    weights = grid_weights(y, z=z)
    if data.shape != weights.shape:
        raise FarfieldInputError(
            f"the integrand is shaped {data.shape} but the grid is {weights.shape} (y, z); "
            "a partial column would broadcast into a plausible full-plane integral"
        )
    if not np.all(np.isfinite(data)):
        return float("nan")
    return float((data * weights).sum())


def grid_coverage(
    integrand: np.ndarray, *, y: np.ndarray | None = None, z: np.ndarray | None = None
) -> float:
    """Fraction of the plane's area whose samples are finite.

    With the grid axes ``y`` and ``z`` each sample is weighted by the area the
    trapezoid rule of :func:`grid_integral` gives it (:func:`grid_weights`:
    a quarter cell at a corner, half at an edge, a whole cell inside), so the
    figure is the fraction of the INTEGRAL's area that was sampled. Without the
    axes every sample counts equally, which is the area fraction only when the
    samples stand for equal areas.

    Parameters
    ----------
    integrand : np.ndarray
        Sampled scalar integrand, with shape (len(y), len(z)) for a grid.
    y, z : np.ndarray, optional
        The grid axes, m; both or neither.

    Returns
    -------
    float
        Fraction of the grid's represented area carrying finite integrand values.

    Raises
    ------
    FarfieldInputError
        If only one axis is given.
    """
    array = np.asarray(integrand)
    if y is None and z is None:
        return float(np.isfinite(array).sum() / array.size)
    if y is None or z is None:
        raise FarfieldInputError("an area-weighted grid coverage needs both axes, y and z")
    weights = grid_weights(y, z=z)
    return float((np.isfinite(array) * weights).sum() / weights.sum())


def strip_edges(
    positions: np.ndarray, *, ends: Literal["station", "extended"] = "extended"
) -> np.ndarray:
    """Return the edges of the strips around the stations of a trace.

    Between two stations the edge is their midpoint. Beyond the end stations:

    - ``"extended"`` (the far field): half the adjacent spacing
      further out, so ``n`` uniform stations at the centers of ``n`` equal
      strips tile exactly the extent the solver divided;
    - ``"station"``: the end stations themselves, so the strips stop at the
      end stations and the span is shortened by half a spacing at each end;
      the integrated sectional loads of the sections table use the
      ``"extended"`` rule too, as the definitions page states.

    Parameters
    ----------
    positions : numpy.ndarray
        Station positions (one coordinate) in the order of the trace, shape
        ``(n,)``, ``n >= 2``; the edges of a trace in ``(y, z)`` are this rule
        applied to each coordinate. A coordinate may be constant or descending;
        no coordinate monotonicity is required. Nonfinite coordinates propagate
        through the midpoint and endpoint arithmetic without validation.
    ends : {"extended", "station"}
        The rule beyond the end stations.

    Returns
    -------
    numpy.ndarray
        The ``n + 1`` strip edges, in the order of ``positions``.

    Raises
    ------
    FarfieldInputError
        If positions is not one-dimensional with at least two stations, or
        ends is neither ``"extended"`` nor ``"station"``.
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise FarfieldInputError("a trace needs at least two stations to place strip edges")
    middle = 0.5 * (p[:-1] + p[1:])
    if ends == "extended":
        first = p[0] - 0.5 * (p[1] - p[0])
        last = p[-1] + 0.5 * (p[-1] - p[-2])
    elif ends == "station":
        first, last = p[0], p[-1]
    else:
        raise FarfieldInputError(f"unknown strip-edge rule {ends!r}; use 'extended' or 'station'")
    return np.concatenate([[first], middle, [last]])


def azimuthal_harmonics(values: np.ndarray, *, psi: np.ndarray, m_max: int = 6) -> np.ndarray:
    """Azimuthal Fourier coefficients of a sampled field, the ledger spine.

    Coefficients follow ``f(psi) = sum_m c_m exp(i m psi)`` with
    ``c_m = (1/N) sum_k f_k exp(-i m psi_k)``; for real fields the negative
    orders are the conjugates and are not stored.

    Parameters
    ----------
    values : numpy.ndarray
        Field whose last axis runs over the uniform lattice azimuths.
    psi : numpy.ndarray
        The azimuths of that axis, radians.
    m_max : int
        Highest stored order; at least 6 is required.

    Returns
    -------
    numpy.ndarray
        Complex coefficients, the last axis replaced by the orders
        ``0..m_max``. Order 0 carries forces, order 1 the in-plane moments,
        orders 2 and above the distortion a downstream surface sees.

    Raises
    ------
    FarfieldInputError
        If the azimuth grid or requested harmonic order is invalid.
    """
    delta_psi(psi)
    data = np.asarray(values)
    n = data.shape[-1]
    if data.ndim < 1 or len(np.asarray(psi)) != n:
        raise FarfieldInputError(
            f"the field has {n} azimuth samples but psi has {len(np.asarray(psi))}; "
            "the spectrum of mismatched grids is not defined"
        )
    if isinstance(m_max, bool) or not isinstance(m_max, numbers.Integral) or m_max < 0:
        raise FarfieldInputError(
            f"m_max={m_max!r} is not an azimuthal order; give a non-negative integer"
        )
    # Order m is unaliased while 2 m < n: an odd grid of n = 2 m + 1 azimuths
    # carries order m, and an even grid's Nyquist order n/2 is refused.
    if 2 * m_max >= n:
        raise FarfieldInputError(
            f"m_max={m_max} needs more than {2 * m_max} uniform azimuths to avoid "
            f"aliasing; this grid has {n}"
        )
    spectrum = np.fft.fft(data, axis=-1) / n
    kept = np.take(spectrum, np.arange(m_max + 1), axis=-1)
    # THE AZIMUTH ORIGIN. The FFT sums f_k exp(-i m 2 pi k / N), which is the
    # c_m of the docstring only when psi_0 = 0; a grid starting at psi_0 needs
    # the factor exp(-i m psi_0), or every order comes back rotated by m psi_0.
    psi_0 = float(np.asarray(psi)[0])
    return kept * np.exp(-1j * np.arange(m_max + 1) * psi_0)


def symmetry_floor(
    values: np.ndarray,
    *,
    psi: np.ndarray,
    m_max: int = 6,
    excluded_orders: Sequence[int] = (),
) -> float:
    """Largest azimuthal harmonic magnitude above order zero (gate G3).

    At zero incidence every retained ``m >= 1`` harmonic must vanish to the numerical
    floor; the recorded floor becomes the detectability threshold for the
    nonzero-incidence step.

    Returns
    -------
    float
        ``max_{m>=1} |c_m|`` over every other index, in field units; NaN where
        any sample is missing.

    Parameters
    ----------
    values : np.ndarray
        Sampled values with azimuth along the last dimension.
    psi : np.ndarray
        Uniform full-turn azimuth coordinates in radians.
    m_max : int
        Highest azimuthal harmonic order to retain, at least 1.
    excluded_orders : sequence of int
        Orders omitted from the floor, such as multiples of a rotor's blade count.
        If no order remains, the floor is NaN rather than a measured zero.

    Raises
    ------
    FarfieldInputError
        If the azimuth grid or the order is invalid, or ``m_max`` is below 1.
    """
    harmonics = azimuthal_harmonics(values, psi=psi, m_max=m_max)
    if harmonics.shape[-1] < 2:
        raise FarfieldInputError(
            f"m_max={m_max} keeps no harmonic above order zero, so there is no floor to "
            "measure; give m_max of at least 1"
        )
    kept = [m for m in range(1, m_max + 1) if m not in excluded_orders]
    return float(np.abs(harmonics[..., kept]).max()) if kept else float("nan")
