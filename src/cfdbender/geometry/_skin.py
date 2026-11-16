"""One skinned surface, split into patches by knot insertion.

Pipeline role: pure numerical stage of the geometry package (S6 and S7). A body band is one
bicubic surface through a lattice of samples that runs through every station; the patches the
mesher and the assembly work on are that surface cut at the stations, so the cuts add no seam of
their own: the pieces are the same surface, and the tangent plane is continuous across each cut
because the whole is.
"""

from __future__ import annotations

import numpy as np

from cfdbender._errors import InputValueError

from ._bspline import clamped_knots, curve_basis, fit_curve, split_patch
from ._patch import Patch, interpolate_patch

#: Samples along each edge of a cap quadrilateral and of a band bay (equal, so a shared rim is one
#: curve), tried in turn: the tangent plane at a pole converges with the count, and a slender end
#: needs more. A body takes the first count whose pole seams turn by at most POLE_TARGET_DEG.
SAMPLE_LADDER = (21, 31, 41, 61, 81)
#: Largest turn across a pole seam, in degrees, that ends the ladder; under the 1 degree of S9.
POLE_TARGET_DEG = 0.8

__all__ = [
    "POLE_TARGET_DEG",
    "SAMPLE_LADDER",
    "band_axial",
    "bay_cuts",
    "cap_edge_ids",
    "clustered",
    "fit_patch",
    "pole_parameter_grids",
    "skin",
    "skin_band",
    "split_along",
]


def skin(points, family: str = "", parameters_v=None, parameters_u=None) -> Patch:
    """Return the one bicubic surface through a lattice of samples ``[u, v, xyz]``.

    ``parameters_u`` and ``parameters_v`` make a direction uniform in a coordinate of the samples
    rather than in their index,
    as :func:`~cfdbender.geometry._patch.interpolate_patch` reads it.
    """
    return interpolate_patch(
        points, family=family, parameters_v=parameters_v, parameters_u=parameters_u
    )


def split_along(patch: Patch, direction: str, fractions) -> list[Patch]:
    """Cut a patch across one parametric direction at ``fractions`` of the whole, increasing.

    Parameters
    ----------
    patch : Patch
        The unsplit surface.
    direction : {"u", "v"}
        The direction to cut across.
    fractions : sequence of float
        Parameters strictly inside (0, 1), strictly increasing.

    Returns
    -------
    list of Patch
        ``len(fractions) + 1`` pieces, each with its own parameter in [0, 1], that evaluate to
        the original surface.

    Raises
    ------
    InputValueError
        When the fractions are not strictly increasing inside (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> from cfdbender.geometry._patch import interpolate_patch
    >>> grid = np.random.default_rng(3).normal(size=(9, 5, 3))
    >>> len(split_along(interpolate_patch(grid), "u", [0.25, 0.5]))
    3
    """
    cuts = [float(f) for f in fractions]
    if any(not 0.0 < f < 1.0 for f in cuts) or any(
        b <= a for a, b in zip(cuts, cuts[1:], strict=False)
    ):
        raise InputValueError("fractions: expected values strictly increasing inside (0, 1)")
    pieces, rest, origin = [], patch, 0.0
    for fraction in cuts:
        left, rest = split_patch(rest, direction, (fraction - origin) / (1.0 - origin))
        pieces.append(left)
        origin = fraction
    return [*pieces, rest]


def bay_cuts(boundaries) -> np.ndarray:
    """Return the parameters, from 0 to 1, of the interior boundaries of a run of bays.

    ``boundaries`` are the coordinates of every bay boundary, ends included, increasing.
    """
    values = np.asarray(boundaries, float)
    return (values[1:-1] - values[0]) / (values[-1] - values[0])


def clustered(samples: int):
    """Return ``samples`` parameters in [0, 1], symmetric and denser at both ends.

    The law is symmetric, so an edge shared by two patches is sampled alike whichever way each
    walks it. The surface is interpolated at these parameters and the parameter stays linear in the
    coordinate, so the node spacing a mesher takes by parameter is unchanged.
    """
    return (1 - np.cos(np.pi * np.linspace(0, 1, samples))) / 2


def band_axial(transitions, samples: int) -> np.ndarray:
    """Return the axial coordinates of a band lattice: ``samples`` per bay, ends shared."""
    bays = zip(transitions, transitions[1:], strict=False)
    parts = [np.linspace(x0, x1, samples)[:-1] for x0, x1 in bays]
    return np.concatenate([*parts, [transitions[-1]]])


def skin_band(points, s, axial, transitions) -> list[Patch]:
    """Skin a band lattice ``[azimuth, axial, xyz]`` once and cut it at the bay boundaries.

    The azimuth parameter follows ``s``, and the axial one follows the axial coordinate, so every
    bay stays uniform in x after the cut. A single bay is returned as the skin itself.
    """
    whole = skin(points, parameters_u=s, parameters_v=(axial - axial[0]) / (axial[-1] - axial[0]))
    if len(transitions) < 3:
        return [whole]
    return split_along(whole, "v", bay_cuts(transitions))


def pole_parameter_grids(samples: int = SAMPLE_LADDER[0]):
    """Split a triangular network cell into three true quadrilaterals.

    Coordinates are barycentric weights of the two station-arc vertices;
    the remaining weight belongs to the pole. Outer arc halves are inherited
    by the adjoining axial bays, so no hanging node is introduced.
    """
    a, b, c = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ab, bc, ca, centre = (a + b) / 2, (b + c) / 2, (c + a) / 2, (a + b + c) / 3
    s = clustered(samples)
    u, v = s[:, None, None], s[None, :, None]
    return [
        (1 - u) * (1 - v) * p0 + u * (1 - v) * p1 + u * v * p2 + (1 - u) * v * p3
        for p0, p1, p2, p3 in [(a, ab, centre, ca), (ab, b, bc, centre), (centre, bc, c, ca)]
    ]


def cap_edge_ids(end, sector, sectors, ring, full):
    """Return the edge names ``(u0, u1, v0, v1)`` of the three quadrilaterals of one cap sector."""
    here, there = (k % sectors if full else k for k in (sector, sector + 1))
    inner = [f"{end}.s{sector}.{name}" for name in ("q01", "q02", "q12")]
    return [
        (f"{end}.seam{there}.in", inner[0], f"{end}.seam{here}.in", inner[1]),
        (inner[0], f"{ring}.h{2 * sector}", f"{end}.seam{here}.out", inner[2]),
        (inner[1], f"{ring}.h{2 * sector + 1}", inner[2], f"{end}.seam{there}.out"),
    ]


#: Step, in the patch parameter, of the one-sided difference that measures the cross-edge slope.
_STEP = 1e-4
_SIDES = ("u0", "u1", "v0", "v1")


def _edge_points(point, side, s):
    """Return the points of one boundary line of a parametric surface, ``s`` along it."""
    fixed = np.full_like(s, 0.0 if side[1] == "0" else 1.0)
    return point(fixed, s) if side[0] == "u" else point(s, fixed)


def _inward_slope(point, side, s):
    """Return the derivative across one boundary line, per unit parameter, pointing inward.

    A second-order one-sided difference of the surface function, so no sample outside the domain
    is read.
    """
    samples = []
    for k in (0.0, 1.0, 2.0):
        a = np.full_like(s, k * _STEP if side[1] == "0" else 1.0 - k * _STEP)
        samples.append(point(a, s) if side[0] == "u" else point(s, a))
    return (-3.0 * samples[0] + 4.0 * samples[1] - samples[2]) / (2.0 * _STEP)


def _end_slopes(point, side, s):
    """Return the derivative along one boundary line, per unit parameter, at its two ends."""
    marks = np.array([0.0, 1.0, 2.0]) * _STEP
    start = _edge_points(point, side, marks)
    stop = _edge_points(point, side, 1.0 - marks)
    return (
        (-3.0 * start[0] + 4.0 * start[1] - start[2]) / (2.0 * _STEP),
        -(-3.0 * stop[0] + 4.0 * stop[1] - stop[2]) / (2.0 * _STEP),
    )


def _boundary(point, knots, degrees, s):
    """Return the fitted controls and residual of each of the four boundary curves.

    Each leaves its ends with the slope of the surface function, so the controls next to a corner
    lie in the tangent plane there whichever edge they are read from.
    """
    axis = {"u0": 1, "u1": 1, "v0": 0, "v1": 0}
    return {
        side: fit_curve(
            _edge_points(point, side, s),
            s,
            knots[axis[side]],
            degrees[axis[side]],
            _end_slopes(point, side, s),
        )
        for side in _SIDES
    }


def _pin_row(controls, fixed, side, rate, step):
    """Set the row (or column) next to a boundary from the cross-edge slope along it."""
    first = side[1] == "0"
    axis = 0 if side[0] == "u" else 1
    edge = 0 if first else controls.shape[axis] - 1
    inner = edge + (1 if first else -1)
    for j in range(1, controls.shape[1 - axis] - 1):
        here, there = ((inner, j), (edge, j)) if axis == 0 else ((j, inner), (j, edge))
        if not fixed[here]:
            controls[here] = controls[there] + step * rate[j]
            fixed[here] = True


def _pin_frame(controls, fixed, edges, *, point, knots, degrees, s, smooth):
    """Fix the boundary controls, and the next row where the edge joins its neighbour smoothly."""
    controls[0, :], controls[-1, :] = edges["u0"][0], edges["u1"][0]
    controls[:, 0], controls[:, -1] = edges["v0"][0], edges["v1"][0]
    fixed[[0, -1], :] = True
    fixed[:, [0, -1]] = True
    for side in _SIDES:
        axis = 0 if side[0] == "u" else 1
        p, k = degrees[axis], knots[axis]
        if not smooth[side] or controls.shape[axis] < 4 or p < 2:
            continue
        along = 1 - axis
        rate = fit_curve(_inward_slope(point, side, s), s, knots[along], degrees[along])[0]
        step = (k[p + 1] if side[1] == "0" else 1.0 - k[-p - 2]) / p
        _pin_row(controls, fixed, side, rate, step)


def fit_patch(point, spans, degrees=(3, 3), smooth=(True, True, True, True), family=""):
    """Fit one tensor-product patch to a parametric surface function, on uniform knots.

    The four boundary curves are fitted first, each from the samples of its own line, so two
    patches that share a line and its span count get the same curve; where an edge is to join its
    neighbour smoothly the next row of controls is set from the cross-edge slope of ``point``, so
    both sides lie in the one tangent plane of the surface they sample. The interior is the least
    squares fit to the samples.

    Parameters
    ----------
    point : callable
        ``point(u, v)`` with ``u`` and ``v`` arrays in [0, 1] that broadcast; returns ``(..., 3)``.
    spans : tuple of int
        Spans in ``u`` and in ``v``, each at least 1.
    degrees : tuple of int, optional
        Degree in ``u`` and in ``v``; default cubic. Degree 1 is exact for a ruled direction.
    smooth : tuple of bool, optional
        For the edges ``u0, u1, v0, v1``, whether the cross-edge slope is imposed.
    family : str, optional
        Family of the patch.

    Returns
    -------
    patch : Patch
    residual : float
        Largest distance from a sample to the patch.
    edge_residual : tuple of float
        Largest boundary-curve residual along ``u`` and along ``v``.
    """
    knots = [
        clamped_knots([i / n for i in range(1, n)], p) for n, p in zip(spans, degrees, strict=True)
    ]
    sizes = [len(k) - p - 1 for k, p in zip(knots, degrees, strict=True)]
    s = np.linspace(0.0, 1.0, max(41, 4 * max(sizes)))
    edges = _boundary(point, knots, degrees, s)
    controls, fixed = np.zeros((*sizes, 3)), np.zeros(sizes, bool)
    _pin_frame(
        controls,
        fixed,
        edges,
        point=point,
        knots=knots,
        degrees=degrees,
        s=s,
        smooth=dict(zip(_SIDES, smooth, strict=True)),
    )
    target = point(s[:, None], s[None, :])
    bu, bv = (curve_basis(k, p, s) for k, p in zip(knots, degrees, strict=True))
    free = ~fixed.ravel()
    if free.any():
        flat = controls.reshape(-1, 3)
        left = target - np.einsum("ia,jb,abx->ijx", bu, bv, controls)
        rhs = np.einsum("ia,ijx,jb->abx", bu, left, bv).reshape(-1, 3)[free]
        gram = np.kron(bu.T @ bu, bv.T @ bv)[np.ix_(free, free)]
        flat[free] = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    error = np.einsum("ia,jb,abx->ijx", bu, bv, controls) - target
    along = (max(edges["v0"][1], edges["v1"][1]), max(edges["u0"][1], edges["u1"][1]))
    patch = _as_patch(controls, knots, degrees, family)
    return patch, float(np.max(np.linalg.norm(error, axis=-1))), along


def _as_patch(controls, knots, degrees, family) -> Patch:
    """Return a :class:`Patch` from controls and full knot vectors."""
    (ku, mu), (kv, mv) = (np.unique(k, return_counts=True) for k in knots)
    return Patch(
        family=family,
        controls=controls,
        knots_u=tuple(float(x) for x in ku),
        knots_v=tuple(float(x) for x in kv),
        multiplicities_u=tuple(int(m) for m in mu),
        multiplicities_v=tuple(int(m) for m in mv),
        degree_u=degrees[0],
        degree_v=degrees[1],
    )
