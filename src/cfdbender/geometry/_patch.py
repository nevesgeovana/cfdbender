"""The B-spline patch value shared by every stage of the parametric geometry.

Pipeline role: pure numerical data type of the geometry package. A patch is a
tensor-product B-spline surface with a family label. It carries only numbers,
so the generation, check and mesh stages exchange it without importing each
other and without a geometry kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from cfdbender._errors import InputValueError

__all__ = [
    "Patch",
    "evaluate_patch",
    "interpolate_patch",
    "interpolate_polynomial_strip",
    "patch_layout",
]


@dataclass(frozen=True, eq=False)
class Patch:
    """A tensor-product B-spline surface with clamped knot vectors.

    Attributes
    ----------
    family : str
        Role of the patch in its component, for example ``"upper_fore"``,
        ``"tip_cap"`` or ``"root_fillet"``.
    controls : numpy.ndarray
        Control points, shape ``(nu, nv, 3)``.
    knots_u, knots_v : tuple of float
        Distinct knots of each direction.
    multiplicities_u, multiplicities_v : tuple of int
        Multiplicity of each distinct knot.
    degree_u, degree_v : int
        Polynomial degree of each direction.
    bay : tuple of float or None
        Normalized span interval ``(eta_min, eta_max)`` a skin patch covers.
    role : str
        What the patch is for in its component (``"band"``, ``"cap"``, ``"base"``, ...);
        empty when no stage has said. A mesher or an assembly reads it instead of list order.
    edge_ids : tuple of str
        Names of the four boundary edges ``(u0, u1, v0, v1)``; empty when no stage has named them.
        Two patches that share an edge carry the same name for it.
    """

    family: str
    controls: np.ndarray
    knots_u: tuple[float, ...]
    knots_v: tuple[float, ...]
    multiplicities_u: tuple[int, ...]
    multiplicities_v: tuple[int, ...]
    degree_u: int
    degree_v: int
    bay: tuple[float, float] | None = None
    role: str = ""
    edge_ids: tuple[str, ...] = ()

    def evaluate(self, count: int = 41) -> np.ndarray:
        """Sample the patch on a ``count`` by ``count`` lattice, shape ``(count, count, 3)``."""
        return evaluate_patch(self, count)

    def to_dict(self) -> dict:
        """Return the plain-data form: lists and numbers only."""
        return {
            "family": self.family,
            "bay": None if self.bay is None else list(self.bay),
            "controls": np.asarray(self.controls).tolist(),
            "knots_u": list(self.knots_u),
            "knots_v": list(self.knots_v),
            "multiplicities_u": list(self.multiplicities_u),
            "multiplicities_v": list(self.multiplicities_v),
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "role": self.role,
            "edge_ids": list(self.edge_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Patch:
        """Rebuild a patch from the form :meth:`to_dict` returns."""
        bay = data.get("bay")
        return cls(
            family=str(data.get("family", "")),
            controls=np.asarray(data["controls"], float),
            knots_u=tuple(float(k) for k in data["knots_u"]),
            knots_v=tuple(float(k) for k in data["knots_v"]),
            multiplicities_u=tuple(int(m) for m in data["multiplicities_u"]),
            multiplicities_v=tuple(int(m) for m in data["multiplicities_v"]),
            degree_u=int(data["degree_u"]),
            degree_v=int(data["degree_v"]),
            bay=None if bay is None else (float(bay[0]), float(bay[1])),
            role=str(data.get("role", "")),
            edge_ids=tuple(str(e) for e in data.get("edge_ids", ())),
        )


def _checked_parameters(values, count, name):
    """Return ``values`` as an array of ``count`` increasing numbers from 0 to 1, or refuse."""
    values = np.asarray(values, float)
    if (
        values.shape != (count,)
        or values[0] != 0
        or values[-1] != 1
        or np.any(np.diff(values) <= 0)
    ):
        raise InputValueError(f"{name}: one increasing value per sample, from 0 to 1")
    return values


def interpolate_patch(
    points,
    family: str = "",
    bay: tuple[float, float] | None = None,
    parameters_v=None,
    parameters_u=None,
) -> Patch:
    """Interpolate a grid of samples ``[u, v, xyz]`` with a bicubic B-spline.

    The samples are uniform in each parameter by default. ``parameters_u`` and ``parameters_v``,
    when given, are the parameter values of the samples, increasing from 0 to 1, so a patch can be
    uniform in a coordinate other than the sample index, with the samples clustered where the
    surface needs them.
    """
    grid = np.asarray(points, float)
    if grid.ndim != 3 or grid.shape[2] != 3 or min(grid.shape[:2]) < 4:
        raise InputValueError("points: at least four by four XYZ samples required")
    u, v = np.linspace(0, 1, grid.shape[0]), np.linspace(0, 1, grid.shape[1])
    if parameters_u is not None:
        u = _checked_parameters(parameters_u, grid.shape[0], "parameters_u")
    if parameters_v is not None:
        v = _checked_parameters(parameters_v, grid.shape[1], "parameters_v")
    first = make_interp_spline(u, grid, axis=0)
    second = make_interp_spline(v, first.c, axis=1)
    knots_u, mult_u = np.unique(first.t, return_counts=True)
    knots_v, mult_v = np.unique(second.t, return_counts=True)
    # make_interp_spline stores its interpolation axis first in c.
    controls = second.c.transpose(1, 0, 2)
    return Patch(
        family=family,
        controls=controls,
        knots_u=tuple(float(k) for k in knots_u),
        knots_v=tuple(float(k) for k in knots_v),
        multiplicities_u=tuple(int(m) for m in mult_u),
        multiplicities_v=tuple(int(m) for m in mult_v),
        degree_u=3,
        degree_v=3,
        bay=bay,
    )


def patch_layout(section_stations, breaks=(), junctions=()) -> list[dict]:
    """List the six skin patches of every spanwise bay, as ``{"bay", "family"}`` entries."""
    stations = sorted({0.0, 1.0, *section_stations, *breaks, *junctions})
    return [
        {"bay": [a, b], "family": side + "_" + region}
        for a, b in zip(stations, stations[1:], strict=False)
        for side, region in [
            ("upper", "fore"),
            ("upper", "aft"),
            ("lower", "fore"),
            ("lower", "aft"),
            ("trailing", "upper"),
            ("trailing", "lower"),
        ]
    ]


def interpolate_polynomial_strip(points, family: str = "", bay=None) -> Patch:
    """Interpolate with a cubic longitudinal spline and an exact transverse Bernstein polynomial."""
    grid = np.asarray(points, float)
    u, v = np.linspace(0, 1, grid.shape[0]), np.linspace(0, 1, grid.shape[1])
    degree = len(v) - 1
    first = make_interp_spline(u, grid, axis=0)
    basis = np.stack(
        [comb(degree, i) * v**i * (1 - v) ** (degree - i) for i in range(degree + 1)], axis=-1
    )
    controls = np.linalg.solve(basis, first.c.transpose(1, 0, 2).reshape(len(v), -1))
    controls = controls.reshape(len(v), len(first.c), 3).transpose(1, 0, 2)
    knots, mult = np.unique(first.t, return_counts=True)
    return Patch(
        family=family,
        controls=controls,
        knots_u=tuple(float(k) for k in knots),
        knots_v=(0.0, 1.0),
        multiplicities_u=tuple(int(m) for m in mult),
        multiplicities_v=(degree + 1, degree + 1),
        degree_u=3,
        degree_v=degree,
        bay=bay,
    )


def evaluate_patch(patch: Patch, count: int = 41) -> np.ndarray:
    """Sample a patch on a square parameter lattice.

    Parameters
    ----------
    patch : Patch
        B-spline surface to sample, in its control points' coordinate frame and units.
    count : int, optional
        Number of samples along each parameter axis, including both ends; default 41.

    Returns
    -------
    numpy.ndarray
        Coordinates with shape ``(count, count, 3)``, in the patch's frame and units.
    """
    controls = np.asarray(patch.controls)
    tu = np.repeat(patch.knots_u, patch.multiplicities_u)
    tv = np.repeat(patch.knots_v, patch.multiplicities_v)
    axis = np.linspace(0, 1, count)
    first = np.moveaxis(BSpline(tu, controls, patch.degree_u)(axis), 0, 1)
    return np.moveaxis(BSpline(tv, first, patch.degree_v, axis=0)(axis), 0, 1)
