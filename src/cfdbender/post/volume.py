"""Locate points in an unstructured volume grid and interpolate its point data at them.

The grid comes from :func:`cfdbender.post.vtu.read_vtu`. The cells that are located are the
tetrahedron (VTK 10), the hexahedron (12), the wedge (13) and the pyramid (14); any other cell
is not searched and is counted in :attr:`VolumeSampler.skipped_cells`.

Interpolation is linear inside the containing cell and uses the point data only. A cell that is
not a tetrahedron is cut into tetrahedra around the mean of its nodes: every triangular face is
the base of one tetrahedron, every quadrilateral face is cut into four triangles around the mean
of its four nodes, and each triangle is the base of one tetrahedron. The tetrahedra of a face
depend on the nodes of that face alone, so two cells that share a face interpolate the same values
on it and the field is continuous across the face, whatever the node numbering of each cell. A
field that is linear in the coordinates is reproduced to round-off in every cell type.

A point that lies in no cell (outside the domain, or inside a body the grid leaves out) gets no
value: its entries are ``nan`` and it is counted as missing. Nothing is extrapolated and no value
is taken from the nearest node.

The search takes the mean of the nodes of each cell as the cell's center and the farthest node
from it as the cell's radius. A point is tried against the nearest centers first; a point that
is not found there is tried, exactly, against every cell whose center is within its radius,
grouping the cells by radius so a grid of widely different cell sizes costs no more per point.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from cfdbender.post.vtu import UnstructuredGrid

__all__ = [
    "SUPPORTED_CELL_TYPES",
    "Location",
    "SampleResult",
    "VolumeError",
    "VolumeSampler",
    "ZonedSampler",
]

TETRA, HEXAHEDRON, WEDGE, PYRAMID = 10, 12, 13, 14

#: The VTK cell types that are searched, with their node counts.
SUPPORTED_CELL_TYPES: dict[int, int] = {TETRA: 4, HEXAHEDRON: 8, WEDGE: 6, PYRAMID: 5}

#: The barycentric tolerance of the containment test. A point within this of a face counts as on
#: it, so a point on a face, an edge or a node is found.
_TOLERANCE = 1e-9
_NEAR_ROUNDS = ((0, 4), (4, 12))
_CHUNK = 10_000

_FACES: dict[int, tuple[tuple[int, ...], ...]] = {
    HEXAHEDRON: (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
    WEDGE: ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    PYRAMID: ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
}


class VolumeError(ValueError):
    """A volume grid or a sampling request that cannot be served."""


@dataclass(frozen=True)
class _Decomposition:
    """The tetrahedra of one cell type, over the cell's nodes and its derived vertices."""

    weights: NDArray[np.float64]
    tets: NDArray[np.int64]


def _decomposition(vtk_type: int) -> _Decomposition:
    n_nodes = SUPPORTED_CELL_TYPES[vtk_type]
    if vtk_type == TETRA:
        return _Decomposition(np.eye(4), np.array([[0, 1, 2, 3]], dtype=np.int64))
    rows: list[NDArray[np.float64]] = [np.eye(n_nodes)[k] for k in range(n_nodes)]
    centre = np.full(n_nodes, 1.0 / n_nodes)
    rows.append(centre)
    centre_index = n_nodes
    tets: list[tuple[int, ...]] = []
    for face in _FACES[vtk_type]:
        if len(face) == 3:
            tets.append((centre_index, *face))
            continue
        middle = np.zeros(n_nodes)
        middle[list(face)] = 0.25
        rows.append(middle)
        middle_index = len(rows) - 1
        for k in range(4):
            tets.append((centre_index, middle_index, face[k], face[(k + 1) % 4]))
    return _Decomposition(np.array(rows), np.array(tets, dtype=np.int64))


_DECOMPOSITIONS = {kind: _decomposition(kind) for kind in SUPPORTED_CELL_TYPES}


@dataclass(frozen=True)
class Location:
    """Where each point of a request lies: its cell and the weights of that cell's nodes.

    Attributes
    ----------
    cell : numpy.ndarray
        The cell number of each point, ``-1`` for a point in no cell.
    nodes : numpy.ndarray
        Shape ``(n, 8)``: the point indices of the containing cell, padded with 0.
    weights : numpy.ndarray
        Shape ``(n, 8)``: the interpolation weight of each node, 0 in the padding and for a
        point in no cell. The weights of a found point sum to 1.
    """

    cell: NDArray[np.int64]
    nodes: NDArray[np.int64]
    weights: NDArray[np.float64]

    @property
    def found(self) -> NDArray[np.bool_]:
        """Whether each point lies in a cell."""
        return self.cell >= 0


@dataclass(frozen=True)
class SampleResult:
    """The interpolated fields at a set of points, and what was found.

    Attributes
    ----------
    values : dict of str to numpy.ndarray
        Each requested point-data array at the points, ``nan`` where no cell holds the point.
    location : Location
        The cells and weights, reusable for another field of the same grid.
    """

    values: dict[str, NDArray[np.float64]]
    location: Location
    n_points: int = field(init=False)
    n_found: int = field(init=False)

    def __post_init__(self) -> None:
        """Count the points and those found."""
        object.__setattr__(self, "n_points", int(self.location.cell.size))
        object.__setattr__(self, "n_found", int(np.count_nonzero(self.location.found)))

    @property
    def n_missing(self) -> int:
        """The number of points that lie in no cell."""
        return self.n_points - self.n_found

    @property
    def found(self) -> NDArray[np.bool_]:
        """Whether each point lies in a cell."""
        return self.location.found


def _cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.cross(a, b)


class VolumeSampler:
    """A spatial index over the searched cells of a volume grid.

    Parameters
    ----------
    grid : UnstructuredGrid
        The grid. Its point data is what :meth:`sample` interpolates.
    neighbors : int, default 12
        How many nearest cell centers are tried before the exact search.

    Attributes
    ----------
    cell_counts : dict of int to int
        The number of cells of each VTK type in the grid, whether searched or not.
    skipped_cells : int
        The cells that are not searched: another type, or a size that is not the type's.

    Examples
    --------
    >>> import numpy as np
    >>> from cfdbender.post.vtu import UnstructuredGrid
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    >>> grid = UnstructuredGrid(
    ...     points,
    ...     np.arange(4),
    ...     np.array([4]),
    ...     np.array([10], dtype=np.uint8),
    ...     {"f": points[:, 0] + 2 * points[:, 1]},
    ... )
    >>> VolumeSampler(grid).sample(np.array([[0.25, 0.25, 0.25]])).values["f"]
    array([0.75])
    """

    def __init__(self, grid: UnstructuredGrid, *, neighbours: int = 12) -> None:
        if neighbours < 1:
            raise VolumeError(f"neighbors is {neighbours}; it is at least 1")
        self._grid = grid
        self._neighbours = neighbours
        starts = np.concatenate(([0], grid.offsets[:-1])).astype(np.int64)
        sizes = grid.offsets - starts
        types = grid.types.astype(np.int64)
        self.cell_counts: dict[int, int] = {
            int(kind): int(count)
            for kind, count in zip(*np.unique(types, return_counts=True), strict=True)
        }
        searched = np.zeros(types.size, dtype=bool)
        for kind, n_nodes in SUPPORTED_CELL_TYPES.items():
            searched |= (types == kind) & (sizes == n_nodes)
        self._cells = np.flatnonzero(searched).astype(np.int64)
        self.skipped_cells = int(types.size - self._cells.size)
        self._starts = starts
        self._types = types
        self._centres = np.zeros((self._cells.size, 3))
        self._radius = np.zeros(self._cells.size)
        self._tree: cKDTree | None = None
        self._classes: dict[int, cKDTree] = {}
        self._class_members: dict[int, NDArray[np.int64]] = {}
        self._build()

    def _cell_nodes(self, cells: NDArray[np.int64], kind: int) -> NDArray[np.int64]:
        n_nodes = SUPPORTED_CELL_TYPES[kind]
        index = self._starts[cells][:, None] + np.arange(n_nodes)[None, :]
        return np.asarray(self._grid.connectivity[index], dtype=np.int64)

    def _build(self) -> None:
        points = self._grid.points
        for kind in SUPPORTED_CELL_TYPES:
            local = np.flatnonzero(self._types[self._cells] == kind)
            for first in range(0, local.size, _CHUNK):
                part = local[first : first + _CHUNK]
                nodes = self._cell_nodes(self._cells[part], kind)
                corners = points[nodes]
                centre = corners.mean(axis=1)
                self._centres[part] = centre
                self._radius[part] = np.linalg.norm(corners - centre[:, None, :], axis=2).max(
                    axis=1
                )
        if self._cells.size:
            self._tree = cKDTree(self._centres, balanced_tree=False, compact_nodes=False)
            self._box_low = (self._centres - self._radius[:, None]).min(axis=0)
            self._box_high = (self._centres + self._radius[:, None]).max(axis=0)

    # ------------------------------------------------------------------ location

    def locate(self, points: Sequence[Sequence[float]] | NDArray[np.float64]) -> Location:
        """Find the cell of each point and the interpolation weights inside it.

        A point on a face, an edge or a node is found once and gets one cell, and the value it
        gets is the same from every cell that holds it. A point in no cell has cell ``-1``.

        Parameters
        ----------
        points : array_like
            Shape ``(n, 3)``.

        Returns
        -------
        Location
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise VolumeError(f"points must have shape (n, 3), got {pts.shape}")
        n = pts.shape[0]
        cell = np.full(n, -1, dtype=np.int64)
        nodes = np.zeros((n, 8), dtype=np.int64)
        weights = np.zeros((n, 8))
        if n == 0 or self._tree is None:
            return Location(cell, nodes, weights)
        inside_box = np.all(np.isfinite(pts), axis=1)
        inside_box &= np.all((pts >= self._box_low - 1e-9) & (pts <= self._box_high + 1e-9), axis=1)
        pending = np.flatnonzero(inside_box)
        if pending.size:
            k = min(self._neighbours, self._cells.size)
            _, near = self._tree.query(pts[pending], k=k, workers=-1)
            near = np.asarray(near).reshape(pending.size, k)
            for first, last in _NEAR_ROUNDS:
                if last > k:
                    last = k
                if first >= last or pending.size == 0:
                    break
                column = near[:, first:last]
                mine = np.repeat(np.arange(pending.size), column.shape[1])
                self._try(pts, pending[mine], column.ravel(), cell, nodes, weights)
                keep = cell[pending] < 0
                pending, near = pending[keep], near[keep]
        if pending.size:
            self._exact(pts, pending, cell, nodes, weights)
        return Location(cell, nodes, weights)

    def _try(
        self,
        pts: NDArray[np.float64],
        point_index: NDArray[np.int64],
        local_cell: NDArray[np.int64],
        cell: NDArray[np.int64],
        nodes: NDArray[np.int64],
        weights: NDArray[np.float64],
    ) -> None:
        """Test candidate pairs (point, searched cell) and record the lowest cell per point."""
        near = np.linalg.norm(pts[point_index] - self._centres[local_cell], axis=1) <= (
            self._radius[local_cell] * (1.0 + 1e-9) + 1e-300
        )
        point_index, local_cell = point_index[near], local_cell[near]
        order = np.argsort(local_cell, kind="stable")
        point_index, local_cell = point_index[order], local_cell[order]
        for kind in SUPPORTED_CELL_TYPES:
            pick = np.flatnonzero(self._types[self._cells[local_cell]] == kind)
            for first in range(0, pick.size, _CHUNK):
                part = pick[first : first + _CHUNK]
                self._test(kind, pts, point_index[part], local_cell[part], cell, nodes, weights)

    def _test(
        self,
        kind: int,
        pts: NDArray[np.float64],
        point_index: NDArray[np.int64],
        local_cell: NDArray[np.int64],
        cell: NDArray[np.int64],
        nodes: NDArray[np.int64],
        weights: NDArray[np.float64],
    ) -> None:
        deco = _DECOMPOSITIONS[kind]
        n_nodes = SUPPORTED_CELL_TYPES[kind]
        global_cell = self._cells[local_cell]
        node_ids = self._cell_nodes(global_cell, kind)
        corners = self._grid.points[node_ids]
        extended = np.einsum("en,mnc->mec", deco.weights, corners)
        tets = extended[:, deco.tets, :]
        origin = tets[:, :, 0, :]
        e0, e1, e2 = (tets[:, :, i, :] - origin for i in (1, 2, 3))
        delta = pts[point_index][:, None, :] - origin
        c12 = _cross(e1, e2)
        det = np.einsum("mtc,mtc->mt", e0, c12)
        scale = np.maximum(
            np.linalg.norm(e0, axis=2) * np.linalg.norm(e1, axis=2) * np.linalg.norm(e2, axis=2),
            1e-300,
        )
        good = np.abs(det) > 1e-13 * scale
        safe = np.where(good, det, 1.0)
        b0 = np.einsum("mtc,mtc->mt", delta, c12) / safe
        b1 = np.einsum("mtc,mtc->mt", e0, _cross(delta, e2)) / safe
        b2 = np.einsum("mtc,mtc->mt", e0, _cross(e1, delta)) / safe
        lam = np.stack([1.0 - b0 - b1 - b2, b0, b1, b2], axis=2)
        worst = np.where(good, lam.min(axis=2), -np.inf)
        best = worst.argmax(axis=1)
        hit = worst[np.arange(worst.shape[0]), best] >= -_TOLERANCE
        if not hit.any():
            return
        rows = np.flatnonzero(hit)
        lam_best = lam[rows, best[rows]]
        ext_weights = deco.weights[deco.tets[best[rows]]]
        node_weights = np.einsum("mk,mkn->mn", lam_best, ext_weights)
        target = point_index[rows]
        candidate = global_cell[rows]
        # one cell per point: the lowest number among the cells that hold it
        taken = cell[target]
        better = (taken < 0) | (candidate < taken)
        order = np.lexsort((candidate, target))
        target, candidate, node_weights, rows = (
            target[order],
            candidate[order],
            node_weights[order],
            rows[order],
        )
        first_of = np.concatenate(([True], target[1:] != target[:-1]))
        keep = first_of & better[order]
        t, c = target[keep], candidate[keep]
        cell[t] = c
        nodes[t, :n_nodes] = node_ids[rows[keep]]
        nodes[t, n_nodes:] = 0
        weights[t, :n_nodes] = node_weights[keep]
        weights[t, n_nodes:] = 0.0

    def _exact(
        self,
        pts: NDArray[np.float64],
        pending: NDArray[np.int64],
        cell: NDArray[np.int64],
        nodes: NDArray[np.int64],
        weights: NDArray[np.float64],
    ) -> None:
        """Test every cell whose center is within its own radius of each pending point."""
        if not self._class_members:
            exponent = np.floor(np.log2(np.maximum(self._radius, 1e-300))).astype(np.int64)
            for value in np.unique(exponent):
                members = np.flatnonzero(exponent == value)
                self._class_members[int(value)] = members
                self._classes[int(value)] = cKDTree(
                    self._centres[members], balanced_tree=False, compact_nodes=False
                )
        for value, tree in self._classes.items():
            if pending.size == 0:
                break
            reach = math.ldexp(1.0, value + 1) * (1.0 + 1e-9)
            lists = tree.query_ball_point(pts[pending], r=reach, workers=-1)
            counts = np.fromiter((len(item) for item in lists), dtype=np.int64, count=len(lists))
            if counts.sum() == 0:
                continue
            members = self._class_members[value]
            local = members[
                np.concatenate([np.asarray(item, dtype=np.int64) for item in lists if item])
            ]
            who = np.repeat(pending, counts)
            self._try(pts, who, local, cell, nodes, weights)
            pending = pending[cell[pending] < 0]

    # ------------------------------------------------------------- interpolation

    @staticmethod
    def interpolate(location: Location, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate one point-data array at a located set of points.

        Parameters
        ----------
        location : Location
            From :meth:`locate`.
        values : numpy.ndarray
            One value per grid point: shape ``(n_points,)`` or ``(n_points, 3)``.

        Returns
        -------
        numpy.ndarray
            One value per located point, ``nan`` where no cell holds it.
        """
        data = np.asarray(values, dtype=np.float64)
        gathered = data[location.nodes]
        if data.ndim == 1:
            out = np.einsum("pk,pk->p", location.weights, gathered)
        else:
            out = np.einsum("pk,pkc->pc", location.weights, gathered)
        out[~location.found] = np.nan
        return np.asarray(out, dtype=np.float64)

    def sample(
        self,
        points: Sequence[Sequence[float]] | NDArray[np.float64],
        fields: Sequence[str] | None = None,
    ) -> SampleResult:
        """Interpolate point-data arrays at ``points``.

        Parameters
        ----------
        points : array_like
            Shape ``(n, 3)``.
        fields : sequence of str, optional
            The point-data arrays to interpolate; every one by default.

        Returns
        -------
        SampleResult

        Raises
        ------
        VolumeError
            If a requested name is not a point-data array of the grid.
        """
        names = list(self._grid.point_data) if fields is None else list(fields)
        unknown = [name for name in names if name not in self._grid.point_data]
        if unknown:
            raise VolumeError(
                f"the grid has no point data {unknown}; it holds {sorted(self._grid.point_data)}"
            )
        location = self.locate(points)
        values = {name: self.interpolate(location, self._grid.point_data[name]) for name in names}
        return SampleResult(values, location)

    @property
    def point_data_names(self) -> Mapping[str, tuple[int, ...]]:
        """The point-data arrays of the grid, with their shapes."""
        return {name: tuple(array.shape) for name, array in self._grid.point_data.items()}


class ZonedSampler:
    """The samplers of the zone volumes of one step, searched in zone order.

    A point is taken from the first zone whose volume holds it; a point in none has cell
    ``-1`` and ``nan`` values, as in one volume. Only the point data every zone carries is
    sampled when no field is named, so a column is never filled from some zones alone. The
    cell number of a found point is that of its zone's volume.

    Parameters
    ----------
    samplers : sequence of VolumeSampler
        One per zone, in zone order.
    """

    def __init__(self, samplers: Sequence[VolumeSampler]) -> None:
        if not samplers:
            raise VolumeError("a multizone volume needs at least one zone")
        self._samplers = tuple(samplers)

    @property
    def skipped_cells(self) -> int:
        """The cells of every zone that are not searched."""
        return sum(sampler.skipped_cells for sampler in self._samplers)

    @property
    def point_data_names(self) -> Mapping[str, tuple[int, ...]]:
        """The point-data arrays every zone holds, with the shape of the first zone's."""
        first, *rest = (dict(s.point_data_names) for s in self._samplers)
        return {n: shape for n, shape in first.items() if all(n in other for other in rest)}

    def sample(
        self,
        points: Sequence[Sequence[float]] | NDArray[np.float64],
        fields: Sequence[str] | None = None,
    ) -> SampleResult:
        """Interpolate point-data arrays at ``points`` from the zone that holds each one."""
        names = list(self.point_data_names) if fields is None else list(fields)
        results = [sampler.sample(points, names) for sampler in self._samplers]
        cell = np.full(results[0].location.cell.shape, -1, dtype=np.int64)
        nodes = np.zeros_like(results[0].location.nodes)
        weights = np.zeros_like(results[0].location.weights)
        values = {name: np.full_like(results[0].values[name], np.nan) for name in names}
        for result in results:
            take = (cell < 0) & result.found
            cell[take] = result.location.cell[take]
            nodes[take] = result.location.nodes[take]
            weights[take] = result.location.weights[take]
            for name in names:
                values[name][take] = result.values[name][take]
        return SampleResult(values, Location(cell, nodes, weights))
