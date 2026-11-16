"""Source-face projection and local surface charts for repitching hub rings."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from cfdbender._errors import GeometryWarning, InputValueError, warn

Array = NDArray[np.float64]

# Ratios use the shortest edge of the source cells incident on each node.
NON_ROOT_SURFACE_TOLERANCE = 1e-8
ROOT_STANDOFF_WARNING = 0.5


def surface_checks(
    surface: HubSurface, old: Array, new: Array, roots: set[int], scale: float
) -> dict[str, Any]:
    """Measure every moved hub node; shared roots belong to the rigid blade.

    A non-root node farther from the source hub than :data:`NON_ROOT_SURFACE_TOLERANCE` local
    cell edges is quality: it is warned with the count, the worst node and its distance, and
    listed under ``warnings``. A zero local cell edge is a broken source and refuses.
    """
    ids = np.flatnonzero(np.isfinite(surface.local_size) & np.any(old != new, axis=1))
    distance = surface.nearest(new[ids])[1] if len(ids) else np.empty(0)
    size = surface.local_size[ids]
    if np.any(size <= 0):
        raise InputValueError(
            "repitch: source hub has a zero local cell edge. No file was written."
        )
    ratio = distance / size
    root = np.array([int(i) in roots for i in ids], dtype=bool)
    failed = np.flatnonzero((~root) & (ratio > NON_ROOT_SURFACE_TOLERANCE))
    off: list[str] = []
    if len(failed):
        # Quality, not a broken mesh: the worst node is warned and the mesh written.
        i = int(failed[np.argmax(ratio[failed])])
        off.append(
            f"{len(failed)} non-root hub node(s) leave the source hub; the worst, non-root hub "
            f"node {ids[i]} (zero-based), is {distance[i] * scale:.9g} m from it "
            f"({ratio[i]:.9g} local cell edges), beyond "
            f"NON_ROOT_SURFACE_TOLERANCE={NON_ROOT_SURFACE_TOLERANCE:g}"
        )
        warn(
            f"repitch: {off[0]}; the mesh is written. A smaller pitch change keeps the hub",
            GeometryWarning,
        )
    result: dict[str, Any] = {
        "warnings": off,
        "non_root_surface_tolerance": NON_ROOT_SURFACE_TOLERANCE,
        "root_standoff_warning": ROOT_STANDOFF_WARNING,
        "local_edge_definition": "shortest edge of incident source hub cells",
        "moved_hub_nodes": [
            {
                "node_index": int(i),
                "shared_root": bool(r),
                "distance_to_source_hub_m": float(d * scale),
                "local_cell_edge_m": float(s * scale),
                "distance_to_source_hub_over_local_edge": float(q),
            }
            for i, r, d, s, q in zip(ids, root, distance, size, ratio, strict=True)
        ],
    }
    for name, mask in (("root", root), ("non_root", ~root)):
        result[f"{name}_max_distance_to_source_hub_m"] = float(
            distance[mask].max(initial=0) * scale
        )
        result[f"{name}_max_distance_to_source_hub_over_local_edge"] = float(
            ratio[mask].max(initial=0)
        )
    if result["root_max_distance_to_source_hub_over_local_edge"] > ROOT_STANDOFF_WARNING:
        warn(
            "repitch: the rigid blade root stands off the source hub surface by up to "
            f"{result['root_max_distance_to_source_hub_m']:.9g} m "
            f"({result['root_max_distance_to_source_hub_over_local_edge']:.6g} local cell edges); "
            "the hub shape near the root changes by this amount. "
            "The root follows the blade; inspect the rebuilt seam.",
            GeometryWarning,
        )
    return result


def _nearest(mesh: Any, points: Array) -> tuple[Array, Array, NDArray[np.int64]]:
    """Choose the distance minimum without a fixed-unit normal-based tie tolerance."""
    from trimesh.proximity import nearby_faces
    from trimesh.triangles import closest_point

    points = points.reshape(-1, 3)
    result, distances, owners = [], [], []
    for point, candidates in zip(points, nearby_faces(mesh, points), strict=True):
        triangles = mesh.triangles[candidates]
        near = closest_point(triangles, np.broadcast_to(point, (len(triangles), 3)))
        distance = np.linalg.norm(near - point, axis=1)
        index = int(np.argmin(distance))
        result.append(near[index])
        distances.append(distance[index])
        owners.append(candidates[index])
    return np.asarray(result), np.asarray(distances), np.asarray(owners, dtype=np.int64)


class HubSurface:
    """Keep the finite source faces as the authority for projection and distance."""

    def __init__(self, points: Array, faces: list[list[int]]) -> None:
        import trimesh

        triangles = [[f[0], f[j], f[j + 1]] for f in faces for j in range(1, len(f) - 1)]
        self.mesh = trimesh.Trimesh(points, triangles, process=False)
        self.local_size = np.full(len(points), np.inf)
        for face in faces:
            q = points[face]
            size = float(np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1).min())
            self.local_size[face] = np.minimum(self.local_size[face], size)

    def nearest(self, points: Array) -> tuple[Array, Array]:
        """Return closest points and Euclidean distances to the source triangles."""
        nearest, distance, _ = _nearest(self.mesh, points)
        return nearest.reshape(points.shape), distance


class RingChart:
    """Unwrap a rotor patch by axial distance and azimuth, or a planar patch by SVD."""

    def __init__(self, points: Array, frame: tuple[Array, Array] | None) -> None:
        self.frame = frame
        self.centre = points.reshape(-1, 3).mean(axis=0)
        if frame is None:
            _, _, basis = np.linalg.svd(points.reshape(-1, 3) - self.centre, full_matrices=False)
            self.basis = basis[:2].T
        else:
            origin, axis = frame
            radial = self.centre - origin
            radial -= (radial @ axis) * axis
            self.radius = float(np.linalg.norm(radial))
            self.radial = radial / self.radius
            self.tangent = np.cross(axis, self.radial)

    def coordinates(self, points: Array) -> Array:
        """Return physical-distance coordinates in the patch's local surface plane."""
        if self.frame is None:
            return (points - self.centre) @ self.basis
        origin, axis = self.frame
        p = points - origin
        return np.stack(
            (p @ axis, self.radius * np.arctan2(p @ self.tangent, p @ self.radial)), axis=-1
        )

    def lift(self, uv: Array, source: Array, faces: NDArray[np.int64]) -> Array:
        """Map chart positions to the source's finite triangles by barycentric coordinates."""
        import trimesh
        from trimesh.triangles import points_to_barycentric

        xy = self.coordinates(source).reshape(-1, 2)
        mesh: Any = trimesh.Trimesh(np.c_[xy, np.zeros(len(xy))], faces, process=False)
        query = uv.reshape(-1, 2)
        near, _, triangle = _nearest(mesh, np.c_[query, np.zeros(len(query))])
        bary = points_to_barycentric(mesh.triangles[triangle], near)
        xyz = source.reshape(-1, 3)[faces[triangle]]
        return np.einsum("ij,ijk->ik", bary, xyz).reshape((*uv.shape[:-1], 3))
