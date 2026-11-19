"""Checking a rebuilt surface against an original: sampled forward and restricted reverse distance.

Pipeline role: the workspace row of the geometry check. Both skins are sampled on
their trimmed faces and edges and each sample is measured to the other skin with
OpenCASCADE witnesses (seeded Gauss-Newton on faces, curve projection on edges,
conservative bounding-box pruning). Forward is original to rebuilt everywhere;
reverse is rebuilt to original, restricted to samples whose nearest original
witness is not on a trim edge, so a rebuilt surface that extends past the original
trim is not penalized. Empty reverse coverage is a rejection, never a vacuous
acceptance. The limit is a named profile of
:data:`cfdbender.geometry.component.DEVIATION_PROFILES` (``study_default``:
the larger of 0.5 mm and 0.1 percent of the local chord). Distances are sampled
upper bounds, not continuous Hausdorff distances.

gmsh is a dependency of the package; the functions raise ``KernelUnavailableError``,
naming the library, when it cannot load.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from cfdbender._errors import InputValueError
from cfdbender.geometry.component import DEVIATION_PROFILES
from cfdbender.mesh._cad import Face
from cfdbender.mesh._gmsh import GmshKernel, gmsh_module, open_kernel

__all__ = ["check_against_original", "local_limits"]

_PURPOSE = "checking a rebuilt surface against an original"


def import_component(kernel, path, name=None):
    """Import an IGES file at meter scale and return the faces of one component, or all of them."""
    named = kernel.import_iges(path)
    selected = [
        face
        for face, label in zip(named.faces, named.names, strict=True)
        if name is None or label == name
    ]
    if not selected:
        raise InputValueError(f"IGES import: no faces named {name}")
    return selected


def surface_tags(surfaces):
    """Return the faces of ``surfaces``, given as OCC tags or ``(2, tag)`` pairs."""
    tags = [
        int(item[1])
        if isinstance(item, (tuple, list)) and len(item) == 2 and item[0] == 2
        else int(item)
        for item in surfaces
    ]
    if not tags or len(set(tags)) != len(tags):
        raise InputValueError("component_surfaces: nonempty unique OCC surface tags required")
    return [Face(tag) for tag in tags]


def resolve_faces(kernel, value, name=None):
    """Return faces for an IGES path (optionally ``path#component``) or a list of tags."""
    if isinstance(value, (str, Path)):
        path, separator, component = str(value).partition("#")
        return import_component(kernel, path, component if separator else name)
    return surface_tags(value)


def edges_of(kernel, faces):
    return sorted({abs(edge) for face in faces for edge in kernel.boundary(face)})


def face_samples(kernel, face, count):
    low, high = map(np.asarray, kernel.uv_bounds(face))
    axis = (np.arange(count) + 0.5) / count
    uv = np.stack(
        np.meshgrid(
            low[0] + axis * (high[0] - low[0]), low[1] + axis * (high[1] - low[1]), indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 2)
    uv = uv[kernel.inside(face, uv)]
    xyz = np.asarray(kernel.values(face, uv)) if len(uv) else np.empty((0, 3))
    return xyz, uv, (low, high)


def edge_samples(kernel, edge, count):
    low, high = kernel.edge_bounds(edge)
    try:
        return np.asarray(kernel.edge_values(edge, np.linspace(low, high, count)))
    except Exception:
        return np.asarray(kernel.edge_ends(edge)).reshape(-1, 3)


def sample_surfaces(kernel, faces, count, edge_count):
    points, coverage = [], []
    for face in faces:
        xyz, _, _ = face_samples(kernel, face, count)
        used = count
        while not len(xyz) and used < 257:
            used = used * 2 + 1
            xyz, _, _ = face_samples(kernel, face, used)
        if not len(xyz):
            raise InputValueError(f"sampling: face {face.tag} has no sampled trimmed interior")
        points.append(xyz)
        coverage.append(dict(face=face.tag, interior_samples=len(xyz), lattice=used))
    for edge in edges_of(kernel, faces):
        points.append(edge_samples(kernel, edge, edge_count))
    result = np.concatenate(points)
    if not np.all(np.isfinite(result)):
        raise InputValueError("sampling: nonfinite coordinates")
    return result, coverage


def box_lower_bound(points, box):
    low, high = (np.asarray(corner) for corner in box)
    return np.linalg.norm(np.maximum(np.maximum(low - points, points - high), 0), axis=1)


def edge_distance(kernel, points, edge):
    """Recover curve parameters from XYZ; batched OCC returned parameters are unreliable."""
    try:
        projected = np.asarray(kernel.edge_project(edge, points))
        distance = np.linalg.norm(points - projected, axis=1)
        endpoints = edge_samples(kernel, edge, 2)
        return np.minimum(distance, cKDTree(endpoints).query(points)[0])
    except Exception:
        xyz = np.asarray(kernel.edge_ends(edge)).reshape(-1, 3)
        if not len(xyz):
            raise InputValueError(
                f"projection: edge {edge} has neither curve nor vertex witness"
            ) from None
        return cKDTree(xyz).query(points)[0]


def surface_distance(kernel, points, face, seed):
    """Return the distance of each point to ``face``; ``seed`` is ``(xyz, uv, bounds)``."""
    seed_xyz, seed_uv, bounds = seed
    low, high = bounds
    xyz, uv = kernel.project(face, points)
    distance = np.where(kernel.inside(face, uv), np.linalg.norm(points - xyz, axis=1), np.inf)
    uv = seed_uv[cKDTree(seed_xyz).query(points)[1]].copy()
    moving = np.arange(len(points))
    for _ in range(20):
        if not len(moving):
            break
        here = uv[moving]
        xyz = np.asarray(kernel.values(face, here))
        derivative = np.asarray(kernel.derivatives(face, here))
        residual = xyz - points[moving]
        a = np.sum(derivative[:, 0] ** 2, axis=1)
        b = np.sum(derivative[:, 0] * derivative[:, 1], axis=1)
        c = np.sum(derivative[:, 1] ** 2, axis=1)
        d = np.sum(derivative[:, 0] * residual, axis=1)
        e = np.sum(derivative[:, 1] * residual, axis=1)
        determinant = np.maximum(a * c - b * b, 1e-30)
        delta = np.column_stack(((c * d - b * e) / determinant, (a * e - b * d) / determinant))
        updated = np.clip(
            here - np.clip(delta, -0.15 * (high - low), 0.15 * (high - low)), low, high
        )
        uv[moving] = updated
        moving = moving[np.max(abs(updated - here) / np.maximum(high - low, 1e-30), axis=1) > 1e-11]
    xyz = np.asarray(kernel.values(face, uv))
    valid = kernel.inside(face, uv)
    return np.minimum(distance, np.where(valid, np.linalg.norm(points - xyz, axis=1), np.inf))


def nearest(kernel, points, faces, boundary_tolerance=1e-8):
    """Reuse SMIGEOM4's bounded candidate selection and seeded projection algorithm."""
    seeds = {}
    for face in faces:
        xyz, uv, bounds = face_samples(kernel, face, 17)
        if not len(xyz):
            xyz = np.concatenate([edge_samples(kernel, e, 17) for e in edges_of(kernel, [face])])
            uv = np.asarray(kernel.parametrize(face, xyz))
        seeds[face] = xyz, uv, bounds
    best = cKDTree(np.concatenate([entry[0] for entry in seeds.values()])).query(points)[0]
    for face, seed in seeds.items():
        near = box_lower_bound(points, kernel.bbox(face)) <= best + boundary_tolerance
        ids = np.flatnonzero(near)
        if len(ids):
            best[ids] = np.minimum(best[ids], surface_distance(kernel, points[ids], face, seed))
    boundary = np.full(len(points), np.inf)
    for edge in edges_of(kernel, faces):
        near = box_lower_bound(points, kernel.edge_bbox(edge)) <= best + boundary_tolerance
        ids = np.flatnonzero(near)
        if len(ids):
            boundary[ids] = np.minimum(boundary[ids], edge_distance(kernel, points[ids], edge))
    distance = np.minimum(best, boundary)
    if not np.all(np.isfinite(distance)):
        raise InputValueError("projection: no finite valid witness")
    return distance, best < boundary - boundary_tolerance


def local_limits(points, limits, profile="study_default"):
    """Return the allowed deviation, in meters, at each of ``points`` under a named profile.

    The local chord comes from ``limits``: ``local_chord`` (a function of the points),
    ``stations_m`` with ``chords_m`` along ``axis`` (default y), or one ``chord_m``.
    """
    constants = _profile(profile)
    if callable(limits.get("local_chord")):
        chord = np.asarray(limits["local_chord"](points), float)
    elif "stations_m" in limits:
        stations = np.asarray(limits["stations_m"], float)
        chords = np.asarray(limits["chords_m"], float)
        if len(stations) < 2 or len(stations) != len(chords) or np.any(np.diff(stations) <= 0):
            raise InputValueError("limits: increasing stations and matching chords required")
        chord = np.interp(points[:, limits.get("axis", 1)], stations, chords)
    elif "chord_m" in limits:
        chord = np.full(len(points), limits["chord_m"])
    else:
        raise InputValueError(
            "limits: give chord_m, stations_m with chords_m, or a local_chord function"
        )
    chord = np.broadcast_to(chord, (len(points),))
    if np.any(~np.isfinite(chord)) or np.any(chord <= 0):
        raise InputValueError("limits: positive finite local chord required")
    return np.maximum(constants["floor_mm"] / 1000, constants["local_fraction"] * chord)


def _kernel() -> tuple[GmshKernel, bool]:
    """Return the kernel of the model already open, or a new one the caller closes."""
    gmsh = gmsh_module(_PURPOSE)
    if gmsh.isInitialized():
        return GmshKernel(gmsh), False
    return open_kernel(_PURPOSE), True


def _profile(profile):
    try:
        return DEVIATION_PROFILES[profile]
    except KeyError:
        raise InputValueError(
            f"deviation profile {profile!r}: unknown; available: {sorted(DEVIATION_PROFILES)}"
        ) from None


def statistics(points, distance, limits, profile="study_default"):
    """Return the maximum, mean and exceedance count of ``distance`` against the local limit."""
    if not len(distance):
        return {
            "samples": 0,
            "max_m": None,
            "mean_m": None,
            "max_mm": None,
            "mean_mm": None,
            "exceedance_count": 0,
            "sampled_pass": False,
        }
    tolerance = local_limits(points, limits, profile)
    ratio = distance / tolerance
    worst = int(np.argmax(distance))
    return {
        "samples": len(distance),
        "max_m": float(distance[worst]),
        "mean_m": float(distance.mean()),
        "max_mm": float(distance[worst] * 1000),
        "mean_mm": float(distance.mean() * 1000),
        "exceedance_count": int(np.count_nonzero(ratio > 1)),
        "maximum_distance_to_limit_ratio": float(ratio.max()),
        "worst_point_m": points[worst].tolist(),
        "local_limit_at_worst_m": float(tolerance[worst]),
        "sampled_pass": bool(np.all(ratio <= 1)),
    }


def _region_results(limits, profile, old, new, forward, *, reverse, interior):
    regions = {}
    for label, mask in (limits.get("regions") or {}).items():
        fm, rm = np.asarray(mask(old), bool), np.asarray(mask(new[interior]), bool)
        fr = statistics(old[fm], forward[fm], limits, profile)
        rr = statistics(new[interior][rm], reverse[interior][rm], limits, profile)
        regions[label] = {
            "forward": fr,
            "restricted_reverse": rr,
            "accepted": bool(fr["sampled_pass"] and (rr["sampled_pass"] or rr["samples"] == 0)),
        }
    return regions


def check_against_original(
    rebuilt: Any, original: Any, limits: Mapping[str, Any], *, profile: str = "study_default"
) -> dict[str, Any]:
    """Compare a rebuilt surface with an original: forward and restricted reverse distance.

    ``rebuilt`` and ``original`` are OCC surface tags in the active model or IGES paths,
    optionally ``path#component``. Distances and local chord inputs are meters. Reverse
    samples whose nearest original witness lies on an original face trim edge are excluded
    explicitly. Empty reverse coverage is a rejection, never a vacuous acceptance.

    Parameters
    ----------
    limits : mapping
        The local chord (see :func:`local_limits`), and optionally ``component``,
        ``exclude_faces``, ``forward_count``, ``reverse_count``, ``edge_count``, ``fold_y``
        and ``regions`` (a mapping of label to a mask of points).
    rebuilt, original
        The two surfaces, as described above.
    profile : str
        Named deviation profile; an unknown name is refused with the available names.

    Returns
    -------
    dict
        ``accepted`` (both directions passed), the ``forward``, ``restricted_reverse`` and
        ``unrestricted_reverse`` statistics, the ``regions`` results, the excluded original
        faces and boundary samples, the sample coverage of each surface, the ``profile`` and
        its ``tolerance_rule``, and the ``method`` (sampled upper bounds, not continuous
        Hausdorff distances).

    Raises
    ------
    InputValueError
        For an unknown profile, limits that name no chord, or a surface that cannot be resolved.
    cfdbender.mesh._extras.MissingExtraError
        When the gmsh library cannot load (``KernelUnavailableError``).
    """
    constants = _profile(profile)
    for key in ("forward_count", "reverse_count", "edge_count"):
        if key in limits and limits[key] < 1:
            raise InputValueError(f"limits.{key}: at least one sample required")
    kernel, own_session = _kernel()
    try:
        new_faces = resolve_faces(kernel, rebuilt, limits.get("component"))
        old_faces = resolve_faces(kernel, original, limits.get("component"))
        for i in limits.get("exclude_faces") or []:
            if not 0 <= i < len(old_faces):
                raise InputValueError(
                    f"limits.exclude_faces: face {i} does not exist; the original has faces 0 to "
                    f"{len(old_faces) - 1}"
                )
        excluded = [
            {"index": int(i), "tag": old_faces[i].tag} for i in (limits.get("exclude_faces") or [])
        ]
        drop = {e["index"] for e in excluded}
        old_faces = [f for i, f in enumerate(old_faces) if i not in drop]
        edge_count = limits.get("edge_count", 101)
        old, old_coverage = sample_surfaces(
            kernel, old_faces, limits.get("forward_count", 45), edge_count
        )
        new, new_coverage = sample_surfaces(
            kernel, new_faces, limits.get("reverse_count", 31), edge_count
        )
        if limits.get(
            "fold_y"
        ):  # half-body original: the rebuilt mirror half folds onto the modeled side
            new = np.c_[new[:, 0], -np.abs(new[:, 1]), new[:, 2]]
        forward, _ = nearest(kernel, old, new_faces)
        reverse, interior = nearest(kernel, new, old_faces)
        f = statistics(old, forward, limits, profile)
        r = statistics(new[interior], reverse[interior], limits, profile)
        return {
            "accepted": bool(f["sampled_pass"] and r["sampled_pass"]),
            "regions": _region_results(
                limits, profile, old, new, forward, reverse=reverse, interior=interior
            ),
            "forward": f,
            "restricted_reverse": r,
            "unrestricted_reverse": statistics(new, reverse, limits, profile),
            "excluded_original_faces": excluded,
            "reverse_excluded_boundary_samples": int(np.count_nonzero(~interior)),
            "reverse_included_samples": int(np.count_nonzero(interior)),
            "source_coverage": old_coverage,
            "rebuilt_coverage": new_coverage,
            "profile": profile,
            "tolerance_rule": (
                f"max({constants['floor_mm']} mm, {constants['local_fraction']} times local chord)"
            ),
            "boundary_tolerance_m": 1e-8,
            "method": (
                "Dense trimmed UV interiors and edges; valid OCC and seeded Gauss-Newton surface "
                "witnesses, curve witnesses and conservative face-box pruning. Sampled upper "
                "bounds, not continuous Hausdorff distances."
            ),
        }
    finally:
        if own_session:
            kernel.close()
