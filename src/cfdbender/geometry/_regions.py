"""Inside tests of the placed components, for choosing which assembled faces are exposed.

Pipeline role: pure numerical stage of the geometry package. A region is a
predicate on a world-frame point: true strictly inside a body, a lifting
surface, or the closed solid a fairing bounds. The fairing solid is closed
virtually, inside its hosts, by triangle fans that are classification data
only and never exported surfaces.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from ._airfoils import cst_airfoil
from ._bodies import body_profile, body_station_table
from ._distributions import blended_section, planform_values
from ._lifting import section_models
from ._patch import Patch


def body_region(component):
    """Return the predicate for points strictly inside the body, in the body frame."""
    table = body_station_table(component)

    def contains(point):
        x, y, z = point
        if not 1e-7 < x < component["length"] - 1e-7:
            return False
        width, height, centre, top, bottom = body_profile(
            component, table, np.asarray(x / component["length"])
        )
        exponent = top if z >= centre else bottom
        return bool(
            (abs(y) / width) ** exponent + (abs(z - centre) / height) ** exponent < 1 - 1e-6
        )

    return contains


def lifting_region(component):
    """Return the predicate for points inside the flat-tip lifting surface, reflected across y=0."""
    models = section_models(component["sections"])

    def contains(point):
        x, y, z = point
        eta = 2 * abs(y) / component["planform"]["span"]
        if not 0 <= eta < 1 - 1e-7:
            return False
        p = planform_values(component, eta)
        angle = np.deg2rad(p["twist"])
        axis = component["planform"]["twist_axis"]
        dx, dz = x - p["leading_edge_x"] - axis * p["chord"], z - p["leading_edge_z"]
        xx = (dx * np.cos(angle) - dz * np.sin(angle)) / p["chord"] + axis
        zz = (dx * np.sin(angle) + dz * np.cos(angle)) / p["chord"]
        if not 1e-7 < xx < 1 - 1e-7:
            return False
        section = blended_section(component["sections"], models, eta)
        up, low = cst_airfoil(
            xx, section["upper"], section["lower"], section["trailing_edge_thickness"]
        )
        return bool(low[1] + 1e-7 < zz < up[1] - 1e-7)

    return contains


def placed_region(contains, rotation, translation):
    """Return ``contains`` evaluated in the frame of a component placed by a rigid motion."""
    return lambda point: contains(rotation.T @ (np.asarray(point) - translation))


def sample_patch(patch: Patch, nu=33, nv=17):
    """Sample a patch on an ``nu`` by ``nv`` lattice, shape ``(nu, nv, 3)``."""
    controls = np.asarray(patch.controls)
    ku = np.repeat(patch.knots_u, patch.multiplicities_u)
    kv = np.repeat(patch.knots_v, patch.multiplicities_v)
    bu = BSpline.design_matrix(np.linspace(0, 1, nu), ku, patch.degree_u).toarray()
    bv = BSpline.design_matrix(np.linspace(0, 1, nv), kv, patch.degree_v).toarray()
    return np.einsum("ai,ijc,bj->abc", bu, controls, bv)


def grid_triangles(grid):
    """Split every cell of a sample grid into two triangles, shape ``(n, 3, 3)``."""
    a, b, c, d = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
    return np.concatenate(
        (
            np.stack((a, b, c), axis=-2).reshape(-1, 3, 3),
            np.stack((a, c, d), axis=-2).reshape(-1, 3, 3),
        )
    )


def fairing_volume_triangles(patches, interior):
    """Close the two contact rings virtually inside the hosts for classification.

    These triangles are transient integration data, never exported surfaces or
    solids. The resulting winding test is used only on the other host skins.
    """
    triangles = []
    for patch in patches:
        grid = sample_patch(patch)
        triangles.append(grid_triangles(grid))
        edge = grid[:, 0] if patch.family == "fairing_body" else grid[:, -1]
        centres = np.broadcast_to(interior, edge[:-1].shape)
        if patch.family == "fairing_body":
            triangles.append(np.stack((centres, edge[1:], edge[:-1]), axis=1))
        else:
            triangles.append(np.stack((centres, edge[:-1], edge[1:]), axis=1))
    skin = np.concatenate(triangles)
    mirrored = skin[:, ::-1] * np.array([1.0, -1.0, 1.0])
    return np.concatenate((skin, mirrored))


def inside_triangles(point, triangles):
    """Return whether ``point`` is inside the closed triangle surface, by the winding number."""
    vectors = triangles - np.asarray(point)
    a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        la * lb * lc
        + np.einsum("ij,ij->i", a, b) * lc
        + np.einsum("ij,ij->i", b, c) * la
        + np.einsum("ij,ij->i", c, a) * lb
    )
    return bool(abs(np.sum(2 * np.arctan2(numerator, denominator))) > 2 * np.pi)
