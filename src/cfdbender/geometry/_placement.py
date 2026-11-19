"""Placement of components: rotation, parent attachment and the wing in its host frame.

Pipeline role: pure numerical stage of the geometry package. A component is
generated in its own frame and moved by its placement and by the chain of
parents it is attached to; shapes are never modified by placement. The wing
of a wing-fuselage fairing is first expressed in the body frame, because the
fairing is built there.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from cfdbender._errors import InputValueError

from ._distributions import planform_values
from ._patch import Patch


def rotation_matrix(angles):
    """Return the rotation of ``angles`` (degrees about x, y, z), applied z after y after x."""
    x, y, z = np.deg2rad(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def attachment_point(parent, station):
    """Return the point of ``parent`` a child attaches to at ``station``."""
    if parent["type"] == "body":
        return np.array([parent["length"] * station, 0.0, 0.0])
    if parent["type"] == "lifting_surface":
        p = planform_values(parent, station)
        return np.array(
            [p["leading_edge_x"], parent["planform"]["span"] / 2 * station, p["leading_edge_z"]]
        )
    if parent["type"] == "propeller":
        return np.array([0.0, parent["radius"] * station, 0.0])
    raise InputValueError("attach.parent: unsupported attachment reference")


def placement_transform(components, name):
    """Return ``(rotation, translation)`` taking the component ``name`` to the world frame.

    Parameters
    ----------
    components : dict
        Resolved components including placement and attachment parents.
    name : str
        Component whose transform is requested.

    Returns
    -------
    tuple of numpy.ndarray
        Rotation matrix of shape (3, 3) and translation in meters of shape (3,).
    """
    component = components[name]
    rotation = rotation_matrix(component["placement"]["rotation"])
    translation = np.asarray(component["placement"]["translation"], float)
    parent = component["attach"]["parent"]
    if parent:
        pr, pt = placement_transform(components, parent)
        anchor = attachment_point(components[parent], component["attach"]["station"])
        return pr @ rotation, pt + pr @ (anchor + translation)
    return rotation, translation


def move(patches, rotation, translation) -> list[Patch]:
    """Return the patches with every control point rotated and translated."""
    return [
        replace(patch, controls=np.asarray(patch.controls) @ rotation.T + translation)
        for patch in patches
    ]


def place(patches, components, name) -> list[Patch]:
    """Return the patches of component ``name`` in the world frame."""
    rotation, translation = placement_transform(components, name)
    return move(patches, rotation, translation)


def wing_in_body_frame(components, body_name, wing_name):
    """Return the wing component with its leading edge expressed in the body frame.

    Parameters
    ----------
    components : dict
        Resolved components by name.
    body_name : str
        Name of the host body.
    wing_name : str
        Name of the lifting surface.

    Returns
    -------
    dict
        A copy of the wing component whose leading-edge distributions are in the body frame.

    Raises
    ------
    InputValueError
        If the wing and body differ by a rigid rotation, or the wing root is off the body
        symmetry plane.
    """
    br, bt = placement_transform(components, body_name)
    wr, wt = placement_transform(components, wing_name)
    if not np.allclose(br.T @ wr, np.eye(3), atol=1e-12):
        raise InputValueError("fairing.hosts: relative rigid rotation is not supported")
    translation = br.T @ (wt - bt)
    if abs(translation[1]) > 1e-12:
        raise InputValueError("fairing.hosts: wing root must lie on the body symmetry plane")
    wing = deepcopy(components[wing_name])
    distributions = wing.setdefault("distributions", {})
    for key, offset in [("leading_edge_x", translation[0]), ("leading_edge_z", translation[2])]:
        if key in distributions:
            distributions[key]["values"] = [
                v + offset / wing["planform"]["span"] for v in distributions[key]["values"]
            ]
        else:
            plan = wing["planform"]
            stations = sorted({*plan["control_stations"], *plan.get("slope_breaks", [])})
            values = planform_values(components[wing_name], np.asarray(stations))[key]
            distributions[key] = {
                "stations": stations,
                "values": ((values + offset) / wing["planform"]["span"]).tolist(),
            }
    return wing


def body_junctions(components, body_name):
    """Return the axial fractions where attached wings start and end, and non-tangent fairing ends.

    A fairing footprint closes at its start and its end with the axial fraction going as the
    square of the contour angle and the azimuth as its power ``2 n`` (``n`` the start or end
    exponent). Below ``n = 1`` the footprint leaves its end point tangent to the station ring
    there, so a ring through that point would bound two body faces with a cusp (a zero corner).
    Such an end is no station: the footprint end then lies inside a band (on an azimuth seam
    there, it meets the seam square), and the band stays cut at the wing root and the body's
    own stations.
    An end with an exponent of 1 or more is a station.
    """
    stations = []
    for name, component in components.items():
        if component["type"] != "lifting_surface" or component["attach"]["parent"] != body_name:
            continue
        wing = wing_in_body_frame(components, body_name, name)
        root = planform_values(wing, 0.0)
        fractions = [0.0, 1.0]
        for other in components.values():
            if other["type"] == "wing_fuselage_fairing" and other["hosts"] == [body_name, name]:
                fractions += [
                    other[end] for end in ("start", "end") if float(other[f"{end}_exponent"]) >= 1
                ]
        stations += [
            (float(root["leading_edge_x"]) + fraction * float(root["chord"]))
            / components[body_name]["length"]
            for fraction in fractions
        ]
    return stations
