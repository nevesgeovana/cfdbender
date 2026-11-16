"""The three identities of a geometry run: parameters, surfaces and build.

Pipeline role: pure helper of the geometry package. Each identity is a SHA-256 of a
canonical form, stored whole and truncated to 16 hex digits where it names a file.

* ``parameter_id`` covers the resolved values only: the source of a value (``given`` or
  ``default``), file paths, comments, open findings and warnings are left out, so stating a
  default explicitly does not change it. A data file a component names enters by its SHA-256, not
  its path. Every number is written by ``float.hex``, so ``9`` and ``9.0`` agree, and so do
  ``0.0`` and ``-0.0``. The order of a configuration's members and of a dominance row's names
  carries no meaning and is sorted.
* ``surface_id`` covers the control nets, knots and degrees of every patch, ordered by
  component name and patch index. It is compared exactly on one platform; across platforms a
  small committed golden is compared to a tolerance.
* ``build_id`` ties a ``parameter_id`` to the package version and the kernel that built it.

The bytes of an IGES file are deliberately not an identity: they depend on creation order and
on the time, and a pure stage has no IGES.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cfdbender._errors import InputValueError
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._patch import Patch

__all__ = ["SCHEMA_MAJOR", "build_id", "canonical", "parameter_id", "short", "surface_id"]

SCHEMA_MAJOR = 1
_PATH_KEYS = frozenset({"rings_file", "sections_file", "original", "file", "open_findings"})
_DROPPED = ("aircraft", "warnings", "findings", "derived", "configurations", "format")


def canonical(value) -> str:
    """Return the canonical JSON of plain data: sorted keys, every number by ``float.hex``."""

    def convert(node):
        if isinstance(node, Mapping):
            return {str(k): convert(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [convert(v) for v in node]
        if isinstance(node, (bool, str)) or node is None:
            return node
        if isinstance(node, (int, float, np.integer, np.floating)):
            return "f:" + (float(node) + 0.0).hex()
        raise InputValueError(f"identity: not plain data: {type(node).__name__}")

    return json.dumps(convert(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def short(identity: str) -> str:
    """Return the 16-hex form of an identity, the form used in names."""
    return identity[:16]


def _strip_paths(node):
    if isinstance(node, dict):
        return {k: _strip_paths(v) for k, v in node.items() if k not in _PATH_KEYS}
    if isinstance(node, list):
        return [_strip_paths(v) for v in node]
    return node


def parameter_id(resolved: Mapping) -> str:
    """Return the ``parameter_id`` of a resolved configuration, as 64 hex digits.

    Parameters
    ----------
    resolved : Mapping
        A configuration as :func:`cfdbender.geometry.resolve_config` returns it. File
        paths, the source of each value and the order of members do not enter the identity.

    Returns
    -------
    str
        The SHA-256 of the canonical form of the values, so two resolved configurations that
        mean the same parameters give the same string.

    Examples
    --------
    >>> from cfdbender.geometry import resolve_config
    >>> from cfdbender.geometry.stages import parameter_id
    >>> resolved = resolve_config({"type": "lifting_surface"})
    >>> len(parameter_id(resolved))
    64
    """
    values = unpack(dict(resolved))
    for key in _DROPPED:
        values.pop(key, None)
    values = _strip_paths(values)
    for row in values.get("registry", {}).values():
        row["members"] = sorted(row["members"], key=lambda member: member["component"])
    for row in values.get("dominance", []):
        row["dominant"], row["dominated"] = sorted(row["dominant"]), sorted(row["dominated"])
    values["schema"] = SCHEMA_MAJOR
    return _digest(canonical(values))


def surface_id(
    patches: Mapping[str, Sequence[Patch]], trims: Mapping[str, Sequence[Any]] | None = None
) -> str:
    """Return the ``surface_id`` of placed patches given per component name, as 64 hex digits.

    ``trims``, per component, are the faces as loops on those bases (:class:`Trim`); each loop
    joins the identity, sorted by base, role and face, so two models on equal bases with other
    faces differ. Without trims the identity is that of the bases alone, as before.

    Parameters
    ----------
    patches : Mapping
        Ordered placed patches per component, with control points in metres.
    trims : Mapping, optional
        Trim loops per component; omitted means only bases enter the digest.

    Returns
    -------
    str
        SHA-256 identity as 64 hexadecimal digits.
    """
    rows: list[dict] = []
    for name in sorted(patches):
        for index, patch in enumerate(patches[name]):
            rows.append(
                {
                    "component": name,
                    "patch": index,
                    "degree": [patch.degree_u, patch.degree_v],
                    "knots_u": list(patch.knots_u),
                    "knots_v": list(patch.knots_v),
                    "mult_u": list(patch.multiplicities_u),
                    "mult_v": list(patch.multiplicities_v),
                    "controls": np.asarray(patch.controls, dtype=float).tolist(),
                }
            )
    trim_sets = trims or {}
    for name in sorted(trim_sets):
        for trim in sorted(trim_sets[name], key=lambda t: (t.base, t.role, t.face)):
            rows.append({"component": name, **trim.to_dict()})
    return _digest(canonical(rows))


def build_id(parameter: str, package_version: str, kernel_name: str, kernel_version: str) -> str:
    """Return the ``build_id``: a parameter identity tied to the package and the kernel.

    Parameters
    ----------
    parameter : str
        Resolved parameter identity.
    package_version : str
        Package version used for generation.
    kernel_name : str
        Geometry kernel name.
    kernel_version : str
        Geometry kernel version.

    Returns
    -------
    str
        SHA-256 identity as 64 hexadecimal digits.
    """
    return _digest(
        canonical(
            {
                "parameter_id": parameter,
                "package": package_version,
                "kernel": [kernel_name, kernel_version],
            }
        )
    )
