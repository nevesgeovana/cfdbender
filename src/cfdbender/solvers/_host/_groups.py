"""Validate named boundary groups against a tracked mesh marker inventory."""

from __future__ import annotations

import importlib
import io
import re
from collections.abc import Mapping
from typing import Any

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.plan import InputFiles


def groups(setup: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """Read [groups] as nonempty marker lists with safe, distinct result names."""
    table = setup.get("groups", {})
    if not isinstance(table, Mapping):
        raise ValueError("setup file [groups] must be a table: name = [markers]")
    result: dict[str, tuple[str, ...]] = {}
    for name, markers in table.items():
        if (
            not isinstance(name, str)
            or re.fullmatch("[A-Za-z][A-Za-z0-9_-]*", name) is None
            or name.upper() == "TOTAL"
        ):
            raise ValueError("setup file [groups]: names must be identifiers other than TOTAL")
        if (
            not isinstance(markers, list)
            or not markers
            or any(
                not isinstance(m, str) or re.fullmatch("[A-Za-z0-9_.:+-]+", m) is None
                for m in markers
            )
        ):
            raise ValueError(
                f"setup file [groups] {name}: expected a nonempty list of marker names"
            )
        if len(set(markers)) != len(markers):
            raise ValueError(f"setup file [groups] {name}: a marker is repeated")
        if name.casefold() in {previous.casefold() for previous in result}:
            raise ValueError(f"setup file [groups] {name}: names must differ ignoring case")
        result[name] = tuple(markers)
    return result


def marker_names(data: bytes, name: str) -> tuple[str, ...]:
    """Read SU2 tags, Gmsh physical names, or HDF5 CGNS boundary nodes."""
    if name.endswith(".cgns"):
        try:
            h5py = importlib.import_module("h5py")
            names = []
            with h5py.File(io.BytesIO(data), "r") as mesh:

                def visit(path: str, node: Any) -> None:
                    label = node.attrs.get("label", b"")
                    if isinstance(label, bytes):
                        label = label.decode("ascii").rstrip(chr(0))
                    if label == "BC_t":
                        names.append(path.rsplit("/", 1)[-1])

                mesh.visititems(visit)
            return tuple(sorted(set(names)))
        except (ImportError, OSError, ValueError) as error:
            raise ValueError(
                f"{name}: cannot read CGNS boundary names; install h5py for HDF5 CGNS"
            ) from error
    text = data.decode("ascii", errors="replace")
    if name.endswith(".su2"):
        names = re.findall(r"^\s*MARKER_TAG\s*=\s*([^%\r\n]+)", text, re.MULTILINE)
        return tuple(sorted({item.strip() for item in names}))
    section = text.partition("$PhysicalNames")[2].partition("$EndPhysicalNames")[0]
    return tuple(sorted(set(re.findall(r'^\s*[12]\s+\d+\s+"([^"]+)"', section, re.MULTILINE))))


def validate_groups(setup: Mapping[str, object], data: bytes, mesh: str) -> tuple[str, ...]:
    """Refuse an unknown marker, naming the markers the mesh actually declares."""
    requested = groups(setup)
    if not requested:
        return ()
    known = marker_names(data, mesh)
    unknown = sorted({m for markers in requested.values() for m in markers} - set(known))
    if unknown:
        raise ValueError(
            f"setup file [groups]: unknown markers {', '.join(unknown)}; "
            f"mesh {mesh} has markers: {', '.join(known) or 'none'}"
        )
    return known


def point_markers(point: AdapterPoint, mesh: str) -> tuple[str, ...]:
    """Read an already planned mesh through the tracked reader to validate group names."""
    files = InputFiles(point.root)
    data = files.read_bytes(mesh)
    if files.digests().get(mesh) != point.point.inputs_sha256.get(mesh):
        raise ValueError(f"{mesh}: input changed after planning; plan again")
    return validate_groups(point.setup, data, mesh)
