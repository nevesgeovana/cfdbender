"""Validate named boundary groups against a tracked mesh marker inventory."""

from __future__ import annotations

import importlib
import io
import re
from collections.abc import Mapping
from typing import Any, cast

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


_MARKER_TAG = re.compile(r"^\s*MARKER_TAG\s*=\s*([^%\r\n]+)", re.MULTILINE)
_LONG_LINE = 1 << 20


def _su2_markers(text: str) -> set[str]:
    return {item.strip() for item in _MARKER_TAG.findall(text)}


class _Su2Markers:
    """Collect the SU2 MARKER_TAG names block by block, never holding more than one line."""

    def __init__(self) -> None:
        self._tail = b""
        self._names: set[str] = set()
        self._skipping = False

    def feed(self, chunk: bytes) -> None:
        """Take the next block of the mesh; whole lines are scanned, a partial one is kept."""
        data = self._tail + chunk
        end = data.rfind(b"\n") + 1
        if self._skipping:
            self._skipping = end == 0
            self._tail = data[end:]
            return
        block, self._tail = data[:end], data[end:]
        if b"MARKER_TAG" in block:
            self._names |= _su2_markers(block.decode("ascii", errors="replace"))
        if len(self._tail) > _LONG_LINE:
            self._tail, self._skipping = b"", True

    def result(self) -> tuple[str, ...]:
        """Return the sorted marker names, after the last block."""
        tail = self._tail.decode("ascii", errors="replace")
        return tuple(sorted(self._names | _su2_markers(tail)))


def _cgns_names(source: Any, name: str) -> tuple[str, ...]:
    """Read HDF5 CGNS boundary node names from a path or a file object."""
    try:
        h5py = importlib.import_module("h5py")
        names = []
        with h5py.File(source, "r") as mesh:

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


def marker_names(data: bytes, name: str) -> tuple[str, ...]:
    """Read SU2 tags, Gmsh physical names, or HDF5 CGNS boundary nodes."""
    if name.endswith(".cgns"):
        return _cgns_names(io.BytesIO(data), name)
    text = data.decode("ascii", errors="replace")
    if name.endswith(".su2"):
        return tuple(sorted(_su2_markers(text)))
    section = text.partition("$PhysicalNames")[2].partition("$EndPhysicalNames")[0]
    return tuple(sorted(set(re.findall(r'^\s*[12]\s+\d+\s+"([^"]+)"', section, re.MULTILINE))))


def _refuse_unknown(
    requested: Mapping[str, tuple[str, ...]], known: tuple[str, ...], mesh: str
) -> None:
    unknown = sorted({m for markers in requested.values() for m in markers} - set(known))
    if unknown:
        raise ValueError(
            f"setup file [groups]: unknown markers {', '.join(unknown)}; "
            f"mesh {mesh} has markers: {', '.join(known) or 'none'}"
        )


def validate_groups(setup: Mapping[str, object], data: bytes, mesh: str) -> tuple[str, ...]:
    """Refuse an unknown marker, naming the markers the mesh actually declares."""
    requested = groups(setup)
    if not requested:
        return ()
    known = marker_names(data, mesh)
    _refuse_unknown(requested, known, mesh)
    return known


def tracked_markers(files: InputFiles, mesh: str) -> tuple[str, ...]:
    """Return a mesh's marker names, tracking the file without holding it in memory.

    A ``.su2`` file is scanned in the pass that digests it and a ``.cgns`` file is opened by
    path, so a large mesh is read at most once per plan and never held; any other format keeps
    the whole-bytes route. The tracked reader caches the result for the plan.
    """
    if mesh.endswith(".su2"):
        return cast(tuple[str, ...], files.scan(mesh, "su2-markers", _Su2Markers))
    if mesh.endswith(".cgns"):
        return cast(
            tuple[str, ...],
            files.derive(mesh, "cgns-markers", lambda path: _cgns_names(path, mesh)),
        )
    return marker_names(files.read_bytes(mesh), mesh)


def validate_tracked_groups(
    setup: Mapping[str, object], files: InputFiles, mesh: str
) -> tuple[str, ...]:
    """Track a mesh and refuse an unknown marker, as ``validate_groups`` does for held bytes."""
    requested = groups(setup)
    if not requested:
        files.track(mesh)
        return ()
    known = tracked_markers(files, mesh)
    _refuse_unknown(requested, known, mesh)
    return known


def point_markers(point: AdapterPoint, mesh: str) -> tuple[str, ...]:
    """Read an already planned mesh through the tracked reader to validate group names."""
    files = InputFiles(point.root)
    if mesh.endswith(".su2") and groups(point.setup):
        tracked_markers(files, mesh)  # one pass digests and scans the file
    else:
        files.track(mesh)
    if files.digests().get(mesh) != point.point.inputs_sha256.get(mesh):
        raise ValueError(f"{mesh}: input changed after planning; plan again")
    return validate_tracked_groups(point.setup, files, mesh)
