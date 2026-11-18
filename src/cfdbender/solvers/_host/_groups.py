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
#: A line longer than this that could matter sends the file to the exact whole-file route.
_LONG_LINE = 1 << 22
#: A reduced text longer than this sends the file to the exact whole-file route.
_REDUCED_LIMIT = 1 << 26
#: The bytes that a regex whitespace class matches in ASCII-decoded text (a line holds no newline).
_WHITESPACE = b" \t\r\x0b\x0c\x1c\x1d\x1e\x1f"


def _su2_markers(text: str) -> set[str]:
    return {item.strip() for item in _MARKER_TAG.findall(text)}


class _Su2Markers:
    r"""Reduce an SU2 mesh to the lines that can take part in a marker match, block by block.

    The pattern is ``^\s*MARKER_TAG\s*=\s*([^%\r\n]+)``; its whitespace may cross lines. A match
    starts at a line start, passes only whitespace, and ends on the line holding the name. After
    the tag a match needs at most two lines that are not whitespace-only: the ``=`` and the
    name. So the only lines that matter are (1) every line containing ``MARKER_TAG`` and (2) after
    such a line, every line up to and including the next two that are not whitespace-only (the
    backtracking ``\s*`` may even give the name a whitespace character of a whitespace-only
    line). Every other line is removed: a whitespace-only line outside that window adds nothing
    to ``\s*``, and any other line cannot start, continue or end a match. The kept lines,
    in order, form a reduced text on which the unchanged whole-file regular expression gives
    exactly the names of the whole file. Memory is one block, one partial line and the reduced
    text.

    ``result`` returns None, and the caller then uses the exact whole-file function, when a line
    longer than ``_LONG_LINE`` could matter or the reduced text exceeds ``_REDUCED_LIMIT``:
    correct and slow, never truncated. A very long line that cannot matter (it starts with
    something other than whitespace and the tag, and no tag is awaited) is skipped.
    """

    def __init__(self) -> None:
        self._tail = b""
        self._kept: list[bytes] = []
        self._size = 0
        self._pending = 0
        self._skipping = False
        self._overflow = False

    def _keep(self, line: bytes) -> None:
        self._kept.append(line)
        self._size += len(line) + (len(self._kept) > 1)
        if self._size > _REDUCED_LIMIT:
            self._overflow = True

    def _line(self, line: bytes) -> None:
        if len(line) > _LONG_LINE and (self._pending or b"MARKER_TAG" in line):
            self._overflow = True
            return
        if b"MARKER_TAG" in line:
            self._keep(line)
            self._pending = 2
        elif self._pending:
            self._keep(line)
            if line.strip(_WHITESPACE):
                self._pending -= 1

    def feed(self, chunk: bytes) -> None:
        """Take the next block of the mesh; whole lines are examined, a partial one is kept."""
        if self._overflow:
            return
        data = self._tail + chunk
        self._tail = b""
        if self._skipping:
            end = data.find(b"\n") + 1
            if not end:
                return
            self._skipping = False
            data = data[end:]
        end = data.rfind(b"\n") + 1
        block, self._tail = data[:end], data[end:]
        if block and (self._pending or b"MARKER_TAG" in block):
            start = 0
            while start < len(block):
                stop = block.find(b"\n", start)
                self._line(block[start:stop])
                start = stop + 1
        if len(self._tail) > _LONG_LINE:
            stripped = self._tail.lstrip(_WHITESPACE)
            if (
                self._pending
                or len(stripped) < len(b"MARKER_TAG")
                or stripped.startswith(b"MARKER_TAG")
            ):
                self._overflow = True
            else:
                self._tail, self._skipping = b"", True
        if self._overflow:
            self._kept, self._tail = [], b""

    def result(self) -> tuple[str, ...] | None:
        """Return the sorted marker names, or None when the exact whole-file route is needed."""
        if self._overflow:
            return None
        if not self._skipping:
            self._line(self._tail)
            self._tail = b""
        if self._overflow:
            return None
        text = b"\n".join(self._kept).decode("ascii", errors="replace")
        return tuple(sorted(_su2_markers(text)))


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
        found = files.scan(mesh, "su2-markers", _Su2Markers)
        if found is None:  # a line the reduction cannot bound: the exact, whole-file route
            found = files.derive(
                mesh, "su2-markers-exact", lambda _path: marker_names(files.read_bytes(mesh), mesh)
            )
        return cast(tuple[str, ...], found)
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
