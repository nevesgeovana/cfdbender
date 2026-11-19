"""Native CFD++ grid folders: validation, family names and staging by link.

A grid folder holds three binary files and, usually, a family-name file. Integers
are 4 bytes and doubles 8 bytes, big-endian, with no record markers. The layout
read here is the documented one; the details that the documentation leaves open
(the node record width, the unit of the cell-group file) are named constants
below as hypotheses not yet verified against a solver-generated grid. The module
is pure and byte-level: it runs no solver tool.

Boundary identity in the solver is the selector number; the family-name file only
gives each selector a name. A grid without that file is keyed by selector number.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cfdbender.core.plan import InputFiles
from cfdbender.solvers._host._mesh import link_mesh

GRID_FILES = ("nodesin.bin", "cellsin.bin", "exbcsin.bin")
OPTIONAL_GRID_FILES = ("cgrpsin.bin.1", "dfcelin.bin", "inbcsin.bin")
BC_FILE = "mcfd.bc"
PARTITION_PREFIX = "mcpusin.bin."
CELL_GROUP_PREFIX = "cgrpsin.bin."

# Layout hypotheses, each to be confirmed by a returned grid.
BYTE_ORDER = ">"
HEADER_INTS = 3  # version, count, array or record length
NODE_TYPES_2D = (2, 3)  # x-y nodes, without and with velocities
NODE_TYPES_3D = (0, 1)  # x-y-z nodes, without and with velocities
MIN_NODE_RECORD = 4 + 2 * 8  # smallest node record: type plus two doubles
MIN_CELL_RECORD = 4 + 2 * 4  # smallest cell record: type plus two vertices
MIN_FACE_RECORD = 4 * 4  # unused, selector, vertex count, one vertex
CELL_GROUP_HAS_HEADER = False  # the cell-group file is one integer per cell, nothing else

# Family-name rules: a letter first, and none of these characters.
_FORBIDDEN_NAME = set(" <>\"'&@#$%^!*()\t")
_SU2_SUFFIX = ".su2"
_SU2_SENTENCE = (
    "{name} is an SU2 mesh, not a native grid folder; convert it once with the solver's own "
    "converter (the convert25 tool) and name the resulting native grid folder, which holds "
    "nodesin.bin, cellsin.bin, exbcsin.bin and mcfd.bc, as the geometry"
)


class GridError(ValueError):
    """A grid folder, a grid file or a family-name file that cannot be used."""


@dataclass(frozen=True)
class GridFacts:
    """What the three grid files state: dimension, counts and the selectors in use."""

    dimension: int
    node_count: int
    cell_count: int
    face_count: int
    selectors: tuple[int, ...]


@dataclass(frozen=True)
class Grid:
    """A validated grid folder; ``facts`` is None when the large-input check kept it unread."""

    folder: Path
    facts: GridFacts | None
    names: dict[int, str]
    files: tuple[Path, ...]
    partitions: dict[int, Path]
    unread: tuple[str, ...] = ()


def read_bc(text: str) -> dict[int, str]:
    """Return selector to family name from the text of a family-name file.

    Layout (documented): a comment line, a header line, then for each family a
    name line (``#`` immediately followed by the name) and a line of four
    integers whose first is the running selector number. Blank lines are ignored.
    """
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#"):
        raise GridError(f"{BC_FILE}: the first line must be a comment starting with '#'")
    if len(lines) < 2 or not lines[1].strip() or lines[1].startswith("#"):
        raise GridError(f"{BC_FILE}: the header line after the comment is missing")
    body = [line.strip() for line in lines[2:] if line.strip()]
    if len(body) % 2:
        raise GridError(
            f"{BC_FILE}: every family needs a name line and an integer line "
            "(an unpaired line ends the file)"
        )
    names: dict[int, str] = {}
    for position in range(0, len(body), 2):
        name_line, number_line = body[position], body[position + 1]
        if not name_line.startswith("#"):
            raise GridError(
                f"{BC_FILE}: a family name line must start with '#', found {name_line!r}"
            )
        name = name_line[1:].strip()
        _check_name(name)
        fields = number_line.split()
        try:
            numbers = [int(field) for field in fields]
        except ValueError:
            numbers = []
        if len(numbers) != 4:
            raise GridError(
                f"{BC_FILE}: family {name!r} must be followed by a line of four integers, "
                f"found {number_line!r}"
            )
        selector = numbers[0]
        if selector in names:
            raise GridError(f"{BC_FILE}: selector {selector} is named twice")
        if name in names.values():
            raise GridError(f"{BC_FILE}: the family name {name!r} is used twice")
        names[selector] = name
    return names


def _check_name(name: str) -> None:
    if not name or not name[0].isalpha():
        raise GridError(f"{BC_FILE}: family name {name!r} must start with a letter")
    bad = sorted(set(name) & _FORBIDDEN_NAME)
    if bad:
        raise GridError(
            f"{BC_FILE}: family name {name!r} holds characters that are not allowed: "
            f"{''.join(bad)!r}"
        )


def selectors(names: Iterable[str], bc: Mapping[int, str]) -> tuple[int, ...]:
    """Return the selector of each family name, in the order given."""
    by_name = {name: selector for selector, name in bc.items()}
    found: list[int] = []
    for name in names:
        if name not in by_name:
            known = ", ".join(sorted(by_name)) or "none"
            raise GridError(f"family {name!r} is not in {BC_FILE}; the families are: {known}")
        found.append(by_name[name])
    return tuple(found)


def _unpack(data: bytes, offset: int, count: int, label: str) -> tuple[int, ...]:
    end = offset + 4 * count
    if end > len(data):
        raise GridError(f"{label} is truncated: {len(data)} bytes, at least {end} needed")
    return struct.unpack_from(f"{BYTE_ORDER}{count}i", data, offset)


def _header(head: bytes, size: int, label: str, record: int) -> tuple[int, ...]:
    """Read the three header integers, refusing a truncated or little-endian file.

    ``head`` holds at least the first 12 bytes of the file and ``size`` is its whole length.
    """
    if size < 4 * HEADER_INTS:
        raise GridError(
            f"{label} is truncated: the header needs {4 * HEADER_INTS} bytes, found {size}"
        )
    version, count, length = struct.unpack_from(f"{BYTE_ORDER}iii", head)
    room = size - 4 * HEADER_INTS
    if count < 0 or count * record > room:
        little = struct.unpack_from("<i", head, 4)[0]
        if 0 < little and little * record <= room:
            raise GridError(
                f"{label} is little-endian; the native format is big-endian "
                "(convert it with the solver's tools)"
            )
        raise GridError(
            f"{label} is truncated or inconsistent: the header states {count} records "
            f"that do not fit {size} bytes"
        )
    if count == 0:
        raise GridError(f"{label} holds no records")
    return version, count, length


#: Identity of each scan in the digest cache; change one when its result layout changes.
_NODES_TAG = "cfdpp-grid-nodes-1"
_CELLS_TAG = "cfdpp-grid-cells-1"
_FACES_TAG = "cfdpp-grid-faces-1"
_GROUP_TAG = "cfdpp-grid-group-1"
#: Bytes read per block when a grid file is streamed without a tracking reader.
_STREAM_BLOCK = 1 << 20
_NODE_HEAD = 4 * HEADER_INTS + 4  # the header and the type of the first node
_FACE_START = 2 * 4  # a face record starts here, as the layout hypothesis has it


class _HeadScan:
    """Keep the first bytes of a file and count all of them: a header needs no more."""

    def __init__(self, keep: int) -> None:
        self._keep = keep
        self._head = b""
        self._size = 0

    def feed(self, chunk: bytes) -> None:
        """Take the next block of the file."""
        self._size += len(chunk)
        if len(self._head) < self._keep:
            self._head += chunk[: self._keep - len(self._head)]

    def result(self) -> tuple[str, ...]:
        """Return the file size and its first bytes in hexadecimal."""
        return (str(self._size), self._head.hex())


class _FaceScan:
    """Read the selectors of the face file block by block, holding less than one record header.

    The face loop of the layout reads three integers at each record start and skips the
    vertices, so a face file is a chain of skips. The scan follows the chain over the blocks it
    is fed and never keeps more than the bytes of one record header. It cannot know the file size
    until the end, so it reads on optimistically and the caller checks the file header first,
    as the whole-file reading did; a failure is recorded, never raised, and reported by
    ``result`` for the caller to raise in the same order.
    """

    def __init__(self) -> None:
        self._head = b""
        self._size = 0
        self._pending = b""
        self._next = _FACE_START  # absolute offset of the next record start
        self._left: int | None = None
        self._selectors: set[int] = set()
        self._inconsistent = False

    def feed(self, chunk: bytes) -> None:
        """Take the next block of the file, in order."""
        self._size += len(chunk)
        if len(self._head) < 4 * HEADER_INTS:
            self._head += chunk[: 4 * HEADER_INTS - len(self._head)]
        if self._left is None and len(self._head) >= _FACE_START:
            self._left = struct.unpack_from(f"{BYTE_ORDER}i", self._head, 4)[0]
        if self._inconsistent or self._left is not None and self._left <= 0:
            return
        data = self._pending + chunk
        if self._left is None:
            self._pending = data
            return
        base = self._size - len(data)
        position = self._next - base
        while self._left > 0 and position + 12 <= len(data):
            _, selector, vertices = struct.unpack_from(f"{BYTE_ORDER}3i", data, position)
            if vertices < 1:
                self._inconsistent = True
                break
            self._selectors.add(selector)
            position += 12 + 4 * vertices
            self._left -= 1
        self._next = base + position
        self._pending = data[position:] if 0 <= position < len(data) else b""

    def result(self) -> tuple[str, ...]:
        """Return the size, the first bytes and how the face chain ended."""
        head = (str(self._size), self._head.hex())
        if self._left is None:
            return (*head, "none")
        if self._inconsistent:
            return (*head, "inconsistent")
        if self._left > 0 or self._size < self._next:  # a face or its vertices run past the end
            need = self._next if self._size < self._next else self._next + 12
            return (*head, "truncated", str(need))
        return (*head, "selectors", *(str(item) for item in sorted(self._selectors)))


class _GroupScan:
    """Say whether a cell-group file states one group or more, block by block."""

    def __init__(self) -> None:
        self._tail = b""
        self._first: bytes | None = None
        self._many = False

    def feed(self, chunk: bytes) -> None:
        """Take the next block of the file."""
        if self._many:
            return
        data = self._tail + chunk
        whole = len(data) - len(data) % 4
        block, self._tail = data[:whole], data[whole:]
        if not block:
            return
        if self._first is None:
            self._first = block[:4]
        if block != self._first * (len(block) // 4):
            self._many = True

    def result(self) -> tuple[str, ...]:
        """Return ``many`` when two integers differ, else ``one``."""
        return ("many",) if self._many else ("one",)


def _streamed(path: Path, scanner: _HeadScan | _FaceScan | _GroupScan) -> tuple[str, ...]:
    with path.open("rb") as stream:
        while chunk := stream.read(_STREAM_BLOCK):
            scanner.feed(chunk)
    return scanner.result()


def _scanned(
    folder: Path,
    name: str,
    tag: str,
    factory: Callable[[], _HeadScan | _FaceScan | _GroupScan],
    files: InputFiles | None,
    mesh: str | None,
) -> tuple[str, ...]:
    """Scan a grid file in the pass that digests it, or stream it when no reader is given."""
    if files is None or mesh is None:
        return _streamed(folder / name, factory())
    key = f"{mesh}/{name}"
    result = files.scan(key, tag, factory)
    if _valid_scan(result, tag, (folder / name).stat().st_size):
        return cast(tuple[str, ...], result)
    # The shared cache validates string lists, not the adapter's result schema.
    # Recompute malformed entries and bind the fallback to the tracked digest.
    path = folder / name
    before = path.stat()
    scanner = factory()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_STREAM_BLOCK):
            digest.update(chunk)
            scanner.feed(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_ino,
    ) or digest.hexdigest() != files.digest(key):
        raise ValueError(f"{key}: input changed while planning; plan again")
    return scanner.result()


def _valid_scan(value: object, tag: str, size_on_disk: int) -> bool:
    """Reject malformed persisted scan results before interpreting their fields."""
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        return False
    try:
        if tag == _GROUP_TAG:
            return value in (("one",), ("many",))
        if len(value) < 2:
            return False
        size = int(value[0])
        head = bytes.fromhex(value[1])
        keep = _NODE_HEAD if tag == _NODES_TAG else 4 * HEADER_INTS
        if size != size_on_disk or len(head) != min(size, keep):
            return False
        if tag != _FACES_TAG:
            return len(value) == 2
        if len(value) < 3:
            return False
        if value[2] in ("none", "inconsistent"):
            return len(value) == 3
        if value[2] == "truncated":
            return len(value) == 4 and int(value[3]) >= 0
        if value[2] == "selectors":
            return all(str(int(item)) == item for item in value[3:])
        return False
    except (ValueError, OverflowError):
        return False


def _facts(nodes: tuple[str, ...], cells: tuple[str, ...], faces: tuple[str, ...]) -> GridFacts:
    """Return the facts of a grid from the scans of its three files, refusing as the layout says."""
    node_size, node_head = int(nodes[0]), bytes.fromhex(nodes[1])
    _, node_count, _ = _header(node_head, node_size, "nodesin.bin", MIN_NODE_RECORD)
    first_type = _unpack(node_head, 4 * HEADER_INTS, 1, "nodesin.bin")[0]
    if first_type in NODE_TYPES_2D:
        dimension = 2
    elif first_type in NODE_TYPES_3D:
        dimension = 3
    else:
        raise GridError(
            f"nodesin.bin: node type {first_type} is not supported; "
            "only x-y (2D) and x-y-z (3D) nodes are"
        )
    _, cell_count, _ = _header(
        bytes.fromhex(cells[1]), int(cells[0]), "cellsin.bin", MIN_CELL_RECORD
    )
    face_size = int(faces[0])
    _, face_count, _ = _header(bytes.fromhex(faces[1]), face_size, "exbcsin.bin", MIN_FACE_RECORD)
    if faces[2] == "inconsistent":
        raise GridError("exbcsin.bin is inconsistent: a face states no vertices")
    if faces[2] == "truncated":
        raise GridError(f"exbcsin.bin is truncated: {face_size} bytes, at least {faces[3]} needed")
    found = tuple(sorted(int(item) for item in faces[3:]))
    return GridFacts(dimension, node_count, cell_count, face_count, found)


def grid_facts(nodes: bytes, cells: bytes, exbcs: bytes) -> GridFacts:
    """Read the headers of the three files and return the facts of the grid.

    The dimension comes from the type of the first node: 2 for the x-y types,
    3 for the x-y-z types. Any other node type is refused in the first release.
    """
    scans = []
    for data, scanner in (
        (nodes, _HeadScan(_NODE_HEAD)),
        (cells, _HeadScan(4 * HEADER_INTS)),
        (exbcs, _FaceScan()),
    ):
        scanner.feed(data)
        scans.append(scanner.result())
    return _facts(*scans)


def _check_cell_group(folder: Path, name: str, files: InputFiles | None, mesh: str | None) -> None:
    tracked = name in OPTIONAL_GRID_FILES
    found = _scanned(folder, name, _GROUP_TAG, _GroupScan, files if tracked else None, mesh)
    if found == ("many",):
        raise GridError(
            f"{name} states more than one cell group; the first release takes one cell group"
        )


_FILE_TAGS = (_NODES_TAG, _CELLS_TAG, _FACES_TAG)


def _unread(files: InputFiles, mesh: str) -> tuple[str, ...]:
    """Return the grid files whose facts are not known because the large-input check is off."""
    return tuple(
        name
        for name, tag in zip(GRID_FILES, _FILE_TAGS, strict=True)
        if files.unread(f"{mesh}/{name}", tag)
    )


def _unread_grid(folder: Path, files: InputFiles, mesh: str, skipped: tuple[str, ...]) -> Grid:
    """Track the grid files by identity, warn once that their facts were not read, open none."""
    used = [folder / name for name in GRID_FILES]
    used.extend(folder / name for name in OPTIONAL_GRID_FILES if (folder / name).is_file())
    for path in used:
        files.skip_content(f"{mesh}/{path.name}", "grid dimension, counts and selectors")
    return Grid(folder, None, {}, tuple(used), _partitions(folder), skipped)


def read_grid(
    folder: Path,
    *,
    require_names: bool = False,
    files: InputFiles | None = None,
    mesh: str | None = None,
) -> Grid:
    """Validate a grid folder and return its facts, names and files.

    With ``require_names`` the family-name file must exist and name every
    selector the grid uses; otherwise a missing file keys the grid by selector
    number.

    The grid files are never held whole. With a tracking reader ``files`` and the
    workspace-relative name ``mesh`` of the folder, each grid file is scanned in the pass that
    digests it, so a large file is read once when first seen and its facts are cached with its
    digest; every file the grid uses is tracked, as planning records it. Without a reader each
    file is streamed once.
    """
    if folder.suffix.lower() == _SU2_SUFFIX:
        raise GridError(_SU2_SENTENCE.format(name=folder.name))
    if not folder.is_dir():
        raise GridError(f"{folder} is not a folder; the geometry must be a native grid folder")
    missing = [name for name in GRID_FILES if not (folder / name).is_file()]
    if missing:
        raise GridError(f"grid folder {folder.name} lacks: {', '.join(missing)}")
    if files is not None and mesh is not None and (skipped := _unread(files, mesh)):
        return _unread_grid(folder, files, mesh, skipped)
    facts = _facts(
        _scanned(folder, GRID_FILES[0], _NODES_TAG, lambda: _HeadScan(_NODE_HEAD), files, mesh),
        _scanned(
            folder, GRID_FILES[1], _CELLS_TAG, lambda: _HeadScan(4 * HEADER_INTS), files, mesh
        ),
        _scanned(folder, GRID_FILES[2], _FACES_TAG, _FaceScan, files, mesh),
    )
    for path in sorted(folder.glob(CELL_GROUP_PREFIX + "*")):
        suffix = path.name[len(CELL_GROUP_PREFIX) :]
        if suffix.isdigit() and int(suffix) > 1:
            raise GridError(f"{path.name}: the first release takes one cell group")
        if files is not None and mesh is not None and path.name in OPTIONAL_GRID_FILES:
            if files.unread(f"{mesh}/{path.name}", _GROUP_TAG):
                files.skip_content(f"{mesh}/{path.name}", "cell group count")
                continue
        _check_cell_group(folder, path.name, files, mesh)
    bc_path = folder / BC_FILE
    used = [folder / name for name in GRID_FILES]
    used.extend(folder / name for name in OPTIONAL_GRID_FILES if (folder / name).is_file())
    if files is not None and mesh is not None:
        for path in used[len(GRID_FILES) :]:
            files.track(f"{mesh}/{path.name}")  # a large file is digested, never held
    if bc_path.is_file():
        bc = read_bc(_family_text(bc_path, files, mesh))
        used.append(bc_path)
    elif require_names:
        raise GridError(
            f"grid folder {folder.name} has no {BC_FILE}, and the boundary names are used"
        )
    else:
        bc = {}
    if require_names:
        for selector in facts.selectors:
            if selector not in bc:
                raise GridError(f"grid selector {selector} has no name in {BC_FILE}")
        names = {selector: bc[selector] for selector in facts.selectors}
    else:
        names = {selector: bc.get(selector, str(selector)) for selector in facts.selectors}
    return Grid(folder, facts, names, tuple(used), _partitions(folder))


def _family_text(path: Path, files: InputFiles | None, mesh: str | None) -> str:
    """Return the text of the family-name file, from the tracked snapshot when there is one."""
    if files is None or mesh is None:
        return path.read_text(encoding="ascii", errors="replace")
    return files.read_bytes(f"{mesh}/{path.name}").decode("ascii", errors="replace")


def _partitions(folder: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in sorted(folder.glob(PARTITION_PREFIX + "*")):
        suffix = path.name[len(PARTITION_PREFIX) :]
        if re.fullmatch(r"[1-9][0-9]*", suffix):
            found[int(suffix)] = path
    return found


def stage_grid(
    folder: Path,
    point: Path,
    digests: Mapping[str, str] | None = None,
    *,
    ncpus: int = 1,
    inputs: InputFiles | None = None,
    mesh: str | None = None,
) -> list[Path]:
    """Link the grid files into a point folder under their fixed names.

    The files are linked, never copied. ``digests`` maps a file name to the
    SHA-256 planned for it; a file whose digest differs is refused and its link
    removed. Without digests each file is checked against itself. The partition
    file for ``ncpus`` is linked when the grid folder holds it.

    With the command's reader ``inputs`` and the workspace-relative name ``mesh`` of the
    folder, a large file already read by this command, or remembered by the digest cache, is
    not read again, neither to scan it nor to check the link; any other file is streamed.
    """
    grid = read_grid(folder, files=inputs, mesh=mesh)
    sources = list(grid.files)
    if ncpus > 1 and ncpus in grid.partitions:
        sources.append(grid.partitions[ncpus])
    if digests is not None and any(source.name not in digests for source in sources):
        raise ValueError("mesh changed after planning; plan again")
    staged: list[Path] = []
    try:
        for source in sources:
            name = f"{mesh}/{source.name}" if mesh is not None else None
            if digests is not None:
                digest = digests[source.name]
            elif inputs is not None and name is not None:
                digest = inputs.digest(name)
            else:
                with source.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            link_mesh(
                source,
                point / source.name,
                digest,
                inputs=inputs if name is not None else None,
                name=name,
            )
            staged.append(point / source.name)
    except ValueError:
        for path in staged:
            path.unlink(missing_ok=True)
        raise
    return staged
