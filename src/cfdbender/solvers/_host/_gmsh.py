"""Read the facts a Gmsh mesh must give a planner without holding the file in memory.

A planner needs two things from a Gmsh mesh: whether it starts with the Gmsh format line, and the
physical names of its ``$PhysicalNames`` section. ``GmshScan`` collects both block by block in the
pass that digests the file, so a large mesh is read once and a warm digest cache reads it not at
all.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol, cast

_START = b"$PhysicalNames"
_END = b"$EndPhysicalNames"
#: The longest header this module distinguishes: ``$MeshFormat`` CR LF ``2.2 0 8`` CR LF.
_HEAD = len(b"$MeshFormat\r\n2.2 0 8\r\n")
#: A names section longer than this sends the file to the exact whole-file route.
_SECTION_LIMIT = 1 << 24
_PHYSICAL = re.compile(r'^\s*[12]\s+\d+\s+"([^"]+)"', re.MULTILINE)


def physical_names(section: str) -> tuple[str, ...]:
    """Return the sorted physical names declared in the text of a ``$PhysicalNames`` section."""
    return tuple(sorted(set(_PHYSICAL.findall(section))))


def section_text(text: str) -> str:
    """Return the text between the first ``$PhysicalNames`` and the next ``$EndPhysicalNames``."""
    return text.partition("$PhysicalNames")[2].partition("$EndPhysicalNames")[0]


class GmshScan:
    """Collect the header kind and the physical names of a Gmsh mesh, block by block.

    The names are those the whole-file route finds: the text after the first ``$PhysicalNames``
    up to the next ``$EndPhysicalNames``, searched with the same pattern. An unterminated
    section takes the exact route. Memory is one block, a tail shorter than a marker and the
    section itself.

    ``result`` returns a tuple of strings so that it can be cached with the digest: the header
    kind (``2.2`` for the Gmsh 2.2 ASCII line, ``gmsh`` for any other Gmsh format line, ``other``),
    then ``names`` followed by the sorted names, or ``overflow`` alone when the section is
    longer than the bound and the caller must use the exact whole-file route. Nothing is
    truncated.
    """

    def __init__(self) -> None:
        self._head = b""
        self._state = 0  # 0 awaiting the start marker, 1 inside the section, 2 section ended
        self._carry = b""
        self._parts: list[bytes] = []
        self._size = 0
        self._overflow = False

    @staticmethod
    def valid_result(value: object) -> bool:
        """Reject malformed persistent facts before using them as scanner results."""
        return (
            isinstance(value, tuple)
            and len(value) >= 2
            and all(isinstance(item, str) for item in value)
            and value[0] in {"2.2", "gmsh", "other"}
            and (value[1] == "names" or (value[1] == "overflow" and len(value) == 2))
        )

    def feed(self, chunk: bytes) -> None:
        """Take the next block of the mesh, in file order."""
        if len(self._head) < _HEAD:
            self._head += chunk[: _HEAD - len(self._head)]
        if self._state == 2 or self._overflow:
            return
        data = self._carry + chunk
        self._carry = b""
        if self._state == 0:
            at = data.find(_START)
            if at < 0:
                self._carry = data[max(0, len(data) - len(_START) + 1) :]
                return
            data = data[at + len(_START) :]
            self._state = 1
        at = data.find(_END)
        if at >= 0:
            self._add(data[:at])
            self._state = 2
            return
        keep = len(_END) - 1
        emit = max(0, len(data) - keep)
        self._add(data[:emit])
        self._carry = data[emit:]

    def _add(self, part: bytes) -> None:
        self._parts.append(part)
        self._size += len(part)
        if self._size > _SECTION_LIMIT:
            self._overflow = True
            self._parts = []

    def result(self) -> tuple[str, ...]:
        """Return the header kind and the names, or ``overflow`` in place of the names."""
        head = self._head
        if head.startswith((b"$MeshFormat\n2.2 0 8\n", b"$MeshFormat\r\n2.2 0 8\r\n")):
            kind = "2.2"
        elif head.startswith((b"$MeshFormat\n", b"$MeshFormat\r\n")):
            kind = "gmsh"
        else:
            kind = "other"
        if self._overflow or self._state == 1:
            return (kind, "overflow")
        text = b"".join(self._parts).decode("ascii", errors="replace")
        return (kind, "names", *physical_names(text))


class _Files(Protocol):
    def scan(self, name: str, tag: str, factory: Callable[[], GmshScan]) -> object: ...


def gmsh_header(files: _Files, mesh: str) -> str:
    """Return ``2.2``, ``gmsh`` or ``other`` for a tracked mesh, reading it at most once."""
    return cast(tuple[str, ...], files.scan(mesh, "gmsh-info", GmshScan))[0]


def gmsh_names(files: _Files, mesh: str) -> tuple[str, ...] | None:
    """Return a tracked mesh's physical names, or None when the exact route is needed."""
    found = cast(tuple[str, ...], files.scan(mesh, "gmsh-info", GmshScan))
    return None if found[1] == "overflow" else found[2:]
