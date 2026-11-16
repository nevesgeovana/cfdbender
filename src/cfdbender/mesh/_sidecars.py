"""The files beside a mesh: its boundary inventory and its declarations sidecar.

An OBJ carries a boundary inventory, ``<name>.obj.boundaries.toml``, that lists its
boundary names in the order of the file, the face count taken when the inventory was
written, and the named ``[trailing_edges.<label>]`` and ``[base_regions.<family>]``
declarations that the mesh verbs read. A legacy ``<stem>.boundaries.toml`` stays readable.
This module writes and checks the inventory (the names read off an OBJ's groups, with the
face counts taken at the same time) and reads each table of the sidecar over one parse.

Only OBJ meshes are read and written here.
"""

from __future__ import annotations

import datetime
import json
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

import cfdbender._textio as _textio
from cfdbender._digest import file_sha256
from cfdbender._errors import GeometryWarning, InputArtifactError, InputValueError, warn
from cfdbender.mesh._sidecar_models import (
    EVERY_SURFACE,
    BoundaryBaseRegion,
    BoundaryDeclarations,
    BoundaryTrailingEdge,
    MeshImport,
    TrailingEdgeMarking,
)

__all__ = [
    "BASE_REGIONS_TABLE",
    "BOUNDARY_FACES_KEY",
    "DETECT_EVERYWHERE",
    "IMPORT_TABLE",
    "INVENTORY_SUFFIX",
    "MESH_FACES_KEY",
    "MESH_SHA256_KEY",
    "OBJ_SUFFIX",
    "TRAILING_EDGES_TABLE",
    "ensure_inventory",
    "face_count_lines",
    "inventory_sidecar",
    "load_toml",
    "obj_boundary_names",
    "obj_face_counts",
    "read_boundary_declarations",
    "read_inventory",
    "read_mesh_import",
    "read_trailing_edges",
    "toml_key",
    "toml_table",
    "toml_value",
    "write_inventory",
]


def _utf8_text(path: Path, what: str) -> str:
    """Decode a TOML file as UTF-8, a leading byte-order mark accepted, naming it otherwise."""
    try:
        return path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise InputArtifactError(
            f"{what} {path} is not UTF-8 text ({error}). Save it as UTF-8."
        ) from error


def load_toml(path: Path, kind: str) -> dict[str, Any]:
    """Read one TOML artifact file, naming the file on a syntax or encoding error."""
    text = _utf8_text(path, f"the {kind} artifact")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise InputArtifactError(
            f"the {kind} artifact {path} is not valid TOML: {error}. Artifacts are "
            "declarative TOML files, one artifact per file."
        ) from error


#: Suffix appended to the mesh file name for its inventory.
INVENTORY_SUFFIX = ".boundaries.toml"


def inventory_sidecar(geometry: str | Path, kind: Literal["obj"] | None = None) -> Path:
    """Return the sidecar path beside ``geometry``, whether or not it exists.

    Parameters
    ----------
    geometry : str or Path
        The mesh file.
    kind : {"obj"} or None
        Include the mesh kind for new inventories (``<stem>.obj.boundaries.toml``). With
        None, find an existing kind-qualified inventory, otherwise return the legacy name
        (``<stem>.boundaries.toml``).

    Returns
    -------
    Path
        The inventory path beside the mesh.
    """
    path = Path(geometry)
    if kind is not None:
        return path.with_name(f"{path.stem}.{kind}{INVENTORY_SUFFIX}")
    if path.suffix.lower() == ".obj":
        qualified = path.with_name(path.name + INVENTORY_SUFFIX)
        if qualified.exists():
            return qualified
    return path.with_name(path.stem + INVENTORY_SUFFIX)


def write_inventory(geometry: str | Path, *, overwrite: bool = False) -> Path:
    r"""Write the boundary inventory beside an OBJ, read from its groups.

    An OBJ's existing sidecar is never rewritten, ``overwrite`` or not: beside a raw mesh
    it also carries the ``[import]`` table and the trailing edges, which a rewrite of the
    list would lose.

    Parameters
    ----------
    geometry : str or Path
        An OBJ file.
    overwrite : bool
        Accepted for the command line's sake; an OBJ's sidecar is refused when it exists
        either way.

    Returns
    -------
    Path
        The sidecar written.

    Raises
    ------
    InputArtifactError
        A file that is not an OBJ or cannot be read; an OBJ's sidecar that already exists;
        an OBJ whose groups cannot be read (:func:`obj_boundary_names`).

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from cfdbender.mesh._sidecars import write_inventory
    >>> with TemporaryDirectory() as folder:
    ...     mesh = Path(folder) / "wing.obj"
    ...     _ = mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\ng Wing\nf 1 2 3\n")
    ...     print(write_inventory(mesh).name)
    wing.obj.boundaries.toml
    """
    del overwrite
    path = Path(geometry)
    sidecar = inventory_sidecar(path, "obj")
    if not path.is_file():
        raise InputArtifactError(
            f"{path} is not a file, so no boundary inventory can be read from it."
        )
    if path.suffix.lower() != OBJ_SUFFIX:
        raise InputArtifactError(
            f"{path.name} is not an OBJ; the boundary inventory is written for an OBJ mesh, "
            "whose boundary names are its groups."
        )
    if sidecar.exists():
        raise InputArtifactError(
            f"{sidecar} already exists, and the sidecar of an OBJ is never rewritten: "
            "beside a raw mesh it also carries the [import] table and the trailing edges, "
            "which a rewrite of the list would lose. Correct the list by hand, or move the "
            "file aside, run this again and copy its tables beneath the new list."
        )
    return ensure_inventory(path, kind="obj")


# --- an OBJ's surface names, read from its groups ---------------------------------------

#: The raw-mesh suffix whose surface names are read from the file itself. An
#: OBJ names its groups, and one boundary is made of each group that holds a face,
#: named by the group, in the order of the file.
OBJ_SUFFIX = ".obj"

#: The two statements that open a group of an OBJ.
_OBJ_GROUP_STATEMENTS = ("o", "g")


def obj_boundary_names(path: str | Path) -> tuple[str, ...]:
    """Return measured OBJ boundary names in native order, preserving duplicates.

    A boundary is a face-bearing group. An o-to-g transition, repeated g groups, a face
    prefix before the first g, and multiple names in a g statement are read; the last uses
    only its first name, with a warning.

    Repeated g groups remain separate boundaries; ambiguous labels must be
    selected by position downstream. Reverse g-to-o transitions, repeated
    o names, unnamed groups and multiword o names are refused.

    Parameters
    ----------
    path : str or Path
        The OBJ mesh file, read as UTF-8 text.

    Returns
    -------
    tuple of str
        The group names in the order the file opens them, repeated names kept.

    Raises
    ------
    InputArtifactError
        If the file cannot be read as text, holds no face under a measured
        group, or uses a variant that has not been measured (a group with no
        name or several words after ``o``, a face before the first ``o``, an
        ``o`` after ``g``, or a name reopened where an ``o`` is involved). A
        face before the first ``g`` is measured and read as ``Boundary-1``. The
        message points to writing the names by hand in the sidecar.
    """
    mesh = Path(path)
    by_hand = (
        "The surface names of such a file are not read from it: write them by hand in "
        f"{inventory_sidecar(mesh).name} beside it, as boundaries = [...] in the order of the "
        "groups."
    )
    names: list[str] = []
    opened: dict[str, tuple[str, int]] = {}
    previous: tuple[str, int] | None = None
    current: str | None = None
    holds_face = False
    prefix_face_line: int | None = None
    try:
        with mesh.open(encoding="utf-8") as lines:
            for number, line in enumerate(lines, start=1):
                if line.startswith("v"):
                    continue
                words = line.split("#", 1)[0].split()
                if not words:
                    continue
                if words[0] == "f":
                    if current is None and prefix_face_line is None:
                        prefix_face_line = number
                    holds_face = True
                    continue
                if words[0] not in _OBJ_GROUP_STATEMENTS:
                    continue
                statement = words[0]
                if len(words) < 2 or (statement == "o" and len(words) != 2):
                    stated = "no group" if len(words) == 1 else "a group of several words"
                    raise InputArtifactError(
                        f"{mesh}: line {number}, {line.strip()!r}, names {stated}, and how "
                        f"such a group is named is not defined. {by_hand}"
                    )
                if prefix_face_line is not None and previous is None and statement != "g":
                    raise InputArtifactError(
                        f"{mesh}: line {prefix_face_line} writes a face before the first "
                        f"{statement!r} group; that variant is not measured. {by_hand}"
                    )
                if previous is not None and previous[0] == "g" and statement == "o":
                    raise InputArtifactError(
                        f"{mesh}: line {number} opens a group with `o` after `g` at "
                        f"line {previous[1]}; this transition is not measured. {by_hand}"
                    )
                name = words[1]
                if name in opened and (statement != "g" or opened[name][0] != "g"):
                    raise InputArtifactError(
                        f"{mesh}: line {number} opens the group {name!r} again, first opened "
                        f"at line {opened[name][1]}: a group in two places involving an "
                        f"`o` statement is not measured. {by_hand}"
                    )
                if len(words) > 2:
                    warn(
                        f"{mesh}: line {number} names multiple OBJ groups; the measured "
                        f"native importer uses only the first, {name!r}.",
                        GeometryWarning,
                        stacklevel=2,
                    )
                opened.setdefault(name, (statement, number))
                if holds_face:
                    names.append(current if current is not None else "Boundary-1")
                current, holds_face = name, False
                previous = (statement, number)
    except (OSError, UnicodeDecodeError) as error:
        raise InputArtifactError(
            f"{mesh} cannot be read as the text of an OBJ: {error}. {by_hand}"
        ) from error
    if current is not None and holds_face:
        names.append(current)
    if not names:
        raise InputArtifactError(
            f"{mesh} holds no face under any measured `o` or `g` group, so no "
            f"boundary inventory is established; check the mesh. {by_hand}"
        )
    return tuple(names)


def ensure_inventory(geometry: str | Path, *, kind: Literal["obj"] | None = None) -> Path:
    """Return the boundary sidecar beside a geometry, writing an OBJ's from its groups.

    THE ONE PLACE A SIDECAR IS ASKED FOR. The matrix binding, which
    ``pyfs-matrix plan`` and ``pyfs-matrix run`` both pass through, and
    ``pyfs-matrix inventory`` (:func:`write_inventory`) reach the sidecar
    through this function, so an OBJ is treated alike wherever a user
    meets it:

    * an OBJ with no sidecar gets one, its ``boundaries`` read by
      :func:`obj_boundary_names` under a comment header naming the file's
      sha256, and a line on stderr says so. It carries the
      list and nothing else: the ``[import]`` units and the trailing
      edges, which only the user can state, go beneath it.
    * an OBJ with a sidecar keeps it, never rewritten, because the file
      may carry tables and names the user wrote. When its ``boundaries``
      differ from the groups, in names or in order, the run cites the
      sidecar's, as before, and a warning names both lists; when the
      groups cannot be read, the sidecar written by hand is that file's
      route and nothing is said.
    * any other geometry is left as it was: a saved simulation's sidecar
      is written by ``pyfs-matrix inventory`` from its mesh block, and an
      STL's by hand.

    Parameters
    ----------
    geometry : str or Path
        The geometry the sidecar sits beside.
    kind : {"obj"} or None
        Filename kind for inventory writers; None keeps the legacy name.

    Returns
    -------
    Path
        The sidecar beside ``geometry``, which exists for every OBJ this
        returns for; for another geometry it exists only if written.

    Raises
    ------
    InputArtifactError
        An OBJ with no sidecar whose groups cannot be read
        (:func:`obj_boundary_names`), naming the line and the sidecar to
        write by hand; an OBJ whose sidecar states no ``boundaries`` list,
        naming the list its groups make.

    Warns
    -----
    GeometryWarning
        An OBJ whose sidecar's ``boundaries`` differ from its groups,
        naming both lists.
    """
    path = Path(geometry)
    sidecar = inventory_sidecar(path, kind)
    if path.suffix.lower() != OBJ_SUFFIX or not path.is_file():
        return sidecar
    if sidecar.exists():
        _compare_with_the_groups(path, sidecar)
        return sidecar
    sealed = _sealed_digest(path)
    names = obj_boundary_names(path)
    if sealed is not None and _sealed_digest(path) != sealed:
        raise InputArtifactError(
            f"{path} changed while its groups were read, so no sidecar is written for it. "
            "Run this again once the file is no longer being replaced."
        )
    body = [
        f"# Written by cfdbender from the groups of {path.name}, which had no sidecar;",
        f"# the OBJ's sha256 was {sealed or file_sha256(path)}.",
        "# One boundary per `o` or `g` group holding a face, named by the group, in the",
        "# order of the file: the order a solver numbers an OBJ's surfaces on import",
        "# The package never rewrites this file.",
        "# Add the [import] units.",
        "# Add named [trailing_edges.<label>] entries, each",
        "# with type and file or detect; optionally add [base_regions.<family>] entries.",
        "# For no trailing edges, declare [trailing_edges] none = true.",
        "boundaries = [",
        *[f"    {_toml_string(name)}," for name in names],
        "]",
        *face_count_lines(path, len(names), sealed=sealed),
    ]
    _textio.write_text(sidecar, "\n".join(body) + "\n")
    print(
        f"wrote {sidecar}: the boundaries of {path.name}, read from its groups in the "
        f"order of the file: {', '.join(names)}. Add [import] units, then named "
        "[trailing_edges.<label>] entries with type and "
        "file or detect, and optional [base_regions.<family>] entries. For no trailing "
        "edges, declare [trailing_edges] none = true.",
        file=sys.stderr,
        flush=True,
    )
    return sidecar


def _compare_with_the_groups(mesh: Path, sidecar: Path) -> None:
    """Warn when an OBJ's own sidecar lists other boundaries than its groups make."""
    stated = _sidecar_data(sidecar).get("boundaries")
    try:
        groups = obj_boundary_names(mesh)
    except InputArtifactError:
        # The file is one whose names are not read from it, so the list
        # written by hand is its route, and there is nothing to compare.
        return
    listed = ", ".join(groups)
    if not isinstance(stated, list) or not stated or not all(isinstance(n, str) for n in stated):
        written = ", ".join(_toml_string(name) for name in groups)
        raise InputArtifactError(
            f"{sidecar} does not state `boundaries` as a non-empty list of strings. The "
            f"groups of {mesh.name} that hold a face, in the order of the "
            f"file, are {listed}: write boundaries = [{written}] at "
            "the top of the file. It is never rewritten by the package, because it may "
            "carry tables you wrote."
        )
    if tuple(stated) != groups:
        warn(
            f"{sidecar.name} lists the boundaries of {mesh.name} as {', '.join(stated)}, and "
            "its groups that hold a face, in the order of the file, "
            f"are {listed}. The run cites the sidecar's list as written, so a "
            "surface it names at a position where the file holds another is cited at the "
            "wrong one. Correct the list, or move the sidecar aside, plan again to have it "
            "written from the groups, and copy its tables beneath the new list.",
            GeometryWarning,
            stacklevel=3,
        )


def _toml_string(name: str) -> str:
    """Quote one name as a TOML basic string; JSON's escapes are TOML's."""
    return json.dumps(name, ensure_ascii=False)


# Face counts taken with the mesh inventory.
# Count while reading the mesh and write beside boundaries:
#     mesh_faces = 1056
#     boundary_faces = [1032, 24]
#     mesh_sha256 = "<sha256 of the counted file>"
# boundary_faces follows boundary order and is written only when available.
# Post-processing carries recorded_mesh_faces into the superfile and
# unsteady polar; it does not recount. A missing field gives NA. A digest
# mismatch warns with both digests but still carries the count. An
# unavailable count is omitted and never prevents writing the inventory.

#: The key of the inventory that states the geometry's face count.
MESH_FACES_KEY = "mesh_faces"
#: The key that states the count of each boundary, in the order of ``boundaries``.
BOUNDARY_FACES_KEY = "boundary_faces"
#: The key that states the sha256 of the file the faces were counted in.
MESH_SHA256_KEY = "mesh_sha256"


def obj_face_counts(path: str | Path) -> tuple[int, ...]:
    """Return the faces of each face-bearing group of an OBJ, in the order of the file.

    The groups are cut where :func:`obj_boundary_names` cuts them: every
    ``o`` or ``g`` statement closes the group before it, which counts when
    it holds a face, and faces before the first group make a group of their
    own. It is called only on a file whose names that reader has read, so
    the counts stand beside those names one for one.

    Parameters
    ----------
    path : str or Path
        The OBJ mesh file, read as UTF-8 text.

    Returns
    -------
    tuple of int
        The face count of each group that holds a face.

    Raises
    ------
    OSError, UnicodeDecodeError
        If the file cannot be read as text.
    """
    counts: list[int] = []
    held = 0
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            if line.startswith("v"):
                continue
            words = line.split("#", 1)[0].split()
            if not words:
                continue
            if words[0] == "f":
                held += 1
            elif words[0] in _OBJ_GROUP_STATEMENTS and held:
                counts.append(held)
                held = 0
    if held:
        counts.append(held)
    return tuple(counts)


def _sealed_digest(path: Path) -> str | None:
    """Return the file's sha256 taken before its names are read, or None if unreadable."""
    try:
        return file_sha256(path)
    except OSError:
        return None


def face_count_lines(
    geometry: Path,
    boundaries: int,
    *,
    digest: Callable[[Path], str] = file_sha256,
    sealed: str | None = None,
) -> list[str]:
    """Return the inventory lines stating a geometry's face count, or none.

    The file is hashed before and after it is counted, and the lines are
    written only when the two digests agree, so the count and the digest
    are of the same bytes. ``boundary_faces`` is written only where a
    count is read for each of the ``boundaries`` the inventory lists.

    Parameters
    ----------
    geometry : Path
        The saved simulation or OBJ the inventory is taken from.
    boundaries : int
        The number of boundaries the inventory lists.
    obj : bool
        Whether the geometry is an OBJ, counted by its groups.
    digest : callable, optional
        The sha256 of a file, read before and after the count (the file's own
        :func:`~cfdbender._digest.file_sha256` unless another is given).
    sealed : str, optional
        The sha256 taken before the inventory's boundary names were read. The
        lines are written only when it equals the digests taken around the count,
        so the names, the count and the digest are of the same bytes.

    Returns
    -------
    list of str
        The TOML lines, or an empty list where the faces cannot be counted.
    """
    try:
        before = digest(geometry)
        counted = _obj_counted(geometry)
        after = digest(geometry)
    except (OSError, UnicodeDecodeError):
        return []
    if counted is None or before != after or (sealed is not None and sealed != before):
        return []
    total, per_boundary = counted
    lines = [
        "# The faces of the mesh, counted when this inventory was taken, and the sha256 of",
        "# the file they were counted in; the post carries mesh_faces as MESH_FACES for a",
        "# run whose recorded sha256 of this geometry is that one.",
        f"{MESH_FACES_KEY} = {total}",
    ]
    if per_boundary is not None and len(per_boundary) == boundaries:
        lines.append(f"{BOUNDARY_FACES_KEY} = [{', '.join(str(n) for n in per_boundary)}]")
    lines.append(f'{MESH_SHA256_KEY} = "{before}"')
    return lines


def _obj_counted(path: Path) -> tuple[int, tuple[int, ...]]:
    """Return an OBJ's face count and its count per group."""
    counts = obj_face_counts(path)
    return sum(counts), counts


IMPORT_TABLE = "import"


def _sidecar_data(sidecar: Path) -> dict[str, Any]:
    """Parse one geometry sidecar, refusing a file that does not read as TOML, by name.

    ONE PARSE, ONE READER PER TABLE. The sidecar holds the boundary order
    (``boundaries``) and, beside a raw mesh, the ``[import]`` table; each
    is read by its own function over this parse, and no reader refuses a
    table it does not read. A table added to the file later is therefore
    one more reader beside these, and the readers already here are left as
    they are.
    """
    try:
        text = _utf8_text(sidecar, "the geometry sidecar")
        return tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise InputArtifactError(
            f"{sidecar} cannot be read as a geometry sidecar: {error}"
        ) from error


def read_inventory(sidecar: str | Path) -> tuple[str, ...]:
    """Return the ordered boundary names a sidecar states.

    Parameters
    ----------
    sidecar : str or Path
        The ``<stem>.boundaries.toml`` file.

    Returns
    -------
    tuple of str
        The boundary names in the order the sidecar lists them.

    Raises
    ------
    InputArtifactError
        A sidecar without a ``boundaries`` list of strings, naming it.
    """
    path = Path(sidecar)
    names = _sidecar_data(path).get("boundaries")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise InputArtifactError(
            f"{path} does not state `boundaries` as a non-empty list of strings; "
            "write the surface names of the mesh in the order of its groups."
        )
    return tuple(names)


def read_mesh_import(sidecar: str | Path) -> MeshImport | None:
    """Return the ``[import]`` table a geometry's sidecar states, or None.

    The table states the length unit a raw mesh is written in, and the
    mesh operations applied right after the import, in the order written::

        boundaries = ["naca"]

        [import]
        units = "MILLIMETER"

        [[import.operations]]
        op = "rename"
        surface = "naca"
        to = "Wing"

    Read at binding, beside :func:`read_inventory`, so a table that does
    not hold its shape is refused with the row before any seat is spent,
    an operation named by its position. Whether the unit is one ``IMPORT``
    takes, whether the geometry is a raw mesh at all, and whether each
    cited surface exists at its step, is the builder's to judge, per build.

    Parameters
    ----------
    sidecar : str or Path
        The ``<stem>.boundaries.toml`` beside the geometry.

    Returns
    -------
    MeshImport or None
        None when the file holds no ``[import]`` table.

    Raises
    ------
    InputArtifactError
        A file that does not read as TOML, an ``import`` key that is not a
        table, a table that states no ``units`` or a key it does not read,
        or an operation that does not hold its shape (its own keys, finite
        numbers, scale factors above zero, a named surface for a rename or
        a mirror), each naming the sidecar and the operation's position.
    """
    path = Path(sidecar)
    table = _sidecar_data(path).get(IMPORT_TABLE)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise InputArtifactError(
            f"{path} states `{IMPORT_TABLE}` as a {type(table).__name__}; write it as the "
            f'table [{IMPORT_TABLE}], with units = "MILLIMETER" (the unit the mesh file is '
            "written in) beneath it."
        )
    try:
        return MeshImport.model_validate(table)
    except ValidationError as error:
        missing = any(
            item["type"] == "missing" and tuple(item["loc"]) == ("units",)
            for item in error.errors()
        )
        if missing:
            raise InputArtifactError(
                f"{path}: the [{IMPORT_TABLE}] table does not state `units`, the length "
                "unit the mesh file is written in; write it beneath the table, as "
                'units = "MILLIMETER". A unit is never assumed.'
            ) from error
        problems = "; ".join(
            f"{_where_in_the_import_table(item['loc'])}: "
            f"{str(item['msg']).removeprefix('Value error, ')}"
            for item in error.errors()
        )
        raise InputArtifactError(
            f"{path}: the [{IMPORT_TABLE}] table is refused: {problems}. It holds `units`, "
            "the length unit the mesh file is written in, and the mesh operations of the "
            f"import as [[{IMPORT_TABLE}.operations]], numbered in the order written "
            "."
        ) from error


#: The tables of a geometry's sidecar that declare a raw mesh's boundary
#: conditions: the trailing edges and the base regions.
TRAILING_EDGES_TABLE = "trailing_edges"
BASE_REGIONS_TABLE = "base_regions"

#: The word a ``detect`` key takes for detection over every surface.
DETECT_EVERYWHERE = "auto"

#: The keys ``[trailing_edges]`` reads: ``file`` with ``type`` and
#: ``tolerance``, the default route, or ``detect`` alone.
_TRAILING_EDGE_KEYS = ("files", "none", "file", "type", "tolerance", "detect")

#: The keys a ``detect = { ... }`` table of ``[trailing_edges]`` reads.
_DETECT_KEYS = ("surfaces", "sweep_angle")

_TRAILING_EDGE_ROUTES = (
    'file = "<points file>", the default route: the mid-point of every trailing-edge '
    'mesh edge, under a line naming their length unit; or detect = "auto" or '
    'detect = { surfaces = ["<name>"], sweep_angle = <degrees> }, detection, which '
    "applies only when written"
)


def read_boundary_declarations(sidecar: str | Path) -> BoundaryDeclarations:
    """Load named marking declarations, refusing obsolete forms with their translation.

    Parameters
    ----------
    sidecar : str or Path
        The OBJ boundary sidecar, including the accepted legacy filename.

    Returns
    -------
    BoundaryDeclarations
        Ordered entries with points paths resolved beside the sidecar.

    Raises
    ------
    InputArtifactError
        Invalid declarations, unknown families or obsolete unnamed tables.
    """
    path = Path(sidecar)
    data = _sidecar_data(path)
    trailing, bases, none = _named_boundary_tables(path, data)
    try:
        entries = (
            ()
            if none
            else tuple(
                _named_trailing_entry(path, label, table) for label, table in trailing.items()
            )
        )
        families = read_inventory(path)
        regions = tuple(
            _named_base_entry(path, family, table, families) for family, table in bases.items()
        )
        merged = {region.family for region in regions if region.merge_into is not None}
        if any(region.merge_into in merged for region in regions):
            raise InputValueError(
                "a merge_into parent may not itself be merged;"
                " point merge_into at a family that is not merged"
            )
        return BoundaryDeclarations(
            trailing_edges=entries, base_regions=regions, no_trailing_edges=none
        )
    except (ValueError, TypeError) as error:
        raise InputArtifactError(f"{path}: invalid boundary declarations: {error}") from error


def _named_boundary_tables(path, data):
    """Separate named tables and refuse obsolete declarations before parsing entries."""
    trailing = data.get("trailing_edges", {})
    bases = data.get("base_regions", {})
    if not isinstance(trailing, dict) or not isinstance(bases, dict):
        raise InputArtifactError(f"{path}: trailing_edges and base_regions must be named tables")
    if any(key in trailing for key in ("file", "files", "detect")):
        files = trailing.get("files", [{"file": trailing.get("file", "wing.te.txt")}])
        labels = (
            [
                Path(str(item.get("file", "wing.te.txt"))).stem
                for item in files
                if isinstance(item, dict)
            ]
            if isinstance(files, list)
            else []
        )
        translated = ", ".join(f"[trailing_edges.{_toml_string(label)}]" for label in labels)
        raise InputArtifactError(
            f"{path}: replace unnamed [trailing_edges] with "
            f"{translated or '[trailing_edges.<label>]'}; "
            "each file becomes its own entry with file and type; detection uses type = STANDARD."
        )
    if "detect" in bases:
        raise InputArtifactError(
            f"{path}: replace [base_regions] detect with one [base_regions.<family>] "
            'per base family, with trailing_edges = "auto".'
        )
    none = "none" in trailing
    if none and (trailing != {"none": True} or trailing["none"] is not True):
        raise InputArtifactError(f"{path}: none = true cannot coexist with named trailing edges")
    return trailing, bases, none


def _named_trailing_entry(path, label, table):
    """Read one named trailing-edge entry and resolve its termination file."""
    if not label.strip() or not isinstance(table, dict):
        raise InputValueError(
            f"trailing_edges.{label} must be a named table; write [trailing_edges.<label>]"
        )
    values = dict(table)
    if "type" not in values:
        raise InputValueError(f"trailing_edges.{label} requires type; add type = ...")
    if "detect" in values:
        if values.pop("type") != "STANDARD":
            raise InputValueError(
                f'trailing_edges.{label}: detect accepts only type STANDARD; set type = "STANDARD"'
            )
    marking = read_trailing_edges(path, values)
    if marking.route == "none":
        raise InputValueError(
            "none = true belongs to the unnamed trailing_edges table; move it there"
        )
    return BoundaryTrailingEdge(label=label, marking=marking)


def _named_base_entry(path, family, table, families):
    """Validate one base family, its rim and its merge destination."""
    if family not in families:
        raise InputValueError(
            f"base_regions.{family}: family must exist in boundaries {families}; name one of them"
        )
    if not isinstance(table, dict):
        raise InputValueError(
            f"base_regions.{family} must be a table; write [base_regions.{family}]"
        )
    values = dict(table)
    rim = values.pop("trailing_edges", None)
    if rim != "auto":
        if not isinstance(rim, dict) or set(rim) != {"file", "type"}:
            raise InputValueError(
                f"base_regions.{family}: trailing_edges needs auto, or file and type"
            )
        rim = read_trailing_edges(path, dict(rim))
    region = BoundaryBaseRegion(family=family, trailing_edges=rim, **values)
    if region.merge_into is not None and (
        region.merge_into not in families or region.merge_into == family
    ):
        raise InputValueError(
            f"base_regions.{family}: merge_into must name another existing family"
        )
    return region


def _trailing_edge_file_fields(
    path: Path, where: str, table: Mapping[str, Any], tolerance: float
) -> dict[str, Any]:
    """Read the ``file`` route of ``[trailing_edges]``: a points file beside the sidecar."""
    written = table["file"]
    if not isinstance(written, str) or not written.strip():
        raise InputArtifactError(
            f"{where} states file = {written!r}; write the name of the points file, "
            'beside the sidecar, as file = "wing.te.txt".'
        )
    named = Path(written.strip())
    if named.is_absolute() or named.root or named.drive or ".." in named.parts:
        raise InputArtifactError(
            f"{where} states file = {written!r}, which is not beside the sidecar; write the "
            'name of a points file in the sidecar\'s folder, as file = "wing.te.txt".'
        )
    if "type" not in table:
        raise InputArtifactError(
            f"{where} states file = {written!r} and no type, and the edge type has no "
            "default: VORTEX_SHEDDING and STANDARD mark different edges. Add "
            'type = "VORTEX_SHEDDING" or type = "STANDARD" beside file.'
        )
    return {
        "route": "file",
        "edge_type": table["type"],
        "tolerance": tolerance,
        "points_file": str(path.parent / named),
    }


def read_trailing_edges(path: Path, table: Mapping[str, Any]) -> TrailingEdgeMarking:
    """Read ``[trailing_edges]``: one route, file or detect, and the keys of that route.

    A ``file`` route states its ``type``: ``VORTEX_SHEDDING`` and ``STANDARD``
    mark different edges, so the type has no default and a file route without one is refused.
    """
    where = f"{path}: the [{TRAILING_EDGES_TABLE}] table"
    if "files" in table:
        return _trailing_edge_sets(path, table, where)
    if "none" in table:
        if table != {"none": True} or table["none"] is not True:
            raise InputArtifactError(f"{where}: none = true must be the only trailing-edge choice")
        return TrailingEdgeMarking(route="none")
    foreign = sorted(set(table) - set(_TRAILING_EDGE_KEYS))
    if foreign:
        raise InputArtifactError(
            f"{where} does not read {', '.join(foreign)}; it reads {_TRAILING_EDGE_ROUTES}, "
            "with type and tolerance beside file."
        )
    if ("file" in table) == ("detect" in table):
        stated = "both file and detect" if "file" in table else "neither file nor detect"
        raise InputArtifactError(
            f"{where} states {stated}, and a trailing edge is marked by one of them: "
            f"{_TRAILING_EDGE_ROUTES}."
        )
    tolerance = table.get("tolerance", 0.0001)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise InputArtifactError(
            f"{where} states tolerance = {tolerance!r}; write the distance, in "
            "metres, within which an edge's mid-point counts as a point "
            "of the file, as a number such as 0.0001."
        )
    tolerance = _as_float(where, "tolerance", tolerance)
    fields: dict[str, Any] = {}
    if "file" in table:
        fields = _trailing_edge_file_fields(path, where, table, tolerance)
    else:
        beside = sorted(key for key in ("type", "tolerance") if key in table)
        if beside:
            raise InputArtifactError(
                f"{where} states {' and '.join(beside)} beside detect, and detection reads "
                "neither: it gives every edge it marks the STANDARD type and matches no "
                "points. Delete them, or mark the edges by a points file (file = ...)."
            )
        fields = {"route": "detect", **_read_trailing_edge_detection(where, table["detect"])}
    try:
        return TrailingEdgeMarking.model_validate(fields)
    except ValidationError as error:
        problems = "; ".join(
            str(item["msg"]).removeprefix("Value error, ") + f" (given {item.get('input')!r})"
            for item in error.errors()
        )
        raise InputArtifactError(f"{where} is refused: {problems}.") from error


def _trailing_edge_sets(path: Path, table: Mapping[str, Any], where: str) -> TrailingEdgeMarking:
    """Read ordered, independently typed files with one shared matching tolerance."""
    foreign = sorted(set(table) - {"files", "tolerance"})
    if foreign:
        raise InputArtifactError(f"{where}: files cannot be combined with {', '.join(foreign)}")
    entries = table["files"]
    if not isinstance(entries, list) or not entries:
        raise InputArtifactError(f"{where}: files must be a nonempty list of file/type tables")
    sets: list = []
    seen: set[str] = set()
    for number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or set(entry) != {"file", "type"}:
            found = sorted(entry) if isinstance(entry, dict) else type(entry).__name__
            raise InputArtifactError(
                f"{where}: files entry {number} has {found}; write exactly "
                '{file = "<name>.te.txt", type = "<type>"}'
            )
        try:
            marking = read_trailing_edges(
                path, {**entry, "tolerance": table.get("tolerance", 0.0001)}
            )
        except InputArtifactError as error:
            raise InputArtifactError(
                f"{where}: files entry {number} ({entry['file']!r}): {error}"
            ) from error
        if marking.points_file is None:
            raise InputArtifactError(f"{where}: files entry {number} names no points file")
        key = str(Path(marking.points_file).resolve()).casefold()
        if key in seen:
            raise InputArtifactError(f"{where}: points file {entry['file']!r} is listed twice")
        seen.add(key)
        sets.extend(marking.sets)
    return TrailingEdgeMarking(route="file", tolerance=marking.tolerance, sets=tuple(sets))


def _read_trailing_edge_detection(where: str, detect: object) -> dict[str, Any]:
    """Read ``detect``: ``"auto"``, or a table of surfaces and an optional sweep angle."""
    if detect == DETECT_EVERYWHERE:
        return {}
    if not isinstance(detect, dict):
        raise InputArtifactError(
            f'{where} states detect = {detect!r}; write detect = "{DETECT_EVERYWHERE}" for '
            'detection over every surface, or detect = { surfaces = ["<name>"], '
            "sweep_angle = <degrees> } for detection on the surfaces named."
        )
    foreign = sorted(set(detect) - set(_DETECT_KEYS))
    if foreign:
        raise InputArtifactError(
            f"{where}: detect does not read {', '.join(foreign)}; it reads surfaces and, "
            "optionally, sweep_angle."
        )
    surfaces = detect.get("surfaces")
    fields: dict[str, Any] = {"detect_surfaces": _surface_names(where, surfaces, every=True)}
    if "sweep_angle" in detect:
        angle = detect["sweep_angle"]
        if isinstance(angle, bool) or not isinstance(angle, (int, float)):
            raise InputArtifactError(
                f"{where}: detect states sweep_angle = {angle!r}; write it in degrees, as a number."
            )
        fields["sweep_angle_deg"] = _as_float(where, "detect sweep_angle", angle)
    return fields


def _as_float(where: str, key: str, value: int | float) -> float:
    """Return a sidecar number as a float, refusing an integer too large for one by name."""
    try:
        return float(value)
    except OverflowError:
        raise InputArtifactError(
            f"{where} states {key} as a number too large to use; write a finite "
            "number of ordinary size."
        ) from None


def _surface_names(where: str, surfaces: object, *, every: bool) -> tuple[str, ...]:
    """Read a ``surfaces`` list of sidecar names; ``"all"`` is every surface where ``every``."""
    if every and surfaces == EVERY_SURFACE:
        return ()
    if (
        not isinstance(surfaces, list)
        or not surfaces
        or not all(isinstance(name, str) and name.strip() for name in surfaces)
    ):
        written = "is not stated" if surfaces is None else f"= {surfaces!r}"
        also = f', or "{EVERY_SURFACE}" for every surface' if every else ""
        raise InputArtifactError(
            f"{where}: detect's surfaces {written}; write the surfaces to detect on as a "
            f'non-empty list of the sidecar\'s names, surfaces = ["<name>"]{also}.'
        )
    return tuple(name.strip() for name in surfaces)


def _where_in_the_import_table(loc: Sequence[int | str]) -> str:
    """Name a place of the ``[import]`` table as its reader counts it: ``operation 2 (factors)``."""
    if len(loc) >= 2 and loc[0] == "operations" and isinstance(loc[1], int):
        rest = ".".join(str(part) for part in loc[2:])
        return f"operation {loc[1] + 1}" + (f" ({rest})" if rest else "")
    return ".".join(str(part) for part in loc) or "the table"


def toml_key(name: str) -> str:
    """Return one TOML key: bare where TOML allows it, a basic string otherwise."""
    bare = name and all(c.isascii() and (c.isalnum() or c in "_-") for c in name)
    return name if bare else toml_value(name)


def toml_value(value: Any) -> str:
    """Return one TOML value: a boolean, a number, a date or time, a string, a table or an array."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime.date, datetime.time)):
        # What the reader returns for a TOML date, time or date-time; ISO 8601 is TOML's form.
        return value.isoformat()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # A basic string that reads back as exactly ``value``: JSON's escapes are TOML's,
        # and DEL, which TOML also refuses raw, is escaped too.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, Mapping):
        pairs = (f"{toml_key(key)} = {toml_value(item)}" for key, item in value.items())
        return "{ " + ", ".join(pairs) + " }"
    return "[" + ", ".join(toml_value(item) for item in value) + "]"


def toml_table(name: str, table: Mapping[str, Any]) -> list[str]:
    """Return the lines of one TOML table, a list of tables written as ``[[name.key]]``."""
    rows = [f"[{name}]"]
    nested: list[str] = []
    for key, value in table.items():
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            for item in value:
                nested += [
                    "",
                    f"[[{name}.{key}]]",
                    *(f"{k} = {toml_value(v)}" for k, v in item.items()),
                ]
        else:
            rows.append(f"{toml_value(key)} = {toml_value(value)}")
    return rows + nested
