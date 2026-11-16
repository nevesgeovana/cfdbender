"""Read and write the component names of an IGES file: its directory and parameter sections only.

An IGES file names a component with a 406 form 15 NAME property on a level: every entity on that
level belongs to the named component, and the text after a ``|`` in the name is an attribute of
the writing system that is not part of it. This module reads the global section (delimiters and
length unit), the directory entries and the parameter records it needs, never the geometry, and
lists the faces in directory order: every trimmed surface (144) and every untrimmed surface at the
top level, which is the order in which an OpenCASCADE import returns them. It also writes those
names back into an IGES that holds the same faces in the same order, by setting each face's level
and appending one NAME per component.
"""

from __future__ import annotations

import codecs
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

import cfdbender._textio as _textio
from cfdbender._errors import InputArtifactError, InputValueError

__all__ = [
    "IgesNames",
    "flag_curves_on_surfaces",
    "name_faces",
    "named_curves",
    "parameter_space_curves",
    "read_iges_names",
    "resolve_components",
    "unit_of_flag",
]

#: The kind of artifact a refusal of this module names.
_IGES_KIND = "geometry"

#: The trimmed surface entity, a face whatever its subordinate switch.
TRIMMED = 144
#: Surface entities that are a face of their own when they stand at the top level: plane,
#: parametric spline, ruled, revolution, tabulated cylinder, rational B-spline, offset,
#: bounded, and the analytic surfaces of the 190 family.
UNTRIMMED = frozenset({108, 114, 118, 120, 122, 128, 140, 143, 190, 192, 194, 196, 198})
#: The name property: entity 406, form 15.
NAME_ENTITY, NAME_FORM = 406, 15

#: The length unit of each unit flag of the global section, in the solver's words.
_UNIT_FLAGS = {
    1: "INCH",
    2: "MILLIMETER",
    4: "FEET",
    5: "MILE",
    6: "METER",
    7: "KILOMETER",
    8: "MILS",
    9: "MICRON",
    10: "CENTIMETER",
    11: "MICROINCH",
}
#: The unit name the global section writes for each unit flag, for a file written back.
_UNIT_WORDS = {
    "INCH": (1, "INCH"),
    "MILLIMETER": (2, "MM"),
    "FEET": (4, "FT"),
    "MILE": (5, "MI"),
    "METER": (6, "M"),
    "KILOMETER": (7, "KM"),
    "MILS": (8, "MIL"),
    "MICRON": (9, "UM"),
    "CENTIMETER": (10, "CM"),
    "MICROINCH": (11, "UIN"),
}


#: Unit flag 3: the unit is the one the unit name field names.
_FLAG_BY_NAME = 3
#: The solver's word for each unit name the global section may write, with flag 3 or any other.
_UNIT_NAMES = {
    **{word: unit for unit, (_, word) in _UNIT_WORDS.items()},
    "IN": "INCH",
}


def unit_of_flag(flag: int) -> str | None:
    """Return the solver's word for an IGES unit flag, or None for a flag it has no word for."""
    return _UNIT_FLAGS.get(flag)


def unit_words(unit: str) -> tuple[int, str]:
    """Return the IGES unit flag and unit name of one of the solver's length words."""
    return _UNIT_WORDS[unit]


@dataclass(frozen=True)
class _Entry:
    """One directory entry: its sequence number and the fields this module reads."""

    number: int
    kind: int
    pstart: int
    level: int
    status: str
    pcount: int
    form: int

    @property
    def subordinate(self) -> str:
        return self.status[2:4] if len(self.status) >= 4 else "00"


@dataclass(frozen=True)
class IgesNames:
    """The faces of one IGES file in directory order, with their component and level.

    Attributes
    ----------
    path : Path
        The file read.
    unit : str or None
        The length unit of the global section, in the solver's words (``MILLIMETER``), or None
        when its flag names no unit the solver knows.
    faces : tuple of str
        The component of each face, in directory order.
    levels : tuple of int
        The level of each face, in the same order.
    entities : tuple of int
        The directory sequence number of each face.
    named : dict
        The component of each level that a NAME names, in file order.
    """

    path: Path
    unit: str | None
    faces: tuple[str, ...]
    levels: tuple[int, ...]
    entities: tuple[int, ...]
    named: dict[int, str]

    @property
    def components(self) -> dict[str, int]:
        """Return the faces of each component, in the order its first face appears."""
        counts: dict[str, int] = {}
        for name in self.faces:
            counts[name] = counts.get(name, 0) + 1
        return counts


def _refuse(path: Path, reason: str) -> InputArtifactError:
    return InputArtifactError(
        f"{path}: {reason}. Fix the file and run again. Nothing was written.", kind=_IGES_KIND
    )


def _split(text: str, delimiter: str, end: str) -> list[str]:
    """Split a free-format record on its delimiter, reading Hollerith strings whole."""
    out: list[str] = []
    i, current = 0, ""
    while i < len(text):
        ch = text[i]
        if ch.isdigit() and not current.strip():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            if j < len(text) and text[j] == "H":
                count = int(text[i:j])
                out.append(text[j + 1 : j + 1 + count])
                i = j + 1 + count
                while i < len(text) and text[i] not in (delimiter, end):
                    i += 1
                if i < len(text) and text[i] == end:
                    return out
                i += 1
                current = ""
                continue
        if ch == end:
            out.append(current.strip())
            return out
        if ch == delimiter:
            out.append(current.strip())
            current = ""
        else:
            current += ch
        i += 1
    if current.strip():
        out.append(current.strip())
    return out


#: The encoding an IGES file is read and written in: one character per byte, so a column is a
#: byte and a Hollerith count is kept, whatever bytes a name outside ASCII holds.
_BYTES = "latin-1"
#: The encoding a NAME payload is decoded with unless the caller states another.
DEFAULT_NAME_ENCODING = "latin-1"


def require_name_encoding(name_encoding: str) -> None:
    """Refuse, by name, a text encoding Python does not know, before any file is read."""
    try:
        codecs.lookup(name_encoding)
    except LookupError as error:
        raise InputValueError(
            f"name_encoding: {name_encoding!r} is not a known text encoding. Nothing was written."
        ) from error


def _decode_name(path: Path, raw: str, encoding: str) -> str:
    """Decode a NAME payload read as one character per byte, strictly, with ``encoding``."""
    try:
        return raw.encode(_BYTES).decode(encoding)
    except UnicodeDecodeError as error:
        raise _refuse(
            path,
            f"a NAME is not valid {encoding} text ({error.reason}); give the encoding the file's "
            "names are written in, or write them again",
        ) from error
    except LookupError as error:
        raise _refuse(path, f"{encoding!r} is not a known text encoding") from error


def _encode_name(name: str, encoding: str) -> str:
    """Return ``name`` as the one-character-per-byte text of its bytes in ``encoding``."""
    try:
        return name.encode(encoding).decode(_BYTES)
    except UnicodeEncodeError as error:
        raise InputArtifactError(
            f"the name {name!r} cannot be written in {encoding} ({error.reason}); choose an "
            "encoding that holds it. Nothing was written.",
            kind=_IGES_KIND,
        ) from error
    except LookupError as error:
        raise InputArtifactError(
            f"{encoding!r} is not a known text encoding. Nothing was written.", kind=_IGES_KIND
        ) from error


def _sections(path: Path) -> dict[str, list[str]]:
    try:
        text = path.read_text(encoding=_BYTES)
    except OSError as error:
        raise _refuse(path, f"not readable ({error})") from error
    sections: dict[str, list[str]] = {"S": [], "G": [], "D": [], "P": [], "T": []}
    for line in text.split("\n"):
        body = line.rstrip("\r\n")
        if len(body) < 73:
            continue
        section = body[72]
        if section in sections:
            sections[section].append(body[:80].ljust(80))
    if not sections["D"]:
        raise _refuse(path, "no directory section; an IGES file has fixed 80-column lines")
    return sections


def _delimiters(glob: str) -> tuple[str, str]:
    delimiter, end = ",", ";"
    if glob.startswith("1H"):
        delimiter = glob[2]
        rest = glob[4:]
        if rest.startswith("1H"):
            end = rest[2]
    elif glob.startswith(delimiter + "1H"):
        # ",1H;": the default parameter delimiter, then the record delimiter at column 4.
        end = glob[3]
    return delimiter, end


def _global_unit(path: Path, glob: str) -> str | None:
    delimiter, end = _delimiters(glob)
    fields = _split(glob, delimiter, end)
    try:
        flag = int(float(fields[13])) if len(fields) > 13 and fields[13] else 1
    except ValueError as error:
        raise _refuse(path, f"the unit flag {fields[13]!r} is not a number") from error
    if flag == _FLAG_BY_NAME:
        name = fields[14].strip().upper() if len(fields) > 14 else ""
        return _UNIT_NAMES.get(name)
    return unit_of_flag(flag)


def _int(path: Path, number: int, word: str) -> int:
    try:
        return int(word) if word.strip() else 0
    except ValueError as error:
        raise _refuse(path, f"directory entry {number} holds {word!r} where a number is") from error


def _entries(path: Path, lines: list[str]) -> list[_Entry]:
    if len(lines) % 2:
        raise _refuse(path, "the directory section holds an odd number of lines")
    out = []
    for i in range(0, len(lines), 2):
        a, b = lines[i], lines[i + 1]
        number = i + 1
        word = lambda line, k: line[8 * k : 8 * k + 8]  # noqa: E731
        out.append(
            _Entry(
                number=number,
                kind=_int(path, number, word(a, 0)),
                pstart=_int(path, number, word(a, 1)),
                level=_int(path, number, word(a, 4)),
                status=word(a, 8).strip().zfill(8),
                pcount=_int(path, number, word(b, 3)),
                form=_int(path, number, word(b, 4)),
            )
        )
    return out


def _parameters(entry: _Entry, plines: list[str], delimiter: str, end: str) -> list[str]:
    record = "".join(
        line[:64] for line in plines[entry.pstart - 1 : entry.pstart - 1 + entry.pcount]
    )
    return _split(record.strip(), delimiter, end)


def _level_list(
    entries: dict[int, _Entry], plines: list[str], level: int, sep: tuple[str, str]
) -> int:
    """Return the first level a negative level field points at (406 form 1), else 0."""
    entry = entries.get(-level)
    if entry is None or entry.kind != NAME_ENTITY:
        return 0
    words = _parameters(entry, plines, *sep)
    try:
        return int(words[2]) if len(words) > 2 else 0
    except ValueError:
        return 0


def read_iges_names(path: str | Path, *, name_encoding: str = DEFAULT_NAME_ENCODING) -> IgesNames:
    """Read the faces of an IGES file in directory order and the component of each.

    The component of a face is the NAME (406 form 15) on the face's level, cut at the first
    ``|`` and stripped; a face whose level carries no NAME belongs to ``<stem>_level<N>``.
    A NAME on level 0 names no level: it is a property another entity points at.

    Parameters
    ----------
    path : str or Path
        The IGES file, read only.
    name_encoding : str, optional
        The text encoding of the NAME bytes. The file is always read byte for byte, so columns
        and Hollerith counts hold whatever the encoding; only the name is decoded, strictly.
        Default ``"latin-1"``, which maps every byte to one character.

    Returns
    -------
    IgesNames
        The faces, their components and levels, and the file's length unit.

    Raises
    ------
    InputArtifactError
        The file is not an IGES file this reader can read; nothing was written.
    """
    parsed = _Parsed(Path(path), name_encoding)
    faces, levels, numbers = [], [], []
    for entry in parsed.entries:
        top = entry.subordinate == "00"
        if entry.kind == TRIMMED or (entry.kind in UNTRIMMED and top):
            level = parsed.level(entry)
            faces.append(parsed.named.get(level, f"{parsed.path.stem}_level{level}"))
            levels.append(level)
            numbers.append(entry.number)
    if not faces:
        raise _refuse(
            parsed.path, "it holds no surface (no 144, and no untrimmed surface at the top level)"
        )
    return IgesNames(
        parsed.path, parsed.unit, tuple(faces), tuple(levels), tuple(numbers), parsed.named
    )


class _Parsed:
    """The sections of one IGES file this module reads, with the NAME of each level."""

    def __init__(self, path: Path, name_encoding: str = DEFAULT_NAME_ENCODING) -> None:
        self.path = path
        sections = _sections(path)
        glob = "".join(line[:72] for line in sections["G"])
        self.sep = _delimiters(glob)
        self.unit = _global_unit(path, glob)
        self.plines = sections["P"]
        self.entries = _entries(path, sections["D"])
        self.by_number = {e.number: e for e in self.entries}
        self.named: dict[int, str] = {}
        for entry in self.entries:
            if entry.kind != NAME_ENTITY or entry.form != NAME_FORM:
                continue
            # Level 0 is no level: a NAME there is a property another entity points at (a
            # feature name of the writing system), never the name of a level. Only the NAME a
            # level takes is decoded, so a name the file never uses cannot refuse it.
            if entry.level <= 0 or entry.level in self.named:
                continue
            words = self.parameters(entry)
            text = words[2] if len(words) > 2 else ""
            name = _decode_name(path, text, name_encoding).split("|", 1)[0].strip()
            if name:
                self.named[entry.level] = name

    def parameters(self, entry: _Entry) -> list[str]:
        return _parameters(entry, self.plines, *self.sep)

    def level(self, entry: _Entry) -> int:
        if entry.level < 0:
            return _level_list(self.by_number, self.plines, entry.level, self.sep)
        return entry.level

    def numbers(self, entry: _Entry) -> list[float]:
        return [
            float(w.replace("D", "E").replace("d", "e")) if w else 0.0
            for w in self.parameters(entry)[1:]
        ]


#: Curve entities a named trailing edge may be: line, rational B-spline curve, composite curve.
_LINE, _BSPLINE_CURVE, _COMPOSITE = 110, 126, 102
#: Points per B-spline curve when it is sampled into a polyline.
_SAMPLES = 400


def _polyline(parsed: _Parsed, entry: _Entry, depth: int = 0) -> list[NDArray[np.float64]]:
    """Sample one curve entity into polylines, in the file's unit (its own transformation aside)."""
    values = parsed.numbers(entry)
    if entry.kind == _LINE:
        return [np.array(values[:6], dtype=float).reshape(2, 3)]
    if entry.kind == _BSPLINE_CURVE:
        k, m = int(values[0]), int(values[1])
        knots = np.array(values[6 : 6 + k + m + 2], dtype=float)
        at = 6 + k + m + 2
        weights = np.array(values[at : at + k + 1], dtype=float)
        at += k + 1
        points = np.array(values[at : at + 3 * (k + 1)], dtype=float).reshape(-1, 3)
        at += 3 * (k + 1)
        start, end = values[at], values[at + 1]
        homogeneous = np.c_[points * weights[:, None], weights]
        spline = BSpline(knots, homogeneous, m)
        sampled = spline(np.linspace(start, end, _SAMPLES))
        return [sampled[:, :3] / sampled[:, 3:]]
    if entry.kind == _COMPOSITE and depth < 8:
        count = int(values[0])
        out: list[NDArray[np.float64]] = []
        for pointer in values[1 : 1 + count]:
            member = parsed.by_number.get(int(pointer))
            if member is not None:
                out += _polyline(parsed, member, depth + 1)
        return out
    return []


def named_curves(
    path: str | Path, *, name_encoding: str = DEFAULT_NAME_ENCODING
) -> dict[str, list[NDArray[np.float64]]]:
    """Return the curves at the top level of an IGES file whose level a NAME names, as polylines.

    Lines (110), rational B-spline curves (126) and composite curves (102) of them are sampled in
    the file's unit; any other curve entity, and a curve on a level no NAME names, is not read.
    ``name_encoding`` is as for :func:`read_iges_names`.
    """
    parsed = _Parsed(Path(path), name_encoding)
    out: dict[str, list[NDArray[np.float64]]] = {}
    for entry in parsed.entries:
        if entry.subordinate != "00" or entry.kind not in (_LINE, _BSPLINE_CURVE, _COMPOSITE):
            continue
        name = parsed.named.get(parsed.level(entry))
        if name is not None:
            out.setdefault(name, []).extend(_polyline(parsed, entry))
    return out


def _line(text: str, section: str, number: int) -> str:
    return f"{text:<72}{section}{number:>7}"


def _hollerith(text: str) -> str:
    return f"{len(text)}H{text}"


def _with_unit(glob: str, unit: str) -> str:
    """Return the global record with its unit flag and unit name set to ``unit``."""
    delimiter, end = _delimiters(glob)
    fields = _split(glob, delimiter, end)
    flag, word = unit_words(unit)
    while len(fields) < 15:
        fields.append("")
    fields[13], fields[14] = str(flag), word
    # The string fields of the global section, which the split returned without their count.
    strings = {2, 3, 4, 5, 11, 14, 17, 20, 21, 24, 25}
    out = [_hollerith(v) if i in strings and v else v for i, v in enumerate(fields)]
    out[0], out[1] = _hollerith(delimiter), _hollerith(end)
    return delimiter.join(out) + end


#: The curve on a parametric surface (142) and the composite curve (102) its boundary may be.
_CURVE_ON_SURFACE = 142
#: The entity use flag, columns 69 and 70 of a directory entry's first line: 00 geometry, 05
#: two-dimensional geometry in a surface's parameter space.
_USE = slice(68, 70)
_GEOMETRY, _PARAMETER_SPACE = "00", "05"


def _set_use(line: str, use: str) -> str:
    return line[: _USE.start] + use + line[_USE.stop :]


def _with_pref(lines: list[str], delimiter: str, end: str) -> list[str]:
    """Return a 142's parameter lines with PREF set to 1, each line keeping its 64 data columns."""
    data = "".join(line[:64] for line in lines)
    record, terminator, rest = data.partition(end)
    fields = record.split(delimiter)
    if len(fields) < 6 or fields[5].strip() == "1":
        return lines
    width = len(fields[5])
    fields[5] = "1".rjust(width) if width > 1 else "1"
    text = delimiter.join(fields) + terminator + rest
    if len(text) > 64 * len(lines):
        return lines
    text = text.ljust(64 * len(lines))
    return [text[64 * i : 64 * i + 64] + line[64:] for i, line in enumerate(lines)]


def parameter_space_curves(
    bptrs: Iterable[int], members: Callable[[int], Sequence[int] | None]
) -> set[int]:
    """Return the entity numbers the solver's CAD import reads as parameter-space curves (use 05).

    The one rule both the IGES writer and ``iges-convert`` apply: each curve a 142's BPTR names
    (``bptrs``, 0 for none), and every member a composite (102) there reaches, at any depth.
    ``members(number)`` returns ``None`` for a number no entity holds, an empty list for a curve
    that is not a composite, and the member numbers of a composite.
    """
    curves: set[int] = set()
    stack = [number for number in bptrs if number]
    while stack:
        number = stack.pop()
        if number in curves:
            continue
        reached = members(number)
        if reached is None:
            continue
        curves.add(number)
        stack += [value for value in reached if value]
    return curves


def flag_curves_on_surfaces(source: str | Path, target: str | Path) -> Path:
    """Write ``source`` to ``target`` with the flags the solver's CAD import reads on each 142.

    Each curve on a parametric surface (142) that has a parameter-space curve (its BPTR) is
    flagged as geometry (use 00) with PREF 1, the parameter-space representation preferred,
    and its BPTR curve, with every member a composite (102) reaches, is flagged as
    parameter-space geometry (use 05). The OpenCASCADE writer flags these the other way round
    (the 142 05, its BPTR curves 00, PREF 3), and the solver refuses every such patch as
    having non-parametric curves. Every line keeps its 80 columns and its sequence number;
    nothing else in the file changes.
    """
    src = Path(source)
    parsed = _Parsed(src)
    sections = _sections(src)
    directory = list(sections["D"])
    parameters = list(sections["P"])
    delimiter, end = parsed.sep
    bptrs: list[int] = []
    for entry in parsed.entries:
        if entry.kind != _CURVE_ON_SURFACE:
            continue
        words = parsed.parameters(entry)
        bptr = int(words[3]) if len(words) > 3 and words[3].strip() else 0
        directory[entry.number - 1] = _set_use(directory[entry.number - 1], _GEOMETRY)
        if bptr == 0:
            continue
        first = entry.pstart - 1
        span = slice(first, first + entry.pcount)
        parameters[span] = _with_pref(parameters[span], delimiter, end)
        bptrs.append(bptr)

    def members(number: int) -> list[int] | None:
        member = parsed.by_number.get(number)
        if member is None:
            return None
        if member.kind != _COMPOSITE:
            return []
        values = parsed.parameters(member)
        count = int(values[1]) if len(values) > 1 and values[1].strip() else 0
        return [int(v) for v in values[2 : 2 + count] if v.strip()]

    curves = parameter_space_curves(bptrs, members)
    for number in curves:
        directory[number - 1] = _set_use(directory[number - 1], _PARAMETER_SPACE)
    out = Path(target)
    lines = [
        *(line[:80] for line in sections["S"]),
        *(line[:80] for line in sections["G"]),
        *directory,
        *parameters,
        *(line[:80] for line in sections["T"]),
    ]
    _textio.write_lines(out, lines, encoding=_BYTES)
    return out


def name_faces(
    source: str | Path,
    target: str | Path,
    components: Sequence[str],
    *,
    unit: str | None = None,
    name_encoding: str = DEFAULT_NAME_ENCODING,
) -> Path:
    """Write ``source`` to ``target`` with the i-th face on its component's level and its NAME.

    Each component, in the order of its first face, takes level 1, 2, ... and one 406 form 15
    NAME on that level; every face (as :func:`read_iges_names` lists them) moves to its
    component's level, and nothing else in the file changes but the terminate counts and,
    when ``unit`` is given, the global unit flag and name. Each name is written as the bytes it
    has in ``name_encoding`` and its Hollerith count is that byte count.

    Raises
    ------
    InputArtifactError
        The file holds another number of faces than ``components`` names; nothing was written.
    """
    src = Path(source)
    listed = read_iges_names(src, name_encoding=name_encoding)
    wire = {name: _encode_name(name, name_encoding) for name in dict.fromkeys(components)}
    if len(listed.faces) != len(components):
        raise _refuse(
            src,
            f"it holds {len(listed.faces)} faces and {len(components)} component names were "
            "given, one per face",
        )
    sections = _sections(src)
    glob = "".join(line[:72] for line in sections["G"])
    if unit is not None:
        glob = _with_unit(glob, unit)
    order: dict[str, int] = {}
    for name in components:
        order.setdefault(name, len(order) + 1)
    level_of = dict(zip(listed.entities, (order[name] for name in components), strict=True))
    directory = list(sections["D"])
    for number, level in level_of.items():
        line = directory[number - 1]
        directory[number - 1] = line[:32] + f"{level:>8}" + line[40:]
    parameters = list(sections["P"])
    for name, level in order.items():
        number = len(directory) + 1
        record = f"{NAME_ENTITY},1,{_hollerith(wire[name])};"
        first = len(parameters) + 1
        for i in range(0, len(record), 64):
            parameters.append(
                _line(f"{record[i : i + 64]:<64}{number:>8}", "P", len(parameters) + 1)
            )
        count = len(parameters) - first + 1
        one = [str(NAME_ENTITY), str(first), "0", "0", str(level), "0", "0", "0", "00000000"]
        two = [str(NAME_ENTITY), "0", "0", str(count), str(NAME_FORM), "", "", "NAME", "0"]
        directory.append(_line("".join(f"{f:>8}" for f in one), "D", number))
        directory.append(_line("".join(f"{f:>8}" for f in two), "D", number + 1))
    start = [line[:80] for line in sections["S"]] or [_line("cfdbender", "S", 1)]
    glines = [_line(glob[i : i + 72], "G", n + 1) for n, i in enumerate(range(0, len(glob), 72))]
    directory = [_line(line[:72], "D", n + 1) for n, line in enumerate(directory)]
    parameters = [_line(line[:72], "P", n + 1) for n, line in enumerate(parameters)]
    counts = f"S{len(start):>7}G{len(glines):>7}D{len(directory):>7}P{len(parameters):>7}"
    out = Path(target)
    lines = start + glines + directory + parameters + [_line(counts, "T", 1)]
    _textio.write_lines(out, lines, encoding=_BYTES)
    return out


def resolve_components(found: Sequence[str], requested: Sequence[str]) -> list[str]:
    """Return the components ``requested`` names, in the order of ``found``, or refuse.

    Raises
    ------
    InputValueError
        ``requested`` is empty, a name matches no component, or a name matches more than one
        component that differs from it only in case or surrounding blanks.
    """
    if not requested:
        raise InputValueError(
            "component: at least one component name required. Nothing was written."
        )
    listing = ", ".join(repr(name) for name in found)
    chosen: set[str] = set()
    for name in requested:
        if name in found:
            chosen.add(name)
            continue
        loose = [f for f in found if f.strip().casefold() == name.strip().casefold()]
        if not loose:
            raise InputValueError(
                f"component {name!r}: no component of the IGES files has this name; found: "
                f"{listing}. Nothing was written."
            )
        if len(loose) > 1:
            raise InputValueError(
                f"component {name!r}: ambiguous, it matches "
                f"{', '.join(repr(m) for m in loose)}; give the exact name. Nothing was written."
            )
        chosen.add(loose[0])
    return [name for name in found if name in chosen]
