"""The named IGES writer: kernel write, then a deterministic re-serialization.

The kernel's own IGES writer stamps the file with the time it ran and the login name of the
account that ran it, and its entity order follows creation order. This module reads the kernel's
file back, names every face with a 406 form 15 NAME (a property pointer and a level per component),
and writes the file again with a normalized global section, so the same faces written by two
processes give the same bytes and no login name reaches a file.
"""

from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.interpolate import BSpline

import cfdbender._textio as _textio
from cfdbender._errors import GeometryWarning, InputValueError, warn
from cfdbender.geometry._iges_names import parameter_space_curves

__all__ = [
    "FIXED_STAMP",
    "PREFERENCES",
    "flag_parameter_space",
    "normalize_global",
    "read_iges",
    "serialize_iges",
    "unbounded_faces",
    "write_iges",
]

#: The date and time every written file states, so a file does not depend on when it was written.
FIXED_STAMP = "15H19700101.000000"
#: The values of PREF in a trimmed face's outer boundary (142): 1 parameter space, 2 model space,
#: 3 either.
PREFERENCES = (1, 2, 3)
#: Zero-based positions of the global section parameters that carry a time or an account name:
#: file generation date (18) and model creation date (25), counted from 1.
_STAMP_POSITIONS = (17, 24)
#: Zero-based position of the author name (parameter 21).
_AUTHOR_POSITION = 20


def normalize_global(global_values: list[str]) -> list[str]:
    """Return the global section with the time stamps fixed and the author emptied."""
    values = list(global_values)
    for position in _STAMP_POSITIONS:
        if position < len(values):
            values[position] = FIXED_STAMP
    if _AUTHOR_POSITION < len(values):
        values[_AUTHOR_POSITION] = ""
    return values


def tokens(payload):
    result, i = [], 0
    while i < len(payload):
        while i < len(payload) and payload[i].isspace():
            i += 1
        match = re.match(r"(\d+)H", payload[i:])
        if match:
            end = i + len(match[0]) + int(match[1])
            result.append(payload[i:end])
            i = end
            while payload[i].isspace():
                i += 1
        else:
            end = i
            while end < len(payload) and payload[end] not in ",;":
                end += 1
            result.append(payload[i:end].strip())
            i = end
        if i == len(payload) or payload[i] == ";":
            break
        i += 1
    return result


def read_iges(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    sections = {
        key: [line for line in lines if len(line) == 80 and line[72] == key] for key in "SGDPT"
    }
    parameters = sections["P"]
    entities = []
    for first, second in zip(sections["D"][::2], sections["D"][1::2], strict=True):
        d1 = [first[i : i + 8] for i in range(0, 72, 8)]
        d2 = [second[i : i + 8] for i in range(0, 72, 8)]
        start, count = int(d1[1]) - 1, int(d2[3])
        payload = "".join(line[:64] for line in parameters[start : start + count])
        entities.append(
            {
                "id": int(first[73:]),
                "d1": d1,
                "d2": d2,
                "parameters": tokens(payload),
                "type": int(d1[0]),
            }
        )
    return {
        "start": [line[:72] for line in sections["S"]],
        "global": tokens("".join(line[:72] for line in sections["G"])),
        "entities": entities,
    }


def surface_key(document, entity):
    by_id = {e["id"]: e for e in document["entities"]}

    def geometry(row):
        values = list(row["parameters"])
        kind = row["type"]
        if kind == 144:
            references = [1, *range(4, 5 + int(values[3]))]
        elif kind == 142:
            references = [2, 3, 4]
        elif kind == 102:
            references = list(range(2, 2 + int(values[1])))
        else:
            references = []
        for index in references:
            pointer = int(values[index])
            values[index] = geometry(by_id[pointer]) if pointer else ()
        transform = int(row["d1"][6])
        return (tuple(values), geometry(by_id[transform]) if transform else ())

    return geometry(entity)


def component_keys(gmsh, groups, directory):
    entities = gmsh.model.getEntities()
    visibility = [(dim, tag, gmsh.model.getVisibility(dim, tag)) for dim, tag in entities]
    old = gmsh.option.getNumber("Geometry.OCCExportOnlyVisible")
    keys = {}
    try:
        gmsh.option.setNumber("Geometry.OCCExportOnlyVisible", 1)
        for index, (name, tags) in enumerate(groups.items()):
            gmsh.model.setVisibility(entities, 0)
            gmsh.model.setVisibility([(2, tag) for tag in tags], 1, recursive=True)
            path = directory / f"component-{index}.igs"
            gmsh.write(str(path))
            document = read_iges(path)
            for entity in document["entities"]:
                if entity["type"] == 144:
                    key = surface_key(document, entity)
                    if key in keys and keys[key] != name:
                        raise InputValueError(
                            "IGES.NAME: coincident bases require unambiguous component ownership"
                        )
                    keys[key] = name
    finally:
        for dim, tag, visible in visibility:
            gmsh.model.setVisibility([(dim, tag)], visible)
        gmsh.option.setNumber("Geometry.OCCExportOnlyVisible", old)
    return keys


def name_properties(document, groups, keys):
    """Name every component twice: a property pointer on each face and a level per component.

    Each trimmed surface (144) points at its component's 406 form 15 NAME and also stands on
    that component's level, on which the NAME stands too, so a reader that names a face by its
    level (this package's own IGES reader) and one that follows property pointers agree.
    """
    result = document
    levels = {name: i + 1 for i, name in enumerate(groups)}
    last = max(entity["id"] for entity in result["entities"])
    property_ids = {name: last + 2 * i + 2 for i, name in enumerate(groups)}
    for entity in result["entities"]:
        if entity["type"] == 144:
            name = keys[surface_key(document, entity)]
            values = entity["parameters"]
            if len(values) != 5 + int(values[3]):
                raise InputValueError("IGES.NAME: unexpected pre-existing surface property lists")
            entity["parameters"] += ["0", "1", str(property_ids[name])]
            entity["d1"][4] = f"{levels[name]:8d}"
    for name, identity in property_ids.items():
        if not name.isascii() or "," in name or ";" in name:
            raise InputValueError("IGES.NAME: ASCII component identifiers required")
        d1 = [f"{406:8d}"] + [f"{0:8d}"] * 7 + ["00000200"]
        d1[4] = f"{levels[name]:8d}"
        d2 = [
            f"{406:8d}",
            f"{0:8d}",
            f"{0:8d}",
            f"{0:8d}",
            f"{15:8d}",
            " " * 8,
            " " * 8,
            f"{'NAME':>8}",
            f"{0:8d}",
        ]
        result["entities"].append(
            {
                "id": identity,
                "type": 406,
                "d1": d1,
                "d2": d2,
                "parameters": ["406", "1", f"{len(name)}H{name}"],
            }
        )
    result["global"][13:15] = ["6", "1HM"]
    return result


def pack_records(values, width):
    """Pack ``values`` into records of at most ``width`` columns, breaking only between values.

    A parameter is never split across records, which the OpenCASCADE reader does not accept for
    numbers; only a single Hollerith string longer than a record is cut.
    """
    records, current = [], ""
    for k, value in enumerate(values):
        text = value + (";" if k == len(values) - 1 else ",")
        if current and len(current) + len(text) > width:
            records.append(current)
            current = ""
        current += text
        while len(current) > width:
            records.append(current[:width])
            current = current[width:]
    return records + [current] if current else records


def serialize_iges(document):
    lines = [f"{payload:<72}S{i + 1:07d}" for i, payload in enumerate(document["start"])]
    global_lines = pack_records(normalize_global(document["global"]), 72)
    lines += [f"{payload:<72}G{i + 1:07d}" for i, payload in enumerate(global_lines)]
    directory, parameters = [], []
    for entity in document["entities"]:
        chunks = pack_records(entity["parameters"], 64)
        d1, d2 = list(entity["d1"]), list(entity["d2"])
        d1[1], d2[3] = f"{len(parameters) + 1:8d}", f"{len(chunks):8d}"
        directory.extend(
            ["".join(d1) + f"D{entity['id']:07d}", "".join(d2) + f"D{entity['id'] + 1:07d}"]
        )
        for chunk in chunks:
            parameters.append(f"{chunk:<64}{entity['id']:8d}P{len(parameters) + 1:07d}")
    counts = (
        f"S{len(document['start']):07d}G{len(global_lines):07d}"
        f"D{len(directory):07d}P{len(parameters):07d}"
    )
    return "\n".join(lines + directory + parameters + [f"{counts:<72}T0000001"]) + "\n"


def set_preference(document, preference):
    """Set PREF (parameter 5) of every composite-curve-on-surface (142) of the document.

    A boundary with no parameter-space curve (BPTR, parameter 3, is 0) can only be read in model
    space, so it states 2 whatever ``preference`` asks.
    """
    for entity in document["entities"]:
        if entity["type"] == 142:
            if len(entity["parameters"]) < 6:
                raise InputValueError("IGES.PREF: a 142 entity without its PREF parameter")
            curveless = int(entity["parameters"][3] or 0) == 0
            entity["parameters"][5] = str(2 if curveless else preference)
    return document


#: The entity use flags a trimming curve carries: 00 geometry (the 142, a model-space curve) and 05
#: 2D parametric (the curves its BPTR names).
_GEOMETRY_USE, _PARAMETER_SPACE_USE = "00", "05"


def _status_with_use(status, use):
    """Return the status number with its entity use flag (digits 5 and 6) set to ``use``."""
    return status[:4] + use + status[6:]


def flag_parameter_space(document):
    """Flag every trimming curve as the solver's CAD import reads it.

    The entity use flag (digits 5 and 6 of the status number, directory entry field 9) of each
    curve a 142's BPTR names, and of every member a composite (102) there reaches, is set to 05
    (2D parametric); the 142 itself, a model-space curve, is set to 00 (geometry). The kernel's
    writer flags them the other way round, and the solver's CAD import then reads no face.
    """
    by_id = {e["id"]: e for e in document["entities"]}
    bptrs = []
    for entity in document["entities"]:
        if entity["type"] != 142:
            continue
        entity["d1"][8] = _status_with_use(entity["d1"][8], _GEOMETRY_USE)
        bptrs.append(int(entity["parameters"][3] or 0))

    def members(number):
        member = by_id.get(number)
        if member is None:
            return None
        if member["type"] != 102:
            return []
        values = member["parameters"]
        return [int(v) for v in values[2 : 2 + int(values[1])] if v]

    curves = parameter_space_curves(bptrs, members)
    for number in curves:
        by_id[number]["d1"][8] = _status_with_use(by_id[number]["d1"][8], _PARAMETER_SPACE_USE)
    return document


def unbounded_faces(document):
    """Count the trimmed faces (144) with no parameter-space outer boundary.

    A 144 with N1 = 0 (its surface's own boundary, no 142 written) or whose outer 142 has no BPTR
    curve (a model-space only boundary, as the kernel writes a plane) is read by the solver's CAD
    import as no face.
    """
    by_id = {e["id"]: e for e in document["entities"]}
    count = 0
    for entity in document["entities"]:
        if entity["type"] != 144:
            continue
        values = entity["parameters"]
        outer = by_id.get(int(values[4] or 0)) if int(values[2] or 0) else None
        if outer is None or not int(outer["parameters"][3] or 0):
            count += 1
    return count


def _write_visible(gmsh, groups, path):
    entities = gmsh.model.getEntities()
    visibility = [(dim, tag, gmsh.model.getVisibility(dim, tag)) for dim, tag in entities]
    old = gmsh.option.getNumber("Geometry.OCCExportOnlyVisible")
    try:
        gmsh.option.setNumber("Geometry.OCCExportOnlyVisible", 1)
        gmsh.model.setVisibility(entities, 0)
        gmsh.model.setVisibility(
            [(2, tag) for tags in groups.values() for tag in tags], 1, recursive=True
        )
        gmsh.write(str(path))
    finally:
        for dim, tag, visible in visibility:
            gmsh.model.setVisibility([(dim, tag)], visible)
        gmsh.option.setNumber("Geometry.OCCExportOnlyVisible", old)


def _nurbs_copies(gmsh, groups):
    """Return the groups as B-spline copies of their faces, which the kernel writes with trims."""
    occ = gmsh.model.occ
    copies = {}
    for name, tags in groups.items():
        copied = occ.copy([(2, tag) for tag in tags])
        occ.convertToNURBS(copied)
        copies[name] = [tag for _, tag in copied]
    occ.synchronize()
    return copies


def _name_and_flag(gmsh, groups, path, preference):
    document = read_iges(path)
    with TemporaryDirectory(prefix="iges-names-", dir=path.parent) as temporary:
        directory = Path(temporary).resolve()
        if not directory.is_relative_to(path.parent.resolve()):
            raise InputValueError("IGES.NAME: temporary folder left the output folder")
        keys = component_keys(gmsh, groups, directory)
    document = name_properties(document, groups, keys)
    document = _reader_boundaries(document)
    document = set_preference(document, preference)
    document = flag_parameter_space(document)
    _textio.write_text(path, serialize_iges(document), encoding="ascii")


def _curve_halves(entity):
    """Split a native curve exactly, retaining its transformation."""
    a, b = (list(entity["parameters"]) for _ in range(2))

    def number(v):
        return float(v.replace("D", "E").replace("d", "e"))

    if entity["type"] == 110:
        middle = [
            format((number(x) + number(y)) / 2, ".17g") for x, y in zip(a[1:4], a[4:7], strict=True)
        ]
        a[4:7], b[1:4] = middle, middle
    elif entity["type"] == 126:
        k, degree = map(int, a[1:3])
        at = 7 + k + degree + 2 + 4 * (k + 1)
        low, high = number(a[at]), number(a[at + 1])
        middle = (low + high) / 2
        a, b = (_bounded_curve(a, x, y) for x, y in ((low, middle), (middle, high)))
    else:
        return None
    return a, b


def _bounded_curve(values, low, high):
    """Clamp an exact rational B-spline subcurve in homogeneous coordinates."""
    k, degree = map(int, values[1:3])
    numbers = np.array([float(v.replace("D", "E").replace("d", "e")) for v in values[7:]])
    count, at = k + 1, k + degree + 2
    weights = numbers[at : at + count]
    points = numbers[at + count : at + 4 * count].reshape(count, 3)
    homogeneous = np.column_stack((points * weights[:, None], weights))
    spline = BSpline(numbers[:at], homogeneous, degree)
    for bound in (low, high):
        missing = degree + 1 - np.count_nonzero(spline.t == bound)
        if missing > 0:
            spline = spline.insert_knot(bound, int(missing))
    first = int(np.flatnonzero(spline.t == low)[0])
    last = int(np.flatnonzero(spline.t == high)[-1]) + 1
    knots = spline.t[first:last]
    poles = spline.c[first : first + len(knots) - degree - 1]
    weights = poles[:, 3]
    points = poles[:, :3] / weights[:, None]
    prefix = [*values[:7]]
    prefix[1], prefix[4], prefix[6] = str(len(poles) - 1), "0", "0"
    data = [*knots, *weights, *points.ravel(), low, high]
    return prefix + [format(v, ".17g") for v in data] + values[7 + at + 4 * count + 2 :]


def _append_copy(document, entity, values):
    copied = {**entity, "id": max(e["id"] for e in document["entities"]) + 2}
    copied.update(d1=list(entity["d1"]), d2=list(entity["d2"]), parameters=values)
    document["entities"].append(copied)
    return str(copied["id"])


def _split_chain(document, chain, index, curve, halves):
    members = [_append_copy(document, curve, values) for values in halves]
    values = list(chain["parameters"])
    values[1] = "4"
    values[2 + index : 3 + index] = members
    return _append_copy(document, chain, values)


def _triangle_chains(boundary, by_id):
    chains = [by_id.get(int(p)) for p in boundary["parameters"][3:5]]
    if all(c and c["type"] == 102 and c["parameters"][1] == "3" for c in chains):
        return chains
    return None


def _split_triangle(document, boundary, chains, by_id):
    for index in range(3):
        curves = [by_id[int(c["parameters"][2 + index])] for c in chains]
        halves = [_curve_halves(c) for c in curves]
        if any(h is None for h in halves):
            continue
        for column, (chain, curve, parts) in enumerate(zip(chains, curves, halves, strict=True)):
            boundary["parameters"][3 + column] = _split_chain(document, chain, index, curve, parts)
        break


def _reader_boundaries(document):
    """Give triangular loops four exact segments in both spaces, without splitting a face.

    IGES 5.3 section 4.23 defines the starting and ending parameters of a 126.
    Two adjacent restrictions cover the original curve. Native lines are
    bisected exactly. Unsupported curves retain their measured profile failure.
    """
    by_id = {e["id"]: e for e in document["entities"]}
    for boundary in list(document["entities"]):
        if boundary["type"] != 142:
            continue
        chains = _triangle_chains(boundary, by_id)
        if chains is None:
            continue
        _split_triangle(document, boundary, chains, by_id)
    return document


def _write_millimetres(gmsh, groups, path, preference):
    """Write scaled temporary copies and state mm without changing the live meter model."""
    write_iges(gmsh, groups, path, preference)
    original = read_iges(path)["global"]
    occ, copies = gmsh.model.occ, {}
    try:
        for name, tags in groups.items():
            copied = occ.copy([(2, tag) for tag in tags])
            copies[name] = [tag for _, tag in copied]
        faces = [(2, tag) for tags in copies.values() for tag in tags]
        occ.dilate(faces, 0, 0, 0, 1000, 1000, 1000)
        occ.synchronize()
        write_iges(gmsh, copies, path, preference)
        document = read_iges(path)
        values = document["global"]
        values[13], values[14] = "2", "2HMM"
        # Global parameters 17 and 19 carry lengths. Preserve physical line width and resolution.
        for index in (16, 18):
            if original[index]:
                value = float(original[index].replace("D", "E").replace("d", "e"))
                values[index] = format(1000 * value, ".17g")
        _textio.write_text(path, serialize_iges(document), encoding="ascii")
    finally:
        if copies:
            occ.remove([(2, tag) for tags in copies.values() for tag in tags], recursive=True)
            occ.synchronize()


def write_iges(gmsh, groups, path, preference=1, *, unit="m"):
    """Write the groups to one named IGES file with parameter space preferred (PREF 1).

    A ``preference`` of 2 (model space) or 3 (either) is warned about and written as 1. The
    solver's CAD import reads no face from a boundary that states PREF 2. Every trimming curve
    is flagged as parameter-space geometry (:func:`flag_parameter_space`). When the kernel
    would write a face
    with no parameter-space outer boundary (a natural boundary, or a plane or other analytic
    surface; :func:`unbounded_faces`), every face is written as a B-spline copy, which carries
    one, and the copies leave the model after the write.
    ``unit`` selects m or mm; the latter writes scaled copies and preserves the live model.
    """
    if unit not in ("m", "mm"):
        raise InputValueError(f"IGES.unit: expected m or mm, got {unit!r}")
    if preference not in PREFERENCES:
        raise InputValueError(f"IGES.PREF: {preference!r} is not one of {PREFERENCES}")
    if preference != 1:
        meaning = "model space" if preference == 2 else "either space"
        warn(
            f"IGES.PREF: {preference} ({meaning}) was requested; PREF 1 is written instead "
            "so the reader uses the parameter-space boundary",
            GeometryWarning,
            stacklevel=3,
        )
        preference = 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if unit == "mm":
        _write_millimetres(gmsh, groups, path, preference)
        return
    _write_visible(gmsh, groups, path)
    if not unbounded_faces(read_iges(path)):
        _name_and_flag(gmsh, groups, path, preference)
        return
    copies = _nurbs_copies(gmsh, groups)
    try:
        _write_visible(gmsh, copies, path)
        _name_and_flag(gmsh, copies, path, preference)
    finally:
        gmsh.model.occ.remove([(2, t) for tags in copies.values() for t in tags], recursive=True)
        gmsh.model.occ.synchronize()
