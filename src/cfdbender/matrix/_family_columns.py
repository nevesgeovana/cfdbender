"""The families, the groups and the column registry of the polar and sections files.

One home for the header of every file written per family or per group: each column's
name, unit, frame, quantity and sign. The writers compose their headers from here and
``products.json`` states the same specification, so a header and its statement cannot
drift apart. Nothing here computes a value.

The kinds of file:

* ``DIM``: the force and moment in the geometry frame about the moment reference point;
* ``ACOEF``: the airframe coefficients in the frames W, S, B and R<id>, then the solver's
  own coefficients under its own names;
* ``RCOEF``: the rotor coefficients of a family or group inside one rotor;
* sections: the sectional distributions of a point, per family.

A family file is ``FAM-<KIND>-P<POL>.csv`` (one row per point and family); a group file is
``P<POL>-<GROUP>-<KIND>.csv`` (one row per point).

FAMILIES. A solver states its integrated loads as one family per point until it states
loads per boundary: ``BODY`` (``FAMILY_ID`` 1) for an airframe solver and ``BLADES``
(``FAMILY_ID`` 1, inside rotor 0) for QPROP's isolated propeller.

GROUPS. ``TOTAL`` (``GROUP_ID`` 0, every family), ``AIRFRAME`` (1, every family not inside
a rotor) and one group per rotor named by its alias (``10 + id``). A group with no member
is a named skip, not a file: a solver without rotors writes no rotor group and QPROP writes
no AIRFRAME group. RCOEF group files exist only for a group whose members all sit inside
one rotor.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from cfdbender.core.axes import frame_names
from cfdbender.core.rotor_coefficients import PER_FRAME_ROTOR_COLUMNS, ROTOR_FRAME_COLUMNS

#: The frame the DIM loads are stated in: the geometry frame about the moment point.
LOAD_FRAME = "MRP"
#: The kinds of polar file, in the order they are written.
KINDS = ("DIM", "ACOEF", "RCOEF")
#: Group names a declaration may not take: the defaults and the file-name prefixes.
RESERVED_GROUPS = ("TOTAL", "AIRFRAME", "FAM", "GRP")


@dataclass(frozen=True)
class ColumnSpec:
    """The statement of one column.

    Attributes
    ----------
    name : str
        The header, exactly as written.
    unit : str
        The unit, ``"-"`` for a dimensionless quantity or a key.
    frame : str or None
        The frame the value is stated in, None for a scalar.
    quantity : str
        What the column holds.
    sign : str or None
        The sense in which the value is positive, where it has one.
    """

    name: str
    unit: str
    frame: str | None
    quantity: str
    sign: str | None = None


#: Per-point convergence evidence carried by SU2 and OpenFOAM family and group tables.
CONVERGENCE_COLUMNS = (
    ColumnSpec("solver_version", "-", None, "reported solver version"),
    ColumnSpec("rule_verdict", "-", None, "polar convergence criterion: R, C or none"),
    ColumnSpec("rule_iteration", "-", None, "first iteration meeting the polar rule"),
    ColumnSpec("rule_drop_orders", "orders", None, "residual drop below its first value"),
)


@dataclass(frozen=True)
class Family:
    """One family: its name, its 1-based id and the rotor it sits in (None off a rotor)."""

    name: str
    family_id: int
    rotor_id: int | None = None


@dataclass(frozen=True)
class Rotor:
    """One rotor: its alias, id and shaft in the geometry frame."""

    alias: str
    rotor_id: int
    shaft: tuple[float, float, float]


@dataclass(frozen=True)
class Group:
    """One group: name, id, origin, member family ids and the rotor all members sit in."""

    name: str
    group_id: int
    origin: str
    members: tuple[int, ...]
    rotor_id: int | None = None


#: The rotors a rotor-capable solver states, by solver.
SOLVER_ROTORS: dict[str, tuple[Rotor, ...]] = {
    "qprop": (Rotor("PROPELLER", 0, (-1.0, 0.0, 0.0)),),
}


def solver_families(solver: str | None) -> tuple[Family, ...]:
    """Return the families a solver's integrated result states."""
    rotors = SOLVER_ROTORS.get(solver or "", ())
    if rotors:
        return (Family("BLADES", 1, rotors[0].rotor_id),)
    return (Family("BODY", 1, None),)


def group_id(origin: str, *, rotor_id: int | None = None) -> int:
    """Return a group's id: 0 TOTAL, 1 AIRFRAME, 10 + id for the group of rotor id."""
    if origin == "TOTAL":
        return 0
    if origin == "AIRFRAME":
        return 1
    if origin == "rotor" and rotor_id is not None and rotor_id >= 0:
        return 10 + rotor_id
    raise ValueError(f"group origin {origin!r} needs a rotor id of zero or more")


def check_group_name(name: str) -> None:
    """Refuse a group or rotor alias that a file name or a default group already takes.

    Raises
    ------
    ValueError
        If the name is reserved, starts with P and a digit, or holds a character a file
        name cannot carry.
    """
    if name.casefold() in {r.casefold() for r in RESERVED_GROUPS}:
        raise ValueError(
            f"group {name!r} is reserved: TOTAL and AIRFRAME are written for every polar, "
            "FAM and GRP are file-name prefixes"
        )
    if re.match(r"^P[0-9]", name) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(
            f"group {name!r} must be letters, digits and underscores and not start with P "
            "and a digit, so its file name P<POL>-<GROUP>-<KIND>.csv reads one way"
        )


def polar_groups(
    families: Sequence[Family], rotors: Sequence[Rotor]
) -> tuple[tuple[Group, ...], tuple[tuple[str, str], ...]]:
    """Return the groups with members, and each group skipped with its reason.

    Returns
    -------
    tuple
        ``(groups, skipped)``: the groups in id order and ``(group, reason)`` pairs.
    """
    every = tuple(f.family_id for f in families)
    rotor_of = {f.family_id: f.rotor_id for f in families}

    def inside(members: Sequence[int]) -> int | None:
        ids = {rotor_of[m] for m in members}
        return ids.pop() if len(ids) == 1 else None

    candidates = [
        Group("TOTAL", 0, "default", every, inside(every) if every else None),
        Group(
            "AIRFRAME",
            1,
            "default",
            tuple(f.family_id for f in families if f.rotor_id is None),
        ),
    ]
    for rotor in sorted(rotors, key=lambda r: r.rotor_id):
        check_group_name(rotor.alias)
        members = tuple(f.family_id for f in families if f.rotor_id == rotor.rotor_id)
        candidates.append(
            Group(
                rotor.alias,
                group_id("rotor", rotor_id=rotor.rotor_id),
                "rotor",
                members,
                rotor.rotor_id,
            )
        )
    groups = tuple(g for g in candidates if g.members)
    skipped = [
        (g.name, f"group {g.name} has no member family, so no {g.name} file is written")
        for g in candidates
        if not g.members
    ]
    if not rotors:
        skipped.append(
            ("ROTOR", "the solver states no rotor, so no rotor group and no RCOEF file is written")
        )
    return groups, tuple(skipped)


def _identity(entity: str) -> tuple[ColumnSpec, ...]:
    if entity == "family":
        return (
            ColumnSpec("FAMILY", "-", None, "family name"),
            ColumnSpec("FAMILY_ID", "-", None, "1-based family index"),
        )
    return (
        ColumnSpec("GROUP", "-", None, "group name"),
        ColumnSpec("GROUP_ID", "-", None, "group number"),
    )


_MESH_FACES = ColumnSpec("MESH_FACES", "-", None, "mesh face count")


def dim_columns(entity: str = "family") -> tuple[ColumnSpec, ...]:
    """Return the DIM columns: identity, ``FX_MRP`` to ``MZ_MRP``, ``MESH_FACES``."""
    forces = tuple(
        ColumnSpec(f"F{a}_{LOAD_FRAME}", "N", LOAD_FRAME, f"force along {a.lower()}") for a in "XYZ"
    )
    moments = tuple(
        ColumnSpec(f"M{a}_{LOAD_FRAME}", "N m", LOAD_FRAME, f"moment about {a.lower()}")
        for a in "XYZ"
    )
    return (*_identity(entity), *forces, *moments, _MESH_FACES)


_ACOEF = (
    ("CD", "drag coefficient", "positive aft"),
    ("CY", "side force coefficient", "positive right"),
    ("CL", "lift coefficient", "positive up"),
    ("Cl", "rolling moment coefficient", "right-hand about x"),
    ("Cm", "pitching moment coefficient", "right-hand about y"),
    ("Cn", "yawing moment coefficient", "right-hand about z"),
)


def acoef_columns(
    rotor_ids: Sequence[int] = (), own: Sequence[str] = (), entity: str = "family"
) -> tuple[ColumnSpec, ...]:
    """Return the ACOEF columns: identity, six per frame, the solver's own, ``MESH_FACES``."""
    blocks = [
        ColumnSpec(f"{stem}{frame}", "-", frame, quantity, sign)
        for frame in frame_names(rotor_ids)
        for stem, quantity, sign in _ACOEF
    ]
    taken = {spec.name for spec in blocks}
    native = [
        ColumnSpec(name, "as the solver states it", None, "the solver's own coefficient")
        for name in own
        if name not in taken
    ]
    return (*_identity(entity), *blocks, *native, _MESH_FACES)


def rcoef_columns(rotor_ids: Sequence[int], entity: str = "family") -> tuple[ColumnSpec, ...]:
    """Return the RCOEF columns: identity, ``ROTOR_ID``, rotor-frame set, per-frame set."""
    if not list(rotor_ids):
        raise ValueError("an RCOEF file needs at least one rotor")
    rotor = ColumnSpec("ROTOR_ID", "-", None, "the rotor's id")
    own = tuple(
        ColumnSpec(name, unit, None if unit == "kW" else "R<id>", quantity, sign)
        for name, unit, quantity, sign in ROTOR_FRAME_COLUMNS
    )
    per_frame = tuple(
        ColumnSpec(f"{stem}{frame}", "-", frame, f"{stem} along the frame's x")
        for frame in frame_names(rotor_ids)
        for stem in PER_FRAME_ROTOR_COLUMNS
    )
    return (*_identity(entity), rotor, *own, *per_frame, _MESH_FACES)


def sections_columns(export: Sequence[ColumnSpec] = ()) -> tuple[ColumnSpec, ...]:
    """Return the sections columns after the run block: identity, then the solver's own."""
    identity = (
        *_identity("family"),
        ColumnSpec("STEP", "-", None, "solver step, NA on a steady point"),
        ColumnSpec("PLANE", "-", None, "cutting plane of the distribution"),
        ColumnSpec("ROTOR_ID", "-", None, "the rotor's id, NA off a rotor"),
        ColumnSpec("AZIMUTH", "deg", None, "blade azimuth, NA off a rotor"),
    )
    return (*identity, *export)


def column_names(columns: Sequence[ColumnSpec]) -> tuple[str, ...]:
    """Return the headers of ``columns``, in order."""
    return tuple(spec.name for spec in columns)


def family_file_name(pol: str, kind: str) -> str:
    """Return ``FAM-<KIND>-P<POL>.csv``."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {', '.join(KINDS)}")
    return f"FAM-{kind}-P{pol}.csv"


def group_file_name(pol: str, group: str, kind: str) -> str:
    """Return ``P<POL>-<GROUP>-<KIND>.csv``."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {', '.join(KINDS)}")
    return f"P{pol}-{group}-{kind}.csv"
