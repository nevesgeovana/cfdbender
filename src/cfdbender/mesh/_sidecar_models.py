"""The models of a mesh sidecar: the import table and the named boundary declarations.

:class:`MeshImport` and :class:`MeshOperation` say how a raw mesh is imported (its length
unit and the operations applied right after); :class:`TrailingEdgeMarking` how its trailing
edges are marked; :class:`BoundaryDeclarations` the ordered trailing-edge and base-region
entries of an OBJ's sidecar.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from cfdbender._errors import InputValueError

__all__ = [
    "EVERY_SURFACE",
    "BoundaryBaseRegion",
    "BoundaryDeclarations",
    "BoundaryTrailingEdge",
    "MeshImport",
    "MeshOperation",
    "TrailingEdgeMarking",
    "TrailingEdgeSet",
]


#: The word a mesh operation's ``surface`` takes for every surface of the file.
EVERY_SURFACE = "all"

#: The keys each mesh operation of an import states besides ``op`` and
#: ``surface``, and the only ones it may state.
_MESH_OPERATION_KEYS: Mapping[str, tuple[str, ...]] = {
    "scale": ("factors",),
    "rename": ("to",),
    "mirror": ("plane",),
    "translate": ("vector",),
    "rotate": ("axis", "angle_deg"),
}


class MeshOperation(BaseModel):
    """One mesh operation applied right after a raw mesh is imported.

    Declared in the geometry's sidecar, beside the unit, as one
    ``[[import.operations]]`` table each, and applied in the order written,
    in the reference frame:

    * ``scale``: ``factors = [fx, fy, fz]``, each greater than zero;
    * ``rename``: ``surface`` and ``to``, the new name;
    * ``mirror``: ``surface`` and ``plane`` (``YZ``, ``XZ`` or ``XY``); the
      mirrored copy joins its source, so the surface count is unchanged;
    * ``translate``: ``vector = [x, y, z]``, in the ``[import]`` unit;
    * ``rotate``: ``axis`` (``X``, ``Y`` or ``Z``) and ``angle_deg``.

    ``surface`` names the surface acted on by the name the file gives it,
    or by a name an earlier rename gave; never by position. It is
    :data:`EVERY_SURFACE` by default, which ``rename`` and ``mirror`` do not
    take: each acts on one named surface.

    Attributes
    ----------
    op : {'scale', 'rename', 'mirror', 'translate', 'rotate'}
        Which operation, and so which of the keys below it states.
    surface : str
        The surface acted on, by the file's name or an earlier rename's;
        ``"all"`` for every surface, which a rename and a mirror do not take.
    factors : tuple of three floats, optional
        A scale's factor along each axis, each greater than zero.
    vector : tuple of three floats, optional
        A translation's vector, in the ``[import]`` unit.
    axis : {'X', 'Y', 'Z'}, optional
        The axis a rotation turns about.
    angle_deg : float, optional
        The angle of a rotation, in degrees.
    plane : {'YZ', 'XZ', 'XY'}, optional
        The plane a mirror reflects in; the copy joins its source.
    to : str, optional
        A rename's new name, one word.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["scale", "rename", "mirror", "translate", "rotate"]
    surface: str = EVERY_SURFACE
    factors: tuple[float, float, float] | None = None
    vector: tuple[float, float, float] | None = None
    axis: Literal["X", "Y", "Z"] | None = None
    angle_deg: float | None = None
    plane: Literal["YZ", "XZ", "XY"] | None = None
    to: str | None = None

    @field_validator("surface", "to", mode="before")
    @classmethod
    def _stripped(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _its_own_keys_and_sound_values(self) -> MeshOperation:
        own = _MESH_OPERATION_KEYS[self.op]
        every = {key for keys in _MESH_OPERATION_KEYS.values() for key in keys}
        missing = [key for key in own if getattr(self, key) is None]
        if missing:
            raise InputValueError(
                f"a {self.op} states {' and '.join(own)}; {', '.join(missing)} is not stated"
            )
        foreign = sorted(key for key in every - set(own) if getattr(self, key) is not None)
        if foreign:
            raise InputValueError(
                f"a {self.op} does not read {', '.join(foreign)}; it states "
                f"{' and '.join(own)} and, optionally, surface"
            )
        if not self.surface:
            raise InputValueError(
                f'surface is empty; name a surface of the file, or write "{EVERY_SURFACE}"'
            )
        if self.op in ("rename", "mirror") and self.surface == EVERY_SURFACE:
            raise InputValueError(
                f'a {self.op} names the surface it acts on, as surface = "<a name of the '
                'file>"; it has no every-surface form'
            )
        numbers = [*(self.factors or ()), *(self.vector or ())]
        if self.angle_deg is not None:
            numbers.append(self.angle_deg)
        if not all(math.isfinite(number) for number in numbers):
            raise InputValueError(
                f"a {self.op} states a value that is not a finite number"
                ". Replace it with a finite number."
            )
        if self.factors is not None and min(self.factors) <= 0:
            raise InputValueError(
                f"each scale factor must be greater than zero, the manual's rule; "
                f"got {list(self.factors)}"
            )
        if self.to is not None and (not self.to or any(char.isspace() for char in self.to)):
            raise InputValueError(
                f"to = {self.to!r} is not a surface name of one word; write a name with no spaces"
            )
        if self.to == EVERY_SURFACE:
            raise InputValueError(
                f'to = "{EVERY_SURFACE}" would name a surface with the word for every surface'
            )
        return self


class MeshImport(BaseModel):
    """How a raw mesh is imported: the ``[import]`` table of its sidecar.

    A raw mesh (``.obj``, ``.stl``) carries no length unit, so the file that names its
    boundaries, beside it, states the unit it is written in; a unit is never assumed.
    The value is only normalised here (stripped, in capitals).

    ``cad`` and ``ccs`` are kept as the tables the sidecar wrote; a mesh verb that reads an
    OBJ refuses a sidecar that states either, because they describe a conversion of another
    file kind.

    ``operations`` are the mesh operations applied right after the import,
    in the order written (:class:`MeshOperation`); empty when the
    table declares none.

    Attributes
    ----------
    units : str
        The raw mesh length unit, or FILE for CAD metadata; never assumed.
    operations : tuple of MeshOperation
        The mesh operations applied right after the import, in the order
        written.

    Examples
    --------
    >>> imported = MeshImport(units="mm")
    >>> imported.units, imported.operations
    ('MM', ())
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    units: str
    #: A CAD conversion table, kept as written.
    cad: Mapping[str, Any] | None = None
    #: A CCS import table, kept as written.
    ccs: Mapping[str, Any] | None = None
    operations: tuple[MeshOperation, ...] = ()

    @field_validator("units", mode="before")
    @classmethod
    def _one_word_in_capitals(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        spelled = value.strip().upper()
        if not spelled:
            raise InputValueError(
                "`units` is empty; write the length unit the mesh file is written in, "
                'as units = "MILLIMETER"'
            )
        return spelled

    @property
    def moving_operations(self) -> tuple[MeshOperation, ...]:
        """The operations that move, scale or copy the body: every one but ``rename``.

        A trailing-edge points file names edges of the mesh as the file
        holds it; one of these would carry the body's edges away from the points.
        """
        return tuple(operation for operation in self.operations if operation.op != "rename")

    def names_after_renames(self, names: Sequence[str]) -> tuple[str, ...]:
        """Return the boundary names as this import's renames leave them, in order.

        This applies the renames and judges nothing: a rename whose surface is absent
        at its step, or carried twice, is passed over.
        """
        renamed = list(names)
        for operation in self.operations:
            if operation.op != "rename" or operation.to is None:
                continue
            found = [index for index, name in enumerate(renamed) if name == operation.surface]
            if len(found) == 1:
                renamed[found[0]] = operation.to
        return tuple(renamed)


#: The ``[trailing_edges]`` routes a mesh's sidecar may take: a file of edge
#: mid-points, the default, or detection, applied only when written.
TrailingEdgeRoute = Literal["file", "detect", "none"]


class TrailingEdgeSet(BaseModel):
    """One points file and the type applied to its trailing edges, in metres."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    edge_type: Literal["STANDARD", "RELAXED", "JET_OUTFLOW", "VORTEX_SHEDDING"] = "STANDARD"
    points_m: tuple[tuple[float, float, float], ...] = ()
    points_file: str | None = Field(default=None, exclude=True)

    @field_validator("edge_type", mode="before")
    @classmethod
    def _uppercase(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class TrailingEdgeMarking(BaseModel):
    """How a mesh's trailing edges are marked: its sidecar's ``[trailing_edges]``.

    ``route = "file"`` is the default route. ``points_m`` are the mid-points
    of the trailing-edge mesh edges, in metres, read from the points file the table
    names, and ``tolerance``, also in metres, is the distance within which an edge's
    mid-point counts as a point of the file. ``points_file`` is where they were read
    from, for a refusal to name; it is kept out of every dump, since it is a path on
    one machine.

    ``route = "detect"`` marks by detection: ``detect_surfaces`` names the surfaces to
    detect on, by the sidecar's names, and is empty for every surface;
    ``sweep_angle_deg`` is set before the detection when stated. Detection gives every
    edge the STANDARD type and reads no tolerance.

    ``route = "none"`` declares a body without a trailing edge.

    Examples
    --------
    >>> marking = TrailingEdgeMarking(route="detect")
    >>> marking.route, marking.edge_type, marking.detect_surfaces
    ('detect', 'STANDARD', ())
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: TrailingEdgeRoute
    tolerance: float = 0.0001
    sets: tuple[TrailingEdgeSet, ...] = ()
    detect_surfaces: tuple[str, ...] = ()
    sweep_angle_deg: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_file(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        legacy = {
            key: data.pop(key) for key in ("edge_type", "points_m", "points_file") if key in data
        }
        if "sets" not in data and (data.get("route") == "file" or legacy):
            data["sets"] = (legacy,)
        return data

    @property
    def edge_type(self) -> str:
        """Return the first file's type for single-file callers."""
        return self.sets[0].edge_type if self.sets else "STANDARD"

    @property
    def points_m(self) -> tuple[tuple[float, float, float], ...]:
        """Return the first file's points in metres."""
        return self.sets[0].points_m if self.sets else ()

    @property
    def points_file(self) -> str | None:
        """Return the first file's local path, excluded from records."""
        return self.sets[0].points_file if self.sets else None

    def model_copy(self, *, update=None, deep=False):
        """Preserve the legacy first-file update spelling for existing callers."""
        values = dict(update or {})
        first = {
            key: values.pop(key)
            for key in ("edge_type", "points_m", "points_file")
            if key in values
        }
        if first:
            entry = self.sets[0] if self.sets else TrailingEdgeSet()
            values["sets"] = (entry.model_copy(update=first, deep=deep), *self.sets[1:])
        return super().model_copy(update=values, deep=deep)

    @model_serializer(mode="wrap")
    def _record(self, handler):
        data = handler(self)
        sets = data.pop("sets")
        record = {
            "route": data.pop("route"),
            "edge_type": self.edge_type,
            "tolerance": data.pop("tolerance"),
            "points_m": sets[0]["points_m"] if sets else [],
            **data,
        }
        if len(sets) > 1:
            record["sets"] = sets
        return record

    @model_validator(mode="after")
    def _one_route_and_sound_values(self) -> TrailingEdgeMarking:
        if self.route == "none":
            if (
                self.points_m
                or self.points_file
                or self.detect_surfaces
                or self.sweep_angle_deg is not None
            ):
                raise InputValueError("no trailing edge cannot also supply points or detection")
            return self
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise InputValueError(
                f"tolerance = {self.tolerance!r}; it is the distance, in metres, "
                "within which an edge's mid-point counts as a point of the "
                "file, and it must be positive and finite"
            )
        if self.route == "file":
            if self.detect_surfaces or self.sweep_angle_deg is not None:
                raise InputValueError(
                    "the file route takes no detection surfaces and no sweep angle"
                    ". Drop them, or use the detect route."
                )
            return self
        if self.points_m or self.points_file is not None:
            raise InputValueError(
                "the detect route reads no points file. Drop the points file from the detect route."
            )
        if not all(name.strip() for name in self.detect_surfaces):
            raise InputValueError(
                "a detection surface is named by an empty string. Name the detection surface."
            )
        if self.sweep_angle_deg is not None and not math.isfinite(self.sweep_angle_deg):
            raise InputValueError(
                f"sweep_angle = {self.sweep_angle_deg!r} is not a finite number"
                ". State a finite sweep_angle."
            )
        return self


class BoundaryTrailingEdge(BaseModel):
    """One labeled trailing-edge import or detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str
    marking: TrailingEdgeMarking


class BoundaryBaseRegion(BaseModel):
    """A mesh family's base model, rim and optional merge destination."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    family: str
    model: Literal["EMPIRICAL", "AXISYMMETRIC", "USER", "CUSTOM"] = "EMPIRICAL"
    cp: float = -0.2
    trailing_edges: TrailingEdgeMarking | Literal["auto"]
    merge_into: str | None = None


class BoundaryDeclarations(BaseModel):
    """Ordered trailing-edge and base-region declarations of an OBJ's sidecar."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    trailing_edges: tuple[BoundaryTrailingEdge, ...] = ()
    base_regions: tuple[BoundaryBaseRegion, ...] = ()
    no_trailing_edges: bool = False
