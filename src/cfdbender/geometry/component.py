"""The component parameter record: the contract between a reconstruction and the library.

Pipeline role: pure data contract of the geometry package. A
:class:`ComponentRecord` states how one aircraft component (a lifting surface,
a body or a body of revolution) is rebuilt from section curves: its stations,
the split points of every section curve, the construction method, the
trailing-edge base and the measured deviation from the source geometry.

Section curves are referenced, not copied: each is an IGES type 126 B-spline
curve named by file path, the SHA-256 of the exact file bytes and the
directory-entry sequence number. The pin check re-reads every referenced
file against its pin, so a record is only as current as the bytes it names.

Coordinate convention: component geometry is in the shared configuration
frame (x downstream, y along span, z upward). All lengths of a record,
including deviations and station coordinates, are in millimetres and all
angles are in degrees. The record model is the authored definition of the
contract; the schema export renders the draft 2020-12 JSON Schema from it
and the packaged copy is compared with the export in the test suite, so the
two cannot drift.

Examples
--------
>>> from cfdbender.geometry.component import DEVIATION_PROFILES
>>> DEVIATION_PROFILES["study_default"]["floor_mm"]
0.5
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cfdbender._errors import InputValueError

__all__ = ["DEVIATION_PROFILES", "ComponentRecord"]

#: Named deviation limits. The limit is a profile, not a universal acceptance
#: threshold: a record names the profile it was measured against, and the
#: constants must equal the profile's. ``study_default`` is the rule
#: ``max(floor_mm, local_fraction * local_size_mm)`` with a 0.5 mm floor and
#: 0.1 percent of the local chord or diameter.
DEVIATION_PROFILES: dict[str, dict[str, float]] = {
    "study_default": {"floor_mm": 0.5, "local_fraction": 0.001},
}

_CODE = r"^[A-Za-z][A-Za-z0-9_-]*$"
_SHA256 = r"^[0-9a-f]{64}$"
_VERSION = r"^[1-9][0-9]*\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"

Code = Annotated[str, Field(pattern=_CODE)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileRef(_Strict):
    """A file named by path (resolved against the asset root) and exact-byte SHA-256."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)


class CurveRef(_Strict):
    """An IGES type 126 B-spline curve: file pin plus directory-entry sequence number."""

    format: Literal["iges_type_126"]
    file: FileRef
    de_sequence: int = Field(ge=1)


class SurfaceRef(_Strict):
    """An IGES type 128 B-spline surface: file pin plus directory-entry sequence number."""

    format: Literal["iges_type_128"]
    file: FileRef
    de_sequence: int = Field(ge=1)


class Sides(_Strict):
    """A curve parameter on each side of a lifting section."""

    upper: Unit
    lower: Unit


class LiftingSplits(_Strict):
    """Split parameters of a lifting section curve (traversal: TE upper, LE, TE lower)."""

    LE: Unit
    max_thickness: Sides
    TE: Sides


class AzimuthSector(_Strict):
    """Start azimuth (degrees about +x, zero at +y toward +z) and curve parameter of a sector."""

    azimuth_deg: float = Field(ge=0.0, le=450.0)
    t: Unit


class BodySplits(_Strict):
    """Azimuth sector starts of a body section curve."""

    azimuth_sectors: list[AzimuthSector] = Field(min_length=2)


class Station(_Strict):
    """One section curve of a component at a coordinate along its station axis."""

    id: Code
    origin: Literal["measured", "inserted", "break", "junction"]
    axis: Literal["x", "span"]
    coordinate: float
    section_curve: CurveRef
    split_points: LiftingSplits | BodySplits


class SurfaceDegree(_Strict):
    """Surface degree in the two parametric directions (``v`` is typically 3)."""

    u: int = Field(ge=1)
    v: int = Field(ge=1)


class Region(_Strict):
    """An ordered x range of a hybrid construction; later regions name their join curve."""

    x_range: tuple[float, float]
    join_section_curve: Code | CurveRef | None = None


class Construction(_Strict):
    """How the surface is generated from the stations."""

    method: Literal["tensor_loft", "revolution", "hybrid"]
    across_station_interpolation: Literal["pchip", "cubic", "linear"]
    surface_degree: SurfaceDegree
    regions: list[Region] = Field(default_factory=list)


class TeBaseSharp(_Strict):
    """A sharp trailing edge: no base surface."""

    type: Literal["sharp"]


class TeBaseFinite(_Strict):
    """A finite trailing edge closed by a type 128 base surface."""

    type: Literal["finite"]
    surface: SurfaceRef


class DirectionResult(_Strict):
    """Maximum distance of one direction and its ratio to the local limit."""

    max_mm: float = Field(ge=0.0)
    max_local_limit_ratio: float = Field(ge=0.0)


class DeviationLimit(_Strict):
    """The limit rule and the named profile whose constants it carries."""

    profile: str = "study_default"
    rule: Literal["max(floor_mm, local_fraction * local_size_mm)"] = (
        "max(floor_mm, local_fraction * local_size_mm)"
    )
    floor_mm: float
    local_fraction: float


class UnresolvedInterval(_Strict):
    """An adjacent-station interval whose forward ratio still exceeds one."""

    station_ids: tuple[Code, Code]
    max_local_limit_ratio: float = Field(gt=1.0)


class StoppingRule(_Strict):
    """The interval floor of the adaptive station insertion and what it left unresolved."""

    interval_floor_fraction: float = Field(ge=0.01, le=0.01)
    component_length_mm: float = Field(gt=0.0)
    unresolved_intervals: list[UnresolvedInterval]


class Deviation(_Strict):
    """The measured deviation of the reconstruction from the source geometry."""

    forward_scope: Literal["reconstruction_to_original_everywhere"]
    reverse_scope: Literal["original_to_reconstruction_inside_trimmed_boundary"]
    local_size: Literal["chord", "diameter"]
    limit: DeviationLimit
    stopping_rule: StoppingRule
    forward: DirectionResult
    reverse: DirectionResult
    evidence: FileRef


class ComponentRecord(_Strict):
    """The parameter record of one component (schema 0.2.0 of the contract).

    ``symmetry`` is required for body and revolution components and absent for
    lifting components; ``te_base`` is explicit for lifting components and null
    for the others. Structural rules are enforced on construction. Checks that
    need the referenced files (that an entity exists, has the declared type and
    that splits lie on their curves) are not made here: the pin check
    covers the pins and the geometric checks belong to the check stage.
    """

    component_code: Code
    version: str = Field(pattern=_VERSION)
    units: Literal["mm"]
    source_iges: FileRef
    kind: Literal["lifting", "body", "revolution"]
    construction: Construction
    symmetry: Literal["full", "half_y0"] | None = None
    stations: list[Station] = Field(min_length=2)
    break_stations: list[Code]
    junction_stations: list[Code]
    te_base: TeBaseSharp | TeBaseFinite | None
    measured_deviation: Deviation

    @model_validator(mode="after")
    def _semantics(self) -> ComponentRecord:
        problems: list[str] = []
        lifting = self.kind == "lifting"
        if lifting and self.symmetry is not None:
            problems.append("symmetry: absent for a lifting component")
        if not lifting and self.symmetry is None:
            problems.append("symmetry: required for a body or revolution component")
        if lifting and self.te_base is None:
            problems.append("te_base: sharp or finite is required for a lifting component")
        if not lifting and self.te_base is not None:
            problems.append("te_base: null for a body or revolution component")
        if self.measured_deviation.local_size != ("chord" if lifting else "diameter"):
            problems.append("measured_deviation.local_size: chord for lifting, diameter otherwise")
        self._station_problems(problems, lifting)
        self._construction_problems(problems)
        self._deviation_problems(problems)
        if problems:
            raise InputValueError("; ".join(problems))
        return self

    def _station_problems(self, problems: list[str], lifting: bool) -> None:
        ids = [s.id for s in self.stations]
        if len(set(ids)) != len(ids):
            problems.append("stations: ids must be unique")
        axis = "span" if lifting else "x"
        if any(s.axis != axis for s in self.stations):
            problems.append(f"stations.axis: every station uses {axis!r} for this kind")
        coords = [s.coordinate for s in self.stations]
        if any(not math.isfinite(c) for c in coords) or any(
            b <= a for a, b in zip(coords, coords[1:], strict=False)
        ):
            problems.append("stations.coordinate: finite and strictly increasing")
        for listing in ("break_stations", "junction_stations"):
            names = getattr(self, listing)
            if len(set(names)) != len(names):
                problems.append(f"{listing}: ids must be unique")
            for name in names:
                if name not in ids:
                    problems.append(f"{listing}: {name!r} is not a station id")
        for station in self.stations:
            splits = station.split_points
            if lifting != isinstance(splits, LiftingSplits):
                problems.append(f"stations[{station.id}].split_points: wrong kind of splits")
            elif isinstance(splits, LiftingSplits):
                order = [
                    splits.TE.upper,
                    splits.max_thickness.upper,
                    splits.LE,
                    splits.max_thickness.lower,
                    splits.TE.lower,
                ]
                if any(b <= a for a, b in zip(order, order[1:], strict=False)):
                    problems.append(
                        f"stations[{station.id}].split_points: must follow the traversal "
                        "TE upper, LE, TE lower with distinct parameters"
                    )
            else:
                self._sector_problems(problems, station.id, splits)

    def _sector_problems(self, problems: list[str], name: str, splits: BodySplits) -> None:
        sectors = splits.azimuth_sectors
        angles = [s.azimuth_deg for s in sectors]
        parameters = [s.t for s in sectors]
        if any(b <= a for a, b in zip(angles, angles[1:], strict=False)) or any(
            b <= a for a, b in zip(parameters, parameters[1:], strict=False)
        ):
            problems.append(
                f"stations[{name}].split_points: sector angles and t strictly increasing"
            )
        if self.symmetry == "half_y0":
            if angles[0] != 270.0 or angles[-1] != 450.0:
                problems.append(
                    f"stations[{name}].split_points: half_y0 spans 270 to 450 degrees inclusive"
                )
            if parameters[0] != 0.0 or parameters[-1] != 1.0:
                problems.append(f"stations[{name}].split_points: half_y0 t spans 0 to 1")
        elif self.symmetry == "full" and angles[-1] >= 360.0:
            problems.append(f"stations[{name}].split_points: full-body angles lie in [0, 360)")

    def _construction_problems(self, problems: list[str]) -> None:
        regions = self.construction.regions
        method = self.construction.method
        if method == "revolution" and self.kind != "revolution":
            problems.append(
                "construction.method: the revolution method belongs to the revolution kind"
            )
        if method != "hybrid":
            if regions:
                problems.append("construction.regions: absent unless the method is hybrid")
            return
        if len(regions) < 2:
            problems.append("construction.regions: a hybrid construction needs at least two")
            return
        ids = {s.id for s in self.stations}
        coords = [s.coordinate for s in self.stations]
        for index, region in enumerate(regions):
            low, high = region.x_range
            if not low < high:
                problems.append(f"construction.regions[{index}].x_range: increasing")
            if index == 0 and region.join_section_curve is not None:
                problems.append("construction.regions[0]: the first region has no join curve")
            if index > 0:
                if region.join_section_curve is None:
                    problems.append(f"construction.regions[{index}]: join_section_curve required")
                elif (
                    isinstance(region.join_section_curve, str)
                    and region.join_section_curve not in ids
                ):
                    problems.append(
                        f"construction.regions[{index}].join_section_curve: not a station id"
                    )
                if low != regions[index - 1].x_range[1]:
                    problems.append(
                        f"construction.regions[{index}].x_range: contiguous with the previous"
                    )
        if coords and (regions[0].x_range[0] > min(coords) or regions[-1].x_range[1] < max(coords)):
            problems.append("construction.regions: must cover the station extent")

    def _deviation_problems(self, problems: list[str]) -> None:
        dev = self.measured_deviation
        constants = DEVIATION_PROFILES.get(dev.limit.profile)
        if constants is None:
            problems.append(
                f"measured_deviation.limit.profile: {dev.limit.profile!r} unknown; "
                f"available: {sorted(DEVIATION_PROFILES)}"
            )
        elif (dev.limit.floor_mm, dev.limit.local_fraction) != (
            constants["floor_mm"],
            constants["local_fraction"],
        ):
            problems.append("measured_deviation.limit: constants differ from the named profile")
        coords = [s.coordinate for s in self.stations]
        if coords and not math.isclose(
            dev.stopping_rule.component_length_mm,
            coords[-1] - coords[0],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            problems.append("stopping_rule.component_length_mm: must equal the station extent")
        order = {s.id: i for i, s in enumerate(self.stations)}
        for interval in dev.stopping_rule.unresolved_intervals:
            first, second = interval.station_ids
            if first not in order or second not in order or abs(order[first] - order[second]) != 1:
                problems.append("stopping_rule.unresolved_intervals: must name adjacent stations")

    def referenced_files(self) -> list[FileRef]:
        """Every pinned file of the record, deduplicated by path and hash, in first-use order."""
        found: list[FileRef] = [self.source_iges]

        def add(reference: FileRef) -> None:
            if reference not in found:
                found.append(reference)

        for station in self.stations:
            add(station.section_curve.file)
        for region in self.construction.regions:
            if isinstance(region.join_section_curve, CurveRef):
                add(region.join_section_curve.file)
        if isinstance(self.te_base, TeBaseFinite):
            add(self.te_base.surface.file)
        add(self.measured_deviation.evidence)
        return found

    def to_json(self) -> str:
        """Serialize to strict JSON, the interchange form."""
        data = self.model_dump(mode="json")
        if data["symmetry"] is None:
            del data["symmetry"]
        for region in data["construction"]["regions"]:
            if region["join_section_curve"] is None:
                del region["join_section_curve"]
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> ComponentRecord:
        """Parse and validate the interchange form, refusing with the validation message."""
        try:
            return cls.model_validate_json(text)
        except ValidationError as error:
            raise InputValueError(f"component record: {error}") from error
