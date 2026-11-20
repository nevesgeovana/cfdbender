"""Read the post-processing products a p file declares, with pyflightstream's keys.

Three tables of ``inputs/pproc/p<id>.toml`` declare products the post writes from the
solver's surface and volume files, solver agnostic:

* ``[sections]`` with ``[[sections.distributions]]`` entries (``families``, ``frame``,
  ``planes``, ``count``, ``plot_direction``, ``integrate``); the table states ``count``
  (50), ``plot_direction`` (1) and ``include_symmetry`` (false) for every entry that states
  none;
* one ``[volume_section]``: a rectangle (``corners_m``, ``refinement_layers``) or a circle
  (``radii_m``, ``points``) in a ``plane`` of a ``frame`` (``MRP``) at ``offset_m``, written
  as ``vtk`` or ``tecplot``; ``kind`` is ``instantaneous`` (every exported step) or
  ``default`` (the last step);
* ``[[probes]]`` entries: ``lines`` (``points`` each), ``rectangles`` and ``circles`` in a
  ``frame`` (empty: the row's rotor hub frame) and a ``scale`` (``m`` or ``rotor_radius``),
  sampling the ``parameters``; ``kind`` is ``default`` (the last step) or ``instantaneous``
  (every exported step).

The keys, the defaults and the refusals are pyflightstream's; a malformed table is refused
with its message. A table the solver's adapter does not supply is warned and skipped, and
so is a key pyflightstream offers that cfdbender does not (``attach = "body"``, a probe
entry's ``field_formats``, ``reusable_inflow``, ``lattices``, ``grids`` and
``points_file``, and ``include_symmetry = true``). Which frame or family a name stands for
is judged when the products are written, from the reference file and the solver's files.
Standard library only: a plan reads this module.
"""

from __future__ import annotations

import difflib
import math
import re
import string
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

from cfdbender.core.refusals import warn_or_refuse

__all__ = [
    "AXES_PLOT_COMPONENTS",
    "AXES_PLOT_GROUP",
    "DEFAULT_FORCE_PLOTS",
    "FORCE_PLOT_PARAMETERS",
    "FLUID_PLOT_PARAMETERS",
    "PlotGroup",
    "Plots",
    "read_plots",
    "PLANES",
    "PRODUCT_TABLES",
    "PROBES",
    "PROBE_KINDS",
    "SECTIONS",
    "VOLUME_SECTION",
    "PostDeclarations",
    "PprocError",
    "ProbeCircle",
    "ProbeLine",
    "ProbeRectangle",
    "Probes",
    "SectionDistribution",
    "Sections",
    "VolumeSection",
    "read_post",
]

SECTIONS = "sections"
VOLUME_SECTION = "volume_section"
PROBES = "probes"
#: The p-file tables that declare a product, in the order the post writes them.
PRODUCT_TABLES: tuple[str, ...] = (SECTIONS, VOLUME_SECTION, PROBES)
#: The planes of a frame, named by the two axes they hold.
PLANES: tuple[str, ...] = ("XY", "XZ", "YZ")
#: How a time-accurate row samples a probe entry or a volume section.
PROBE_KINDS: tuple[str, ...] = ("default", "instantaneous")
#: The parameters a probe entry may sample, pyflightstream's fluid plot parameters.
FLUID_PLOT_PARAMETERS: tuple[str, ...] = (
    "CP_FREE",
    "CP_REF",
    "MACH",
    "VELOCITY",
    "VX",
    "VY",
    "VZ",
    "STATIC_PRESSURE_RATIO",
    "BL_MOMENTUM_THICKNESS",
    "BL_DISPLACEMENT_THICKNESS",
    "BL_TOTAL_THICKNESS",
    "BL_SHAPE_FACTOR",
    "BL_SKIN_FRICTION",
    "BL_TRANSITION_MARKER",
)
_EACH_BLADE = (
    "families = 'each_blade' of a post-processing entry was renamed to frame = 'LOCAL_AXIS' "
    "in v0.15.0 and is no longer accepted. Write frame = 'LOCAL_AXIS'. The FRAME decides how "
    "an entry expands, and a blade's own axes are one per blade, so the selector that said so "
    "is a second statement of one fact."
)
_PROPELLER_RADIUS = (
    'scale = "propeller_radius" of a probe table is no longer accepted since v0.15.0. Write '
    'scale = "rotor_radius". This package says ROTOR everywhere, one word for the block, the '
    "key, the frame and the point: a propeller is a rotor, and so is a lift fan, so the "
    "general word is the one that never has to change again."
)
_PROBES_TABLE = (
    "`[probes]` is a LIST of tables, so that one artifact can probe several frames on one "
    "row, as `[[plots.groups]]` and `[[sections.distributions]]` already do. Write "
    "`[[probes]]` instead of `[probes]`, once per frame you are sampling in; everything inside "
    "the table is unchanged, `frame`, `scale`, `parameters`, `points` and the "
    "`[[probes.lines]]` beneath it."
)
_FRAME_EMPTY = (
    "frame names one the run creates (MRP, BLADE_AXIS, LOCAL_AXIS, or a rotor's own "
    "<ALIAS>_SMRP and <ALIAS>_RMRP) or one the REFERENCE's [[frames]] table declares; it is "
    "empty"
)
#: Probe-entry keys pyflightstream offers and cfdbender skips with a warning.
_PROBE_SKIPPED = ("field_formats", "reusable_inflow", "lattices", "grids", "points_file")


class PprocError(ValueError):
    """A product table of the p file that pyflightstream would refuse."""


@dataclass(frozen=True)
class SectionDistribution:
    """One ``[[sections.distributions]]`` entry with ``count`` and ``plot_direction`` resolved.

    ``position`` is the entry's 1-based place in the p file; ``families`` is the word or the
    list as written.
    """

    families: str | tuple[str, ...]
    frame: str
    planes: tuple[str, ...]
    count: int
    plot_direction: int
    integrate: bool
    position: int


@dataclass(frozen=True)
class Sections:
    """The ``[sections]`` table."""

    count: int = 50
    plot_direction: int = 1
    include_symmetry: bool = False
    distributions: tuple[SectionDistribution, ...] = ()


@dataclass(frozen=True)
class VolumeSection:
    """The ``[volume_section]`` table, lengths in meters in ``frame``."""

    shape: str
    plane: str
    kind: str = "instantaneous"
    frame: str = "MRP"
    offset_m: float = 0.0
    corners_m: tuple[float, float, float, float] | None = None
    refinement_layers: int = 1
    radii_m: tuple[float, float] | None = None
    points: tuple[int, int] | None = None
    format: str = "vtk"


@dataclass(frozen=True)
class ProbeLine:
    """One probe line from ``start`` to ``end``, in the entry's frame and scale."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]


@dataclass(frozen=True)
class ProbeRectangle:
    """A probe plane given by three corners and the stations toward two of them."""

    origin: tuple[float, float, float]
    along_u: tuple[float, float, float]
    along_v: tuple[float, float, float]
    points_u: int
    points_v: int


@dataclass(frozen=True)
class ProbeCircle:
    """A polar probe disk: center, normal, radius and its stations."""

    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float
    points_radial: int
    points_azimuth: int


@dataclass(frozen=True)
class Probes:
    """One ``[[probes]]`` entry; ``position`` is its 1-based place in the p file."""

    parameters: tuple[str, ...]
    frame: str = ""
    points: int = 25
    scale: str = "m"
    lines: tuple[ProbeLine, ...] = ()
    rectangles: tuple[ProbeRectangle, ...] = ()
    circles: tuple[ProbeCircle, ...] = ()
    kind: str = "default"
    position: int = 1


@dataclass(frozen=True)
class PostDeclarations:
    """The product tables of one p file that the solver supplies, and what was skipped."""

    sections: Sections | None = None
    volume_section: VolumeSection | None = None
    probes: tuple[Probes, ...] = ()
    skipped: tuple[str, ...] = field(default=())

    @property
    def products(self) -> tuple[str, ...]:
        """The product tables declared and kept, in :data:`PRODUCT_TABLES` order."""
        kept = {
            SECTIONS: self.sections is not None,
            VOLUME_SECTION: self.volume_section is not None,
            PROBES: bool(self.probes),
        }
        return tuple(name for name in PRODUCT_TABLES if kept[name])


def _fail(loc: str, text: str) -> PprocError:
    return PprocError(f"{loc}: {text}" if loc else text)


def _table(value: object, loc: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail(loc, "Input should be a valid dictionary")
    return value


def _known(table: Mapping[str, object], keys: Sequence[str], loc: str) -> None:
    for key in table:
        if key not in keys:
            close = difflib.get_close_matches(str(key), sorted(keys), n=1, cutoff=0.75)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise _fail(f"{loc}.{key}", f"Extra inputs are not permitted.{hint}")


def _int(value: object, loc: str, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(loc, "Input should be a valid integer")
    if value < least:
        raise _fail(loc, f"Input should be greater than or equal to {least}")
    return value


def _float(value: object, loc: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(loc, "Input should be a valid number")
    if not math.isfinite(float(value)):
        raise _fail(loc, "Input should be a finite number")
    return float(value)


def _bool(value: object, loc: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(loc, "Input should be a valid boolean")
    return value


def _choice(value: object, loc: str, choices: Sequence[object]) -> object:
    if value not in choices or isinstance(value, bool) != isinstance(choices[0], bool):
        listed = " or ".join(repr(c) for c in choices)
        raise _fail(loc, f"Input should be {listed}")
    return value


def _numbers(value: object, loc: str, count: int) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != count:
        raise _fail(loc, f"Tuple should have {count} items")
    return tuple(_float(v, f"{loc}.{i}") for i, v in enumerate(value))


def _frame(value: object, loc: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _fail(loc, "Input should be a valid string")
    if not value.strip():
        if empty:
            return ""
        raise _fail(loc, _FRAME_EMPTY)
    return value


def _families(value: object, loc: str) -> str | tuple[str, ...]:
    if isinstance(value, str):
        if not value.strip():
            raise _fail(
                loc,
                "families is a selector word, an alias of the setup, a family name, or a "
                "non-empty list of those; it is empty",
            )
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        if any(item in ("each", "each_blade") for item in value):
            raise _fail(
                loc,
                "'each' and 'each_blade' expand an entry and stand alone. Remove the other "
                "words from that list.",
            )
        if any(not item.strip() for item in value):
            raise _fail(loc, "families lists an empty word. Remove the empty word.")
        return tuple(value)
    raise _fail(
        loc,
        "families is a selector word, an alias of the setup, a family name, or a non-empty "
        "list of those",
    )


def _planes(value: object, loc: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _fail(loc, "Input should be a valid list")
    if not value:
        raise _fail(loc, "List should have at least 1 item after validation, not 0")
    return tuple(str(_choice(v, f"{loc}.{i}", PLANES)) for i, v in enumerate(value))


def _kind(value: object, loc: str, what: str, warn: Callable[[str, str], None]) -> str:
    aliases = {"normal": "default", "unsteady": "instantaneous"}
    if isinstance(value, str) and value in aliases:
        warn(
            f"{loc}|{value}",
            f"{what} kind {value!r} is deprecated; use kind = {aliases[value]!r} instead.",
        )
        return aliases[value]
    if value not in PROBE_KINDS:
        if what == "probe":
            tail = (
                'write kind = "instantaneous" for a fluid plot per point and parameter, '
                'evaluated at every time step, or kind = "default" for probe points '
                "sampled once, at the last time step."
            )
        else:
            tail = (
                'write kind = "instantaneous" for fluid plots at every time step, or kind = '
                '"default" for probe points sampled once, at the last time step.'
            )
        raise _fail(
            loc,
            f"{what} kind {value!r} is not one of 'default' or 'instantaneous': {tail} "
            "Deprecated aliases: 'unsteady' and 'normal', respectively.",
        )
    return str(value)


def _attach(table: Mapping[str, object], loc: str) -> bool:
    """Return whether the entry is attached in space; ``body`` is skipped by the caller."""
    return _choice(table.get("attach", "space"), f"{loc}.attach", ("space", "body")) == "space"


def _sections(value: object, loc: str, warn: Callable[[str, str], None]) -> Sections:
    table = _table(value, loc)
    _known(table, ("count", "plot_direction", "include_symmetry", "distributions"), loc)
    count = _int(table.get("count", 50), f"{loc}.count", 1)
    direction = int(
        _as_int(_choice(table.get("plot_direction", 1), f"{loc}.plot_direction", (1, 2)))
    )
    symmetry = _bool(table.get("include_symmetry", False), f"{loc}.include_symmetry")
    if symmetry:
        warn(
            f"{loc}.include_symmetry",
            f"{loc}: include_symmetry = true is skipped; cfdbender makes no symmetry copy of a "
            "section, so each distribution cuts the meshed surface only",
        )
    raw = table.get("distributions", [])
    if not isinstance(raw, list):
        raise _fail(f"{loc}.distributions", "Input should be a valid list")
    entries: list[SectionDistribution] = []
    for index, item in enumerate(raw):
        where = f"{loc}.distributions.{index}"
        entry = _table(item, where)
        _known(
            entry,
            ("families", "frame", "planes", "count", "plot_direction", "integrate", "attach"),
            where,
        )
        if "families" not in entry:
            raise _fail(f"{where}.families", "Field required")
        if "planes" not in entry:
            raise _fail(f"{where}.planes", "Field required")
        families = _families(entry["families"], f"{where}.families")
        planes = _planes(entry["planes"], f"{where}.planes")
        frame = _frame(entry.get("frame", "MRP"), f"{where}.frame")
        own = entry.get("count")
        own_count = count if own is None else _int(own, f"{where}.count", 1)
        own_dir = entry.get("plot_direction")
        direction_of = (
            direction
            if own_dir is None
            else int(_as_int(_choice(own_dir, f"{where}.plot_direction", (1, 2))))
        )
        integrate = _bool(entry.get("integrate", False), f"{where}.integrate")
        if families == "each_blade":
            raise _fail(where, f"section distribution over {list(planes)}: {_EACH_BLADE}")
        if not _attach(entry, where):
            warn(
                where, f"{where}: attach = 'body' is skipped; cfdbender cuts the surface as written"
            )
            continue
        entries.append(
            SectionDistribution(
                families, frame, planes, own_count, direction_of, integrate, index + 1
            )
        )
    return Sections(count, direction, False, tuple(entries))


def _as_int(value: object) -> int:
    """Return an integer choice as an int (``_choice`` has checked it)."""
    return value if isinstance(value, int) else 0


def _volume_section(
    value: object, loc: str, warn: Callable[[str, str], None]
) -> VolumeSection | None:
    table = _table(value, loc)
    keys = (
        "kind",
        "shape",
        "frame",
        "plane",
        "offset_m",
        "corners_m",
        "refinement_layers",
        "radii_m",
        "points",
        "format",
        "attach",
    )
    _known(table, keys, loc)
    for required in ("shape", "plane"):
        if required not in table:
            raise _fail(f"{loc}.{required}", "Field required")
    kind = _kind(table.get("kind", "instantaneous"), f"{loc}.kind", "volume section", warn)
    shape = str(_choice(table["shape"], f"{loc}.shape", ("rectangle", "circle")))
    plane = str(_choice(table["plane"], f"{loc}.plane", PLANES))
    frame = _frame(table.get("frame", "MRP"), f"{loc}.frame")
    offset = _float(table.get("offset_m", 0.0), f"{loc}.offset_m")
    corners = (
        None
        if table.get("corners_m") is None
        else _numbers(table["corners_m"], f"{loc}.corners_m", 4)
    )
    layers = _int(table.get("refinement_layers", 1), f"{loc}.refinement_layers", 1)
    radii = (
        None if table.get("radii_m") is None else _numbers(table["radii_m"], f"{loc}.radii_m", 2)
    )
    points: tuple[int, int] | None = None
    if table.get("points") is not None:
        raw = table["points"]
        if not isinstance(raw, list | tuple) or len(raw) != 2:
            raise _fail(f"{loc}.points", "Tuple should have 2 items")
        first, second = (_int(v, f"{loc}.points.{i}", 1) for i, v in enumerate(raw))
        points = (first, second)
    fmt = str(_choice(table.get("format", "vtk"), f"{loc}.format", ("vtk", "tecplot")))
    if points is not None and (min(points) < 2 or (shape == "circle" and points[1] < 3)):
        raise _fail(
            loc,
            "volume grid needs at least 2 samples per axis and 3 around a circle. State the "
            "larger sample counts.",
        )
    own = {"rectangle": ("corners_m", "refinement_layers"), "circle": ("radii_m", "points")}
    other = "circle" if shape == "rectangle" else "rectangle"
    stray = [
        key
        for key in own[other]
        if key != "points"
        and table.get(key) is not None
        and not (key == "refinement_layers" and layers == 1)
    ]
    if stray:
        raise _fail(
            loc,
            f"[volume_section] shape = {shape!r} states {', '.join(stray)}, which "
            f"{'is' if len(stray) == 1 else 'are'} a {other}'s; a {shape} takes "
            f"{' and '.join(own[shape])}",
        )
    stated = {"corners_m": corners, "radii_m": radii, "points": points}
    needed = [k for k in own[shape] if k != "refinement_layers" and stated[k] is None]
    if needed:
        raise _fail(
            loc,
            f"[volume_section] shape = {shape!r} needs {' and '.join(needed)}; "
            + (
                "a rectangle takes corners_m, the two diagonal corners x1, y1, x2, y2 in the "
                "plane, in meters"
                if shape == "rectangle"
                else "a circle takes radii_m, the inner and outer radius r1, r2 in meters, and "
                "points, the ipts radial and jpts azimuthal segments"
            ),
        )
    if corners is not None:
        x1, y1, x2, y2 = corners
        if x1 == x2 or y1 == y2:
            raise _fail(
                loc,
                f"[volume_section] corners_m = {list(corners)} enclose no area: two diagonal "
                "corners differ in both coordinates. State two corners that differ in both "
                "coordinates.",
            )
    if radii is not None and not 0.0 <= radii[0] < radii[1]:
        raise _fail(
            loc,
            f"[volume_section] radii_m = {list(radii)}: the inner radius r1 is at least 0 and "
            "smaller than the outer r2. State 0 <= r1 < r2.",
        )
    if not _attach(table, loc):
        warn(loc, f"{loc}: attach = 'body' is skipped; cfdbender samples the plane in space")
        return None
    return VolumeSection(
        shape,
        plane,
        kind,
        frame,
        offset,
        None if corners is None else (corners[0], corners[1], corners[2], corners[3]),
        layers,
        None if radii is None else (radii[0], radii[1]),
        points,
        fmt,
    )


def _vec3(value: object, loc: str) -> tuple[float, float, float]:
    a, b, c = _numbers(value, loc, 3)
    return (a, b, c)


def _shapes(entry: Mapping[str, object], where: str) -> tuple[list[object], ...]:
    out: dict[str, list[object]] = {"lines": [], "rectangles": [], "circles": []}
    keys = {
        "lines": ("start", "end"),
        "rectangles": ("origin", "along_u", "along_v", "points_u", "points_v"),
        "circles": ("center", "normal", "radius", "points_radial", "points_azimuth"),
    }
    for name, fields in keys.items():
        raw = entry.get(name, [])
        if not isinstance(raw, list):
            raise _fail(f"{where}.{name}", "Input should be a valid list")
        for index, item in enumerate(raw):
            loc = f"{where}.{name}.{index}"
            shape = _table(item, loc)
            _known(shape, fields, loc)
            for key in fields:
                if key not in shape:
                    raise _fail(f"{loc}.{key}", "Field required")
            if name == "lines":
                out[name].append(
                    ProbeLine(
                        _vec3(shape["start"], f"{loc}.start"), _vec3(shape["end"], f"{loc}.end")
                    )
                )
            elif name == "rectangles":
                origin = _vec3(shape["origin"], f"{loc}.origin")
                u = _vec3(shape["along_u"], f"{loc}.along_u")
                v = _vec3(shape["along_v"], f"{loc}.along_v")
                pu = _int(shape["points_u"], f"{loc}.points_u", 2)
                pv = _int(shape["points_v"], f"{loc}.points_v", 2)
                first = [b - a for a, b in zip(origin, u, strict=True)]
                second = [b - a for a, b in zip(origin, v, strict=True)]
                cross = (
                    first[1] * second[2] - first[2] * second[1],
                    first[2] * second[0] - first[0] * second[2],
                    first[0] * second[1] - first[1] * second[0],
                )
                if sum(x * x for x in cross) <= 0.0:
                    raise _fail(
                        loc,
                        f"a probe rectangle at {origin} spans no plane: its three corners are "
                        "collinear or two of them coincide, so there is no rectangle to grid. "
                        "`origin`, `along_u` and `along_v` are three DIFFERENT corners and the "
                        "fourth follows from them. Give three different corners.",
                    )
                out[name].append(ProbeRectangle(origin, u, v, pu, pv))
            else:
                center = _vec3(shape["center"], f"{loc}.center")
                normal = _vec3(shape["normal"], f"{loc}.normal")
                radius = _float(shape["radius"], f"{loc}.radius")
                if radius <= 0.0:
                    raise _fail(f"{loc}.radius", "Input should be greater than 0")
                pr = _int(shape["points_radial"], f"{loc}.points_radial", 2)
                pa = _int(shape["points_azimuth"], f"{loc}.points_azimuth", 3)
                if sum(x * x for x in normal) <= 0.0:
                    raise _fail(
                        loc,
                        f"a probe circle at {center} states `normal = {normal}`, which is no "
                        "direction and so names no plane. The normal is the axis the disk is "
                        "perpendicular to. State a non-zero normal.",
                    )
                out[name].append(ProbeCircle(center, normal, radius, pr, pa))
    return out["lines"], out["rectangles"], out["circles"]


def _probes(value: object, loc: str, warn: Callable[[str, str], None]) -> tuple[Probes, ...]:
    if isinstance(value, Mapping):
        raise _fail(loc, _PROBES_TABLE)
    if not isinstance(value, list):
        raise _fail(loc, "Input should be a valid list")
    keys = (
        "frame",
        "parameters",
        "field_formats",
        "reusable_inflow",
        "points",
        "scale",
        "lines",
        "rectangles",
        "circles",
        "lattices",
        "grids",
        "points_file",
        "kind",
        "attach",
    )
    entries: list[Probes] = []
    for index, item in enumerate(value):
        where = f"{loc}.{index}"
        entry = _table(item, where)
        _known(entry, keys, where)
        frame = _frame(entry.get("frame", ""), f"{where}.frame", empty=True)
        raw = entry.get("parameters", [])
        if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
            raise _fail(f"{where}.parameters", "Input should be a valid list of strings")
        unknown = [name for name in raw if name not in FLUID_PLOT_PARAMETERS]
        if unknown:
            raise _fail(
                f"{where}.parameters",
                f"fluid plot parameter(s) {', '.join(unknown)} are not among "
                f"{', '.join(FLUID_PLOT_PARAMETERS)}. Use one of the parameters listed.",
            )
        points = _int(entry.get("points", 25), f"{where}.points", 2)
        scale = entry.get("scale", "m")
        if scale == "propeller_radius":
            raise _fail(f"{where}.scale", _PROPELLER_RADIUS)
        scale = str(_choice(scale, f"{where}.scale", ("m", "rotor_radius")))
        kind = _kind(entry.get("kind", "default"), f"{where}.kind", "probe", warn)
        lines, rectangles, circles = _shapes(entry, where)
        parameters = list(raw)
        formats = entry.get("field_formats", [])
        if formats or entry.get("reusable_inflow", False):
            parameters = list(dict.fromkeys([*parameters, "VX", "VY", "VZ"]))
        drawn = len(lines) + len(rectangles) + len(circles)
        for key in ("lattices", "grids"):
            stated = entry.get(key, [])
            drawn += len(stated) if isinstance(stated, list) else 0
        if entry.get("points_file") and drawn:
            raise _fail(
                where,
                f"a probe entry in {frame or 'the rotor hub frame'} states both `points_file = "
                f"{entry['points_file']!r}` and {drawn} shape(s) of its own, which is the "
                "survey written twice with nothing keeping the two in agreement. State the "
                "shapes, or cite the file and delete them.",
            )
        if (drawn or entry.get("points_file")) and not parameters:
            raise _fail(
                where,
                "a probe entry drawing or citing points requires nonempty parameters; state "
                "the variables to sample, or request field_formats or reusable_inflow to "
                "sample VX, VY and VZ",
            )
        for key in _PROBE_SKIPPED:
            if entry.get(key):
                warn(
                    f"{where}.{key}",
                    f"{where}.{key} is skipped; cfdbender samples the lines, rectangles and "
                    "circles an entry draws and writes the probes table only",
                )
        if not _attach(entry, where):
            warn(where, f"{where}: attach = 'body' is skipped; cfdbender samples probes in space")
            continue
        if not (lines or rectangles or circles):
            continue
        entries.append(
            Probes(
                tuple(parameters),
                frame,
                points,
                scale,
                tuple(lines),  # type: ignore[arg-type]
                tuple(rectangles),  # type: ignore[arg-type]
                tuple(circles),  # type: ignore[arg-type]
                kind,
                index + 1,
            )
        )
    return tuple(entries)


def read_post(
    post: Mapping[str, object],
    *,
    supported: Collection[str] = PRODUCT_TABLES,
    where: str = "the PPROC file",
    supplier: str = "the solver",
    warn: Callable[[str, str], None] | None = None,
) -> PostDeclarations:
    """Read the product tables of a p file; refuse a malformed one, skip an unsupplied one.

    Parameters
    ----------
    post : mapping
        The parsed p file.
    supported : collection of str
        The products the solver's adapter supplies (see :mod:`cfdbender.core.contract`).
    where : str
        The file's name in messages, for example ``inputs/pproc/p1.toml``.
    supplier : str
        Who does not supply a skipped table, for example ``solver 'openfoam'``.
    warn : callable, optional
        ``warn(key, message)`` for a skip; the key identifies it so that a caller warns once.
        By default each skip goes through :func:`cfdbender.core.refusals.warn_or_refuse`.

    Returns
    -------
    PostDeclarations
        The kept tables and the text of every skip.

    Raises
    ------
    PprocError
        A table pyflightstream refuses, with its message.
    """
    said: list[str] = []

    def skip(key: str, message: str) -> None:
        said.append(message)
        if warn is None:
            warn_or_refuse(message, UserWarning, stacklevel=4)
        else:
            warn(key, message)

    loc = {SECTIONS: "[sections]", VOLUME_SECTION: "[volume_section]", PROBES: "[[probes]]"}
    found: dict[str, object] = {}
    for table in PRODUCT_TABLES:
        if table not in post:
            continue
        reader = {SECTIONS: _sections, VOLUME_SECTION: _volume_section, PROBES: _probes}[table]
        try:
            found[table] = reader(post[table], loc[table], skip)
        except PprocError as error:
            raise PprocError(f"{where} {error}") from None
        if table not in supported:
            offered = ", ".join(t for t in PRODUCT_TABLES if t in supported) or "none"
            skip(
                f"{where}|{table}",
                f"{loc[table]} of {where} is skipped; {supplier} does not supply the files it is "
                f"written from (products it supplies: {offered})",
            )
            found.pop(table)
    probes = found.get(PROBES, ())
    return PostDeclarations(
        found.get(SECTIONS),  # type: ignore[arg-type]
        found.get(VOLUME_SECTION),  # type: ignore[arg-type]
        probes,  # type: ignore[arg-type]
        tuple(said),
    )


#: The force-plot parameters of ``[plots]``, pyflightstream's: the plot is named
#: ``<parameter>_<group>``.
FORCE_PLOT_PARAMETERS: tuple[str, ...] = (
    "CL",
    "CDI",
    "CDO",
    "CDP",
    "CDV",
    "CD",
    "FX",
    "FY",
    "FZ",
    "MX",
    "MY",
    "MZ",
)
#: What a group plots when ``[plots]`` names no parameter: all but the drag split CDP and CDV.
DEFAULT_FORCE_PLOTS: tuple[str, ...] = tuple(
    name for name in FORCE_PLOT_PARAMETERS if name not in ("CDP", "CDV")
)
#: The six components of the axes of a polar; a run adds ``MRP_TOTAL`` when no group plots them.
AXES_PLOT_COMPONENTS: tuple[str, ...] = ("FX", "FY", "FZ", "MX", "MY", "MZ")
AXES_PLOT_GROUP = "MRP_TOTAL"
#: The frames whose group is one per rotor or per blade, in that rotor's or blade's own axes.
EXPANDING_FRAMES: dict[str, str] = {"SMRP": "rotor", "RMRP": "rotor", "LOCAL_AXIS": "blade"}
PLOTS = "plots"


@dataclass(frozen=True)
class PlotGroup:
    """One ``[[plots.groups]]`` entry: a name, a frame and the families summed."""

    name: str
    frame: str
    families: str | tuple[str, ...]

    @property
    def emits_per(self) -> str:
        """Return ``once``, ``rotor``, ``blade`` or ``family``: what the entry is one of."""
        kind = EXPANDING_FRAMES.get(self.frame.strip().upper())
        if kind is not None:
            return kind
        return "family" if self.families == "each" else "once"


@dataclass(frozen=True)
class Plots:
    """The ``[plots]`` table: the parameters and the groups, in the order written."""

    parameters: tuple[str, ...] = DEFAULT_FORCE_PLOTS
    groups: tuple[PlotGroup, ...] = ()
    #: ``[plots.families]``: each named family and the wall markers it holds, in order.
    families: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _plot_group(value: object, loc: str) -> PlotGroup:
    table = _table(value, loc)
    _known(table, ("name", "frame", "families"), loc)
    if "name" not in table:
        raise _fail(f"{loc}.name", "Field required")
    if "families" not in table:
        raise _fail(f"{loc}.families", "Field required")
    name = table["name"]
    if not isinstance(name, str) or not name:
        raise _fail(f"{loc}.name", "Input should be a non-empty string")
    frame = _frame(table.get("frame", "MRP"), f"{loc}.frame")
    families = _families(table["families"], f"{loc}.families")
    if families == "each_blade":
        raise _fail(
            loc,
            f"plot group {name!r}: families = 'each_blade' was renamed to frame = 'LOCAL_AXIS'. "
            "Change the key named above.",
        )
    group = PlotGroup(name, frame, families)
    if (group.emits_per != "once") != ("{family}" in name):
        expands = (
            f"an entry in {frame} is one per {group.emits_per}"
            if group.emits_per != "once"
            else f"an entry in {frame} is ONE group over the families it names"
        )
        carries = "carries" if group.emits_per != "once" else "carries no"
        raise _fail(
            loc,
            f"plot group {name!r}: {expands}, so its name {carries} {{family}}, which names "
            f"what each emission is about. The frames that expand are "
            f"{', '.join(sorted(EXPANDING_FRAMES))}, and the selector 'each' expands in a "
            "common frame.",
        )
    fields = [f for _, f, _, _ in string.Formatter().parse(name) if f is not None]
    if any(f != "family" for f in fields) or re.search(r"\{family[^}]", name):
        raise _fail(
            loc,
            f"plot group {name!r}: a plot name may carry only the bare placeholder {{family}}, "
            "with no format spec and no conversion",
        )
    if re.fullmatch(r"(?:[A-Za-z0-9_]|\{family\})+", name) is None:
        raise _fail(
            loc,
            f"plot group {name!r}: a plot name is letters, digits and underscores, and the "
            "bare placeholder {family}; no space and no other character",
        )
    if EXPANDING_FRAMES.get(frame.strip().upper()) and families == "each":
        raise _fail(
            loc,
            f"plot group {name!r}: the frame {frame} already expands, one per "
            f"{group.emits_per}, and families = 'each' expands too, so the entry states how "
            "many it is twice. Name the rotors or the alias whose families they own, or 'all' "
            "for every rotor; write families = 'each' only in a frame that does not expand.",
        )
    return group


def _plot_families(value: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Read ``[plots.families]``: a family name to the wall markers it holds, no marker in two."""
    loc = "[plots].families"
    table = _table(value, loc)
    found: list[tuple[str, tuple[str, ...]]] = []
    claimed_by: dict[str, str] = {}
    for name, markers in table.items():
        if re.fullmatch(r"[A-Za-z0-9_]+", str(name)) is None:
            raise _fail(
                f"{loc}.{name}", "a family name is letters, digits and underscores, one word"
            )
        if (
            not isinstance(markers, list)
            or not markers
            or not all(isinstance(m, str) and m.strip() for m in markers)
        ):
            raise _fail(f"{loc}.{name}", "Input should be a non-empty list of marker names")
        if len(set(markers)) != len(markers):
            raise _fail(f"{loc}.{name}", "a marker is repeated")
        for marker in markers:
            if marker in claimed_by:
                raise _fail(
                    f"{loc}.{name}",
                    f"marker {marker!r} is claimed by two families, {claimed_by[marker]!r} and "
                    f"{name!r}; a family is a set of markers and no marker is in two",
                )
            claimed_by[marker] = str(name)
        found.append((str(name), tuple(markers)))
    return tuple(found)


def read_plots(post: Mapping[str, object], *, where: str = "the PPROC file") -> Plots | None:
    """Read the ``[plots]`` table of a p file with pyflightstream's keys, or None without one.

    ``parameters`` defaults to every force parameter but ``CDP`` and ``CDV``; each group has
    a ``name`` (the plot is ``<parameter>_<name>``), a ``frame`` (default ``MRP``) and
    ``families``. A name carries ``{family}`` exactly when the entry expands.

    Raises
    ------
    PprocError
        A malformed table, with pyflightstream's message.
    """
    if PLOTS not in post:
        return None
    try:
        table = _table(post[PLOTS], "[plots]")
        _known(table, ("parameters", "groups", "families"), "[plots]")
        parameters: tuple[str, ...] = DEFAULT_FORCE_PLOTS
        if "parameters" in table:
            stated = table["parameters"]
            if not isinstance(stated, list) or not all(isinstance(p, str) for p in stated):
                raise _fail("[plots].parameters", "Input should be a valid list of strings")
            unknown = [p for p in stated if p not in FORCE_PLOT_PARAMETERS]
            if unknown:
                raise _fail(
                    "[plots].parameters",
                    f"force plot parameter(s) {', '.join(unknown)} are not among "
                    f"{', '.join(FORCE_PLOT_PARAMETERS)}. Use one of the parameters listed.",
                )
            parameters = tuple(dict.fromkeys(stated))
        entries = table.get("groups", [])
        if not isinstance(entries, list):
            raise _fail("[plots].groups", "Input should be a valid list")
        groups = tuple(
            _plot_group(entry, f"[plots].groups.{index}") for index, entry in enumerate(entries)
        )
        families = _plot_families(table.get("families", {}))
    except PprocError as error:
        raise PprocError(f"{where} {error}") from None
    return Plots(parameters, groups, families)
