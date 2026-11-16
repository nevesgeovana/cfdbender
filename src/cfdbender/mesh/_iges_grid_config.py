"""The grid control file of ``cfd-mesh iges-grid``: one table per component, read and checked.

``grid.toml`` holds a ``[defaults]`` table (``size``,
``tolerance``, ``te_angle_deg``, ``remove_inner_faces``) and one ``[component.<Name>]`` table
per component named in the IGES. A key that does not apply is warned and skipped; a value that
cannot be read is refused before anything is imported, with nothing written.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cfdbender._errors import InputValueError, NotApplicableWarning, warn

__all__ = [
    "LAWS",
    "TE_ROLES",
    "ComponentGrid",
    "Direction",
    "FaceLayout",
    "GridConfig",
    "Layout",
    "read_grid",
    "read_layout",
    "sidecar_path",
]

#: The spacing laws a direction may name.
LAWS = ("cosine", "geometric", "tanh", "uniform")
#: The ratio a law takes when none is given: geometric, the growth of a cell over the next one
#: away from the clustered end; tanh, the coefficient of the one-sided law (nearer 1 clusters
#: harder). cosine has none: its end-to-middle ratio follows the count.
DEFAULT_RATIO = {"geometric": 1.1, "tanh": 1.05}
#: The angle between the normals of two faces above which their shared edge is a trailing edge
#: when ``trailing_edge = "auto"`` and their bisector faces downstream: 60 degrees takes the sharp
#: trailing edge and both rims of a blunt one (each near 90 degrees), of which one is marked.
DEFAULT_TE_ANGLE_DEG = 60.0

_TOP = {"defaults", "component"}
_DEFAULTS = {"size", "tolerance", "te_angle_deg", "remove_inner_faces", "symmetry"}
_SYMMETRY = ("none", "x", "y", "z")
_COMPONENT = {
    "size",
    "chord",
    "span",
    "span_axis",
    "le_cluster",
    "te_cluster",
    "structured",
    "trailing_edge",
    "te_angle_deg",
}
_DIRECTION = {"count", "law", "ratio"}
_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise InputValueError(f"grid: {message}. Nothing was written.")


def _positive(value: object, where: str) -> float:
    _need(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0,
        f"{where} must be a positive number",
    )
    return float(value)  # type: ignore[arg-type]


def _skip(keys: set[str], allowed: set[str], where: str) -> None:
    for key in sorted(keys - allowed):
        warn(f"{where}.{key} does not apply and is skipped", NotApplicableWarning)


@dataclass(frozen=True)
class Direction:
    """The cells along one parameter direction of a four-sided patch and their spacing law."""

    count: int | None
    law: str = "uniform"
    ratio: float | None = None


@dataclass
class ComponentGrid:
    """The grid one component asks for."""

    name: str
    size: float | None = None
    chord: Direction | None = None
    span: Direction | None = None
    span_axis: tuple[float, float, float] | None = None
    le_cluster: float | None = None
    te_cluster: float | None = None
    structured: bool = True
    trailing_edge: str | list[str] | None = None
    te_angle_deg: float | None = None


@dataclass
class GridConfig:
    """The whole control file: defaults and the components in the file's order."""

    size: float | None
    tolerance: float | None
    te_angle_deg: float
    components: dict[str, ComponentGrid] = field(default_factory=dict)
    #: Remove the faces that lie inside the assembly where components cross (true by default).
    remove_inner_faces: bool = True
    #: The origin plane a half model is mirrored across for that removal: ``"none"`` (default),
    #: or the axis ``"x"``, ``"y"`` or ``"z"`` normal to it. Never inferred from the geometry.
    symmetry: str = "none"


def _direction(value: object, where: str) -> Direction:
    _need(
        isinstance(value, dict), f'{where} must be a table such as {{count = 40, law = "cosine"}}'
    )
    assert isinstance(value, dict)
    _skip(set(value), _DIRECTION, where)
    count = value.get("count")
    _need(
        count is None or (type(count) is int and count >= 1),
        f"{where}.count must be a whole number of cells, 1 or more",
    )
    law = value.get("law", "uniform")
    _need(law in LAWS, f"{where}.law must be one of {', '.join(LAWS)}")
    ratio = value.get("ratio")
    if ratio is not None:
        ratio = _positive(ratio, f"{where}.ratio")
        if law in ("geometric", "tanh"):
            _need(ratio >= 1.0, f"{where}.ratio of the {law} law must be 1 or more")
        if law == "uniform":
            warn(
                f"{where}.ratio does not apply to the uniform law and is skipped",
                NotApplicableWarning,
            )
            ratio = None
    return Direction(count=count, law=str(law), ratio=ratio)


def _axis(value: object, where: str) -> tuple[float, float, float]:
    if isinstance(value, str):
        _need(value.lower() in _AXES, f"{where} must be x, y, z or three numbers")
        return _AXES[value.lower()]
    _need(
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value),
        f"{where} must be x, y, z or three numbers",
    )
    assert isinstance(value, list)
    norm = math.sqrt(sum(float(v) ** 2 for v in value))
    _need(norm > 0, f"{where} must not be zero")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def _trailing(value: object, where: str) -> str | list[str] | None:
    if value is None or value is False:
        return None
    if value == "auto":
        return "auto"
    _need(
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, str) and bool(v) for v in value),
        f'{where} must be "auto" or a list of IGES curve names',
    )
    assert isinstance(value, list)
    return [str(v) for v in value]


def _component(name: str, table: object) -> ComponentGrid:
    where = f"component.{name}"
    _need(isinstance(table, dict), f"[{where}] must be a table")
    assert isinstance(table, dict)
    _skip(set(table), _COMPONENT, where)
    out = ComponentGrid(name=name)
    if "size" in table:
        out.size = _positive(table["size"], f"{where}.size")
    for key in ("chord", "span"):
        if key in table:
            setattr(out, key, _direction(table[key], f"{where}.{key}"))
    if "span_axis" in table:
        out.span_axis = _axis(table["span_axis"], f"{where}.span_axis")
    for key in ("le_cluster", "te_cluster"):
        if key in table:
            value = _positive(table[key], f"{where}.{key}")
            _need(value <= 1.0, f"{where}.{key} is the end cell over the middle cell, 1 or less")
            setattr(out, key, value)
    if "structured" in table:
        _need(isinstance(table["structured"], bool), f"{where}.structured must be true or false")
        out.structured = bool(table["structured"])
    out.trailing_edge = _trailing(table.get("trailing_edge"), f"{where}.trailing_edge")
    if "te_angle_deg" in table:
        out.te_angle_deg = _positive(table["te_angle_deg"], f"{where}.te_angle_deg")
    return out


def read_grid(path: str | Path) -> GridConfig:
    """Read and check a grid control file.

    Raises
    ------
    InputValueError
        The file is missing, not TOML, or a value cannot be read; nothing was written.
    """
    source = Path(path)
    _need(source.is_file(), f"{source} is not a file; name the grid control file")
    try:
        data: dict[str, Any] = tomllib.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise InputValueError(
            f"grid: {source} is not TOML ({error}). Nothing was written."
        ) from error
    _skip(set(data), _TOP, "grid")
    defaults = data.get("defaults", {})
    _need(isinstance(defaults, dict), "[defaults] must be a table")
    _skip(set(defaults), _DEFAULTS, "defaults")
    size = _positive(defaults["size"], "defaults.size") if "size" in defaults else None
    tolerance = (
        _positive(defaults["tolerance"], "defaults.tolerance") if "tolerance" in defaults else None
    )
    angle = (
        _positive(defaults["te_angle_deg"], "defaults.te_angle_deg")
        if "te_angle_deg" in defaults
        else DEFAULT_TE_ANGLE_DEG
    )
    inner = defaults.get("remove_inner_faces", True)
    _need(isinstance(inner, bool), "defaults.remove_inner_faces must be true or false")
    symmetry = defaults.get("symmetry", "none")
    _need(
        symmetry in _SYMMETRY,
        f"defaults.symmetry must be one of {', '.join(repr(v) for v in _SYMMETRY)}",
    )
    tables = data.get("component", {})
    _need(isinstance(tables, dict), "[component.<Name>] tables must be tables")
    components = {name: _component(name, table) for name, table in tables.items()}
    return GridConfig(
        size=size,
        tolerance=tolerance,
        te_angle_deg=angle,
        components=components,
        remove_inner_faces=inner,
        symmetry=symmetry,
    )


#: The roles of a layout sidecar's faces that are the blunt trailing-edge base of a lifting
#: surface; the trailing edge of the component is the rim of such a face.
TE_ROLES = frozenset({"trailing_edge", "te"})
#: Two lines of a base are one when they differ by less than this in parameter.
_LINE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FaceLayout:
    """One face of a layout sidecar: its base, role and the iso lines that cut it into blocks.

    ``u`` and ``v`` hold the lines inside the face's own parameter bounds with both bounds
    included, ascending; ``u_names`` and ``v_names`` give the named lines strictly inside.
    """

    name: str
    base: str
    role: str
    area_m2: float | None
    u: tuple[float, ...]
    v: tuple[float, ...]
    u_names: dict[str, float] = field(default_factory=dict)
    v_names: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Layout:
    """A layout sidecar read: the component it belongs to and its faces in the IGES's order."""

    path: Path
    component: str
    faces: tuple[FaceLayout, ...]

    def faces_by_name(self) -> dict[str, dict[str, Any]]:
        """Return ``{face: {"base", "role", "u", "v", "u_names", "v_names"}}``.

        The ``u`` and ``v`` lists are the form the strategy 1 junction block classifier reads.
        """
        return {
            f.name: {
                "base": f.base,
                "role": f.role,
                "u": list(f.u),
                "v": list(f.v),
                "u_names": dict(f.u_names),
                "v_names": dict(f.v_names),
            }
            for f in self.faces
        }


def sidecar_path(iges: str | Path) -> Path:
    """Return the layout sidecar of an IGES file: ``<stem>.layout.json`` beside it."""
    path = Path(iges)
    return path.with_name(f"{path.stem}.layout.json")


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise _UnusableError("layout numbers must be finite")
    return number


def _bounds(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return the (u0, u1, v0, v1) box of the face's source trims, or the unit square."""
    trims = [
        t for t in entry["source_faces"] if isinstance(t, dict) and t.get("base") == entry["base"]
    ]
    same = [t for t in trims if t.get("role") == entry["role"]] or trims
    points = [p for t in same for pair in t["loop"] for p in pair]
    if not points:
        return 0.0, 1.0, 0.0, 1.0
    us, vs = [_finite(p[0]) for p in points], [_finite(p[1]) for p in points]
    return min(us), max(us), min(vs), max(vs)


def _inside(values: set[float], low: float, high: float) -> tuple[float, ...]:
    kept = sorted(v for v in values if low + _LINE_TOLERANCE < v < high - _LINE_TOLERANCE)
    return (low, *kept, high)


def _face(entry: dict[str, Any], ends: dict[str, dict[str, set[float]]], names: dict[str, Any]):
    u0, u1, v0, v1 = _bounds(entry)
    held = ends.get(entry["base"], {"u": set(), "v": set()})
    u, v = _inside(held["u"], u0, u1), _inside(held["v"], v0, v1)

    def named(direction: str, lines: tuple[float, ...]) -> dict[str, float]:
        inner = lines[1:-1]
        return {
            str(k): _finite(x)
            for k, x in names.get(direction, {}).items()
            if any(abs(_finite(x) - y) <= _LINE_TOLERANCE for y in inner)
        }

    area = entry.get("area_m2")
    return FaceLayout(
        name=str(entry["name"]),
        base=str(entry["base"]),
        role=str(entry["role"]),
        area_m2=None if area is None else _finite(area),
        u=u,
        v=v,
        u_names=named("u", u),
        v_names=named("v", v),
    )


class _UnusableError(Exception):
    """A layout sidecar that cannot be used; the message says why."""


def _layout_of(data: Any, path: Path) -> Layout:
    """Build the layout from the decoded sidecar, or raise :class:`_UnusableError` (why not)."""
    if not isinstance(data, dict):
        raise _UnusableError("it is not a JSON object")
    entries = data.get("faces")
    if not isinstance(entries, list) or not entries:
        raise _UnusableError("it lists no faces")
    blocks = data.get("blocks", [])
    if not isinstance(blocks, list):
        raise _UnusableError("its blocks are not a list")
    ends: dict[str, dict[str, set[float]]] = {}
    for block in blocks:
        held = ends.setdefault(str(block["base"]), {"u": set(), "v": set()})
        held["u"].update(_finite(x) for x in block["u"])
        held["v"].update(_finite(x) for x in block["v"])
    lines = data.get("lines", {})
    names = lines if isinstance(lines, dict) else {}
    faces = []
    for entry in entries:
        for key in ("name", "base", "role", "source_faces"):
            if not isinstance(entry, dict) or key not in entry:
                raise _UnusableError(f"a face lacks {key!r}")
        faces.append(_face(entry, ends, names))
    components = {f.name.rpartition("_")[0] for f in faces}
    if len(components) != 1 or "" in components:
        raise _UnusableError("its faces do not belong to one component")
    return Layout(path=path, component=components.pop(), faces=tuple(faces))


def read_layout(iges: str | Path, notes: list[str]) -> Layout | None:
    """Read the layout sidecar beside an IGES file; ``None`` for a file that has none.

    A file with no sidecar is a foreign file and is handled with no layout, silently. A sidecar
    that cannot be used (not JSON, no faces, faces of two components, a trim missing) is warned
    by a line appended to ``notes`` and the file is handled as a foreign one: nothing is refused.
    """
    path = sidecar_path(iges)
    if not path.is_file():
        return None
    try:
        return _layout_of(json.loads(path.read_text(encoding="utf-8")), path)
    except json.JSONDecodeError as error:
        reason = f"it is not JSON ({error})"
    except (OSError, UnicodeDecodeError) as error:
        reason = f"it cannot be read ({error})"
    except (_UnusableError, KeyError, TypeError) as error:
        reason = str(error.args[0]) if isinstance(error, _UnusableError) else f"it lacks {error}"
    except (ValueError, IndexError, AttributeError) as error:
        reason = f"invalid layout data ({error})"
    notes.append(
        f"{path.name}: {reason}; no layout is used and the file is gridded as a foreign one"
    )
    return None
