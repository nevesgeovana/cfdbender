"""Draw a panel mesh in three views and an isometric view, with marked points, into a PDF.

``cfd-mesh visual-inspect``. The views are a
technical drawing: one scale, the top view above the side view, the front view
beside it, the isometric view in the remaining corner, no axes and a scale bar.
Every patch (every ``g`` or ``o`` group of an OBJ, every property of a NASTRAN
export) is drawn in its own color from a matplotlib colormap, its perimeter in a
thick line. Node lists saved from an ANSA SET (``Id, Name, X, Y, Z``) and
trailing-edge points files in any of the three forms the package writes (a unit
line then rows; the solver's count, its unused ``0,0,0`` line, then rows; plain
rows) are drawn over the surface in every view, hidden points included. A table
gives each patch's faces and marked points (a point lies on a patch within a
quarter of the local edge length of one of its vertices or edge mid-points).

Pages: the four views with the table, each view on its own page, the patch list.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender._errors import warn as _warn
from cfdbender._lengths import METRES_PER_UNIT

#: Colors and markers of the points files, chosen to stand out from a colormap.
POINT_COLORS = [
    "#e41a1c",
    "#ff7f00",
    "#f781bf",
    "#00bfff",
    "#000000",
    "#a65628",
    "#984ea3",
    "#4daf4a",
]
POINT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

#: Page size, A4 landscape, in inches.
PAGE = (11.69, 8.27)

_SPLIT = re.compile(r"[,;\s]+")


class InspectError(InputArtifactError):
    """An input that cannot be drawn; the command exits 2 and writes nothing."""

    def __init__(self, message: str) -> None:
        super().__init__(f"{message}. Nothing was written.", kind="mesh")


@dataclass
class Mesh:
    """The vertices and the faces of an OBJ, the faces grouped by patch in file order."""

    path: Path
    verts: np.ndarray
    patches: dict[str, list[list[int]]]


@dataclass
class Points:
    """Points read from one file, in the mesh's coordinates."""

    path: Path
    kind: str
    xyz: np.ndarray


@dataclass
class View:
    """An orthographic view: the screen axes, the viewing direction and their labels."""

    title: str
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    labelled: bool


def _skipped(message: str) -> None:
    """Warn and go on: an entry that cannot be used is skipped, never refused."""
    _warn(message, GeometryWarning, stacklevel=3)


def _face(words: list[str], rows: list[int]) -> list[int] | None:
    """Return a face's 0-based vertex rows, or None when it is unusable.

    ``rows`` maps each vertex number of the file (in order) to its row in the vertex array,
    or -1 for a vertex line that did not read. A face with fewer than three vertices, a
    missing vertex, an unreadable vertex or an index that is not an integer is unusable.
    """
    count = len(rows)
    face = []
    try:
        for word in words:
            index = int(word.split("/")[0])
            face.append(index - 1 if index > 0 else count + index)
    except ValueError:
        return None
    if len(face) < 3 or min(face) < 0 or max(face) >= count:
        return None
    face = [rows[k] for k in face]
    return None if min(face) < 0 else face


def _vertex(words: list[str]) -> list[float] | None:
    """Return a ``v`` line's three finite coordinates, or None."""
    try:
        xyz = [float(w) for w in words[1:4]]
    except ValueError:
        return None
    return xyz if len(xyz) == 3 and all(math.isfinite(v) for v in xyz) else None


def _parse_obj(lines: list[str]) -> tuple[list[list[float]], dict[str, list[list[int]]], int]:
    """Return an OBJ's vertices, its faces by group name, and the count of lines skipped."""
    verts: list[list[float]] = []
    rows: list[int] = []
    patches: dict[str, list[list[int]]] = {}
    current = "default"
    skipped = 0
    for line in lines:
        words = line.split()
        key = words[0] if words else ""
        if key == "v":
            xyz = _vertex(words)
            rows.append(-1 if xyz is None else len(verts))
            if xyz is None:
                skipped += 1
            else:
                verts.append(xyz)
        elif key in ("g", "o"):
            current = " ".join(words[1:]) or "default"
        elif key == "f":
            face = _face(words[1:], rows)
            if face is None:
                skipped += 1
            else:
                patches.setdefault(current, []).append(face)
    return verts, patches, skipped


def read_obj(path: Path) -> Mesh:
    """Read the vertices and the faces of an OBJ, grouped by its ``g`` or ``o`` names.

    The reader is lenient: it draws what it can and warns. A file that is not UTF-8 is read
    as latin-1 (every byte kept, so distinct group names stay distinct); a ``v`` or ``f``
    line that is not numbers is skipped and counted with the faces that cannot be used.
    """
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise InspectError(f"{path}: the mesh is not readable ({error})") from error
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        lines = raw.decode("latin-1").splitlines()
        _skipped(f"{path}: the file is not UTF-8; it is read as latin-1")
    verts, patches, skipped = _parse_obj(lines)
    if skipped:
        _skipped(
            f"{path}: {skipped} line(s) are skipped: a face with fewer than three vertices, a "
            "missing or unreadable vertex or a non-integer index, or a v line that is not "
            "three numbers"
        )
    if not patches:
        raise InspectError(f"{path}: holds no face to draw")
    return Mesh(path, np.array(verts, dtype=float).reshape(-1, 3), patches)


def _rows(lines: list[str], path: Path) -> np.ndarray:
    """Return the ``x,y,z`` rows of ``lines``; a row not three numbers is warned and skipped."""
    out, bad = [], 0
    for line in lines:
        words = [w for w in _SPLIT.split(line.strip()) if w]
        try:
            xyz = [float(w) for w in words[:3]]
        except ValueError:
            bad += 1
            continue
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            bad += 1
            continue
        out.append(xyz)
    if bad:
        _skipped(f"{path}: {bad} line(s) are not three numbers and are skipped")
    return np.array(out, dtype=float).reshape(-1, 3)


def read_node_list(path: Path) -> Points:
    """Read an ANSA SET node-window export (``#  Id, Name, X, Y, Z``); X, Y and Z by name."""
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as error:
        raise InspectError(f"{path}: the node list is not readable ({error})") from error
    columns = {"X": 2, "Y": 3, "Z": 4}
    points, bad = [], 0
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            names = [c.strip().upper() for c in text.lstrip("#").split(",")]
            if {"X", "Y", "Z"} <= set(names):
                columns = {k: names.index(k) for k in ("X", "Y", "Z")}
            continue
        words = [w.strip() for w in text.split(",")]
        try:
            xyz = [float(words[columns[k]]) for k in ("X", "Y", "Z")]
        except (ValueError, IndexError):
            bad += 1
            continue
        points.append(xyz)
    if bad:
        _skipped(f"{path}: {bad} line(s) are not Id, Name, X, Y, Z and are skipped")
    return Points(path, "nodes", np.array(points, dtype=float).reshape(-1, 3))


def read_points_file(path: Path, mesh_unit: str | None) -> Points:
    """Read a trailing-edge points file in the package, solver or plain form."""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as error:
        raise InspectError(f"{path}: the points file is not readable ({error})") from error
    lines = [x for x in text.splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if not lines:
        return Points(path, "te", np.zeros((0, 3)))
    first = lines[0].strip()
    if first.isdigit():
        count, rows = int(first), lines[1:]
        xyz = _rows(rows, path)
        if len(xyz) == count + 1 and np.allclose(xyz[0], 0.0):
            xyz = xyz[1:]
        elif len(xyz) != count:
            _skipped(
                f"{path}: the first line counts {count} point(s) and {len(xyz)} row(s) are read"
            )
        return Points(path, "te (solver form)", xyz)
    if re.fullmatch(r"[A-Za-z_]+", first):
        unit = first.upper()
        xyz = _rows(lines[1:], path)
        if mesh_unit is not None and unit != mesh_unit:
            if unit in METRES_PER_UNIT and mesh_unit in METRES_PER_UNIT:
                xyz = xyz * float(METRES_PER_UNIT[unit] / METRES_PER_UNIT[mesh_unit])
            else:
                _skipped(f"{path}: the unit {unit} has no scale; the points are drawn as written")
        return Points(path, f"te ({unit})", xyz)
    return Points(path, "te", _rows(lines, path))


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def make_view(title: str, forward, up, labelled: bool = True) -> View:
    """Return a view looking along ``forward`` with ``up`` as the screen's vertical."""
    f = _unit(np.asarray(forward, dtype=float))
    right = _unit(np.cross(f, np.asarray(up, dtype=float)))
    return View(title, right, np.cross(right, f), f, labelled)


def standard_views(iso_from) -> list[View]:
    """Top, front and side views along the axes, and an isometric view from ``iso_from``."""
    eye = np.asarray(iso_from, dtype=float)
    sign = lambda v: "+" if v >= 0 else "-"  # noqa: E731
    origin = " ".join(f"{sign(v)}{a}" for v, a in zip(eye, "XYZ", strict=True) if abs(v) > 1e-12)
    return [
        make_view("Top (looking -Z)", (0, 0, -1), (0, 1, 0)),
        make_view("Front (looking +X)", (1, 0, 0), (0, 0, 1)),
        make_view("Side (looking +Y)", (0, 1, 0), (0, 0, 1)),
        make_view(f"Isometric (from {origin})", -eye, (0, 0, 1), labelled=False),
    ]


def face_geometry(verts: np.ndarray, faces: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Return each face's centroid and its unit normal (Newell's rule, so any polygon works)."""
    centroids = np.zeros((len(faces), 3))
    normals = np.zeros((len(faces), 3))
    by_size: dict[int, list[int]] = {}
    for number, face in enumerate(faces):
        by_size.setdefault(len(face), []).append(number)
    for ids in by_size.values():
        corners = verts[np.array([faces[i] for i in ids])]
        centroids[ids] = corners.mean(axis=1)
        normals[ids] = np.cross(corners, np.roll(corners, -1, axis=1)).sum(axis=1)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return centroids, normals / np.where(length > 0, length, 1.0)


def patch_colors(names: list[str], cmap_name: str | None) -> tuple[dict[str, tuple], str]:
    """Return one color per patch from the colormap, in patch order, and the colormap's name."""
    from matplotlib import colormaps, rcParams
    from matplotlib.colors import ListedColormap

    name = cmap_name or rcParams["image.cmap"]
    if name not in colormaps:
        _skipped(
            f"--cmap {name}: matplotlib has no colormap of that name; its default "
            f"{rcParams['image.cmap']} is used"
        )
        name = rcParams["image.cmap"]
    cmap = colormaps[name]
    n = len(names)
    if isinstance(cmap, ListedColormap) and cmap.N >= n and cmap.N <= 32:
        return {p: cmap(i) for i, p in enumerate(names)}, name
    return {p: cmap(i / max(n - 1, 1)) for i, p in enumerate(names)}, name


class Scene:
    """The faces kept for drawing, their colors, and the points over them."""

    def __init__(
        self,
        mesh: Mesh,
        names: list[str],
        colors: dict[str, tuple],
        box,
        *,
        edges: bool = True,
        perimeter_width: float = 0.9,
    ):
        self.perimeter_width = perimeter_width
        faces, patch_of = [], []
        for number, name in enumerate(names):
            faces += mesh.patches[name]
            patch_of += [number] * len(mesh.patches[name])
        self.verts = mesh.verts
        centroids, normals = face_geometry(mesh.verts, faces)
        keep = np.ones(len(faces), dtype=bool)
        if box is not None:
            lo, hi = np.array(box[0::2]), np.array(box[1::2])
            keep = np.all((centroids >= lo) & (centroids <= hi), axis=1)
        self.faces = [f for f, k in zip(faces, keep, strict=True) if k]
        self.patch_of = np.array(patch_of)[keep]
        self.centroids, self.normals = centroids[keep], normals[keep]
        self.names = names
        self.base = (
            np.array([colors[n] for n in names])[self.patch_of]
            if len(self.faces)
            else np.zeros((0, 4))
        )
        self.edges = edges
        self.box = box
        self.highlight: set[str] = set()  # names whose label is drawn bold in a red frame
        self.highlight_labels: dict[str, str] = {}
        self.labelled: set[str] | None = None  # names labeled when labels are drawn (None: all)
        self.perimeter_color: tuple[float, ...] = (0.05, 0.05, 0.05, 1.0)
        self.direction_arcs: list[tuple[np.ndarray, str, str, bool]] = []
        self.perimeter, self.perimeter_faces = self._perimeters()

    def _perimeters(self) -> tuple[np.ndarray, list[list[int]]]:
        """Return the edges bounding a patch (open or shared with another) and the faces beside."""
        beside: dict[tuple[int, int], list[int]] = {}
        for number, face in enumerate(self.faces):
            for k in range(len(face)):
                a, b = face[k], face[(k + 1) % len(face)]
                beside.setdefault((a, b) if a < b else (b, a), []).append(number)
        edges, faces = [], []
        for edge, near in beside.items():
            patches_of = [int(self.patch_of[f]) for f in near]
            if any(patches_of.count(o) == 1 for o in set(patches_of)):
                edges.append(edge)
                faces.append(near)
        return np.array(edges, dtype=int).reshape(-1, 2), faces

    def clip(self, points: Points) -> np.ndarray:
        """Return the points inside the box (all of them without one)."""
        if self.box is None or len(points.xyz) == 0:
            return points.xyz
        lo, hi = np.array(self.box[0::2]), np.array(self.box[1::2])
        return points.xyz[np.all((points.xyz >= lo) & (points.xyz <= hi), axis=1)]


#: Symbols of the length units, for the scale bar.
UNIT_SYMBOL = {"METER": "m", "CENTIMETER": "cm", "MILLIMETER": "mm", "INCH": "in", "FEET": "ft"}

#: Page layout, in inches: margins, the room a view's title takes, the gap between views, the
#: column the table and the legend take, and the band the scale bar takes.
LEFT, TOP, TITLE, GAP, RIGHT, BOTTOM = 0.35, 0.55, 0.22, 0.3, 2.75, 0.8


def view_extent(scene: Scene, view: View, points: list[Points]) -> np.ndarray:
    """Return ``(hmin, hmax, vmin, vmax)`` of the drawn faces and points on the view's screen."""
    to_screen = np.stack([view.right, view.up], axis=1)
    spans = []
    if scene.faces:
        used = np.unique(np.concatenate([np.asarray(f) for f in scene.faces]))
        spans.append(scene.verts[used] @ to_screen)
    for pts in points:
        xyz = scene.clip(pts)
        if len(xyz):
            spans.append(xyz @ to_screen)
    if not spans:
        return np.array([-1.0, 1.0, -1.0, 1.0])
    both = np.vstack(spans)
    lo, hi = both.min(axis=0), both.max(axis=0)
    return np.array([lo[0], hi[0], lo[1], hi[1]])


def draw_view(
    ax, scene: Scene, view: View, points: list[Points], styles: list[dict], *, labels: bool, extent
):
    """Draw the faces back to front, shaded by their angle to the view, then the points; no axes."""
    from matplotlib.collections import PolyCollection

    to_screen = np.stack([view.right, view.up], axis=1)
    flat = scene.verts @ to_screen
    depth = scene.centroids @ view.forward
    shade = 0.45 + 0.55 * np.abs(scene.normals @ view.forward)
    fills = scene.base.copy()
    fills[:, :3] *= shade[:, None]
    lines = fills if not scene.edges else np.tile((0.15, 0.15, 0.15, 0.6), (len(fills), 1))
    polygons = [flat[f] for f in scene.faces]
    widths = [0.12 if scene.edges else 0.0] * len(polygons)
    if scene.perimeter_width > 0 and len(scene.perimeter):
        # A patch's perimeter is drawn just after the nearest face beside it, so a surface in
        # front still hides it and the faces it bounds do not paint over it.
        nudge = 1e-6 * float(np.ptp(scene.verts, axis=0).max() or 1.0)
        depth = np.concatenate(
            [depth, [depth[near].min() - nudge for near in scene.perimeter_faces]]
        )
        polygons += [flat[e] for e in scene.perimeter]
        fills = np.vstack([fills, np.zeros((len(scene.perimeter), 4))])
        lines = np.vstack([lines, np.tile(scene.perimeter_color, (len(scene.perimeter), 1))])
        widths += [scene.perimeter_width] * len(scene.perimeter)
    order = np.argsort(-depth, kind="stable")
    collection = PolyCollection(
        [polygons[i] for i in order],
        facecolors=fills[order],
        edgecolors=lines[order],
        linewidths=np.asarray(widths)[order],
    )
    collection.set_rasterized(True)
    ax.add_collection(collection)
    _draw_directions(ax, scene, to_screen)
    for pts, style in zip(points, styles, strict=True):
        xyz = scene.clip(pts)
        if len(xyz):
            xy = xyz @ to_screen
            ax.scatter(xy[:, 0], xy[:, 1], zorder=5, clip_on=False, **style)
    if labels and len(scene.faces):
        for number, name in enumerate(scene.names):
            mine = scene.patch_of == number
            if mine.any() and (scene.labelled is None or name in scene.labelled):
                at = scene.centroids[mine].mean(axis=0) @ to_screen
                lead = name in scene.highlight
                ax.text(
                    at[0],
                    at[1],
                    f"{name}\n{scene.highlight_labels.get(name, 'GOVERNING')}" if lead else name,
                    fontsize=6 if lead else 5,
                    fontweight="bold" if lead else "normal",
                    ha="center",
                    va="center",
                    zorder=7 if lead else 6,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "fc": "white",
                        "ec": "#d62728" if lead else "none",
                        "lw": 1.2 if lead else 0.0,
                        "alpha": 0.85 if lead else 0.7,
                    },
                )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()
    ax.set_title(view.title, fontsize=8, pad=4)


def _nice(length: float) -> float:
    """Return the largest 1, 2 or 5 times a power of ten not above ``length``."""
    base = 10.0 ** math.floor(math.log10(length))
    return next(m * base for m in (5, 2, 1, 0.5) if m * base <= length)


def scale_bar(
    fig, x_in: float, y_in: float, scale: float, unit: str | None, *, longest_in: float
) -> None:
    """Draw a scale bar of four alternating segments, and the ratio when the unit is known."""
    from matplotlib.patches import Rectangle

    length = _nice(longest_in / scale)
    pw, ph = fig.get_size_inches()
    ax = fig.add_axes((x_in / pw, y_in / ph, length * scale / pw, 0.07 / ph))
    ax.set_xlim(0, length)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    for k in range(4):
        ax.add_patch(
            Rectangle(
                (k * length / 4, 0),
                length / 4,
                1,
                fc="black" if k % 2 == 0 else "white",
                ec="black",
                lw=0.6,
                clip_on=False,
            )
        )
    for value in (0.0, length / 2, length):
        ax.text(value, -0.8, f"{value:g}", ha="center", va="top", fontsize=6.5)
    words = UNIT_SYMBOL.get(unit or "", "mesh length units")
    if unit is not None and unit in METRES_PER_UNIT:
        ratio = float(METRES_PER_UNIT[unit]) / (scale * 0.0254)
        shown = f"{ratio:.3g}" if ratio < 1 else f"{round(float(f'{ratio:.3g}')):d}"
        words += f"      scale 1:{shown} (A4 printed at 100%)"
    ax.text(length * 1.03, 0.5, words, ha="left", va="center", fontsize=6.5)


def _place(fig, x_in: float, y_in: float, extent, scale: float):
    pw, ph = fig.get_size_inches()
    w, h = (extent[1] - extent[0]) * scale, (extent[3] - extent[2]) * scale
    return fig.add_axes((x_in / pw, y_in / ph, w / pw, h / ph))


def _padded(extents: list[np.ndarray]) -> list[np.ndarray]:
    pad = 0.02 * max(max(e[1] - e[0], e[3] - e[2]) for e in extents)
    pad = pad or 0.02
    return [e + np.array([-pad, pad, -pad, pad]) for e in extents]


def views_page(fig, scene, views, points, styles, *, labels, unit) -> list:
    """Lay the views out as a technical drawing, all at one scale.

    The top view stands above the side view and shares its X; the front view stands beside
    the side view and shares its Z; the isometric view fills the remaining corner.
    """
    pw, ph = fig.get_size_inches()
    top, front, side, iso = _padded([view_extent(scene, v, points) for v in views])
    width = lambda e: e[1] - e[0]  # noqa: E731
    height = lambda e: e[3] - e[2]  # noqa: E731
    w1, w2 = width(side), max(width(front), width(iso))
    h1, h2 = max(height(top), height(iso)), height(side)
    room_w = pw - LEFT - RIGHT - GAP
    room_h = ph - TOP - BOTTOM - GAP - 2 * TITLE
    scale = min(room_w / (w1 + w2), room_h / (h1 + h2))
    x1 = LEFT + (room_w - scale * (w1 + w2)) / 2
    x2 = x1 + w1 * scale + GAP
    y1 = ph - TOP - TITLE - h1 * scale
    y2 = y1 - GAP - TITLE - h2 * scale
    axes = []
    for view, extent, x, y in (
        (views[0], top, x1, y1),
        (views[1], front, x2, y2),
        (views[2], side, x1, y2),
        (views[3], iso, x2 + (w2 - width(iso)) * scale / 2, y1),
    ):
        ax = _place(fig, x, y, extent, scale)
        draw_view(ax, scene, view, points, styles, labels=labels, extent=extent)
        axes.append(ax)
    scale_bar(fig, x1, y2 - 0.45, scale, unit, longest_in=min(3.0, w1 * scale))
    return axes


def single_view_page(fig, scene, view, points, styles, *, labels, unit) -> list:
    """One view, as large as the page allows, with its scale bar."""
    pw, ph = fig.get_size_inches()
    (extent,) = _padded([view_extent(scene, view, points)])
    room_w, room_h = pw - LEFT - RIGHT, ph - TOP - BOTTOM - TITLE
    scale = min(room_w / (extent[1] - extent[0]), room_h / (extent[3] - extent[2]))
    x = LEFT + (room_w - scale * (extent[1] - extent[0])) / 2
    y = ph - TOP - TITLE - scale * (extent[3] - extent[2])
    ax = _place(fig, x, y, extent, scale)
    draw_view(ax, scene, view, points, styles, labels=labels, extent=extent)
    scale_bar(fig, LEFT, y - 0.45, scale, unit, longest_in=min(3.0, room_w / 2))
    return [ax]


def side_column(fig, axes, scene, colors, marked, *, points, table: bool) -> None:
    """Draw the right-hand column: the patch table on the first page, and the points legend."""
    pw, ph = fig.get_size_inches()
    legend_h = 0.0
    if points:
        legend_h = 0.32 + 0.16 * len(points)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="lower left",
            fontsize=6,
            title_fontsize=6,
            title="points (drawn over the surface, hidden ones too)",
            bbox_to_anchor=((pw - RIGHT + 0.1) / pw, 0.3 / ph),
        )
    if table:
        y = 0.3 + legend_h + 0.15
        draw_table(
            fig.add_axes(
                ((pw - RIGHT + 0.15) / pw, y / ph, (RIGHT - 0.3) / pw, (ph - TOP - y) / ph)
            ),
            scene,
            colors,
            marked,
        )


def point_styles(points: list[Points], scene: Scene) -> list[dict]:
    """Return the styles: node lists white edged black, each points file its own color."""
    styles, te = [], 0
    for number, pts in enumerate(points):
        inside = len(scene.clip(pts))
        count = (
            f"{len(pts.xyz)} points"
            if inside == len(pts.xyz)
            else f"{inside} of {len(pts.xyz)} points in the box"
        )
        label = f"{pts.path.name} ({pts.kind}, {count})"
        if pts.kind == "nodes":
            styles.append(
                {
                    "s": 14,
                    "marker": POINT_MARKERS[number % len(POINT_MARKERS)],
                    "facecolors": "white",
                    "edgecolors": "black",
                    "linewidths": 0.6,
                    "label": label,
                }
            )
        else:
            styles.append(
                {
                    "s": 9,
                    "marker": POINT_MARKERS[te % len(POINT_MARKERS)],
                    "color": POINT_COLORS[te % len(POINT_COLORS)],
                    "linewidths": 0.3,
                    "label": label,
                }
            )
            te += 1
    return styles


#: A marked point lies on a patch when it is closer to one of the patch's vertices, or to the
#: mid-point of one of its edges, than this fraction of the local edge length.
MATCH_EDGE_FRACTION = 0.25


@dataclass
class Marked:
    """Per drawn patch, its faces and the distinct marked points on it; the totals."""

    faces: list[int]
    points: list[int]
    total_points: int
    on_no_patch: int


def _targets(verts: np.ndarray, faces: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Return a patch's vertices and edge mid-points, and the distance each is matched within."""
    edges = {tuple(sorted((f[k], f[(k + 1) % len(f)]))) for f in faces for k in range(len(f))}
    pairs = np.array(sorted(edges), dtype=int).reshape(-1, 2)
    length = np.linalg.norm(verts[pairs[:, 0]] - verts[pairs[:, 1]], axis=1)
    shortest = np.full(len(verts), np.inf)
    np.minimum.at(shortest, pairs[:, 0], length)
    np.minimum.at(shortest, pairs[:, 1], length)
    used = np.unique(pairs)
    where = np.vstack([verts[used], 0.5 * (verts[pairs[:, 0]] + verts[pairs[:, 1]])])
    within = MATCH_EDGE_FRACTION * np.concatenate([shortest[used], length])
    return where, within


def _on(points: np.ndarray, where: np.ndarray, within: np.ndarray) -> np.ndarray:
    """Return which points lie within the matching distance of a target (x window, then exact)."""
    order = np.argsort(where[:, 0])
    where, within = where[order], within[order]
    reach = float(within.max()) if len(within) else 0.0
    lo = np.searchsorted(where[:, 0], points[:, 0] - reach, side="left")
    hi = np.searchsorted(where[:, 0], points[:, 0] + reach, side="right")
    hit = np.zeros(len(points), dtype=bool)
    for k in range(len(points)):
        if hi[k] > lo[k]:
            d = np.linalg.norm(where[lo[k] : hi[k]] - points[k], axis=1)
            hit[k] = bool(np.any(d <= within[lo[k] : hi[k]]))
    return hit


def marked_points(scene: Scene, points: list[Points]) -> Marked:
    """Count, per drawn patch, the distinct points of every node list and points file on it."""
    faces = [int(np.sum(scene.patch_of == n)) for n in range(len(scene.names))]
    inside = [scene.clip(p) for p in points]
    every = (
        np.vstack([p for p in inside if len(p)])
        if any(len(p) for p in inside)
        else np.zeros((0, 3))
    )
    if len(every) == 0 or not scene.faces:
        return Marked(faces, [0] * len(scene.names), len(every), len(every))
    size = float(np.ptp(scene.verts, axis=0).max()) or 1.0
    _, first = np.unique(np.round(every / (size * 1e-7)), axis=0, return_index=True)
    every = every[np.sort(first)]
    on_any = np.zeros(len(every), dtype=bool)
    counts = []
    for number in range(len(scene.names)):
        mine = [scene.faces[i] for i in np.flatnonzero(scene.patch_of == number)]
        if not mine:
            counts.append(0)
            continue
        hit = _on(every, *_targets(scene.verts, mine))
        on_any |= hit
        counts.append(int(hit.sum()))
    return Marked(faces, counts, len(every), int((~on_any).sum()))


def _readable(color) -> dict:
    """Text in a patch's color, outlined in gray when the color is too light to read on white."""
    from matplotlib import patheffects

    r, g, b = color[:3]
    if 0.299 * r + 0.587 * g + 0.114 * b < 0.7:
        return {"color": color}
    return {
        "color": color,
        "path_effects": [patheffects.withStroke(linewidth=1.2, foreground="0.35")],
    }


def draw_table(ax, scene: Scene, colors: dict[str, tuple], marked: Marked, rows: int = 40) -> None:
    """Draw a small box: each patch's name in its color, faces and marked points, and totals."""
    from matplotlib.patches import FancyBboxPatch

    ax.axis("off")
    names = scene.names
    shown = names[:rows]
    lines = len(shown) + 3 + (1 if len(names) > rows else 0) + (1 if marked.on_no_patch else 0)
    step = min(0.035, 0.94 / max(lines, 1))
    size = max(4.0, min(7.0, step * 200))
    top = 0.5 + step * lines / 2
    ax.add_patch(
        FancyBboxPatch(
            (0.02, top - step * (lines + 0.3)),
            0.96,
            step * (lines + 0.8),
            boxstyle="round,pad=0.01",
            fc="white",
            ec="0.4",
            lw=0.6,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    y = top - step * 0.2
    for text, x, ha in (
        ("Family", 0.04, "left"),
        ("Faces", 0.70, "right"),
        ("Marked", 0.96, "right"),
    ):
        ax.text(
            x, y, text, fontsize=size, fontweight="bold", ha=ha, va="top", transform=ax.transAxes
        )
    y -= step
    for number, name in enumerate(shown):
        ax.text(
            0.04,
            y,
            name,
            fontsize=size,
            ha="left",
            va="top",
            transform=ax.transAxes,
            **_readable(colors[name]),
        )
        ax.text(
            0.70,
            y,
            str(marked.faces[number]),
            fontsize=size,
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
        ax.text(
            0.96,
            y,
            str(marked.points[number]),
            fontsize=size,
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
        y -= step
    if len(names) > rows:
        ax.text(
            0.04,
            y,
            f"... {len(names) - rows} more on the patch list page",
            fontsize=size,
            style="italic",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        y -= step
    if marked.on_no_patch:
        ax.text(
            0.04,
            y,
            "on no patch",
            fontsize=size,
            style="italic",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        ax.text(
            0.96,
            y,
            str(marked.on_no_patch),
            fontsize=size,
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
        y -= step
    ax.plot([0.03, 0.97], [y + step * 0.1] * 2, color="0.4", lw=0.5, transform=ax.transAxes)
    for text, x, ha in (
        ("Total", 0.04, "left"),
        (str(sum(marked.faces)), 0.70, "right"),
        (str(marked.total_points), 0.96, "right"),
    ):
        ax.text(
            x,
            y - step * 0.1,
            text,
            fontsize=size,
            fontweight="bold",
            ha=ha,
            va="top",
            transform=ax.transAxes,
        )


def legend_pages(
    pdf,
    scene: Scene,
    colors: dict[str, tuple],
    marked: Marked,
    mesh: Mesh,
    *,
    cmap_label: str,
    dpi: int,
) -> None:
    """Write the list of patches with their colors, faces and marked points, 180 to a page."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows, cols = 45, 4
    names = scene.names
    for start in range(0, len(names), rows * cols):
        fig = plt.figure(figsize=PAGE)
        ax = fig.add_axes((0.03, 0.03, 0.94, 0.9))
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.axis("off")
        fig.suptitle(
            f"{mesh.path.name}: patches and colors (colormap {cmap_label}); "
            "faces drawn, marked points",
            fontsize=10,
        )
        for k, name in enumerate(names[start : start + rows * cols]):
            c, r = divmod(k, rows)
            number = start + k
            ax.add_patch(Rectangle((c + 0.02, r + 0.15), 0.08, 0.7, color=colors[name]))
            ax.text(
                c + 0.13,
                r + 0.5,
                f"{name}  ({marked.faces[number]} faces, {marked.points[number]} marked)",
                fontsize=6,
                va="center",
            )
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)


def render(
    mesh: Mesh,
    points: list[Points],
    out: Path,
    *,
    cmap_name,
    families,
    box,
    edges,
    labels,
    iso_from,
    dpi,
    perimeter_width: float = 0.9,
    mesh_unit: str | None = None,
    png: bool = False,
):
    """Write the PDF: the four views on one page, each view on its own page, then the patch list.

    With ``png``, each page of views is also written as a PNG beside the PDF:
    ``<stem>.png`` for the four views, ``<stem>.<view>.png`` for each one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    names = _chosen(mesh, families)
    colors, cmap_label = patch_colors(names, cmap_name)
    scene = Scene(mesh, names, colors, box, edges=edges, perimeter_width=perimeter_width)
    if not scene.faces:
        _skipped(f"{mesh.path}: no face lies inside --box; only the points are drawn")
    views = standard_views(iso_from)
    styles = point_styles(points, scene)
    faces = sum(len(f) for f in mesh.patches.values())
    header = f"{mesh.path.name}: {len(mesh.patches)} patches, {faces} faces"
    if families or box is not None:
        header += f"; drawn: {len(names)} patches, {len(scene.faces)} faces"
    marked = marked_points(scene, points)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=PAGE)
        axes = views_page(fig, scene, views, points, styles, labels=labels, unit=mesh_unit)
        side_column(fig, axes, scene, colors, marked, points=points, table=True)
        fig.suptitle(header, fontsize=10, y=1 - 0.2 / PAGE[1], va="top")
        pdf.savefig(fig, dpi=dpi)
        if png:
            fig.savefig(out.with_suffix(".png"), dpi=dpi)
        plt.close(fig)
        for view, key in zip(views, ("top", "front", "side", "iso"), strict=True):
            fig = plt.figure(figsize=PAGE)
            axes = single_view_page(fig, scene, view, points, styles, labels=labels, unit=mesh_unit)
            side_column(fig, axes, scene, colors, marked, points=points, table=False)
            fig.suptitle(header, fontsize=9, y=1 - 0.2 / PAGE[1], va="top")
            pdf.savefig(fig, dpi=dpi)
            if png:
                fig.savefig(out.with_name(f"{out.stem}.{key}.png"), dpi=dpi)
            plt.close(fig)
        legend_pages(pdf, scene, colors, marked, mesh, cmap_label=cmap_label, dpi=dpi)


def _draw_directions(ax, scene: Scene, to_screen: np.ndarray) -> None:
    """Draw retained planner arcs and band labels above the neutral surface."""
    from matplotlib.collections import LineCollection

    for xyz, direction, label, lead in scene.direction_arcs:
        xy = xyz @ to_screen
        color = DIRECTION_COLORS[direction]
        ax.add_collection(
            LineCollection([xy], colors=[color], linewidths=3.5 if lead else 2, zorder=4)
        )
        if label:
            lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            distance = np.concatenate(([0.0], np.cumsum(lengths)))
            at = np.array([np.interp(distance[-1] / 2, distance, xy[:, i]) for i in (0, 1)])
            ax.text(
                *at,
                label,
                color=color,
                fontsize=7,
                ha="center",
                va="center",
                zorder=8,
                fontweight="bold" if lead else "normal",
                bbox={"fc": "white", "ec": color, "boxstyle": "round,pad=0.2"} if lead else None,
            )


DIRECTION_COLORS = {"parallel": "#0072b2", "normal": "#d55e00"}


def patches_pdf(
    mesh: str | Path,
    groups: dict[str, list[str]],
    governing: dict[str, tuple[bool, bool]] | set[str],
    out: Path,
    *,
    mesh_unit: str | None = None,
    dpi: int = 200,
    layouts: Mapping[str, Any] | None = None,
) -> Path:
    """Write patches and planner directions pages for each original family."""
    from cfdbender.mesh._inspect_pages import patches_pdf as write_pages

    return write_pages(mesh, groups, governing, out, mesh_unit=mesh_unit, dpi=dpi, layouts=layouts)


def _chosen(mesh: Mesh, families) -> list[str]:
    """Return the patches to draw: the named ones the mesh holds, else every patch."""
    names = list(mesh.patches)
    if not families:
        return names
    unknown = [f for f in families if f not in mesh.patches]
    if unknown:
        _skipped(f"{mesh.path}: holds no patch {', '.join(unknown)}; skipped")
    chosen = [n for n in names if n in families]
    if not chosen:
        _skipped(f"{mesh.path}: none of --families is a patch of the mesh; every patch is drawn")
    return chosen or names


def visual_inspect(
    mesh: str | Path,
    *,
    nodes: Sequence[str | Path] = (),
    te: Sequence[str | Path] = (),
    out: str | Path | None = None,
    cmap: str | None = None,
    families: Sequence[str] | None = None,
    box: Sequence[float] | None = None,
    mesh_unit: str | None = None,
    points_unit: str | None = None,
    nastran_unit: str = "MILLIMETER",
    iso_from: Sequence[float] = (1.0, -1.0, 1.0),
    labels: bool = False,
    edges: bool = True,
    perimeter_width: float = 0.9,
    dpi: int = 300,
    overwrite: bool = False,
    png: bool = False,
) -> Path:
    """Write the PDF of a mesh's views with its marked points and return its path.

    Parameters
    ----------
    mesh : str or pathlib.Path
        An OBJ, or a NASTRAN export written in ``nastran_unit`` (default MILLIMETER) and
        drawn in ``mesh_unit`` (default METER).
    nodes : sequence of path
        Node lists saved from an ANSA SET.
    te : sequence of path
        Trailing-edge points files, each drawn in its own color.
    out : path, optional
        The PDF (default: ``<mesh>.inspect.pdf`` beside the mesh).
    cmap : str, optional
        The matplotlib colormap of the patches (default: matplotlib's default).
    families : sequence of str, optional
        Draw only these patches; a name the mesh does not hold is warned and skipped.
    box : six floats, optional
        ``xmin xmax ymin ymax zmin zmax``: draw only what lies inside.
    mesh_unit : str, optional
        The mesh's length unit: the scale bar names it and gives the ratio, and a
        points file that states another unit is converted to it. For a NASTRAN mesh
        it is the unit the drawing is converted to.
    points_unit : str, optional
        The unit of the node lists and points files that state none; needs ``mesh_unit``.
    nastran_unit : str
        A NASTRAN mesh's length unit (default MILLIMETER; CLI: ``--unit``).
    iso_from : three floats
        The direction the isometric view is seen from (default (1, -1, 1), from +x, -y, +z).
    labels, edges : bool
        Write each patch's name at its center; draw the face edges.
    perimeter_width : float
        Width of the patch perimeters in points (0 draws none).
    dpi : int
        Resolution of the rasterized surface.
    overwrite : bool
        Replace an existing PDF (and PNGs).
    png : bool
        Also write each page of views as a PNG beside the PDF.

    Returns
    -------
    pathlib.Path
        The PDF written.

    Raises
    ------
    InspectError
        (an InputArtifactError) The mesh cannot be read, the PDF exists without
        ``overwrite``.
    """
    source = Path(mesh)
    target = Path(out) if out is not None else source.with_name(source.stem + ".inspect.pdf")
    if target.exists() and not overwrite:
        raise InspectError(f"{target}: exists; pass overwrite (CLI: --overwrite) to replace it")
    mesh_unit = mesh_unit.upper() if mesh_unit else mesh_unit
    points_unit = points_unit.upper() if points_unit else points_unit
    nastran_unit = nastran_unit.upper()
    drawn = _read_mesh(source, mesh_unit, nastran_unit)
    unit = mesh_unit or ("METER" if drawn.path.suffix.lower() != ".obj" else None)
    points = load_points(nodes, te, mesh_unit=unit, points_unit=points_unit)
    chosen = [f for f in families if f] if families else None
    window = _ordered_box(box)
    render(
        drawn,
        points,
        target,
        cmap_name=cmap,
        families=chosen,
        box=window,
        edges=edges,
        labels=labels,
        iso_from=tuple(iso_from),
        dpi=dpi,
        perimeter_width=perimeter_width,
        mesh_unit=unit,
        png=png,
    )
    return target


def load_points(
    nodes: Sequence[str | Path],
    te: Sequence[str | Path],
    *,
    mesh_unit: str | None,
    points_unit: str | None,
) -> list[Points]:
    """Return the node lists and points files, in the mesh's unit where both units are known.

    Unit names are case-insensitive. A file that states its unit is converted to
    ``mesh_unit``; one that states none is taken in ``points_unit``. An unknown unit,
    or ``points_unit`` without ``mesh_unit``, is warned and the points kept as written;
    a file with no point is warned.
    """
    mesh_unit = mesh_unit.upper() if mesh_unit else mesh_unit
    points_unit = points_unit.upper() if points_unit else points_unit
    points = [read_node_list(Path(p)) for p in nodes]
    points += [read_points_file(Path(p), mesh_unit) for p in te]
    _convert_unitless(points, points_unit, mesh_unit)
    for pts in points:
        if len(pts.xyz) == 0:
            _skipped(f"{pts.path}: holds no point; it is listed and draws nothing")
    return points


def _read_mesh(source: Path, mesh_unit: str | None, nastran_unit: str) -> Mesh:
    """Read an OBJ, or a NASTRAN export through the refinement's reader."""
    from cfdbender.mesh._refine import _nastran

    if not _nastran.is_nastran(source):
        return read_obj(source)
    drawn_in = mesh_unit if mesh_unit and mesh_unit in METRES_PER_UNIT else "METER"
    if nastran_unit not in METRES_PER_UNIT:
        _skipped(_unknown_unit("--unit", nastran_unit) + "; the mesh is read without scaling")
        nastran_unit = drawn_in
    read = _nastran.read_nastran(source, unit=nastran_unit, obj_unit=drawn_in)
    return Mesh(source, read.mesh.verts, dict(read.mesh.families))


def _unknown_unit(flag: str, name: str) -> str:
    """Return the message for a length unit name that has no scale."""
    return f"{flag} {name} is not a length unit (accepted: {', '.join(sorted(METRES_PER_UNIT))})"


def _convert_unitless(points: list[Points], points_unit: str | None, mesh_unit: str | None) -> None:
    """Scale the files that state no unit from ``points_unit`` to the mesh's unit."""
    if not points_unit:
        return
    if not mesh_unit:
        _skipped("--points-unit is given without --mesh-unit; the points are drawn as written")
        return
    if points_unit == mesh_unit:
        return
    for flag, name in (("--points-unit", points_unit), ("--mesh-unit", mesh_unit)):
        if name not in METRES_PER_UNIT:
            _skipped(_unknown_unit(flag, name) + "; the points are drawn as written")
            return
    factor = float(METRES_PER_UNIT[points_unit] / METRES_PER_UNIT[mesh_unit])
    for pts in points:
        if pts.kind == "nodes" or "(" not in pts.kind or pts.kind == "te (solver form)":
            pts.xyz = pts.xyz * factor


def beam_pdf(
    out: Path, vertices_m, faces, table: Mapping[str, Any], *, title: str, dpi: int = 200
) -> Path:
    """Write the drawing of a blade's sections cut from its mesh.

    The first page draws the blade as ``cfd-mesh visual-inspect`` draws a mesh, with every
    station's contour placed on it where the beam takes it (the span along +z, the section
    turned by its pitch, toward the leading edge ``(-sin b, -cos b, 0)``); the next pages
    draw each station's contour in its section frame, with its chord line and pitch axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = list(
        zip(
            table["station_radii_m"],
            table["chord_m"],
            table["geometric_pitch_deg"],
            table["sections_m"],
            strict=True,
        )
    )
    mesh = Mesh(out, np.asarray(vertices_m, dtype=float), {"blade": np.asarray(faces).tolist()})
    placed = [_on_the_blade(radius, pitch, contour) for radius, _, pitch, contour in rows]
    points = [Points(out.with_suffix(".toml"), "sections", np.vstack(placed))]
    colors, _ = patch_colors(["blade"], None)
    scene = Scene(mesh, ["blade"], colors, None)
    styles = point_styles(points, scene)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=PAGE)
        views = standard_views((1.0, -1.0, 1.0))
        axes = views_page(fig, scene, views, points, styles, labels=False, unit="METER")
        marked = marked_points(scene, points)
        side_column(fig, axes, scene, colors, marked, points=points, table=True)
        fig.suptitle(title, fontsize=10, y=1 - 0.2 / PAGE[1], va="top")
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)
        for first in range(0, len(rows), 12):
            fig = plt.figure(figsize=PAGE)
            for slot, number in enumerate(range(first, min(first + 12, len(rows)))):
                _station_axes(fig.add_subplot(3, 4, slot + 1), number + 1, rows[number])
            fig.suptitle(title, fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
    return out


def _on_the_blade(radius: float, pitch_deg: float, contour) -> np.ndarray:
    """Return a section contour placed on the blade: x, y, z [m] of its points."""
    beta = math.radians(pitch_deg)
    lead = np.array([-math.sin(beta), -math.cos(beta), 0.0])
    suction = np.array([-math.cos(beta), math.sin(beta), 0.0])
    points = np.asarray(contour, dtype=float)
    return points[:, :1] * lead + points[:, 1:2] * suction + np.array([0.0, 0.0, radius])


def _station_axes(ax, number: int, row: tuple) -> None:
    """Draw one station's contour in its section frame, its chord line and its pitch axis.

    ``row`` is the station's (radius, chord, pitch, contour).
    """
    radius, chord, pitch, contour = row
    points = np.asarray(contour, dtype=float)
    closed = np.vstack([points, points[:1]])
    ax.plot(closed[:, 0] * 1e3, closed[:, 1] * 1e3, color="#0072b2", linewidth=0.8)
    trailing = points[int(np.argmin(points[:, 0]))]
    leading = points[int(np.argmax(np.hypot(*(points - trailing).T)))]
    line = np.array([trailing, leading]) * 1e3
    ax.plot(line[:, 0], line[:, 1], color="#d55e00", linewidth=0.6, linestyle="--")
    ax.plot([0.0], [0.0], marker="+", color="black", markersize=6)
    ax.set_title(
        f"station {number}: r = {radius:.4f} m\nchord {chord * 1e3:.1f} mm, pitch {pitch:.2f} deg",
        fontsize=7,
    )
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("toward the leading edge [mm]", fontsize=6)
    ax.set_ylabel("toward the suction side [mm]", fontsize=6)


def _ordered_box(box: Sequence[float] | None) -> list[float] | None:
    """Return the box, each minimum before its maximum; a reversed pair is warned and swapped."""
    if box is None:
        return None
    values = [float(v) for v in box]
    if any(values[i] > values[i + 1] for i in (0, 2, 4)):
        _skipped("--box: a minimum exceeds its maximum; the two are swapped")
        values = [v for i in (0, 2, 4) for v in sorted(values[i : i + 2])]
    return values
