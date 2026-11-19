"""Build preparation, refinement and refusal inspection pages from planner records."""

from __future__ import annotations

import json
import textwrap
import warnings
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh._visual_inspect import (
    BOTTOM,
    DIRECTION_COLORS,
    PAGE,
    RIGHT,
    TOP,
    Mesh,
    Points,
    Scene,
    _skipped,
    patch_colors,
    read_obj,
    standard_views,
    views_page,
)


class Inspection:
    """Carry the outer refusal hook and optional level PDF through refinement."""

    def __init__(self, source, obj, enabled):
        self.source, self.obj, self.enabled = source, obj, enabled
        self.work: Any = None

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if isinstance(error, InputArtifactError):
            refusal_diagnostic(self.source, self.obj, error, self.work)
        return False

    def write(self, writer, work, *args):
        """Write the level and append its inspection before the final record."""
        written = writer(work, *args)
        if self.enabled:
            written.insert(-1, refined_pdf(work, written[0], written[-1]))
        return written


def _directions_page(
    pdf,
    drawn,
    names,
    layouts,
    views,
    *,
    family,
    unit,
    dpi,
    governing,
    arc_verts=None,
):
    """Render directions exclusively from planner records, without classifying geometry."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {n: (0.85, 0.85, 0.85, 1.0 if n in layouts else 0.25) for n in names}
    scene = Scene(drawn, names, colors, None, edges=False, perimeter_width=0.3)
    leads = {
        f"{n} leads {d}"
        for n in names
        for d, yes in zip(("normal", "parallel"), governing.get(n, (False, False)), strict=True)
        if yes
    }
    seen = set()
    for name in names:
        layout = layouts.get(name)
        if layout is None:
            continue
        labels = getattr(layout, "labels", {})
        dominant = getattr(layout, "leads", {})
        for patch, records in layout.directions.items():
            if layout.names[layout.holder[patch]] != name:
                continue
            for arc, record in records.items():
                key = (id(layout), arc, record.direction)
                if key in seen:
                    continue
                seen.add(key)
                label = labels.get(arc, "")
                lead = dominant.get(label)
                is_lead = lead is not None and lead[1] == arc
                if lead is not None:
                    leads.add(f"{lead[0]} leads {label}")
                vertices = drawn.verts if arc_verts is None else arc_verts
                xyz = vertices[list(layout.blocks.arcs[arc])]
                scene.direction_arcs.append((xyz, record.direction, label, is_lead))
    fig = plt.figure(figsize=PAGE)
    views_page(fig, scene, views, [], [], labels=False, unit=unit)
    handles = [Line2D([], [], color=c, lw=2, label=d) for d, c in DIRECTION_COLORS.items()]
    handles += [Line2D([], [], color="none", label=s) for s in sorted(leads)]
    fig.legend(
        handles=handles,
        loc="upper left",
        fontsize=6,
        title_fontsize=7,
        title="parallel / normal to the reference\n(trailing edge, else the longest side)",
        bbox_to_anchor=((PAGE[0] - RIGHT) / PAGE[0], (PAGE[1] - TOP) / PAGE[1]),
    )
    fig.suptitle(f"{drawn.path.name}: family {family}, directions", fontsize=10)
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def patches_pdf(
    mesh: str | Path | Mesh,
    groups: dict[str, list[str]],
    governing: dict[str, tuple[bool, bool]] | set[str],
    out: Path,
    *,
    mesh_unit: str | None = None,
    dpi: int = 200,
    layouts: Mapping[str, Any] | None = None,
    refusal: InputArtifactError | None = None,
    notes: Mapping[str, str] | None = None,
    arc_verts: np.ndarray | None = None,
    first: Callable[[Any, Mesh, list], None] | None = None,
) -> Path:
    """Write one page per original family of a prepared mesh: its patches colored and named.

    ``groups`` maps each original family to its patches (the families of the
    patches OBJ ``mesh``); ``governing`` maps a patch to its normal and parallel
    flags. Each lead is labeled GOVERNING NORMAL or GOVERNING PARALLEL in a red
    frame and listed so beside the views. A set of names means both directions.
    The pages are the four views of the
    technical drawing; a family whose patches hold no face is skipped.
    ``first`` draws its own pages ahead of them, given the open PDF, the mesh and the views.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    drawn = mesh if isinstance(mesh, Mesh) else read_obj(Path(mesh))
    flags = governing if isinstance(governing, dict) else dict.fromkeys(governing, (True, True))
    labels = {
        n: "\n".join(
            f"GOVERNING {direction}"
            for direction, leads in (("PARALLEL", pair[1]), ("NORMAL", pair[0]))
            if leads
        )
        for n, pair in flags.items()
    }
    views = standard_views((1.0, -1.0, 1.0))
    with PdfPages(out) as pdf:
        if refusal is not None:
            _refusal_page(pdf, drawn, refusal, views, unit=mesh_unit, dpi=dpi)
        if first is not None:
            first(pdf, drawn, views)
        for family, patches in groups.items():
            names = [p for p in patches if drawn.patches.get(p)]
            if not names:
                continue
            colors, _ = patch_colors(names, "tab20" if len(names) <= 20 else "turbo")
            scene = Scene(drawn, names, colors, None, edges=True, perimeter_width=1.2)
            scene.highlight = {n for n in names if n in governing}
            scene.highlight_labels = labels
            if refusal is not None:
                involved = set(refusal.where.get("families", ())) & set(names)
                scene.highlight |= involved
                scene.highlight_labels = {**labels, **dict.fromkeys(involved, "REFUSAL")}
            fig = plt.figure(figsize=PAGE)
            views_page(fig, scene, views, [], [], labels=True, unit=mesh_unit)
            pw, ph = fig.get_size_inches()
            handles = [
                Patch(
                    facecolor=colors[n],
                    edgecolor="#d62728" if n in scene.highlight else "0.2",
                    linewidth=1.5 if n in scene.highlight else 0.5,
                    label=f"{n}  ({scene.highlight_labels[n].replace(chr(10), ', ')})"
                    if n in scene.highlight
                    else n,
                )
                for n in names
            ]
            add_legend = partial(
                fig.legend,
                handles=handles,
                loc="upper left",
                title=f"{family}: {len(names)} patches, {len(scene.highlight)} governing",
                bbox_to_anchor=((pw - RIGHT + 0.1) / pw, (ph - TOP - 0.05) / ph),
                ncol=1 if len(names) <= 40 else 2,
                borderaxespad=0,
            )
            legend = add_legend(fontsize=6, title_fontsize=7)
            bounds = legend.get_window_extent()
            width, height = bounds.width / fig.dpi, bounds.height / fig.dpi
            fit = min(1.0, (RIGHT - 0.2) / width, (ph - TOP - BOTTOM - 0.05) / height)
            if fit < 1:
                legend.remove()
                # Leave room for font metric rounding between the screen and PDF renderers.
                fit *= 0.9
                add_legend(fontsize=6 * fit, title_fontsize=7 * fit)
            fig.suptitle(
                f"{drawn.path.name}: family {family}", fontsize=10, y=1 - 0.2 / PAGE[1], va="top"
            )
            if notes:
                fig.text(0.03, 0.02, "\n".join(notes[n] for n in names if n in notes), fontsize=6)
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
            if layouts is not None:
                _directions_page(
                    pdf,
                    drawn,
                    names,
                    layouts,
                    views,
                    family=family,
                    unit=mesh_unit,
                    dpi=dpi,
                    governing=flags,
                    arc_verts=arc_verts,
                )
    return out


def _refusal_page(pdf, mesh, error, views, *, unit, dpi):
    """Draw the whole source with only structurally supplied locations highlighted."""
    import matplotlib.pyplot as plt

    names = list(mesh.patches)
    involved = set(error.where.get("families", ()))
    colors = {n: (0.9, 0.3, 0.2, 1.0) if n in involved else (0.8, 0.8, 0.8, 0.4) for n in names}
    scene = Scene(mesh, names, colors, None)
    xyz = list(error.where.get("points", ()))
    xyz += [mesh.verts[i] for i in error.where.get("nodes", ()) if 0 <= i < len(mesh.verts)]
    points = [Points(mesh.path, "refusal", np.asarray(xyz, dtype=float).reshape(-1, 3))]
    fig = plt.figure(figsize=PAGE)
    views_page(fig, scene, views, points, [{"color": "red", "s": 14}], labels=False, unit=unit)
    fig.suptitle(f"{mesh.path.name}: refinement refused", fontsize=10)
    message = str(error).removesuffix("Nothing was written.").rstrip()
    text = textwrap.fill(f"{message} No level was written.", width=42, break_long_words=True)
    fig.text(
        (PAGE[0] - RIGHT + 0.1) / PAGE[0],
        0.9,
        text,
        va="top",
        fontsize=min(7, 410 / max(len(text.splitlines()), 1)),
    )
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def refusal_diagnostic(source, obj, error: InputArtifactError, work=None) -> None:
    """Attempt the diagnostic without ever replacing the original refinement refusal."""
    target = source.with_name(source.stem + ".refine-refusal.pdf")
    try:
        drawn = Mesh(source, obj.verts, dict(obj.families) if work is None else work.original)
        governing = (
            {} if work is None else {n: s.leads for n, s in work.specs.items() if any(s.leads)}
        )
        patches_pdf(
            drawn,
            _family_groups(drawn.patches, {} if work is None else work.request.components),
            governing,
            target,
            layouts={} if work is None else work.layouts,
            refusal=error,
        )
    except Exception as drawing_error:
        try:
            _skipped(f"{source}: could not write the diagnostic PDF ({drawing_error})")
        except Warning:
            pass  # Warnings-as-errors must not replace the original refusal either.
    else:
        message = str(error).removesuffix("Nothing was written.").rstrip()
        error.args = (f"{message} No level was written. A diagnostic PDF: {target}.",)


def refined_pdf(work, source: Path, record: Path) -> Path:
    """Draw the governors, then the completed parts under their output components.

    The PDF is ``<level>.refine-governors.pdf`` beside ``record``: a summary page, the
    requested and achieved factor pages, one page per governing family and direction,
    then the patches and directions pages with the recorded continuity notes.
    """
    data = json.loads(record.read_text(encoding="utf-8"))
    vertices = [work.mesh.verts]
    patches = {n: list(faces) for n, faces in work.mesh.families.items() if n not in work.bands}
    offset = len(work.mesh.verts)
    for points, families, _ in work.parts.values():
        vertices.append(points)
        for name, faces in families.items():
            moved = [[v + offset for v in face] for face in faces]
            if name in work.bands:
                patches[work.bands[name]] += moved
            else:
                patches[name] = moved
        offset += len(points)
    drawn = Mesh(source, np.vstack(vertices), patches)
    grouped = _family_groups(patches, data.get("components", {}))
    governing = {n: s.leads for n, s in work.specs.items() if any(s.leads)}
    notes = {
        n: f"{n}: continuity {entry['continuity']}"
        if entry.get("continuity")
        else f"{n}: continuity remeshed band ({entry['band']} source faces)"
        for n, entry in data["families"].items()
        if entry.get("continuity") or entry.get("band")
    }
    pages = _governors(data, work.layouts, work.specs)
    try:
        factors = factor_rows(data, work)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        # The table is a reading aid: a record it cannot read leaves it out, never the level.
        warnings.warn(
            f"{record.name}: the requested and achieved factor table was left out of the "
            f"inspection PDF ({type(exc).__name__}: {exc})",
            GeometryWarning,
            stacklevel=2,
        )
        factors = []
    return patches_pdf(
        drawn,
        grouped,
        governing,
        record.parent / (source.stem + GOVERNORS_SUFFIX),
        layouts=work.layouts,
        notes=notes,
        mesh_unit=work.te_unit,
        arc_verts=work.mesh.verts,
        first=partial(
            _governor_pages, pages=pages, specs=work.specs, unit=work.te_unit, factors=factors
        ),
    )


#: The refinement's inspection PDF beside its record; the standalone visual-inspect
#: writes ``<mesh>.inspect.pdf``, so the two never replace each other.
GOVERNORS_SUFFIX = ".refine-governors.pdf"

#: What each direction means for a family, as the mesh reference words it.
MEANING = {
    "parallel": "PARALLEL runs along the family's reference perimeter "
    "(its trailing edge, else its longest side).",
    "normal": "NORMAL runs across the family's reference perimeter "
    "(its trailing edge, else its longest side).",
}

_FAINT = (0.85, 0.85, 0.85, 0.3)


def _governors(data, layouts, specs) -> dict[tuple[str, str], dict[str, Any]]:
    """Group the recorded bands by lead and direction, with the families each governs.

    A band governs every family holding one of its arcs. Under component governing,
    a component's governor of a direction also governs its other refined grids.
    """
    held = {lay.names: lay for lay in layouts.values()}
    pages: dict[tuple[str, str], dict[str, Any]] = {}

    def page(lead, direction):
        return pages.setdefault((lead, direction), {"bands": [], "families": set()})

    for record in data.get("layouts", ()):
        layout = held.get(tuple(record["families"]))
        for band in record["bands"]:
            entry = page(band["lead"], band["direction"])
            entry["bands"].append(f"{band['label']} ({band['count']})")
            entry["families"] |= _band_families(layout, band["arcs"])
    components = data.get("components", {})
    for component, leads in data.get("component_governors", {}).items():
        refined = {
            n
            for n in components.get(component, ())
            if getattr(specs.get(n), "method", "") in ("grid", "grid_auto")
        }
        for direction, lead in leads.items():
            page(lead, direction)["families"] |= refined
    for (lead, _), entry in pages.items():
        entry["families"].discard(lead)
    return dict(sorted(pages.items(), key=lambda item: (item[0][0], item[0][1] != "parallel")))


def _band_families(layout, arcs) -> set[str]:
    """Return the families of the layout whose patches hold one of the band's arcs."""
    if layout is None:
        return set()
    wanted = set(arcs)
    return {
        layout.names[patch_of]
        for patch_of, sides in zip(layout.holder, layout.blocks.sides, strict=False)
        if wanted & {arc for arc, _ in sides}
    }


def _factor(specs, lead, direction) -> str:
    """Return the factor the lead refines its direction by, as text."""
    spec = specs.get(lead)
    if spec is None:
        return "not stated"
    return f"{spec.directions[direction == 'parallel']:g}"


def _governor_pages(pdf, drawn, views, *, pages, specs, unit, factors=(), dpi=200):
    """Draw the summary page, the factor comparison, then one page per governor and direction."""
    from matplotlib import colormaps

    _governors_summary(pdf, drawn.path.name, pages, specs, dpi=dpi)
    factor_pages(pdf, drawn.path.name, factors, dpi=dpi)
    names = list(drawn.patches)
    scene = Scene(drawn, names, dict.fromkeys(names, _FAINT), None, edges=False)
    scene.perimeter_width, scene.perimeter_color = 0.25, (0.6, 0.6, 0.6, 1.0)
    palette = colormaps["tab10"]
    for number, ((lead, direction), entry) in enumerate(pages.items()):
        strong = palette(number % 10)
        _governor_page(
            pdf,
            scene,
            views,
            (lead, direction, entry),
            strong=strong,
            factor=_factor(specs, lead, direction),
            unit=unit,
            dpi=dpi,
        )


def _governors_summary(pdf, name, pages, specs, *, dpi):
    """Tabulate each governing family, its direction, bands and governed families."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=PAGE)
    fig.suptitle(f"{name}: governing families", fontsize=16, fontweight="bold")
    intro = (
        "Each governing family sets the element count of every curve it shares in one "
        "direction. One page per governing family and direction follows.\n"
        f"{MEANING['parallel']}\n{MEANING['normal']}"
    )
    fig.text(0.05, 0.90, intro, fontsize=8, va="top")
    ax = fig.add_axes((0.05, 0.05, 0.9, 0.72))
    ax.set_axis_off()
    rows = [
        [
            lead,
            direction.upper(),
            ", ".join(entry["bands"][:6]) + (" ..." if len(entry["bands"]) > 6 else ""),
            _factor(specs, lead, direction),
            str(len(entry["families"])),
        ]
        for (lead, direction), entry in pages.items()
    ]
    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=["Governing family", "Direction", "Bands (count)", "Factor", "Governed"],
            loc="upper center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(min(8.0, 280 / len(rows)))
        table.auto_set_column_width(range(5))
    else:
        ax.text(0.5, 0.9, "No band of this level records a governing family.", ha="center")
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def _governor_page(pdf, scene, views, governor, *, strong, factor, unit, dpi):
    """Fill the governor strongly and each family it governs in its own color."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lead, direction, entry = governor
    governed = [n for n in scene.names if n in entry["families"]]
    # tab20 pairs a dark and a light shade of each tab10 hue: leave out both shades of
    # the governor's hue so no governed family reads as the governor.
    hues = plt.get_cmap("tab10").colors
    hue = min(range(len(hues)), key=lambda k: float(np.sum((np.array(hues[k]) - strong[:3]) ** 2)))
    palette = [c for i, c in enumerate(plt.get_cmap("tab20").colors) if i // 2 != hue]
    colors = {n: (*palette[i % len(palette)], 1.0) for i, n in enumerate(governed)}
    fills = {**dict.fromkeys(scene.names, _FAINT), **colors, lead: strong}
    if len(scene.faces):
        scene.base = np.array([fills[n] for n in scene.names])[scene.patch_of]
    # The legend names every colored family and the title the direction; labels over the
    # mesh would repeat them and hide it.
    scene.labelled = set()
    scene.highlight = set()
    scene.highlight_labels = {}
    fig = plt.figure(figsize=PAGE)
    views_page(fig, scene, views, [], [], labels=False, unit=unit)
    pw, ph = fig.get_size_inches()
    fig.suptitle(
        f"{lead} governs the {direction.upper()} direction",
        fontsize=16,
        fontweight="bold",
        y=1 - 0.12 / ph,
        va="top",
    )
    words = (
        f"{MEANING[direction]} Bands it leads, with their element count: "
        f"{', '.join(entry['bands']) or 'none recorded'}. {direction.capitalize()} factor "
        f"{factor}. It governs {len(governed)} other "
        f"famil{'y' if len(governed) == 1 else 'ies'}."
    )
    fig.text(
        (pw - RIGHT + 0.1) / pw,
        (ph - TOP - 0.05) / ph,
        textwrap.fill(words, width=44),
        va="top",
        fontsize=7,
    )
    handles = [
        Patch(facecolor=strong, edgecolor="black", linewidth=1.5, label=f"{lead} (GOVERNING)")
    ]
    handles += [Patch(facecolor=colors[n], edgecolor="0.4", label=n) for n in governed]
    handles.append(Patch(facecolor=_FAINT, edgecolor="0.6", label="other families (outline)"))
    size = 6 if len(handles) <= 40 else max(3.0, 6 * 40 / len(handles))
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        fontsize=size,
        title_fontsize=7,
        title=f"governed in the {direction} direction",
        bbox_to_anchor=((pw - RIGHT + 0.1) / pw, (ph - TOP - 1.6) / ph),
        ncol=1 if len(handles) <= 40 else 2,
        borderaxespad=0,
    )
    legend.get_texts()[0].set_fontweight("bold")
    legend.get_texts()[0].set_fontsize(size + 1)
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


#: Rows per comparison page, so A4 landscape keeps a readable font.
FACTOR_ROWS_PER_PAGE = 35

FACTOR_COLUMNS = (
    "Family",
    "Direction",
    "Governed by",
    "Requested\nfactor",
    "Governor's\nfactor",
    "Intervals\nbefore -> after",
    "Achieved\nfactor",
    "Faces\nbefore -> after",
)


def _requested(data, work, name, direction) -> float | None:
    """Return a family's own factor in a direction, from the record's specs, else the plan's."""
    if direction not in ("parallel", "normal"):
        return None
    spec = data.get("specs", {}).get(name)
    if spec is not None:
        own = spec.get("spanwise" if direction == "parallel" else "chordwise")
        return float(own if own is not None else spec.get("factor", 1.0))
    planned = work.specs.get(name)
    return None if planned is None else float(planned.directions[direction == "parallel"])


def _band_leads(data) -> dict[tuple[str, ...], dict[int, tuple[str, str]]]:
    """Per recorded layout, each arc's band lead and the lead's direction."""
    return {
        tuple(record["families"]): {
            arc: (band["lead"], band["direction"])
            for band in record["bands"]
            for arc in band["arcs"]
        }
        for record in data.get("layouts", ())
    }


def _block_axes(entry, layout, name, leads):
    """Yield (direction, before, after, lead, lead direction) per patch axis of a multiblock.

    A patch's columns are counted along its first row's arc and its rows along its first
    column's (``_refine._blocks.SIDES``); the planner's per-patch arc direction names
    each axis. Without that record the direction is ``?``.
    """
    mine = [] if layout is None else layout.patches_of(name)
    for i, block in enumerate(entry["blocks"]):
        patch = mine[i] if i < len(mine) else None
        for key, side in (("columns", 0), ("rows", 2)):
            arc = None if patch is None else layout.blocks.sides[patch][side][0]
            told = None if patch is None else layout.directions.get(patch, {}).get(arc)
            direction = "?" if told is None else told.direction
            lead = leads.get(arc, (name, direction))
            yield (direction, *block[key], *lead)


def _axes(data, work, name, entry):
    """Yield every interval axis of a refined family with the family governing it."""
    if "intervals" in entry:
        chord, span = entry["intervals"]["chordwise"], entry["intervals"]["spanwise"]
        axes = [("parallel", *span, name, "parallel"), ("normal", *chord, name, "normal")]
    elif "blocks" in entry:
        layout = work.layouts.get(name)
        leads = _band_leads(data).get(tuple(layout.names), {}) if layout is not None else {}
        axes = list(_block_axes(entry, layout, name, leads))
    else:
        return []
    component = next((c for c, ns in data.get("components", {}).items() if name in ns), None)
    over = data.get("component_governors", {}).get(component, {})
    return [
        (d, before, after, *((over[d], d) if d in over else (lead, ld)))
        for d, before, after, lead, ld in axes
    ]


def _faces(work, entry, name) -> str:
    faces = entry.get("faces")
    if isinstance(faces, list) and len(faces) == 2:
        return f"{faces[0]} -> {faces[1]}"
    count = len(work.mesh.families.get(name, ()))
    return f"{count} -> {count}"


def _factor_row(data, work, name, direction, axes) -> dict[str, Any]:
    """Sum one family's axes of a direction and compare the achieved factor with the governor's."""
    before = sum(a[1] for a in axes)
    after = sum(a[2] for a in axes)
    governors = {(a[3], a[4]) for a in axes}
    factors = {g: _requested(data, work, *g) for g in governors}
    # One whole interval rounds each patch axis: beyond half an interval of the governor's
    # count, rounding does not explain the achieved count.
    flagged = any(
        (f := factors[(a[3], a[4])]) is not None and abs(a[2] - a[1] * f) > 0.5 + 1e-9 for a in axes
    )
    requested = None if direction == "?" else _requested(data, work, name, direction)
    shown = sorted({"itself" if lead == name else lead for lead, _ in governors})
    values = sorted({f"{f:g}" for f in factors.values() if f is not None})
    return {
        "cells": [
            name,
            direction.upper(),
            ", ".join(shown),
            "-" if requested is None else f"{requested:g}",
            ", ".join(values) or "-",
            f"{before} -> {after}",
            (f"{after / before:.3f}" if before else "-") + (" *" if flagged else ""),
            _faces(work, data["families"][name], name),
        ],
        "flagged": flagged,
    }


def factor_rows(data, work) -> list[dict[str, Any]]:
    """Return the requested and achieved factor rows, sorted by family then direction.

    A family without interval axes (copied, unchanged or remeshed) is one row with
    direction ``-`` and its method in place of its governor.
    """
    rows = []
    for name in sorted(data.get("families", {})):
        entry = data["families"][name]
        axes = _axes(data, work, name, entry)
        if not axes:
            method = entry.get("method", "copied")
            cells = [name, "-", method, "-", "-", "-", "-", _faces(work, entry, name)]
            rows.append({"cells": cells, "flagged": False})
            continue
        order = sorted(
            {a[0] for a in axes},
            key=lambda d: ("parallel", "normal").index(d) if d in ("parallel", "normal") else 2,
        )
        rows += [_factor_row(data, work, name, d, [a for a in axes if a[0] == d]) for d in order]
    return rows


def factor_pages(pdf, name, rows, *, dpi):
    """Tabulate requested against achieved factors, at most FACTOR_ROWS_PER_PAGE per page."""
    import matplotlib.pyplot as plt

    chunks = [rows[i : i + FACTOR_ROWS_PER_PAGE] for i in range(0, len(rows), FACTOR_ROWS_PER_PAGE)]
    notes = [
        "* The achieved factor departs from the governor's by more than rounding to whole "
        "intervals explains (over half an interval in some patch).",
        "-  A family without interval axes (copied, unchanged or remeshed): its method "
        "stands under Governed by.",
    ]
    if any(r["cells"][1] == "?" for r in rows):
        notes.append("?  The planner recorded no direction for that patch axis.")
    for number, chunk in enumerate(chunks, 1):
        fig = plt.figure(figsize=PAGE)
        fig.suptitle(
            f"{name}: requested and achieved factors ({number} of {len(chunks)})",
            fontsize=16,
            fontweight="bold",
        )
        if number == 1:
            fig.text(
                0.04,
                0.905,
                "Achieved factor = intervals after / intervals before in that direction, "
                "summed over the family's patches; a governed family follows its governor.",
                fontsize=10,
                va="top",
            )
        fig.text(0.04, 0.075, "\n".join(notes), fontsize=9, va="top")
        _factor_table(fig, chunk)
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)


def _factor_table(fig, chunk):
    """Draw one page of rows: header bold on gray, zebra rows, flagged rows bold red."""
    ax = fig.add_axes((0.04, 0.12, 0.92, 0.76))
    ax.set_axis_off()
    table = ax.table(
        cellText=[r["cells"] for r in chunk],
        colLabels=list(FACTOR_COLUMNS),
        colWidths=[0.21, 0.08, 0.17, 0.08, 0.09, 0.12, 0.08, 0.17],
        cellLoc="left",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # Every page keeps one row height; the header holds two lines.
    height = 1 / (FACTOR_ROWS_PER_PAGE + 2)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_height(2 * height if row == 0 else height)
        if row == 0:
            cell.set_facecolor("0.85")
            cell.get_text().set_fontweight("bold")
            continue
        flagged = chunk[row - 1]["flagged"]
        cell.set_facecolor("#fde0dc" if flagged else ("0.95" if row % 2 == 0 else "white"))
        if flagged:
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#b00020")


def _family_groups(patches, components):
    """Read component membership, keeping ungrouped families as their own pages."""
    grouped = {n: list(members) for n, members in components.items()}
    members = {n for group in grouped.values() for n in group}
    grouped.update({n: [n] for n in patches if n not in members})
    return grouped
