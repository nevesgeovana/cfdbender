"""Write the products a point's p file declares, from the files its solver wrote.

For each recorded point with a result, the post reads the p file's ``[sections]``,
``[volume_section]`` and ``[[probes]]`` (:mod:`cfdbender.core.pproc`), asks the adapter
which files and fields it supplies (:mod:`cfdbender.core.contract`) and writes, under
``post/<matrix>/``, with pyflightstream's names, in a folder per polar:

* ``sections/P<POL>/<point>_cp_<name>.csv`` and ``sections/P<POL>/<point>_sloads_<name>.csv``
  per distribution entry. A row leads with ``POL, STEP, time_s, FAMILY, PLANE, ROTOR,
  AZIMUTH``, then the condition block, then the export's columns. On a steady point the
  file is one instant and ``STEP`` the solver iteration of the files; on a time-accurate point
  with an export window the file holds every exported step, in order, and is a history;
  without one it holds the last step, one instant;
* ``fields/P<POL>/<point>_vsec.vtk`` (or ``.dat``), with ``_step_<STEP>`` per exported step of
  a time-accurate point, each with its provenance file;
* ``probes/P<POL>/<point>_probes.csv`` (and ``_probes_history.csv`` beside it on a
  time-accurate row that mixes the two kinds of entry).

A column the adapter cannot fill, or whose field its file lacks, is absent and named with its
reason; a product whose file or frame is missing is skipped with its reason, and never stops
another product or the polar. The polar folder ``P<POL>/`` is cfdbender's, because a cfdbender
point name does not carry its polar.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.contract import (
    FreeStream,
    ProductInputs,
    ProductSupport,
    StepFile,
    product_support,
)
from cfdbender.core.plan import workspace_path
from cfdbender.core.pproc import (
    Probes,
    SectionDistribution,
    VolumeSection,
    read_post,
)
from cfdbender.post.product_frames import product_frames, rotor_table, section_frame
from cfdbender.post.sections import (
    CP_COLUMNS,
    INTEGRATED_COLUMNS,
    PARTS_COLUMNS,
    PARTS_INTEGRATED_COLUMNS,
    SLOADS_COLUMNS,
    Distribution,
    Reference,
    SectionError,
    SectionsResult,
    Surface,
    compute_sections,
    distribution_names,
    write_section_series,
)
from cfdbender.post.volume_files import (
    ProbeEntry,
    VolumeStep,
    write_probe_table,
    write_volume_section,
)
from cfdbender.post.volume_frames import Frame
from cfdbender.post.volume_layouts import (
    circle_points,
    line_points,
    rectangle_points,
    volume_section_points,
)
from cfdbender.post.vtu import read_vtu

#: The leading identity columns of a sections file, before the condition block.
SECTION_LEAD = ("POL", "STEP", "time_s", "FAMILY", "PLANE", "ROTOR", "AZIMUTH")
#: The Cp-table columns a surface field fills, by the sections cut's names.
_FIELD_COLUMNS = ("Mach", "vx", "vy", "vz", "Cp_ref", "CF", "Theta", "Delta*", "Delta", "H")
_NA = "NA"


@dataclass
class PointProducts:
    """What the products of one point wrote and skipped.

    ``files`` maps each written product (its final path) to what ``products.json`` states
    about it; ``skipped`` holds ``(product, reason)``.
    """

    files: dict[Path, dict[str, object]] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _fields(
    grid_data: Mapping[str, Any],
    columns: Mapping[str, object],
    free: FreeStream | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve each mapped column from the file's fields; return the values and the absent."""
    found: dict[str, Any] = {}
    absent: dict[str, str] = {}
    for name, source in columns.items():
        try:
            if isinstance(source, str):
                found[name] = grid_data[source]
            elif free is None:
                absent[name] = "the run states no free stream to compute it from"
            else:
                found[name] = cast(Callable[..., Any], source)(grid_data, free)
        except KeyError as error:
            absent[name] = f"the file holds no field {error.args[0]}"
        except (IndexError, TypeError, ValueError, ArithmeticError) as error:
            raise ValueError(f"column {name!r} cannot be computed: {error}") from error
    return found, absent


def _word_families(
    word: str, reference: Mapping[str, object], families: Sequence[str]
) -> list[str] | None:
    """Return the families a word stands for, None for one that stands for nothing."""
    if word == "all":
        return list(families)
    aliases = reference.get("aliases")
    if isinstance(aliases, Mapping) and word in aliases:
        members = aliases[word]
        out: list[str] = []
        for member in members if isinstance(members, list) else [members]:
            out += _word_families(str(member), reference, families) or []
        return out or None
    found = rotor_table(reference)
    if found is not None and word == found[0].name:
        _, table = found
        owned = [
            *cast(list[str], table.get("families_blades") or []),
            *cast(list[str], table.get("families_general") or []),
        ]
        return [name for name in owned if name in families] or None
    return [word] if word in families else None


def _selected(
    entry: SectionDistribution, reference: Mapping[str, object], inputs: ProductInputs
) -> tuple[list[str], str | None]:
    """Return the families an entry cuts, or the reason it cuts none."""
    words = [entry.families] if isinstance(entry.families, str) else list(entry.families)
    if "each" in words:
        return [], "families = 'each' is not offered by cfdbender; name the families"
    chosen: list[str] = []
    for word in words:
        for name in _word_families(word, reference, inputs.families) or []:
            if name not in chosen:
                chosen.append(name)
    if not chosen:
        return [], (
            f"families {words} name no family of the surface file "
            f"(it holds: {', '.join(inputs.families) or 'none'})"
        )
    if inputs.zoned:
        unplaced = [name for name in chosen if name not in inputs.family_zones]
        if unplaced:
            return [], (
                f"the zone of {unplaced} is not known (the point folder's mesh.su2 states "
                f"{sorted(inputs.family_zones) or 'no marker of the surface files'})"
            )
        for zone in sorted({inputs.family_zones[name] for name in chosen}):
            held = [name for name in inputs.families if inputs.family_zones.get(name) == zone]
            if set(held) - set(chosen):
                return [], (
                    f"the surface file of zone {zone} does not say which cell is of which "
                    f"family, so a distribution cuts all of {held} or none; this entry cuts "
                    f"{chosen}"
                )
    elif not inputs.families_separable and set(chosen) != set(inputs.families):
        return [], (
            f"the surface file does not say which cell is of which family, so a distribution "
            f"cuts all of {list(inputs.families)} or none; this entry cuts {chosen}"
        )
    return chosen, None


def _surface(
    paths: Sequence[Path], support: ProductSupport, inputs: ProductInputs, chosen: Sequence[str]
) -> tuple[Surface, dict[str, str]]:
    """Read the surface file of a step: one file, or the files of the zones joined in one."""
    read = [_zone_surface(path, support, inputs, chosen) for path in paths]
    if len(read) == 1:
        return read[0]
    absent: dict[str, str] = {}
    for _, own in read:
        absent.update(own)
    offsets = np.cumsum([0, *(len(surface.points) for surface, _ in read)])
    surfaces = [surface for surface, _ in read]
    names = set.intersection(*(set(surface.fields) for surface in surfaces))
    shear = (
        np.concatenate([cast(Any, s.shear) for s in surfaces])
        if all(s.shear is not None for s in surfaces)
        else None
    )
    return (
        Surface(
            np.concatenate([s.points for s in surfaces]),
            np.concatenate([s.triangles + int(o) for s, o in zip(surfaces, offsets, strict=False)]),
            np.concatenate([s.quads + int(o) for s, o in zip(surfaces, offsets, strict=False)]),
            np.concatenate([s.pressure for s in surfaces]),
            {name: np.concatenate([s.fields[name] for s in surfaces]) for name in sorted(names)},
            surfaces[0].families,
            None,
            None,
            inputs.normal_sign,
            shear,
        ),
        absent,
    )


def _zone_surface(
    path: Path, support: ProductSupport, inputs: ProductInputs, chosen: Sequence[str]
) -> tuple[Surface, dict[str, str]]:
    grid = read_vtu(path)
    triangles, _ = grid.cells_of_type(5)
    quads, _ = grid.cells_of_type(9)
    data = dict(grid.point_data)
    values, absent = _fields(data, support.columns, inputs.free_stream)
    if "pressure" not in values:
        raise SectionError(absent.get("pressure", "the adapter maps no pressure"))
    fields = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in values.items()
        if name not in ("pressure", "shear")
    }
    if inputs.families_separable and inputs.family_field is not None:
        if inputs.family_field not in grid.cell_data:
            raise SectionError(
                f"the surface file holds no cell family field {inputs.family_field!r}"
            )
        cell_family = np.asarray(grid.cell_data[inputs.family_field], dtype=np.int64)
        _, tri_cells = grid.cells_of_type(5)
        _, quad_cells = grid.cells_of_type(9)
        surface = Surface(
            grid.points,
            triangles.reshape(-1, 3) if triangles.size else np.zeros((0, 3), np.int64),
            quads.reshape(-1, 4) if quads.size else np.zeros((0, 4), np.int64),
            np.asarray(values["pressure"], dtype=np.float64),
            fields,
            tuple(inputs.families),
            cell_family[tri_cells],
            cell_family[quad_cells],
            inputs.normal_sign,
        )
    else:
        surface = Surface(
            grid.points,
            triangles.reshape(-1, 3) if triangles.size else np.zeros((0, 3), np.int64),
            quads.reshape(-1, 4) if quads.size else np.zeros((0, 4), np.int64),
            np.asarray(values["pressure"], dtype=np.float64),
            fields,
            ("+".join(chosen),),
            normal_sign=inputs.normal_sign,
        )
    if "shear" in values:
        surface = replace(surface, shear=np.asarray(values["shear"], dtype=np.float64))
    return surface, absent


def _fillable(support: ProductSupport, absent: Mapping[str, str], integrate: bool) -> set[str]:
    wanted = (
        set(CP_COLUMNS) | set(SLOADS_COLUMNS) | (set(INTEGRATED_COLUMNS) if integrate else set())
    )
    for name in _FIELD_COLUMNS:
        if name in support.unfillable or name in absent or name not in support.columns:
            wanted.discard(name)
    if not all(c in support.columns and c not in absent for c in ("vx", "vy", "vz")):
        wanted.discard("vtot")
    if "Cp_ref" in support.unfillable:
        wanted.discard("Cp_ref")
    if "shear" in support.columns and "shear" not in absent:
        wanted |= set(PARTS_COLUMNS) | (set(PARTS_INTEGRATED_COLUMNS) if integrate else set())
    return wanted


def _friction_absent(
    support: ProductSupport, absent: Mapping[str, str], integrate: bool
) -> dict[str, str]:
    """Name the friction columns of a sectional-loads file that are absent, with the reason."""
    if "shear" in support.columns and "shear" not in absent:
        return {}
    reason = (
        absent.get("shear")
        or support.unfillable.get("shear")
        or "the adapter supplies no wall shear field"
    )
    columns = (*PARTS_COLUMNS, *(PARTS_INTEGRATED_COLUMNS if integrate else ()))
    return dict.fromkeys(columns, reason)


@dataclass(frozen=True)
class _Context:
    point: AdapterPoint
    pol: str
    name: str
    lead: Mapping[str, str]
    frames: Mapping[str, Frame]
    moment_point: tuple[float, float, float]
    cell: Callable[[object], str]


def _azimuth(
    reference: Mapping[str, object], chosen: Sequence[str], step: int | None, spr: float | None
) -> tuple[str, float | None]:
    found = rotor_table(reference)
    if found is None:
        return _NA, None
    block, table = found
    blades = cast(list[str], table.get("families_blades") or [])
    owned = {*blades, *cast(list[str], table.get("families_general") or [])}
    if not chosen or not set(chosen) <= owned:
        return _NA, None
    blade = blades.index(chosen[0]) + 1 if len(chosen) == 1 and chosen[0] in blades else 1
    if spr and step is not None:
        return block.alias, block.blade_azimuth_deg(blade, step=step, steps_per_revolution=spr)
    return block.alias, block.blade_azimuth_deg(blade, step=0, steps_per_revolution=1.0)


def _sections(
    ctx: _Context,
    entries: Sequence[SectionDistribution],
    support: ProductSupport,
    inputs: ProductInputs,
    stage: Path,
    out: PointProducts,
    target: Callable[[str], Path],
) -> None:
    names = distribution_names([e.families for e in entries])
    files = _by_step(inputs.surface)
    kind = "history" if inputs.windowed else "instant"
    if inputs.time_accurate and not inputs.windowed:
        files = files[-1:]
    relative = {
        e.position: (
            f"sections/P{ctx.pol}/{ctx.name}_cp_{n}.csv",
            f"sections/P{ctx.pol}/{ctx.name}_sloads_{n}.csv",
        )
        for e, n in zip(entries, names, strict=True)
    }
    if not files:
        reason = _missing(inputs)
        for pair in relative.values():
            out.skipped += [(path, f"no surface file: {reason}") for path in pair]
        return
    reference = ctx.point.reference
    for entry, name in zip(entries, names, strict=True):
        cp_rel, sl_rel = relative[entry.position]
        chosen, why = _selected(entry, reference, inputs)
        frame = ctx.frames.get(entry.frame)
        if why is None and frame is None:
            why = (
                f"frame {entry.frame!r} is not one this point places (frames: "
                f"{', '.join(sorted(ctx.frames))})"
            )
        if why is None and inputs.free_stream is not None and inputs.free_stream.q_ref <= 0.0:
            why = "the free stream is at rest, so no Cp has a reference"
        if (
            why is None
            and inputs.free_stream is not None
            and not math.isfinite(inputs.free_stream.q_ref)
        ):
            why = "the reference dynamic pressure is not finite"
        if why is None and inputs.free_stream is None:
            why = "the run states no free stream for the Cp columns"
        if why is not None:
            out.skipped += [(cp_rel, why), (sl_rel, why)]
            continue
        assert frame is not None and inputs.free_stream is not None
        steps: list[tuple[SectionsResult, dict[str, str]]] = []
        absent_all: dict[str, str] = {
            column: reason for column, reason in support.unfillable.items() if column in CP_COLUMNS
        }
        skipped_stations: list[str] = []
        tabled: list[int | None] = []
        absent_loads: dict[str, str] = {}
        try:
            cp_target, sl_target = target(cp_rel), target(sl_rel)
            zones = {inputs.family_zones[name] for name in chosen} if inputs.zoned else None
            used: list[StepFile] = []
            for group in files:
                # A multizone step is cut from the zone files that hold the entry's markers.
                members = [f for f in group if zones is None or f.zone in zones]
                if zones is not None and len(members) < len(zones):
                    continue
                item = members[0]
                used.extend(members)
                surface, absent = _surface(
                    [workspace_path(ctx.point.root, f.path) for f in members],
                    support,
                    inputs,
                    chosen,
                )
                absent_all.update({k: v for k, v in absent.items() if k in CP_COLUMNS})
                absent_loads.update(_friction_absent(support, absent, entry.integrate))
                result = compute_sections(
                    surface,
                    Reference(inputs.free_stream.p_inf, inputs.free_stream.q_ref),
                    Distribution(
                        tuple(chosen) if inputs.families_separable else (),
                        "F",
                        entry.planes,
                        entry.count,
                        entry.plot_direction,
                        entry.integrate,
                    ),
                    frames={"F": section_frame(frame, item.step)},
                    fillable=_fillable(support, absent, entry.integrate),
                    moment_point=ctx.moment_point,
                )
                result.family_label = "+".join(chosen)
                step = item.step if inputs.time_accurate else inputs.iteration
                rotor, azimuth = _azimuth(reference, chosen, item.step, inputs.steps_per_revolution)
                lead = {
                    "POL": ctx.pol,
                    "STEP": ctx.cell(step),
                    "time_s": ctx.cell(item.time_s),
                    "FAMILY": "",
                    "PLANE": "",
                    "ROTOR": rotor,
                    "AZIMUTH": ctx.cell(azimuth),
                    **ctx.lead,
                }
                skipped_stations += [f"STEP {lead['STEP']}: {text}" for text in result.skips]
                tabled.append(step)
                steps.append((result, lead))
            if not steps:
                reason = (
                    f"no step has the surface file of every zone the entry cuts: {_missing(inputs)}"
                )
                out.skipped += [(cp_rel, reason), (sl_rel, reason)]
                continue
            cp_path, sl_path = write_section_series(steps, stage, ctx.name, name)
        except (OSError, ValueError) as error:
            out.skipped += [(cp_rel, str(error)), (sl_rel, str(error))]
            continue
        common: dict[str, object] = {
            "point": ctx.point.point.point_id,
            "distribution": entry.position,
            "families": list(chosen),
            "frame": entry.frame,
            "kind": kind,
            "steps_tabled": tabled,
            "skipped_stations": skipped_stations,
            "cp_reference": inputs.free_stream.cp_reference,
            "sources": [item.path for item in used],
        }
        written_cp = set(steps[0][0].cp_columns)
        absent_cp = {
            column: absent_all.get(column, "the adapter does not fill it")
            for column in CP_COLUMNS
            if column not in written_cp
        }
        out.files[cp_target] = {
            **common,
            "product": "sections_cp",
            "absent_columns": absent_cp,
            "staged": str(cp_path),
        }
        out.files[sl_target] = {
            **common,
            "product": "sections_sloads",
            "absent_columns": absent_loads,
            "staged": str(sl_path),
        }


def _missing(inputs: ProductInputs) -> str:
    """Name the files the adapter could not supply, or say it supplies none."""
    if not inputs.missing:
        return "the adapter supplies none"
    return "; ".join(f"{path}: {reason}" for path, reason in inputs.missing[:3])


def _by_step(files: Sequence[StepFile]) -> list[list[StepFile]]:
    """Group the files of a point by step, in order: one file each, or one per zone."""
    groups: dict[int | None, list[StepFile]] = {}
    for item in files:
        groups.setdefault(item.step, []).append(item)
    return list(groups.values())


def _volumes(inputs: ProductInputs, root: Path) -> list[VolumeStep]:
    """Return the volume of each step; a multizone step holds its zones' files, in zone order.

    A step that lacks the volume file of a zone is left out: a point is never sampled from
    some of the zones of a step alone.
    """
    steps: list[VolumeStep] = []
    for group in _by_step(inputs.volume):
        if not inputs.zoned:
            source = workspace_path(root, group[0].path)
            steps.append(VolumeStep(group[0].step, group[0].time_s, source))
        elif len(group) == inputs.zones:
            steps.append(
                VolumeStep(
                    group[0].step, group[0].time_s, [workspace_path(root, f.path) for f in group]
                )
            )
    return steps


def _wrapped(
    support: ProductSupport, wanted: Sequence[str], free: FreeStream | None
) -> tuple[dict[str, object], dict[str, str]]:
    columns: dict[str, object] = {}
    absent: dict[str, str] = {}
    for name in wanted:
        if name in support.unfillable:
            absent[name] = support.unfillable[name]
        elif name not in support.columns:
            absent[name] = "the adapter does not fill it"
        else:
            source = support.columns[name]
            if isinstance(source, str):
                columns[name] = source
            elif free is None:
                absent[name] = "the run states no free stream to compute it from"
            else:
                columns[name] = lambda fields, s=source: cast(Callable[..., Any], s)(fields, free)
    return columns, absent


def _volume_section(
    ctx: _Context,
    entry: VolumeSection,
    support: ProductSupport,
    inputs: ProductInputs,
    volumes: Sequence[VolumeStep],
    stage: Path,
    out: PointProducts,
    target: Callable[[str], Path],
) -> None:
    extension = "vtk" if entry.format == "vtk" else "dat"
    label = f"fields/P{ctx.pol}/{ctx.name}_vsec.{extension}"
    frame = ctx.frames.get(entry.frame)
    chosen = list(volumes)
    if inputs.time_accurate and (entry.kind == "default" or not inputs.windowed):
        chosen = chosen[-1:]
    if not chosen:
        reason = _missing(inputs)
        out.skipped.append((label, f"no volume file: {reason}"))
        return
    if frame is None:
        out.skipped.append(
            (
                label,
                f"frame {entry.frame!r} is not one this point places (frames: "
                f"{', '.join(sorted(ctx.frames))})",
            )
        )
        return
    columns, absent = _wrapped(support, ("Velocity",), inputs.free_stream)
    if "Velocity" in columns:
        source = columns["Velocity"]

        def velocity(fields: Mapping[str, Any]) -> Any:
            values = (
                fields[source]
                if isinstance(source, str)
                else cast(Callable[..., Any], source)(fields)
            )
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(f"Velocity has shape {array.shape}; expected a 3-vector per point")
            return array

        columns["Velocity"] = velocity
    try:
        points = volume_section_points(
            entry.shape,
            entry.plane,
            corners_m=entry.corners_m,
            radii_m=entry.radii_m,
            points=entry.points,
            refinement_layers=entry.refinement_layers,
            offset_m=entry.offset_m,
        )
        report = write_volume_section(
            stage,
            ctx.name,
            points,
            frame if frame.spin is None or inputs.time_accurate else replace(frame, spin=None),
            chosen,
            cast(Mapping[str, Any], columns),
            format=entry.format,
            description={"shape": entry.shape, "plane": entry.plane, "kind": entry.kind},
        )
    except (OSError, ValueError) as error:
        out.skipped.append((label, str(error)))
        return
    for step in report.steps:
        if step.file is None:
            out.skipped.append(
                (f"{label}#step={step.step}", "no laid point lies in a cell of the volume")
            )
            continue
        relative = f"fields/P{ctx.pol}/{step.file.name}"
        try:
            file_target = target(relative)
            provenance_target = target(relative + ".provenance.json")
        except (OSError, ValueError) as error:
            out.skipped.append((relative, str(error)))
            continue
        details: dict[str, object] = {
            "point": ctx.point.point.point_id,
            "product": "volume_section",
            "kind": "instant",
            "steps_tabled": [step.step],
            "frame": entry.frame,
            "declared_points": step.n_points,
            "points_found": step.n_found,
            "absent_columns": {
                **absent,
                **{c: "the volume file lacks its field" for c in step.absent_columns},
            },
        }
        out.files[file_target] = {**details, "staged": str(step.file)}
        provenance = step.file.with_name(step.file.name + ".provenance.json")
        out.files[provenance_target] = {
            "point": ctx.point.point.point_id,
            "product": "volume_section_provenance",
            "staged": str(provenance),
        }


def _heading(parameter: str) -> str:
    """Return a probe parameter's column: ``MACH`` is ``MACH_LOCAL``, as pyflightstream."""
    return "MACH_LOCAL" if parameter.upper() == "MACH" else parameter


def _probe_points(entry: Probes, scale: float) -> np.ndarray:
    parts = [line_points(line.start, line.end, entry.points, scale) for line in entry.lines]
    parts += [
        rectangle_points(r.origin, r.along_u, r.along_v, r.points_u, r.points_v, scale)
        for r in entry.rectangles
    ]
    parts += [
        circle_points(c.center, c.normal, c.radius, c.points_radial, c.points_azimuth, scale)
        for c in entry.circles
    ]
    return np.vstack(parts)


def _probes(
    ctx: _Context,
    entries: Sequence[Probes],
    support: ProductSupport,
    inputs: ProductInputs,
    volumes: Sequence[VolumeStep],
    stage: Path,
    out: PointProducts,
    target: Callable[[str], Path],
) -> None:
    table = f"probes/P{ctx.pol}/{ctx.name}_probes.csv"
    if not volumes:
        reason = _missing(inputs)
        out.skipped.append((table, f"no volume file: {reason}"))
        return
    found = rotor_table(ctx.point.reference)
    radius = None if found is None else found[0].diameter_m / 2.0
    hub = None if found is None else f"{found[0].alias}_SMRP"
    laid: dict[str, list[ProbeEntry]] = {"default": [], "instantaneous": []}
    wanted: dict[str, dict[str, None]] = {"default": {}, "instantaneous": {}}
    for entry in entries:
        name = entry.frame or hub
        what = f"{table}#probes.{entry.position}"
        if name is None or name not in ctx.frames:
            out.skipped.append(
                (
                    what,
                    f"frame {entry.frame or 'the rotor hub frame'!r} is not one this point "
                    f"places (frames: {', '.join(sorted(ctx.frames))})",
                )
            )
            continue
        if entry.scale == "rotor_radius" and radius is None:
            out.skipped.append((what, 'scale = "rotor_radius" needs a rotor in the REF file'))
            continue
        frame = ctx.frames[name]
        if frame.spin is not None and not inputs.time_accurate:
            frame = replace(frame, spin=None)
        group = entry.kind if inputs.time_accurate else "default"
        scale = radius if entry.scale == "rotor_radius" and radius else 1.0
        laid[group].append(ProbeEntry(_probe_points(entry, scale), frame))
        wanted[group].update(dict.fromkeys(entry.parameters))
    both = bool(laid["default"]) and bool(laid["instantaneous"])
    for group, history in (("default", False), ("instantaneous", both)):
        if not laid[group]:
            continue
        chosen = (
            list(volumes) if group == "instantaneous" and inputs.windowed else list(volumes)[-1:]
        )
        sources, missing_columns = _wrapped(support, tuple(wanted[group]), inputs.free_stream)
        # pyflightstream writes the sampled MACH as MACH_LOCAL beside the condition's MACH
        columns = {_heading(name): source for name, source in sources.items()}
        absent = {_heading(name): reason for name, reason in missing_columns.items()}
        relative = f"probes/P{ctx.pol}/{ctx.name}_probes{'_history' if history else ''}.csv"
        try:
            probe_target = target(relative)
            report = write_probe_table(
                stage,
                ctx.name,
                laid[group],
                chosen,
                cast(Mapping[str, Any], columns),
                pol=ctx.pol,
                condition=dict(ctx.lead),
                history=history,
            )
        except (OSError, ValueError) as error:
            out.skipped.append((relative, str(error)))
            continue
        missing = [
            f"STEP {step.step}: {step.n_missing} point(s) in no cell"
            for step in report.steps
            if step.n_missing
        ]
        out.files[probe_target] = {
            "point": ctx.point.point.point_id,
            "product": "probes",
            "kind": "history" if group == "instantaneous" and inputs.windowed else "instant",
            "steps_tabled": [step.step for step in report.steps],
            "absent_columns": {
                **absent,
                **{c: "the volume file lacks its field" for c in report.absent_columns},
            },
            "points_not_found": missing,
            "staged": str(report.file),
        }


def point_products(
    root: Path,
    pol: str,
    name: str,
    point: AdapterPoint,
    adapter: object,
    *,
    lead: Mapping[str, str],
    moment_point: Sequence[float],
    cell: Callable[[object], str],
    target: Callable[[str], Path],
) -> PointProducts:
    """Write one point's declared products to a staging folder; say what goes where.

    The caller moves each staged file to its final path (``target(relative)``), so that the
    collect's overwrite rules apply to these files as to every other product.
    """
    out = PointProducts()
    support = product_support(adapter)
    declared = read_post(point.post, supported=tuple(support), warn=lambda key, text: None)
    out.skipped.extend(
        (f"products of {point.point.point_id}", reason) for reason in declared.skipped
    )
    if not declared.products:
        return out
    provider = getattr(adapter, "product_inputs", None)
    if not callable(provider):
        return out
    try:
        inputs = cast(ProductInputs, provider(point))
    except (OSError, ValueError) as error:
        for product in declared.products:
            out.skipped.append((f"{product}/P{pol}/{name}", f"the adapter's files: {error}"))
        return out
    # A partially available series still writes its remaining steps. Keep the adapter's
    # missing-source evidence even when the product has at least one usable file.
    out.skipped.extend((f"source {path}", reason) for path, reason in inputs.missing)
    frames = product_frames(
        point.reference,
        moment_point=moment_point,
        steps_per_revolution=inputs.steps_per_revolution,
    )
    ctx = _Context(
        point,
        pol,
        name,
        lead,
        frames,
        (float(moment_point[0]), float(moment_point[1]), float(moment_point[2])),
        cell,
    )
    stage = Path(tempfile.mkdtemp(prefix="cfdbender-products-"))
    out_files = out.files
    try:
        if declared.sections is not None and declared.sections.distributions:
            _sections(
                ctx,
                declared.sections.distributions,
                support["sections"],
                inputs,
                stage,
                out,
                target,
            )
        volumes = _volumes(inputs, root)
        if declared.volume_section is not None:
            _volume_section(
                ctx,
                declared.volume_section,
                support["volume_section"],
                inputs,
                volumes,
                stage,
                out,
                target,
            )
        if declared.probes:
            _probes(ctx, declared.probes, support["probes"], inputs, volumes, stage, out, target)
        for volume in volumes:
            volume.release()
        for details in out_files.values():
            staged = Path(cast(str, details.pop("staged")))
            details["text"] = staged.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return out


__all__ = ["SECTION_LEAD", "PointProducts", "point_products"]
