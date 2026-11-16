"""The skin patches of every component of a resolved configuration, and their inside regions.

Pipeline role: pure numerical stage of the geometry package, below the geometry
kernel. :func:`generate_components` turns a resolved configuration into placed
patches per component; :func:`assembly_regions` gives the inside predicate of
each component, which the workspace assembly uses to drop faces buried in
another component.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, overload

import numpy as np
from scipy.interpolate import BSpline

from cfdbender._errors import InputValueError

from ._blister import blister_model, blister_region
from ._bodies import body, body_station_table
from ._boundary import unpack
from ._bspline import full_knots
from ._caps import lifting_model, tip_cap, wing_contour_point
from ._fairings import wing_fuselage_fairing, wing_junction
from ._fillet import fillet_hosts, fillet_model
from ._identity import surface_id
from ._lifting import layout_record, lifting_surface, section_law, section_models, thickness_station
from ._patch import Patch, evaluate_patch, interpolate_patch
from ._placement import (
    body_junctions,
    move,
    place,
    placement_transform,
    wing_in_body_frame,
)
from ._propeller import Propeller
from ._regions import (
    body_region,
    fairing_volume_triangles,
    inside_triangles,
    lifting_region,
    placed_region,
)
from ._skin import split_along
from ._trims import Trim, iso_trim

__all__ = [
    "GeneratedPiece",
    "assembly_regions",
    "generate_components",
    "generate_pieces",
    "junction_lifting_surface",
    "source_cells",
    "stored_sections_config",
]


def stored_sections_config(resolved, sections):
    """Store fitted lifting sections as B-splines for exact, fit-free regeneration.

    The resolved parameters and derived values stay in their resolved form,
    so writing and reloading does not recompute their floating-point values.
    Optional unset mapping values are omitted because TOML has no null value.

    Parameters
    ----------
    resolved : dict
        Resolved geometry configuration.
    sections : dict
        Stored B-spline section tables keyed by lifting component name.

    Returns
    -------
    dict
        Plain TOML-compatible configuration retaining the supplied section coefficients.
    """
    config = unpack(resolved)
    for name, component in config["components"].items():
        if component["type"] != "lifting_surface":
            continue
        component["sections"]["items"] = sections[name]
        component["sections"]["airfoils"] = []
        component["sections"].pop("physical", None)
        component["sections"]["form"] = "bspline"
    return _toml_values(config)


def _toml_values(value):
    if isinstance(value, dict):
        return {k: _toml_values(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_toml_values(v) for v in value]
    return value


def _loop_mask(loop, uv):
    inside, boundary = np.zeros(len(uv), bool), np.zeros(len(uv), bool)
    for first, last in loop:
        a, b = np.asarray(first), np.asarray(last)
        edge, delta = b - a, uv - a
        cross = edge[0] * delta[:, 1] - edge[1] * delta[:, 0]
        boundary |= (np.abs(cross) <= 1e-12) & (delta @ edge >= 0) & (delta @ edge <= edge @ edge)
        if b[1] != a[1]:
            crossing = (a[1] > uv[:, 1]) != (b[1] > uv[:, 1])
            inside ^= crossing & (uv[:, 0] < a[0] + (uv[:, 1] - a[1]) * edge[0] / edge[1])
    return inside | boundary


def source_cells(patch: Patch, *, spacing: float, scale: float, trims=None) -> dict:
    """Sample source UV cells, integrating their physical areas by Gauss quadrature.

    Grid counts follow the sampled arc lengths and the requested share of the
    piece's extent. Original trim exclusions are retained as a separate mask.
    The cell-area sum can therefore be checked against the kernel's source area.

    Parameters
    ----------
    patch : Patch
        Source base with control points in metres.
    spacing : float
        Positive cell spacing as a fraction of the source extent.
    scale : float
        Positive source extent in metres.
    trims : sequence of Trim, optional
        Original included faces; omitted means the entire parameter domain.

    Returns
    -------
    dict
        Cell centres, normals, areas in square metres and the inclusion mask.

    Raises
    ------
    InputValueError
        If spacing or scale is nonpositive, or the grid exceeds one million cells.
    """
    if spacing <= 0 or scale <= 0:
        raise InputValueError("coverage requires positive spacing and source extent")
    grid = evaluate_patch(patch, 33)
    lengths = [
        float(np.linalg.norm(np.diff(grid, axis=i), axis=2).sum(axis=i).max()) for i in (0, 1)
    ]
    counts = [max(2, int(np.ceil(length / (spacing * scale)))) for length in lengths]
    if counts[0] * counts[1] > 1_000_000:
        raise InputValueError("coverage spacing requires over one million cells on one base")
    us = np.linspace(patch.knots_u[0], patch.knots_u[-1], counts[0] + 1)
    vs = np.linspace(patch.knots_v[0], patch.knots_v[-1], counts[1] + 1)
    uc, vc = (us[:-1] + us[1:]) / 2, (vs[:-1] + vs[1:]) / 2
    controls = np.asarray(patch.controls, float)
    bu = BSpline(
        full_knots(patch.knots_u, patch.multiplicities_u), np.eye(controls.shape[0]), patch.degree_u
    )
    bv = BSpline(
        full_knots(patch.knots_v, patch.multiplicities_v), np.eye(controls.shape[1]), patch.degree_v
    )

    def net(u, v, du=0, dv=0):
        return np.einsum("ui,vj,ijc->uvc", bu(u, du), bv(v, dv), controls, optimize=True)

    points = net(uc, vc)
    normals = np.cross(net(uc, vc, du=1), net(uc, vc, dv=1))
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-300)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    areas = np.zeros(tuple(counts))
    for a, wa in zip(nodes, weights, strict=True):
        for b, wb in zip(nodes, weights, strict=True):
            u, v = uc + a * np.diff(us) / 2, vc + b * np.diff(vs) / 2
            jacobian = np.cross(net(u, v, du=1), net(u, v, dv=1))
            areas += wa * wb * np.linalg.norm(jacobian, axis=2)
    areas *= np.diff(us)[:, None] * np.diff(vs)[None, :] / 4
    uv = np.stack(np.meshgrid(uc, vc, indexing="ij"), axis=-1).reshape(-1, 2)
    included = np.ones(len(uv), bool) if trims is None else np.zeros(len(uv), bool)
    for trim in trims or ():
        mask = _loop_mask(trim.loop, uv)
        for inner in trim.inner:
            mask &= ~_loop_mask(inner, uv)
        included |= mask
    return {
        "uv": uv,
        "points": points.reshape(-1, 3),
        "normals": normals.reshape(-1, 3),
        "areas": areas,
        "included": included,
    }


@dataclass(frozen=True)
class GeneratedPiece:
    """Placed bases and UV faces, with the version and identity actually generated.

    Layout lines are metadata. They never split or resample these bases. Assembly
    propagates the base index of each retained face back to this source record.
    """

    name: str
    version: str
    bases: dict[str, Patch]
    trims: tuple[Trim, ...]
    layout: dict[str, Any]
    hull_patches: tuple[Patch, ...] = ()
    inert_bases: tuple[str, ...] = ()

    @property
    def source_surface_id(self) -> str:
        """Identity of the placed bases and their original loops."""
        return surface_id({self.name: list(self.bases.values())}, {self.name: self.trims})

    @classmethod
    def from_patches(cls, name: str, version: str, patches: list[Patch]) -> GeneratedPiece:
        """Wrap full-domain patches from the legacy component interface."""
        bases = {f"{p.family}_{i}": p for i, p in enumerate(patches)}
        trims = tuple(
            iso_trim(
                key,
                key,
                (p.knots_u[0], p.knots_u[-1]),
                (p.knots_v[0], p.knots_v[-1]),
                p.role or p.family,
            )
            for key, p in bases.items()
        )
        return cls(name, version, bases, trims, {"lines": {"u": {}, "v": {}}})


def generate_pieces(config, *, variant="clean", selection=None) -> dict[str, GeneratedPiece]:
    """Build placed source pieces for trimmed assembly and export, without layout fitting.

    The legacy patch generator remains the mesher's interface. Here lifting
    skins, fillets and blisters carry their original bases and parameter loops.

    Parameters
    ----------
    config : dict
        Resolved geometry configuration including placement and component versions.
    variant : str, optional
        Registry variant selecting active component controls.
    selection : dict, optional
        Moving piece label for each active component.control, one angle per control.

    Returns
    -------
    dict of str to GeneratedPiece
        Source bases in world coordinates, in metres, with their trims and layouts.

    Raises
    ------
    InputValueError
        If a component type has no source-piece generator.
    """
    components = unpack(config)["components"]
    pieces = {}
    for name, component in components.items():
        kind, frame = component["type"], name
        layout: dict[str, Any] = {"lines": {"u": {}, "v": {}}}
        if kind == "lifting_surface":
            from ._fowler import with_pins

            component = with_pins(component)
            model = lifting_model(component)
            bases, trims = model.bases(), model.trims
            layout = layout_record(model)
            layout["sections"] = [
                {
                    "eta": eta,
                    "bspline": {k: section.to_stored()[k] for k in ("degree", "knots", "poles")},
                }
                for eta, section in zip(
                    component["sections"]["eta"], section_law(component).given, strict=True
                )
            ]
        elif kind == "fillet":
            fillet = fillet_model(component, components, name)
            _, _, frame = fillet_hosts(components, name)
            # Use the model's own base name, including the optional TE closure.
            bases = {fillet.trims[0].base: fillet.base}
            if fillet.trailing_edge is not None:
                bases["te"] = fillet.trailing_edge
            trims = fillet.trims
        elif kind == "blister":
            frame = component["host"]
            blister = blister_model(component, components[frame])
            bases, trims = blister.bases, blister.trims
        else:
            if kind == "propeller":
                patches = propeller_patches(component)
            elif kind == "body":
                patches = body(component, body_junctions(components, name))
            elif kind == "wing_fuselage_fairing":
                frame, wing_name = component["hosts"]
                wing = wing_in_body_frame(components, frame, wing_name)
                patches = wing_fuselage_fairing(component, components[frame], wing)
            else:
                raise InputValueError(f"{name}.type: unsupported source piece {kind!r}")
            bases = {f"{patch.family}_{i}": patch for i, patch in enumerate(patches)}
            options = component["definition"]["options"] if kind == "propeller" else None
            trims = tuple(source_trim(key, p, options=options) for key, p in bases.items())
        placed = dict(
            zip(
                bases,
                move(list(bases.values()), *placement_transform(components, frame)),
                strict=True,
            )
        )
        pieces[name] = GeneratedPiece(name, str(component["version"]), placed, tuple(trims), layout)
    if variant != "clean":
        _apply_controls(unpack(config), pieces, variant, selection or {})
    return pieces


def _apply_controls(config, pieces, variant, selection):
    """Retain placed control bases and original loops, with exact restricted hulls."""
    from . import _control_surfaces as controls

    components = config["components"]
    active = config["variants"][variant]
    for name, source in list(pieces.items()):
        entries = [
            e
            for e in components[name].get("control_surfaces", [])
            if f"{name}.{e['name']}" in active
        ]
        if not entries:
            continue
        component = {**components[name], "control_surfaces": entries}
        model = controls.placed({**components, name: component}, name)
        # Keep the noncollapsed, exact flat-cap construction already generated.
        main = list(model.pieces["main"])
        for end in ("root", "tip"):
            caps = [t for t in source.trims if t.base.startswith(f"{end}_cap")]
            if caps and any(t.base == end for t in main):
                main = [t for t in main if t.base != end] + caps
                model.bases.update({t.base: source.bases[t.base] for t in caps})
        model.pieces["main"] = tuple(main)
        chosen = ["main"]
        for entry in entries:
            labels = [k for k in model.pieces if k.startswith(f"{entry['name']}-d")]
            label = selection.get(f"{name}.{entry['name']}", labels[0])
            if label not in labels:
                raise InputValueError(f"{name}.{entry['name']}: unknown applied piece {label!r}")
            chosen.append(label)
        trims = tuple(_control_trim(t, label) for label in chosen for t in model.pieces[label])
        bases = {key: model.bases[key] for key in sorted({t.base for t in trims})}
        restricted = tuple(controls.face_patches(model, trims))
        trims = tuple(_reader_trim(t, p) for t, p in zip(trims, restricted, strict=True))
        inert = tuple(
            sorted(
                {
                    t.base
                    for t in trims
                    if t.face.startswith(tuple(chosen[1:]))
                    or t.role in ("cove_wall", "cove_roof", "lip_base")
                }
            )
        )
        pieces[name] = GeneratedPiece(
            name,
            source.version,
            bases,
            trims,
            source.layout,
            restricted,
            inert,
        )


def _control_trim(trim, label):
    role = trim.role
    if role == "skin":
        us = [p[0] for edge in trim.loop for p in edge]
        role = "lower" if max(us) <= 0.5 else "upper"
    return replace(trim, face=f"{label}/{trim.face}", role=role)


def _reader_trim(trim, patch):
    """Give a collapsed end face four nonzero rim segments without changing its surface."""
    grid = evaluate_patch(patch, 5)
    edges = (grid[:, 0], grid[-1], grid[::-1, -1], grid[0, ::-1])
    lengths = [float(np.linalg.norm(np.diff(e, axis=0), axis=1).sum()) for e in edges]
    if min(lengths) > 1e-12:
        return trim
    split = int(np.argmax(lengths))
    loop = []
    for i, (a, b) in enumerate(trim.loop):
        if i == split:
            middle = tuple((x + y) / 2 for x, y in zip(a, b, strict=True))
            loop.extend(((a, middle), (middle, b)))
        else:
            loop.append((a, b))
    return replace(trim, loop=tuple(loop))


@overload
def generate_components(
    config: Any, *, trimmed: Literal[False] = False
) -> dict[str, list[Patch]]: ...


@overload
def generate_components(config: Any, *, trimmed: Literal[True]) -> dict[str, GeneratedPiece]: ...


def generate_components(
    config, *, trimmed=False
) -> dict[str, list[Patch]] | dict[str, GeneratedPiece]:
    """Return the placed patches of every component of a resolved configuration.

    Parameters
    ----------
    config : dict
        Resolved configuration; its components are unpacked to plain values.
    trimmed : bool, optional
        Return source pieces with their bases and UV loops for assembly and export.
        False preserves the mesher's patch-list interface.

    Returns
    -------
    dict
        Component name to its list of placed patches, in the world frame.

    Raises
    ------
    InputValueError
        If a component is of a type other than body, lifting_surface or wing_fuselage_fairing.
    """
    if trimmed:
        return generate_pieces(config)
    components = unpack(config)["components"]
    result = {}
    for name, component in components.items():
        kind = component["type"]
        if kind == "propeller":
            patches = propeller_patches(component)
        elif kind == "body":
            patches = body(component, body_junctions(components, name))
        elif kind == "lifting_surface":
            fairings = [
                other
                for other in components.values()
                if other["type"] == "wing_fuselage_fairing" and other["hosts"][1] == name
            ]
            if fairings:
                fairing = fairings[0]
                body_name = fairing["hosts"][0]
                host_wing = wing_in_body_frame(components, body_name, name)
                patches = junction_lifting_surface(host_wing, components[body_name], fairing)
                patches += tip_cap(host_wing)
                rotation, translation = placement_transform(components, body_name)
                result[name] = move(patches, rotation, translation)
                continue
            patches = lifting_surface(component) + tip_cap(component)
        elif kind == "wing_fuselage_fairing":
            body_name, wing_name = component["hosts"]
            wing = wing_in_body_frame(components, body_name, wing_name)
            patches = wing_fuselage_fairing(component, components[body_name], wing)
        elif kind in ("fillet", "blister"):
            # Built in the frame of the body they lie on; their own placement is not read.
            patches, body_name = blend_faces(components, name)
            result[name] = move(patches, *placement_transform(components, body_name))
            continue
        else:
            raise InputValueError(
                f"{name}.type: this assembly driver supports body, lifting_surface, "
                "wing_fuselage_fairing, fillet and blister"
            )
        result[name] = place(patches, components, name)
    return result


def propeller_patches(component):
    """Build the selected blade set, converting the definition's mm to workspace metres."""
    if "definition" not in component:
        raise InputValueError("propeller.definition: schema 1 definition required for generation")
    design = Propeller(component["definition"])
    model = design.blade() if design.config["output"]["blades"] == "one" else design.rotor()
    return [replace(p, controls=p.controls * 1e-3) for p in model.patches]


def source_trim(key, patch, *, options=None):
    """Give a cap's two curved rims an explicit midpoint vertex before kernel export.

    The tensor cap has a collapsed nose edge. Dividing both curved rims keeps
    its outer wire outside the reader profile's unsupported three-edge form,
    including when a sharp trailing edge collapses its other end too.
    """
    trim = iso_trim(
        key,
        key,
        (patch.knots_u[0], patch.knots_u[-1]),
        (patch.knots_v[0], patch.knots_v[-1]),
        patch.role or patch.family,
    )
    if options is None or patch.role not in ("root_cap", "tip", "upper", "lower", "trailing_edge"):
        return trim
    if patch.role == "trailing_edge" and not _collapsed_end(patch):
        return trim
    loop = []
    for i, (a, b) in enumerate(trim.loop):
        cap = "root_cap" if i == 0 else "tip"
        divide = (
            i in (1, 3)
            if patch.role == "trailing_edge"
            else i in (0, 2) and (patch.role in ("root_cap", "tip") or options[cap] == "planar")
        )
        if divide:
            middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            loop.extend(((a, middle), (middle, b)))
        else:
            loop.append((a, b))
    return replace(trim, loop=tuple(loop))


def _collapsed_end(patch):
    """Whether a propeller trailing face collapses to a point at either radial end, in m."""
    return any(
        float(np.linalg.norm(p - p[0], axis=1).max()) <= 1e-12
        for p in (patch.controls[:, 0], patch.controls[:, -1])
    )


def blend_faces(components, name) -> tuple[list[Patch], str]:
    """Return the faces a fillet or a blister writes today, and the body whose frame they are in.

    Until the assembly writes trimmed faces, each face is its base cut at iso lines: a fillet's
    two faces lie between its contact lines, split at the leading edge; a blister's base is cut
    at its mean crease into the face outside the host and a band inside it, which the assembly
    drops as buried. No count is assumed: every face carries its family and role.
    """
    component = components[name]
    if component["type"] == "fillet":
        model = fillet_model(component, components, name)
        _, _, body_name = fillet_hosts(components, name)
        start = model.trims[0].loop[0][0][0]
        middle = split_along(model.base, "v", [model.contacts["v_host"], model.contacts["v_wing"]])[
            1
        ]
        cuts = [start, 0.5] if start > 0.0 else [0.5]
        pieces = split_along(middle, "u", cuts)[-2:]
        faces = [
            replace(piece, family=f"fillet_{side}", role="fillet")
            for piece, side in zip(pieces, ("lower", "upper"), strict=True)
        ]
        closing = [] if model.trailing_edge is None else [model.trailing_edge]
        return faces + closing, body_name
    host = component["host"]
    if components[host]["type"] != "body":
        raise InputValueError(f"{name}.host: a blister lies on a body; {host!r} is not one")
    loft = blister_model(component, components[host])
    patches = []
    for base, patch in loft.bases.items():
        crease = float(np.mean(loft.crease[base][1:-1, 0]))
        outer, inner = split_along(patch, "u", [crease])
        patches.append(replace(outer, family=f"blister_{base}", role="blister"))
        patches.append(replace(inner, family=f"blister_{base}_inside", role="blister_band"))
    return patches, host


def junction_lifting_surface(wing, host, fairing) -> list[Patch]:
    """Imprint the fairing contact as a tensor-patch edge on both wing skins.

    The same contour samples and knots are used at the junction. The thickness
    split blends to the tip split with zero slope at the junction, preserving
    the span tangent there and the existing tip-cap boundary.
    """
    models, table = section_models(wing["sections"]), body_station_table(host)
    root_max, tip_max = (thickness_station(wing, models, eta) for eta in (0.0, 1.0))
    families = [
        "upper_fore",
        "upper_aft",
        "trailing_upper",
        "trailing_lower",
        "lower_aft",
        "lower_fore",
    ]
    patches = []
    for index, family in enumerate(families):
        contacts = [
            wing_junction(fairing, host, table, wing, models, maximum=root_max, index=index, u=u)
            for u in np.linspace(0, 1, 49)
        ]
        for bay in ("inner", "outer", "outboard"):
            grid = []
            for u, (contact, _, width) in zip(np.linspace(0, 1, 49), contacts, strict=True):
                junction_eta = 2 * contact[1] / wing["planform"]["span"]
                ring_eta = junction_eta + min(
                    2
                    * max(width, 0.05 * wing["planform"]["root_chord"])
                    / wing["planform"]["span"],
                    (1 - junction_eta) / 2,
                )
                row = []
                for v in np.linspace(0, 1, 9):
                    low, high = {
                        "inner": (0.0, junction_eta),
                        "outer": (junction_eta, ring_eta),
                        "outboard": (ring_eta, 1.0),
                    }[bay]
                    eta = low + (high - low) * v
                    maximum = (
                        root_max + (tip_max - root_max) * (3 * v * v - 2 * v * v * v)
                        if bay == "outboard"
                        else root_max
                    )
                    row.append(wing_contour_point(wing, models, maximum, index, u, eta))
                grid.append(row)
            patches.append(interpolate_patch(grid, family=family))
    return patches


def assembly_regions(config, patches) -> dict:
    """Return the inside predicate of every component, in the world frame.

    Parameters
    ----------
    config : dict
        Resolved configuration of the assembly.
    patches : dict
        Placed patches per component name, as :func:`generate_components` returns; the fairing
        volume is built from them.

    Returns
    -------
    dict
        Component name to a function of a world point that tells whether it is inside.
    """
    components = unpack(config)["components"]
    regions = {}
    for name, component in components.items():
        rotation, translation = placement_transform(components, name)
        if component["type"] == "body":
            regions[name] = placed_region(body_region(component), rotation, translation)
        elif component["type"] == "lifting_surface":
            regions[name] = placed_region(lifting_region(component), rotation, translation)
        elif component["type"] == "wing_fuselage_fairing":
            wing_name = component["hosts"][1]
            wr, wt = placement_transform(components, wing_name)
            interior = wt + wr @ np.array(
                [components[wing_name]["planform"]["root_chord"] / 2, 0.0, 0.0]
            )
            triangles = fairing_volume_triangles(patches[name], interior)
            regions[name] = lambda point, skin=triangles: inside_triangles(point, skin)
        elif component["type"] == "blister":
            host = component["host"]
            regions[name] = placed_region(
                blister_region(component, components[host]),
                *placement_transform(components, host),
            )
        elif component["type"] == "fillet":
            # The blend buries nothing until the dominance stage trims its hosts by it.
            regions[name] = lambda point: False
    return regions
