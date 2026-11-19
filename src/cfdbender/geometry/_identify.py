"""The kernel entry points of the geometry pipeline: generate, check, mesh and identify.

Pipeline role: the public entry points of the stages that need the geometry kernel, on the gmsh
kernel. :func:`run_geometry` runs the stages S0 to S13 for a configuration
(:func:`cfdbender.geometry.load_config`), stops at the first failed check and returns a
report that names its stage and check, with the exposed skin as a named IGES file (one NAME per
component) from stage S11; :func:`run_doe` runs a loop of parameter sets and continues after a
failed one; :func:`check_against_original` measures a rebuilt surface against an original under
a named deviation profile; :func:`mesh_generated` meshes a configuration into an all-quad
surface mesh with its acceptance checks; :func:`mesh_configuration` meshes it by strategy 1 or 2
from its per-component mesh controls (or plans it on a dry run) and writes the mesh and its
record; and :func:`identify_lifting_surface` reads the planform and the sections of a lifting
surface back from an original IGES, in the form a configuration takes
(:class:`LiftingIdentification`). The pure parameter contract
(:class:`~cfdbender.geometry.ComponentRecord`) and the TOML input boundary live in
:mod:`cfdbender.geometry` and import no kernel.

A planform is reported with ``accepted`` set only when a coarse and a fine pass agree within
the tolerances of the result (``convergence``); sections carry their own ``accepted`` against
their residual tolerance.

Each function raises ``KernelUnavailableError`` (gmsh is a dependency of the package) when the
kernel library cannot load, and the message names the library that failed. The command lines
are ``cfd-geom run | doe | validate | schema | configurations | stages | identify | check |
iges-convert`` and ``cfd-mesh mesh | mesh-generated``.

Examples
--------
>>> from cfdbender.geometry._identify import __all__
>>> sorted(__all__)[:2]
['ConfigurationMesh', 'GeneratedMesh']
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cfdbender._errors import InputValueError
from cfdbender.geometry._geometry_check import (
    check_against_original,
    resolve_faces,
    sample_surfaces,
)
from cfdbender.geometry._geometry_mesh import (
    ConfigurationMesh,
    GeneratedMesh,
    mesh_configuration,
    mesh_generated,
)
from cfdbender.geometry._geometry_reverse import (
    COARSE,
    CUTS,
    PURPOSE,
    ROOTS,
    convergence,
    identify_pass,
    refuse_tolerances,
    resolve_hosts,
    span_direction,
    surface_grid,
)
from cfdbender.geometry.doe import run_doe
from cfdbender.geometry.pipeline import (
    built_stages,
    describe_config,
    list_configurations,
    run_geometry,
)
from cfdbender.geometry.stages import fit_section
from cfdbender.mesh._gmsh import gmsh_module, open_kernel

__all__ = [
    "ConfigurationMesh",
    "GeneratedMesh",
    "LiftingIdentification",
    "built_stages",
    "check_against_original",
    "describe_config",
    "identify_lifting_surface",
    "list_configurations",
    "mesh_configuration",
    "mesh_generated",
    "run_doe",
    "run_geometry",
]


@dataclass(frozen=True)
class LiftingIdentification:
    """The identified planform, sections and evidence of one lifting surface (meters, degrees).

    ``planform`` is the result of :func:`cfdbender.geometry.stages.reference_planform`;
    ``sections`` are the root and tip CST fits and ``midspan`` the extra measured section.
    ``accepted`` is true when the coarse and fine passes agree and every section met its
    tolerance.
    """

    component: str
    source: str
    seed: str
    planform: dict[str, Any]
    sections: list[dict[str, Any]]
    midspan: dict[str, Any]
    accepted: bool
    root_kind: str
    clean_cuts: int
    convergence: dict[str, Any] | None = None
    scan: dict[str, Any] = field(default_factory=dict)

    def to_component(self) -> dict[str, Any]:
        """Return a ``lifting_surface`` component of a configuration that rebuilds this planform.

        The configuration describes sections by NACA designation, so the seed designation
        stands for the section shape and the measured CST fits stay in ``sections``. A
        configuration twists each section about its quarter chord, so the placement, the sweep
        and the dihedral come from the quarter-chord line, which incidence and twist leave in
        place, not from the measured leading edge, which they move. A left half is described by
        its mirror image. A span axis tilted 89 degrees or more from the y axis (a fin) is
        refused: place such a surface by rotation instead.
        """
        p = self.planform
        if abs(p["dihedral_deg"]) >= 89.0:
            raise InputValueError(
                f"to_component: dihedral {p['dihedral_deg']:.3g} degrees is outside the "
                "configuration's range of +-89; describe this surface with a placement rotation"
            )
        span = p["reference_plane"]["span_direction"]
        side = math.copysign(1.0, span[1])
        rows = p["distributions"]

        def untwisted_leading_edge(at: int) -> tuple[float, float, float]:
            """Return the root (0) or tip (-1) leading edge before the section turned about c/4."""
            chord, angle = rows["chord_m"][at], math.radians(rows["incidence_deg"][at])
            # The chord lies in the cut plane: along x and along the upward normal of the span.
            down = 0.25 * chord * math.sin(angle)
            return (
                rows["le_x_m"][at] + 0.25 * chord * (math.cos(angle) - 1.0),
                side * rows["le_y_m"][at] + down * span[2],
                rows["le_z_m"][at] - down * abs(span[1]),
            )

        root, tip = untwisted_leading_edge(0), untwisted_leading_edge(-1)
        semi_y = tip[1] - root[1]
        return {
            "type": "lifting_surface",
            "planform": {
                "semispan_mm": 1000.0 * semi_y,
                "root_chord_mm": 1000.0 * p["root_chord_m"],
                "tip_chord_mm": 1000.0 * p["tip_chord_m"],
                "le_sweep_deg": math.degrees(math.atan2(tip[0] - root[0], semi_y)),
                "dihedral_deg": math.degrees(math.atan2(tip[2] - root[2], semi_y)),
                "incidence": p["incidence_deg"],
                "tip_twist": p["tip_twist_deg"],
            },
            "stations": [
                {"eta": 0.0, "section": {"kind": "naca", "code": self.seed.removeprefix("NACA")}}
            ],
            "placement": {"translation": list(root)},
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the result as plain data, ready for JSON."""
        return {
            "component": self.component,
            "source": self.source,
            "seed": self.seed,
            "accepted": self.accepted,
            "root_kind": self.root_kind,
            "clean_cuts": self.clean_cuts,
            "planform": self.planform,
            "sections": self.sections,
            "midspan": self.midspan,
            "convergence": self.convergence,
            "scan": self.scan,
        }


def identify_lifting_surface(
    source: str | Path,
    component: str,
    *,
    hosts: Sequence[str] = (),
    root: str = "auto",
    mirrored: bool | None = None,
    cut: str = "plain",
    seed: str = "NACA0012",
    span_axis: Sequence[float] | None = None,
    stations: int = 161,
    grid: int = 257,
    section_tolerance_mm: float | None = None,
    section_degree: int = 5,
    section_max_degree: int = 20,
    host_tolerance: float = 5e-5,
    converge: bool = True,
) -> LiftingIdentification:
    """Identify one lifting-surface component of an IGES file from its trimmed skin.

    Parameters
    ----------
    source : str or Path
        The IGES file, read only. Its faces are named by the NAME of their level.
    component : str
        The component to identify.
    hosts : sequence of str, optional
        Components the surface sits on, as names in ``source`` or ``path#name``. Given, the
        root is where the extended leading edge meets them (``root="auto"``).
    root : {"auto", "plane", "host", "boundary"}
        ``plane``: the symmetry plane through the origin normal to the span axis. ``host``:
        the host surface. ``boundary``: the span extent of the original's own boundary curves
        (a surface open inside its host). ``auto``: ``host`` when ``hosts`` is given, else
        ``plane``.
    mirrored : bool, optional
        Whether the surface has a mirror image across its root (doubles span and area);
        defaults to true for a plane root and false otherwise.
    cut : {"plain", "bridge_te", "mirror"}
        ``bridge_te`` closes an open blunt trailing-edge slit; ``mirror`` adds the reflection
        of a half-body original.
    seed : str
        The NACA designation the section fit starts from.
    span_axis : sequence of float, optional
        The first estimate of the span axis; the larger of y and z extent when omitted.
    stations, grid : int
        Stations of the fine pass and cut-grid points per side of each face.
    section_tolerance_mm : float, optional
        Section residual tolerance; the larger of 0.5 mm and 0.1 percent of the chord when
        omitted.
    section_degree, section_max_degree : int
        The CST degree a section fit starts at and the highest it may rise to.
    host_tolerance : float
        The largest gap, in meters, at which the extended leading edge counts as meeting a host.
    converge : bool
        Run a coarse pass as well and require the planforms to agree for ``accepted``.

    Returns
    -------
    LiftingIdentification

    Raises
    ------
    MissingExtraError
        The gmsh library cannot load (``KernelUnavailableError``).
    InputValueError
        An argument is refused, the component is unknown, or no clean band exists.
    """
    if root not in ROOTS:
        raise InputValueError(f"root: one of {list(ROOTS)} required, got {root!r}")
    if cut not in CUTS:
        raise InputValueError(f"cut: one of {list(CUTS)} required, got {cut!r}")
    if stations < 20 or grid < 33:
        raise InputValueError("stations, grid: at least 20 stations and a 33-point grid required")
    refuse_tolerances(section_tolerance_mm, host_tolerance)
    if root == "host" and not hosts:
        raise InputValueError("root: 'host' needs hosts")
    gmsh_module(PURPOSE)
    resolved = ("host" if hosts else "plane") if root == "auto" else root
    kernel = open_kernel(PURPOSE)
    try:
        faces = resolve_faces(kernel, f"{source}#{component}")
        host_faces = resolve_hosts(kernel, hosts, source) if resolved == "host" or hosts else []
        samples, _ = sample_surfaces(kernel, faces, 65, 257)
        axis = span_direction(samples, span_axis)
        spec = {
            "span": axis,
            "root": resolved,
            "cut": cut,
            "mirrored": resolved == "plane" if mirrored is None else mirrored,
            "host_tolerance": host_tolerance,
        }
        if resolved == "boundary":
            spec["faces"] = faces
        grids = [surface_grid(kernel, t, grid) for t in faces]
        agreement = None
        if converge:
            small = [surface_grid(kernel, t, COARSE[1]) for t in faces]
            coarse = identify_pass(
                kernel, small, samples, host_faces, spec, count=min(stations, COARSE[0])
            )[0]
        planform, band, features, stations_m, span, diagnostics = identify_pass(
            kernel, grids, samples, host_faces, spec, count=stations
        )
        if converge:
            agreement = convergence(coarse, planform)
        chosen = [int(band[0]), int(band[len(band) // 2]), int(band[-1])]
        fits = []
        for role, index in zip(("root", "midspan", "tip"), chosen, strict=True):
            fit = fit_section(
                features[index]["contour"], span, seed,
                degree=section_degree, max_degree=max(section_degree, section_max_degree),
                tolerance_m=None if section_tolerance_mm is None else section_tolerance_mm / 1000,
            )  # fmt: skip
            fit["role"], fit["station_m"] = role, float(stations_m[index])
            fit["eta"] = (fit["station_m"] - planform["root_station_m"]) / planform["semi_span_m"]
            fits.append(fit)
    finally:
        kernel.close()
    accepted = all(f["accepted"] for f in fits) and (agreement is None or agreement["accepted"])
    return LiftingIdentification(
        component=component,
        source=Path(source).name,
        seed=seed,
        planform=planform,
        sections=[fits[0], fits[2]],
        midspan=fits[1],
        accepted=accepted,
        root_kind=planform["root_kind"],
        clean_cuts=len(band),
        convergence=agreement,
        scan={**diagnostics, "stations": stations, "grid": grid},
    )
