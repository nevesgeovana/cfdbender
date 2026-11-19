"""What SU2 supplies to the products the post writes: its surface and volume files.

SU2 writes the surface of the plotted markers (``MARKER_PLOTTING``) to
``<SURFACE_FILENAME>.vtu`` (``surface_flow``) and the volume to ``<VOLUME_FILENAME>.vtu``
(``flow``) when ``OUTPUT_FILES`` holds ``SURFACE_PARAVIEW`` and ``PARAVIEW``. A
time-accurate run appends its 0-based time iteration as five digits, ``_<k:05d>``, which is
step ``k + 1``. :func:`product_outputs` adds the two outputs to the deck when a declared
product needs them, and only then.

The fields: ``Pressure`` (Pa), ``Density``, ``Momentum`` (the velocity is ``Momentum /
Density``; an incompressible run writes ``Velocity``), ``Mach``, ``Pressure_Coefficient``
and the vector ``Skin_Friction_Coefficient``. The surface file does not say which cell is
of which marker, so a section distribution cuts every plotted marker or none of them.

PROVISIONAL readings, each behind one named parameter of this module:

* :data:`CP_REFERENCE`: the Cp columns divide by the free stream's ``0.5 rho V^2`` and
  subtract its static pressure, also in a rotating frame; ``Cp_ref`` and ``CP_REF`` are
  SU2's own ``Pressure_Coefficient``, whose reference speed is ``MACH_MOTION``'s when the
  deck states it.
* :data:`FRICTION`: ``CF`` is the magnitude of ``Skin_Friction_Coefficient``.
* :data:`NORMAL_SIGN`: the winding of a surface cell, as SU2 writes it, points into the
  fluid.
* :data:`VELOCITY_FRAME`: the velocity of a rotating-frame run is absolute (measured once
  on a steady rotating-frame file).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.contract import (
    SURFACE,
    VOLUME,
    FreeStream,
    ProductInputs,
    ProductSupport,
    StepFile,
)
from cfdbender.core.plan import workspace_path
from cfdbender.core.pproc import PROBES, SECTIONS, VOLUME_SECTION
from cfdbender.core.run_types import TIME_ACCURATE

#: The Cp reference (PROVISIONAL): the free stream of the deck.
CP_REFERENCE = (
    "free stream: Cp = (p - p_inf) / (0.5 rho_inf V_inf^2), p_inf = rho_inf R T_inf of the deck"
)
#: How CF is read (PROVISIONAL): the magnitude of SU2's skin friction coefficient vector.
FRICTION = "magnitude of Skin_Friction_Coefficient"
#: The surface normal sign (PROVISIONAL): +1, the winding normal points into the fluid.
NORMAL_SIGN = 1
#: The frame of the velocity SU2 writes (PROVISIONAL for a sliding zone): absolute.
VELOCITY_FRAME = "absolute"
_BOUNDARY_LAYER = "SU2 writes no boundary-layer integral thickness or shape factor"


def _velocity(fields: Mapping[str, Any], free: FreeStream) -> Any:
    if "Velocity" in fields:
        return fields["Velocity"]
    return fields["Momentum"] / fields["Density"][:, None]


def _component(index: int) -> Any:
    return lambda fields, free: _velocity(fields, free)[:, index]


def _speed(fields: Mapping[str, Any], free: FreeStream) -> Any:
    return (_velocity(fields, free) ** 2).sum(axis=1) ** 0.5


def _friction(fields: Mapping[str, Any], free: FreeStream) -> Any:
    return (fields["Skin_Friction_Coefficient"] ** 2).sum(axis=1) ** 0.5


#: The wall shear stress (Pa), the one place it is converted: SU2's skin friction
#: coefficient vector times the dynamic pressure of the Cp reference (:data:`CP_REFERENCE`).
SHEAR = "Skin_Friction_Coefficient times q_ref of the Cp reference"


def _shear(fields: Mapping[str, Any], free: FreeStream) -> Any:
    return fields["Skin_Friction_Coefficient"] * free.q_ref


def _cp_free(fields: Mapping[str, Any], free: FreeStream) -> Any:
    return (fields["Pressure"] - free.p_inf) / free.q_ref


def _pressure_ratio(fields: Mapping[str, Any], free: FreeStream) -> Any:
    return fields["Pressure"] / free.p_inf


PRODUCTS: Mapping[str, ProductSupport] = {
    SECTIONS: ProductSupport(
        SURFACE,
        {
            "pressure": "Pressure",
            "Mach": "Mach",
            "vx": _component(0),
            "vy": _component(1),
            "vz": _component(2),
            "Cp_ref": "Pressure_Coefficient",
            "CF": _friction,
            "shear": _shear,
        },
        {name: _BOUNDARY_LAYER for name in ("Theta", "Delta*", "Delta", "H")},
    ),
    VOLUME_SECTION: ProductSupport(VOLUME, {"Velocity": _velocity}),
    PROBES: ProductSupport(
        VOLUME,
        {
            "CP_FREE": _cp_free,
            "CP_REF": "Pressure_Coefficient",
            "MACH": "Mach",
            "VELOCITY": _speed,
            "VX": _component(0),
            "VY": _component(1),
            "VZ": _component(2),
            "STATIC_PRESSURE_RATIO": _pressure_ratio,
        },
        {
            name: _BOUNDARY_LAYER
            for name in (
                "BL_MOMENTUM_THICKNESS",
                "BL_DISPLACEMENT_THICKNESS",
                "BL_TOTAL_THICKNESS",
                "BL_SHAPE_FACTOR",
                "BL_SKIN_FRICTION",
                "BL_TRANSITION_MARKER",
            )
        },
    ),
}


def _listed(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip().strip("()")
    return [item.strip() for item in text.split(",") if item.strip()]


def product_outputs(values: dict[str, object], post: Mapping[str, object]) -> None:
    """Add SURFACE_PARAVIEW and PARAVIEW to OUTPUT_FILES when a declared product reads them.

    Nothing changes for a p file that declares no product. A list the setup states keeps its
    entries and gains only what is missing; without one, SU2's own default list is written
    with the additions.
    """
    from cfdbender.solvers.su2._run_types import SU2_OUTPUT_FILES

    wanted = [
        name
        for name, needed in (
            ("SURFACE_PARAVIEW", SECTIONS in post),
            ("PARAVIEW", VOLUME_SECTION in post or PROBES in post),
        )
        if needed
    ]
    if not wanted:
        return
    outputs = _listed(values.get("OUTPUT_FILES", list(SU2_OUTPUT_FILES)))
    upper = {item.upper() for item in outputs}
    values["OUTPUT_FILES"] = [*outputs, *(name for name in wanted if name not in upper)]


_MESH_LINE = re.compile(rb"^[ \t]*(IZONE|MARKER_TAG)[ \t]*=[ \t]*([^\r\n%]*)", re.MULTILINE)


def zone_markers(path: Path) -> dict[str, int]:
    """Return the zone (0-based, the index in SU2's file names) of each marker of a SU2 mesh.

    ``IZONE= 2`` opens the second zone and each ``MARKER_TAG`` after it belongs to it; a mesh
    that states no ``IZONE`` is one zone. The file is read in blocks, so a large mesh costs no
    memory. A marker name repeated in a later zone keeps the first zone.
    """
    found: dict[str, int] = {}
    zone = 0

    def scan(text: bytes) -> None:
        nonlocal zone
        for key, value in _MESH_LINE.findall(text):
            word = value.decode("ascii", errors="replace").strip()
            if key == b"IZONE":
                zone = int(word) - 1 if word.isdigit() and int(word) > 0 else zone
            else:
                found.setdefault(word, zone)

    tail = b""
    with path.open("rb") as stream:
        while block := stream.read(1 << 24):
            data = tail + block
            cut = data.rfind(b"\n") + 1
            tail = data[cut:]
            scan(data[:cut])
    scan(tail)
    return found


def _stem(values: Mapping[str, object], key: str, default: str) -> str:
    name = str(values.get(key, default)).strip()
    return name.removesuffix(".vtu") or default


def _last_iteration(folder: Path) -> int | None:
    from cfdbender.solvers.su2._history import read

    try:
        history = read(folder)[0].history
    except (OSError, ValueError, KeyError, StopIteration):
        return None
    return history[-1].iteration if history else None


def product_inputs(point: AdapterPoint) -> ProductInputs:
    """Return the surface and volume files of a recorded point and the facts they need.

    A steady point (``steady``, ``qsteady_rotor``) has one file of each kind; a
    time-accurate point has one per step of its export window, or the last step's without
    one. A file that is absent is named with the reason. A sliding-mesh multizone point has one
    file of each kind per zone and step, ``flow_<zone>_<iteration>.vtu`` (the order of SU2 8.3.0's
    files), listed in zone order with their ``zone``.
    """
    from cfdbender.solvers.su2._config import config, free_stream, gas_state

    record: dict[str, object] = {}
    values = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
        restarted=point.point.restart_from is not None,
        precursor=point.point.precursor,
        record=record,
    )
    folder = workspace_path(point.root, point.point.directory)
    relative = point.point.directory
    stream = free_stream(values)
    state = gas_state(values)
    free: FreeStream | None = None
    if stream is not None and stream.density is not None and state is not None:
        gas, temperature = state
        free = FreeStream(
            stream.density * gas * temperature,
            0.5 * stream.density * stream.velocity**2,
            stream.density,
            stream.velocity,
            CP_REFERENCE,
        )
    surface_stem = _stem(values, "SURFACE_FILENAME", "surface_flow")
    volume_stem = _stem(values, "VOLUME_FILENAME", "flow")
    timed = point.point.run_type in TIME_ACCURATE
    missing: list[tuple[str, str]] = []
    steps: list[tuple[int | None, float | None, str]] = []
    if timed:
        clock = cast(Mapping[str, object], record.get("clock") or {})
        total = int(cast(int, clock.get("time_iterations", 0)))
        delta = cast(float | None, clock.get("delta_time_s"))
        window = cast(Mapping[str, object] | None, record.get("export_window"))
        first = int(cast(int, window["first_step"])) if window else total
        # Step s is SU2's time iteration s - 1 + offset (RESTART_ITER of a restarted point).
        offset = int(cast(int, record.get("step_offset", 0)))
        for step in range(first, total + 1):
            suffix = f"_{step - 1 + offset:05d}"
            steps.append((step, None if delta is None else step * delta, suffix))
    else:
        steps.append((None, None, ""))
    multizone = bool(values.get("MULTIZONE"))
    zones: list[int | None] = (
        list(range(len(cast(list[object], values["CONFIG_LIST"])))) if multizone else [None]
    )
    found: dict[str, list[StepFile]] = {SURFACE: [], VOLUME: []}
    for kind, stem in ((SURFACE, surface_stem), (VOLUME, volume_stem)):
        for at, time_s, suffix in steps:
            for zone in zones:
                # SU2 writes the zone before the time iteration: flow_<zone>_<iteration>.vtu.
                name = f"{stem}{'' if zone is None else f'_{zone}'}{suffix}.vtu"
                if (folder / name).is_file():
                    found[kind].append(StepFile(at, time_s, f"{relative}/{name}", zone))
                else:
                    missing.append((f"{relative}/{name}", "the run did not write it"))
    markers = _listed(values.get("MARKER_PLOTTING", []))
    family_zones: dict[str, int] = {}
    if multizone and (folder / "mesh.su2").is_file():
        owned = zone_markers(folder / "mesh.su2")
        family_zones = {name: owned[name] for name in markers if name in owned}
    clock = cast(Mapping[str, object], record.get("clock") or {})
    spr = clock.get("steps_per_revolution")
    return ProductInputs(
        surface=tuple(found[SURFACE]),
        volume=tuple(found[VOLUME]),
        free_stream=free,
        normal_sign=NORMAL_SIGN,
        families=tuple(markers),
        families_separable=False,
        iteration=None if timed else _last_iteration(folder),
        steps_per_revolution=float(cast(float, spr)) if isinstance(spr, int | float) else None,
        time_accurate=timed,
        windowed=timed and record.get("export_window") is not None,
        missing=tuple(missing),
        zones=len(zones),
        family_zones=family_zones,
        notes={
            "velocity_frame": VELOCITY_FRAME,
            "cp_reference": CP_REFERENCE,
            "friction": FRICTION,
            "normal_sign": json.dumps(NORMAL_SIGN),
        },
    )
