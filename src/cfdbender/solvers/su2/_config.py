"""Validate s/r/p inputs and render deterministic SU2 configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import cast

from cfdbender.core.flight import (
    ATTITUDE_KEYS,
    STATE_KEYS,
    resolve_flow,
    speed_of_sound,
    states_flow,
)
from cfdbender.solvers._host._config import number as number
from cfdbender.solvers.su2.routes import ROUTES

NEUTRAL_KEYS = {
    "mach": "MACH_NUMBER",
    "alpha": "AOA",
    "angle_of_attack": "AOA",
    "beta": "SIDESLIP_ANGLE",
    "sideslip": "SIDESLIP_ANGLE",
    "freestream_temperature": "FREESTREAM_TEMPERATURE",
    "freestream_pressure": "FREESTREAM_PRESSURE",
    "freestream_density": "FREESTREAM_DENSITY",
    "freestream_velocity": "FREESTREAM_VELOCITY",
    "reynolds": "REYNOLDS_NUMBER",
    "reynolds_length": "REYNOLDS_LENGTH",
    "area": "REF_AREA",
    "length": "REF_LENGTH",
    "iterations": "ITER",
    "convergence_field": "CONV_FIELD",
    "convergence_residual": "CONV_RESIDUAL_MINVAL",
    "output_frequency": "OUTPUT_WRT_FREQ",
    "mesh": "MESH_FILENAME",
}
_OPTIONS = {
    "route",
    "executable",
    "pysu2_path",
    "threads",
    "timeout_s",
    "ram_estimate_gib",
    "mpi_launcher",
}
_NUMERIC = {
    "MACH_NUMBER",
    "AOA",
    "SIDESLIP_ANGLE",
    "FREESTREAM_TEMPERATURE",
    "FREESTREAM_PRESSURE",
    "FREESTREAM_DENSITY",
    "REYNOLDS_NUMBER",
    "REYNOLDS_LENGTH",
    "REF_AREA",
    "REF_LENGTH",
    "REF_ORIGIN_MOMENT_X",
    "REF_ORIGIN_MOMENT_Y",
    "REF_ORIGIN_MOMENT_Z",
    "ITER",
    "OUTPUT_WRT_FREQ",
    "CONV_RESIDUAL_MINVAL",
}


def options(
    setup: Mapping[str, object], workspace: Mapping[str, object] | None = None
) -> dict[str, object]:
    value = setup.get("solver", {})
    if not isinstance(value, dict):
        raise ValueError("s file [solver] must be a table")
    if unknown := set(value) - _OPTIONS:
        raise ValueError("s file [solver] has unknown options: " + ", ".join(sorted(unknown)))
    if workspace and (unknown := set(workspace) - _OPTIONS):
        raise ValueError("init.toml [solver] has unknown options: " + ", ".join(sorted(unknown)))
    result = dict(workspace or {})
    result.update(value)
    if "route" in result and result["route"] not in ROUTES:
        raise ValueError("solver.route must be pysu2 or executable")
    launcher = result.setdefault("mpi_launcher", [])
    if not isinstance(launcher, list) or any(
        not isinstance(item, str) or not item or any(ord(c) < 32 for c in item) for item in launcher
    ):
        raise ValueError("solver.mpi_launcher must be a list of nonempty strings")
    for path_key in ("executable", "pysu2_path"):
        if path_key in result and (not isinstance(result[path_key], str) or not result[path_key]):
            raise ValueError(f"solver.{path_key} must be a nonempty path")
    threads = result.setdefault("threads", 1)
    if type(threads) is not int or threads < 1:
        raise ValueError("solver.threads must be a positive integer")
    for key, default in (("timeout_s", 600.0), ("ram_estimate_gib", 0.25)):
        result.setdefault(key, default)
        if number(result[key], key) <= 0:
            raise ValueError(f"solver.{key} must be positive")
    return result


def render_value(value: object, name: str) -> str:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (int, float)):
        return format(number(value, name), ".15g")
    if isinstance(value, list):
        return "( " + ", ".join(render_value(item, name) for item in value) + " )"
    if (
        isinstance(value, str)
        and value
        and value.isascii()
        and not any(ord(c) < 32 or c in "%=" for c in value)
    ):
        return value
    raise ValueError(f"{name} needs an ASCII SU2 scalar or list without control characters, % or =")


def _reference_length(*tables: Mapping[str, object]) -> float | None:
    for table in tables:
        for key in ("length", "REF_LENGTH"):
            if key in table:
                return number(table[key], key)
    return None


def _flow(
    condition: Mapping[str, float], setup: Mapping[str, object], reference: Mapping[str, object]
) -> tuple[dict[str, float], dict[str, object]]:
    """Split the matrix condition into SU2 keys and the resolved free stream."""
    kept = {key: value for key, value in condition.items() if key not in STATE_KEYS}
    for key in kept:
        if key in ATTITUDE_KEYS and key not in {"ALPHA", "BETA"}:
            raise ValueError(f"the SU2 adapter does not read FLIGHT_CONDITION key {key}")
    if not states_flow(condition):
        if "MACH" in condition:
            kept["MACH"] = condition["MACH"]
        return kept, {}
    state = resolve_flow(
        condition,
        reference_length=_reference_length(reference, setup),
        label="FLIGHT_CONDITION",
    )
    derived: dict[str, object] = {
        "MACH_NUMBER": state.mach,
        "FREESTREAM_TEMPERATURE": state.temperature,
        "FREESTREAM_PRESSURE": state.pressure,
        "FREESTREAM_DENSITY": state.density,
    }
    if state.reynolds is not None:
        derived.update(REYNOLDS_NUMBER=state.reynolds, REYNOLDS_LENGTH=state.reynolds_length)
    if "MUPas" in state.stated:
        derived.update(VISCOSITY_MODEL="CONSTANT_VISCOSITY", MU_CONSTANT=state.viscosity)
    if "ASMPS" in state.stated and not math.isclose(
        state.sonic_velocity, speed_of_sound(state.temperature), rel_tol=1e-3
    ):
        raise ValueError(
            "ASMPS pins a speed of sound SU2 cannot hold: SU2 derives it from the "
            "temperature; pin TK instead"
        )
    return kept, derived


def _same(stated: object, derived: object) -> bool:
    if isinstance(derived, float) and isinstance(stated, (int, float)):
        return not isinstance(stated, bool) and math.isclose(stated, derived, rel_tol=1e-9)
    return stated == derived


def config(
    setup: Mapping[str, object],
    reference: Mapping[str, object],
    post: Mapping[str, object],
    condition: Mapping[str, float],
    *,
    geometry: str | None = None,
) -> dict[str, object]:
    options(setup)
    if type(setup.get("mesh_link", True)) is not bool:
        raise ValueError("mesh_link must be a boolean")
    if setup.get("mesh_link", True) is False:
        raise ValueError(
            "mesh_link = false asks for a mesh copy; meshes are always linked, "
            "never copied: remove mesh_link"
        )
    if post.get("history") is not True:
        raise ValueError("pproc history must be true")
    condition, derived = _flow(condition, setup, reference)
    if geometry is not None:
        derived["MESH_FILENAME"] = geometry
    values: dict[str, object] = {}

    def add(key: str, value: object) -> None:
        if key in values:
            raise ValueError(f"conflict: {key} is supplied more than once")
        if key != "MESH_FILENAME":
            render_value(value, key)
        values[key] = value

    for table in (setup, reference, post, condition):
        for key, value in table.items():
            if (
                key in {"solver", "mesh_link"}
                and table is setup
                or key == "history"
                and table is post
            ):
                continue
            if key == "moment_origin":
                if not isinstance(value, list) or len(value) != 3:
                    raise ValueError("moment_origin needs three finite coordinates")
                for axis, coordinate in zip("XYZ", value, strict=True):
                    add(f"REF_ORIGIN_MOMENT_{axis}", number(coordinate, key))
                continue
            target = NEUTRAL_KEYS.get(key.lower())
            if target is None:
                if re.fullmatch("[A-Z][A-Z0-9_]*", key) is None:
                    raise ValueError(
                        f"unsupported input {key}; SU2-specific keys must be uppercase"
                    )
                target = key
            add(target, value)
    for key, value in derived.items():
        if key in values and not _same(values[key], value):
            source = "GEOMETRY" if key == "MESH_FILENAME" else "FLIGHT_CONDITION"
            raise ValueError(
                f"conflict: {key} is {values[key]!r} in the s/r/p files and "
                f"{value!r} from the matrix {source}; state it in one place"
            )
        values[key] = value
    for key in ("REF_AREA", "REF_LENGTH", "ITER", "MESH_FILENAME"):
        if key not in values:
            raise ValueError(f"missing {key}")
    for key in values.keys() & _NUMERIC:
        value = number(values[key], key)
        if key in {"REF_AREA", "REF_LENGTH", "ITER", "OUTPUT_WRT_FREQ"} and value <= 0:
            raise ValueError(f"{key} must be positive")
        if key in {"ITER", "OUTPUT_WRT_FREQ"} and value != int(value):
            raise ValueError(f"{key} must be an integer")
        if key == "MACH_NUMBER" and value < 0:
            raise ValueError("MACH_NUMBER must be non-negative")
        if (
            key
            in {
                "FREESTREAM_TEMPERATURE",
                "FREESTREAM_PRESSURE",
                "FREESTREAM_DENSITY",
                "REYNOLDS_NUMBER",
                "REYNOLDS_LENGTH",
            }
            and value <= 0
        ):
            raise ValueError(f"{key} must be positive")
    if "FREESTREAM_VELOCITY" in values:
        velocity = values["FREESTREAM_VELOCITY"]
        if not isinstance(velocity, list) or len(velocity) != 3:
            raise ValueError("freestream_velocity needs three finite coordinates")
        for coordinate in velocity:
            number(coordinate, "freestream_velocity")
    if not all(f"REF_ORIGIN_MOMENT_{axis}" in values for axis in "XYZ"):
        raise ValueError("moment_origin needs three finite coordinates")
    mesh = values["MESH_FILENAME"]
    if not isinstance(mesh, str) or not mesh.endswith((".su2", ".cgns")):
        raise ValueError("mesh must name a workspace-relative .su2 or .cgns file")
    # Classification needs a fixed CSV history contract. Conflicting output choices refuse.
    for key, value in {
        "MESH_FORMAT": "CGNS" if mesh.endswith(".cgns") else "SU2",
        "CONV_FILENAME": "history",
        "TABULAR_FORMAT": "CSV",
    }.items():
        if key in values and values[key] != value:
            raise ValueError(f"conflict: {key} must be {value}")
        values.setdefault(key, value)
    return values


def mesh_name(values: Mapping[str, object]) -> str:
    return cast(str, values["MESH_FILENAME"])


def staged_mesh(values: Mapping[str, object]) -> str:
    """Return the stable point-local name for the selected mesh format."""
    return "mesh.cgns" if mesh_name(values).endswith(".cgns") else "mesh.su2"


def render(values: Mapping[str, object]) -> str:
    return "".join(
        f"{key}= {render_value(staged_mesh(values) if key == 'MESH_FILENAME' else value, key)}"
        + chr(10)
        for key, value in sorted(values.items())
    )
