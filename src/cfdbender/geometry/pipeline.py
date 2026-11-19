"""The runner of the geometry pipeline: the stages in order, stop at the first failed check.

Pipeline role: executes the stages of :mod:`cfdbender.geometry.stage_table` for one resolved
configuration and returns the report a design loop reads. A stage returns its checks, and the
runner writes one stage record per stage (``S<n>.json``), a human-readable log and a report
last (``report.json``). The first failed check stops the run and the report names its stage and
its check, with the value measured and the limit; a refused input stops it as ``refused``. A
report is returned, never raised, so a loop of parameter sets continues after a failed one.

The pure stages need numpy and scipy only: S0 (definition), S8 (closures) and its blend
sub-steps, S10 (the junction plan) and the pure propeller sub-steps; the checks of S1 to S7
and S9 are measured inside the construction of the components. The kernel stages open gmsh
in a worker process: the S10 assembly sub-steps, S11 (the exported IGES and its measured
sub-steps: registry, profile, read-back, sew and the propeller checks), S12 (the package's
own convert and grid read the export back) and S13 (the round trip: the stored sections
regenerated and the export identified). A stage with no function (S1 to S7 and S9, and the
control-surface sub-steps S10.F0 to S10.F7, which belong to a solver set-up) is recorded with
every check ``skipped`` and the reason; the report then lists each such check in ``not_run``
and its status is ``incomplete`` (exit code 3), never ``accepted``. The default ``until`` is
``S13``; ``until="S10"`` stops before the export.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from cfdbender._errors import (
    CheckFailedError,
    GeometryError,
    InputArtifactError,
    InputValueError,
)
from cfdbender.geometry._propeller import Propeller
from cfdbender.geometry._run_log import RunLog
from cfdbender.geometry.config import _read_config, resolve_config
from cfdbender.geometry.stage_table import STAGE_ROWS, StageRow, stage_index, stage_row
from cfdbender.geometry.stages import (
    CAP_EDGE_GAP,
    adopt_propeller,
    adopt_surface_file,
    blister_model,
    cap_edge_gap,
    check_entry,
    fillet_model,
    geometry_report,
    junctions,
    lifting_model,
    parameter_id,
    propeller_measurement,
    stage_record,
    unpack,
    write_record,
)
from cfdbender.mesh._gmsh_worker import _ENV as WORKER_ENV
from cfdbender.mesh._gmsh_worker import run_isolated

__all__ = [
    "PROPELLER_STAGES",
    "RunContext",
    "Stage",
    "StageOutcome",
    "built_stages",
    "default_stages",
    "describe_config",
    "list_configurations",
    "run_geometry",
]


@dataclass
class StageOutcome:
    """What one stage returns: its checks, findings, warnings and the values it hands on."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    wall_s: float | None = None


@dataclass
class RunContext:
    """The inputs every stage reads and the values a stage hands to the next.

    ``resolved`` is the resolved configuration (with every override applied), ``members`` the
    component names of the selected configuration, ``state`` the scratch the stages share, and
    ``out`` the folder the records are written to.
    """

    resolved: dict[str, Any]
    configuration: str | None
    variant: str
    out: Path
    parameter_id: str
    members: list[str]
    state: dict[str, Any] = field(default_factory=dict)


StageFn = Callable[[RunContext], StageOutcome]


@dataclass(frozen=True)
class Stage:
    """One stage of a run: its row of the table and its function, when this release has one."""

    row: StageRow
    run: StageFn | None


def _definition(context: RunContext) -> StageOutcome:
    """S0: the configuration resolved, so every key is known and every relationship holds."""
    resolved = context.resolved
    return StageOutcome(
        checks=[check_entry("input.valid", "passed", 0, 0, unit="refused keys")],
        findings=[str(item) for item in resolved.get("findings", [])],
        warnings=[str(item) for item in resolved.get("warnings", [])],
        outputs={"components": list(resolved["components"]), "members": context.members},
    )


#: The inputs of each blend piece recorded with its sub-step checks.
_BLEND_INPUTS = {
    "fillet": ("hosts", "laws", "extend_lower_to_te", "patches", "wing_band", "host_band"),
    "blister": ("host", "stations", "profile", "sections", "sections_file", "tips"),
}

#: The stage ids of each blend piece's sub-steps start with its prefix.
_BLEND_PREFIX = {"fillet": "S8.F", "blister": "S8.B"}


def _blend_model(kind: str, component: Any, components: Any, name: str) -> Any:
    """Return the model of one piece, or the failed check its build stopped at."""
    try:
        if kind == "fillet":
            return fillet_model(component, components, name)
        host = component["host"]
        if components[host]["type"] != "body":
            raise InputValueError(f"{name}.host: a blister lies on a body; {host!r} is not one")
        return blister_model(component, components[host])
    except CheckFailedError as failure:
        return failure.check


def _blend_models(context: RunContext, kind: str) -> dict[str, Any]:
    """Return the fillet or blister models of the selected members, built once per run."""
    key = f"blend.{kind}"
    if key not in context.state:
        components = unpack(context.resolved)["components"]
        context.state[key] = {
            name: _blend_model(kind, components[name], components, name)
            for name in context.members
            if components[name]["type"] == kind
        }
    models: dict[str, Any] = context.state[key]
    return models


def _holder(kind: str, check_id: str) -> str:
    """Return the sub-step of ``kind`` whose row names ``check_id``; its first one otherwise."""
    rows = [r for r in STAGE_ROWS if r.stage.startswith(_BLEND_PREFIX[kind])]
    return next((r.stage for r in rows if check_id in {c for c, _ in r.checks}), rows[0].stage)


def _stopped_piece(
    kind: str, stage: str, wanted: list[str], name: str, failed: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the checks of ``stage`` for a piece whose build stopped at the check ``failed``."""
    holder = _holder(kind, failed["id"])
    if holder == stage:
        return [{**failed, "where": {**failed["where"], "component": name}}]
    note = f"not measured: {name} stopped at {holder} {failed['id']}"
    return [check_entry(cid, "skipped", where={"component": name}, note=note) for cid in wanted]


def _blend_step(kind: str, stage: str) -> StageFn:
    """Return the function of one sub-step of a blend piece: its own checks of each piece."""

    def run(context: RunContext) -> StageOutcome:
        wanted = [cid for cid, _ in stage_row(stage).checks]
        models = _blend_models(context, kind)
        if not models:
            note = f"no {kind} among the members of this configuration"
            return StageOutcome(checks=[check_entry(cid, "skipped", note=note) for cid in wanted])
        components = unpack(context.resolved)["components"]
        checks: list[dict[str, Any]] = []
        inputs: dict[str, Any] = {}
        for name, model in models.items():
            if isinstance(model, dict):  # the build stopped at this failed check
                checks += _stopped_piece(kind, stage, wanted, name, model)
            else:
                for record in model.records:
                    if record["id"] in wanted:
                        checks.append({**record, "where": {**record["where"], "component": name}})
            inputs[name] = {
                k: components[name][k]
                for k in _BLEND_INPUTS[kind]
                if components[name].get(k) not in (None, "")
            }
        return StageOutcome(checks=checks, outputs={"inputs": inputs})

    return run


def _closures(context: RunContext) -> StageOutcome:
    """S8: measure the lifting models' cap closures, building absent models once."""
    checks: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    models = context.state.setdefault("lifting_models", {})
    components = unpack(context.resolved)["components"]
    for name in context.members:
        component = components[name]
        if component["type"] != "lifting_surface":
            continue
        if name not in models:
            models[name] = lifting_model(component)
        model = models[name]
        if not model.caps:
            continue
        gap = cap_edge_gap(model)
        failed = gap > CAP_EDGE_GAP
        checks.append(
            check_entry(
                "closure.cap_edge_gap",
                "failed" if failed else "passed",
                gap,
                CAP_EDGE_GAP,
                unit="m",
                where={"component": name},
                note="BUG: a cap boundary leaves its neighboring edge"
                if failed
                else "4001 samples on each shared cap boundary",
            )
        )
        inputs[name] = {"cap": component["tip_cap"], "root": component["root"]}
    if not checks:
        checks.append(check_entry("closure.cap_edge_gap", "skipped", note="no lifting caps"))
    return StageOutcome(checks=checks, outputs={"inputs": inputs})


def _dominance(context: RunContext) -> StageOutcome:
    """S10: the dominance rows read on the members; a cycle was refused at S0 already."""
    warnings: list[str] = []
    rows = unpack(context.resolved).get("dominance", [])
    plan = junctions.dominance_plan(rows, context.members, warnings)
    names = [f"{op.operation} {op.dominant} > {op.dominated}" for op in plan]
    return StageOutcome(
        checks=[check_entry("dominance.plan", "passed", 0, 0, unit="cycles")],
        warnings=warnings,
        outputs={"operations": names},
    )


#: The checks each pure propeller sub-step reports, from the propeller's own measurements.
_PROPELLER_CHECKS = {
    "S1.P": ("propeller.section_fit",),
    "S4.P": (
        "propeller.solidity",
        "propeller.activity_factor",
        "propeller.beta_reference",
        "propeller.beta_0p7R",
    ),
    "S5.P": ("propeller.section_plane", "propeller.handedness"),
    "S6.P": ("propeller.loft_exact", "propeller.loft_c2"),
    "S8.P": ("propeller.closure",),
}

#: Every propeller sub-step of the table; those of S11 need the kernel.
PROPELLER_STAGES = frozenset({*_PROPELLER_CHECKS, "S11.PU", "S11.PC", "S11.PS"})


def _propellers(context: RunContext) -> dict[str, Propeller]:
    """Return the propeller models of the selected members, built once per run."""
    if "propellers" not in context.state:
        components = unpack(context.resolved)["components"]
        context.state["propellers"] = {
            n: Propeller(components[n]["definition"])
            for n in context.members
            if components[n]["type"] == "propeller" and "definition" in components[n]
        }
    models: dict[str, Propeller] = context.state["propellers"]
    return models


def _propeller_step(stage: str) -> StageFn:
    """Return the function of one pure propeller sub-step: its checks of each propeller."""

    def run(context: RunContext) -> StageOutcome:
        models = _propellers(context)
        if not models:
            note = "no propeller definition in this run"
            return StageOutcome(
                checks=[
                    check_entry(cid, "skipped", note=note) for cid, _ in stage_row(stage).checks
                ]
            )
        checks: list[dict[str, Any]] = []
        for name, model in models.items():
            blade = model.blade()
            candidates = model.report()["checks"] + [
                propeller_measurement(
                    "propeller.loft_exact", blade.info["loft_error_mm"], 1e-8, "mm"
                ),
                propeller_measurement(
                    "propeller.loft_c2", blade.info["loft_c2_violations"], 0, "knots"
                ),
            ]
            checks += [
                {**c, "where": {"component": name}}
                for c in candidates
                if c["id"] in _PROPELLER_CHECKS[stage]
            ]
        inputs = {name: model.config for name, model in models.items()}
        return StageOutcome(checks=checks, outputs={"inputs": inputs})

    return run


# ---------------------------------------------------------------- stages with the kernel
# S10 assembly sub-steps, S11 export and its measured sub-steps, S12 reader gate and S13 round
# trip. Each opens the kernel in a worker process (cfdbender.mesh._gmsh_worker) unless the run
# already is one, so a native fault of the kernel fails the stage and the caller's loop goes on.


def _generate(resolved: dict[str, Any], target: Path, configuration: str | None) -> Any:
    """Write the IGES with the kernel in a worker process, unless this already is one."""
    arguments = {"config": resolved, "path": str(target), "configuration": configuration}
    if os.environ.get(WORKER_ENV) == "1":
        return _generate_record(resolved, str(target), configuration)
    return run_isolated(f"{__name__}:_generate_record", arguments, _generate_record)


def _generate_record(config: Any, path: str, configuration: str | None) -> dict[str, Any]:
    """Write the IGES and return the record of ``generate_iges`` as JSON values (a worker's)."""
    from cfdbender.geometry._geom_export import generate_record

    written = generate_record(config, path, configuration)
    result: dict[str, Any] = json.loads(json.dumps(written, default=str))
    return result


def _export(context: RunContext) -> StageOutcome:
    """S11: the exposed skin of the selected members as one named IGES, read by the gate."""
    name = context.configuration or str(
        unpack(context.resolved["aircraft"]).get("name", "geometry")
    )
    target = context.out / f"{name}.igs"
    try:
        assembled = context.state.get("assembly")
        written = assembled["export"] if assembled else _generate_variant(context, target)
    except InputArtifactError as error:
        if "solid entities" not in str(error):
            raise
        return StageOutcome(
            checks=[check_entry("profile.surfaces_only", "failed", 1, 0, note=str(error))]
        )
    warnings = [
        str(item) for item in written["warnings"] if item not in context.resolved["warnings"]
    ]
    context.state["export"] = written
    measured = _export_step("S11")(context)
    return StageOutcome(
        checks=measured.checks,
        warnings=warnings + measured.warnings,
        outputs={
            **measured.outputs,
            "iges": written["iges"],
            "manifest": written["manifest"],
            "surface_id": written["surface_id"],
            "build_id": written["build_id"],
            "faces": written["faces"],
            "audit": written["audit"],
            "variant_exports": written.get("variant_exports", []),
        },
    )


def _export_step(stage: str) -> StageFn:
    """Return one measured export sub-step, read against the retained source identity."""

    def run(context: RunContext) -> StageOutcome:
        from cfdbender.geometry._geom_export import check_export

        return StageOutcome(
            **check_export(
                stage,
                context.resolved,
                configuration=context.configuration,
                members=context.members,
                out=context.out,
                source=context.state["export"],
            )
        )

    return run


def _generate_variant(context: RunContext, target: Path) -> Any:
    if context.variant == "clean":
        return _generate(context.resolved, target, context.configuration)
    from cfdbender.geometry._geom_variant import generate

    arguments = {
        "config": context.resolved,
        "path": str(target),
        "configuration": context.configuration,
        "variant": context.variant,
        "members": context.members,
    }
    return (
        generate(**arguments)
        if os.environ.get(WORKER_ENV) == "1"
        else run_isolated("cfdbender.geometry._geom_variant:generate", arguments, generate)
    )


def _plan(context: RunContext) -> tuple[list[Any], list[str]]:
    """Return the dominance operations on the members of the run, and the rows it skipped."""
    warnings: list[str] = []
    rows = unpack(context.resolved).get("dominance", [])
    return junctions.dominance_plan(rows, context.members, warnings), warnings


def _assembly(context: RunContext) -> dict[str, Any] | None:
    """Return the sub-step records of the assembly, run once per run in a worker.

    None when no dominance row applies to the members: nothing is assembled, no kernel opens.
    """
    if "assembly" not in context.state:
        from cfdbender.geometry._geom_variant import assemble_positions

        plan, _ = _plan(context)
        operation: Callable[..., Any] = (
            _assembly_record if context.variant == "clean" else assemble_positions
        )
        arguments: dict[str, Any] = {
            "config": context.resolved,
            "members": context.members,
            "out": str(context.out / "S10-assembly.json"),
            "configuration": context.configuration,
            "variant": context.variant,
        }
        if not plan:
            context.state["assembly"] = None
        elif os.environ.get(WORKER_ENV) == "1":
            context.state["assembly"] = operation(**arguments)
        else:
            context.state["assembly"] = run_isolated(
                f"{operation.__module__}:{operation.__name__}", arguments, operation
            )
    assembly: dict[str, Any] | None = context.state["assembly"]
    return assembly


def _assembly_record(
    config: Any,
    members: list[str],
    out: str,
    configuration: str | None = None,
    variant: str = "clean",
    *,
    selection: Any = None,
) -> dict[str, Any]:
    """Build the members on a kernel and run the S10 sub-steps; return their records as JSON."""
    from cfdbender.geometry._geom_assembly import Assembly
    from cfdbender.geometry._geom_export import write_assembly
    from cfdbender.geometry.stages import (
        GeneratedPiece,
        generate_components,
        generate_pieces,
    )
    from cfdbender.mesh._gmsh import open_kernel

    resolved = resolve_config(copy.deepcopy(config))
    every = (
        generate_components(resolved, trimmed=True)
        if variant == "clean"
        else generate_pieces(resolved, variant=variant, selection=selection)
    )
    again = (
        generate_components(resolve_config(copy.deepcopy(config)), trimmed=True)
        if variant == "clean"
        else generate_pieces(
            resolve_config(copy.deepcopy(config)), variant=variant, selection=selection
        )
    )
    plain = unpack(resolved)
    pieces = {name: every[name] for name in members if name in every}
    half = all(
        float(p.controls[..., 1].min()) >= -1e-9
        for ps in pieces.values()
        for p in (ps.bases.values() if isinstance(ps, GeneratedPiece) else ps)
    )
    kernel = open_kernel("assembling the pieces by the dominance rows")
    try:
        assembly = Assembly(
            kernel,
            pieces,
            plain.get("dominance", []),
            out=Path(out).parent,
            members=members,
            kinds={name: plain["components"][name]["type"] for name in pieces},
            plane_axis=1 if half else None,
            regenerated={
                name: list(p.bases.values()) if isinstance(p, GeneratedPiece) else list(p)
                for name, p in again.items()
                if name in pieces
            },
            coverage_fraction=float(plain["acceptance"]["coverage_fraction"]),
            coverage_spacing=float(plain["acceptance"]["coverage_spacing"]),
        )
        records = assembly.run()
        written: dict[str, Any] = {}
        if not any(c["status"] == "failed" for r in records for c in r["checks"]):
            for name in pieces:
                if name not in assembly.models:
                    assembly.models[name] = GeneratedPiece.from_patches(
                        name, plain["components"][name]["version"], assembly.sources[name]
                    )
            written = write_assembly(
                assembly,
                Path(out).parent / "assembly.igs",
                resolved=resolved,
                configuration=configuration,
                variant=variant,
                selection=selection,
            )
            written["sections"] = {
                n: p.layout["sections"]
                for n, p in every.items()
                if isinstance(p, GeneratedPiece) and "sections" in p.layout
            }
    finally:
        kernel.close()
    result: dict[str, Any] = json.loads(
        json.dumps({"records": records, "export": written}, default=str)
    )
    write_record(Path(out), result)
    return result


def _assembly_step(stage: str) -> StageFn:
    """Return the function of one sub-step of S10: its record from the assembly's run."""

    def run(context: RunContext) -> StageOutcome:
        ids = [cid for cid, _ in stage_row(stage).checks]
        result = _assembly(context)
        if result is None:
            note = "no dominance row applies to the members of this configuration"
            return StageOutcome(checks=[check_entry(c, "skipped", note=note) for c in ids])
        record = next((r for r in result["records"] if r["stage"] == stage), None)
        if record is None:
            note = f"not measured: the assembly stopped at {result['records'][-1]['stage']}"
            return StageOutcome(checks=[check_entry(c, "skipped", note=note) for c in ids])
        return StageOutcome(
            checks=record["checks"],
            warnings=record["warnings"],
            outputs=record["outputs"],
            wall_s=float(record["wall_s"]),
        )

    return run


def _kernel_propeller_step(stage: str) -> StageFn:
    """Return one propeller sub-step of S11, measured on the exported file."""

    def run(context: RunContext) -> StageOutcome:
        from cfdbender.geometry._geom_propeller import stage_result

        return StageOutcome(**stage_result(context, stage))

    return run


#: The control-surface sub-steps of S10 (S10.F0 to S10.F7) build the solver set-up's
#: controls and are not part of this package; they are recorded skipped with this note.
_CONTROL_NOTE = "control-surface assembly is not built in this package"


#: The stage functions of this release, by stage id. A stage not named here is recorded skipped.
STAGE_FUNCTIONS: dict[str, StageFn] = {
    **{stage: _propeller_step(stage) for stage in _PROPELLER_CHECKS},
    "S0": _definition,
    "S8": _closures,
    "S10": _dominance,
    **{f"S8.F{n}": _blend_step("fillet", f"S8.F{n}") for n in (1, 2, 3)},
    **{f"S8.B{n}": _blend_step("blister", f"S8.B{n}") for n in (1, 2, 3)},
    **{
        row.stage: _assembly_step(row.stage)
        for row in STAGE_ROWS
        if row.stage.startswith("S10.") and not row.stage.startswith("S10.F")
    },
    **{stage: _kernel_propeller_step(stage) for stage in ("S11.PU", "S11.PC", "S11.PS")},
    "S11": _export,
    **{
        stage: _export_step(stage)
        for stage in ("S11.V", "S11.P", "S11.R", "S11.S", "S12.C", "S12.G", "S12", "S13.R", "S13")
    },
}


def _applicable(planned: list[Stage], resolved: dict[str, Any], members: list[str]) -> list[Stage]:
    """Select the propeller-only workflow; a mixed assembly keeps its full plan."""
    components = unpack(resolved)["components"]
    propellers = [
        n for n in members if components[n]["type"] == "propeller" and "definition" in components[n]
    ]
    if not propellers:
        return [stage for stage in planned if stage.row.stage not in PROPELLER_STAGES]
    if len(propellers) != len(members):
        return planned
    keep = {"S0", "S11", "S11.V", "S11.P", "S11.R", "S11.S"} | PROPELLER_STAGES
    return [stage for stage in planned if stage.row.stage in keep]


def default_stages(*, include_propeller: bool = False) -> list[Stage]:
    """Return the stages of the table with this release's functions.

    Parameters
    ----------
    include_propeller : bool, optional
        Whether the propeller sub-steps are included.

    Returns
    -------
    list of Stage
        Every row of the table, with its function or None when this release builds none.
    """
    return [
        Stage(row, STAGE_FUNCTIONS.get(row.stage))
        for row in STAGE_ROWS
        if include_propeller or row.stage not in PROPELLER_STAGES
    ]


def built_stages() -> set[str]:
    """Return the ids of the stages this release has a function for.

    Returns
    -------
    set of str
        The stage ids that run; the other stages of the table are reported as not measured.

    Examples
    --------
    >>> from cfdbender.geometry.pipeline import built_stages
    >>> "S0" in built_stages(), "S11" in built_stages()
    (True, False)
    """
    return set(STAGE_FUNCTIONS)


def _package_version() -> str:
    try:
        return metadata.version("cfdbender")
    except metadata.PackageNotFoundError:
        return "0.0.0+uninstalled"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _set_path(raw: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``component.key.key`` of the raw ``components`` table, refusing an unknown component."""
    parts = dotted.split(".")
    components = raw.get("components")
    if len(parts) < 2 or not isinstance(components, dict) or parts[0] not in components:
        known = sorted(components) if isinstance(components, dict) else []
        raise InputValueError(
            f"overrides: {dotted!r} must be <component>.<key>[.<key>] of a declared component "
            f"(declared: {known})"
        )
    node = components[parts[0]]
    for part in parts[1:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise InputValueError(f"overrides: {dotted!r}: {part!r} is not a table")
    node[parts[-1]] = value


def _reported_unknown(warnings: Sequence[Any], dotted: str) -> bool:
    """Whether the resolver warned that ``dotted``, or a table on its path, is an unknown key."""
    parts = dotted.split(".")
    prefixes = [".".join(parts[:n]) for n in range(2, len(parts) + 1)]
    return any(
        re.search(rf"(?:^|\s)(?:components\.)?{re.escape(prefix)}: unknown key", str(w))
        for prefix in prefixes
        for w in warnings
    )


def _resolve(config: Any, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the resolved configuration of a path or mapping with ``overrides`` applied."""
    stem: str | None
    base: Path | None
    if isinstance(config, (str, Path)):
        path = Path(config)
        raw = _read_config(path)
        stem, base = path.stem, path.parent
    else:
        raw, stem, base = copy.deepcopy(dict(config)), None, None
    if overrides and raw.get("format") == "cfdbender-geometry-resolved":
        raise InputValueError("overrides: give a path or a raw configuration, not a resolved one")
    if overrides:
        # a [surface] or [body] file is one component named by its table: overrides address it
        raw = adopt_propeller(adopt_surface_file(raw))
    for dotted, value in (overrides or {}).items():
        _set_path(raw, dotted, value)
    resolved: dict[str, Any] = resolve_config(raw, stem=stem, base=base)
    for dotted in overrides or {}:
        if _reported_unknown(resolved["warnings"], dotted):
            raise InputValueError(f"overrides: {dotted!r}: unknown parameter, nothing was run")
    return resolved


def _members(resolved: dict[str, Any], configuration: str | None, variant: str) -> list[str]:
    configurations = resolved["configurations"]
    if configuration is None:
        members = list(resolved["components"])
    elif configuration not in configurations:
        raise InputValueError(
            f"configuration {configuration!r}: unknown; available: {sorted(configurations)}"
        )
    else:
        members = list(configurations[configuration])
    if variant not in unpack(resolved["variants"]):
        raise InputValueError(
            f"variant {variant!r}: unknown; available: {sorted(unpack(resolved['variants']))}"
        )
    if configuration is not None:
        allowed = unpack(resolved["registry"])[configuration]["variants"]
        if variant not in allowed:
            raise InputValueError(
                f"configuration {configuration!r}: variant {variant!r} is excluded; "
                f"available: {allowed}"
            )
    return members


def _leaves(node: Any, path: str) -> Iterator[dict[str, Any]]:
    """Yield ``{path, value, unit, source}`` for every wrapped value under ``node``."""
    if isinstance(node, dict) and {"value", "unit", "source"} <= set(node):
        yield {"path": path, **node}
    elif isinstance(node, dict):
        for key, child in node.items():
            yield from _leaves(child, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from _leaves(child, f"{path}[{index}]")


def describe_config(config: Any, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a configuration (S0 only) and return every value with its unit and source.

    Parameters
    ----------
    config : Mapping or str or Path
        A configuration, or the path of a TOML file.
    overrides : Mapping, optional
        Values to set before resolving, by dotted path.

    Returns
    -------
    dict
        ``status``, the ``parameter_id``, the ``values`` with unit and source, and the
        ``warnings`` and ``findings`` of the resolution.

    Examples
    --------
    >>> from cfdbender.geometry.pipeline import describe_config
    >>> describe_config({"type": "lifting_surface"})["status"]
    'valid'
    """
    resolved = _resolve(config, overrides)
    values = [
        leaf
        for section in ("aircraft", "acceptance", "reader", "components")
        for leaf in _leaves(resolved[section], section)
    ]
    return {
        "status": "valid",
        "parameter_id": parameter_id(resolved),
        "values": values,
        "warnings": list(resolved["warnings"]),
        "findings": list(resolved["findings"]),
    }


def list_configurations(config: Any) -> dict[str, Any]:
    """Return the configurations of a configuration file with their members and variants.

    Parameters
    ----------
    config : Mapping or str or Path
        A configuration, or the path of a TOML file.

    Returns
    -------
    dict
        The ``components``, the ``variants`` and, per configuration, its ``members``.

    Examples
    --------
    >>> from cfdbender.geometry.pipeline import list_configurations
    >>> list_configurations({"type": "lifting_surface"})["components"]
    ['COMPONENT']
    """
    resolved = unpack(dict(_resolve(config, None)))
    return {
        "components": list(resolved["components"]),
        "variants": sorted(resolved["variants"]),
        "configurations": {
            name: {"members": members, "variants": resolved["registry"][name].get("variants")}
            for name, members in resolved["configurations"].items()
        },
    }


#: The note of a check of the stage table that a built stage did not return.
_OMITTED = "not measured by this stage in this release"


def _skipped(row: StageRow) -> StageOutcome:
    note = _CONTROL_NOTE if row.stage.startswith("S10.F") else "stage not built in this release"
    return StageOutcome(
        checks=[check_entry(cid, "skipped", note=note) for cid, _ in row.checks],
        warnings=[f"{row.stage} {row.name}: {note}"],
    )


def _with_table_checks(row: StageRow, outcome: StageOutcome) -> list[dict[str, Any]]:
    """Return the stage's checks plus a ``skipped`` entry for each check of its row it omitted."""
    given = {check["id"] for check in outcome.checks}
    missing = [
        check_entry(cid, "skipped", note=_OMITTED) for cid, _ in row.checks if cid not in given
    ]
    return [*outcome.checks, *missing]


def _not_measured(record: Mapping[str, Any], built: bool) -> list[str]:
    """Return ``"<stage> <check>"`` for each check of a record the stage did not measure.

    Every check of a stage that is not built, and each check a built stage omitted; a check a
    built stage itself reports skipped (it does not apply, with its reason) is not listed.
    """
    return [
        f"{record['stage']} {check['id']}"
        for check in record["checks"]
        if check["status"] == "skipped" and (not built or check["note"] == _OMITTED)
    ]


def _refused(
    message: str, *, parameter: str = "", stage: str = "S0", check: str = "input.valid"
) -> dict[str, Any]:
    report: dict[str, Any] = geometry_report(
        "refused", parameter or None, stage=stage, check=check, message=message
    )
    return report


def run_geometry(
    config: Any,
    *,
    out: str | Path,
    configuration: str | None = None,
    variant: str = "clean",
    overrides: Mapping[str, Any] | None = None,
    until: str = "S13",
    stages: Sequence[Stage] | None = None,
) -> dict[str, Any]:
    """Run the stages S0 to ``until`` and return the report; it stops at the first failed check.

    Parameters
    ----------
    config : path or mapping
        A configuration file or its raw mapping.
    out : path
        The folder the stage records (``S<n>.json``), the log and the report (``report.json``)
        are written to.
    configuration, variant : str
        The entry of the configuration registry to build, and the variant of it.
    overrides : mapping, optional
        ``{"main.planform.sref_m2": 12.0}``: component, then key path. A name no component holds
        is refused and nothing runs.
    until : str
        The last stage to run, ``S0`` to ``S13``. The stages after S10 need the geometry kernel
        and are recorded as not measured in this release.
    stages : sequence of Stage, optional
        The stages to run, for a caller that replaces one; the stages of this release by default.

    Returns
    -------
    dict
        The report: ``status`` accepted (every check of the stages run was measured and none
        failed), incomplete (none failed, and ``not_run`` names each check that was not
        measured), failed (its ``stage``, ``check``, ``measured`` and ``limit`` name the first
        failed check) or refused (the input is not usable). Never raised for a failed or a
        refused run.

    Examples
    --------
    >>> import tempfile
    >>> from cfdbender.geometry.pipeline import run_geometry
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     report = run_geometry({"type": "lifting_surface"}, out=folder, until="S0")
    >>> report["status"]
    'accepted'
    """
    last = stage_index(until)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    log = RunLog(folder)
    log.start(
        config=str(config) if isinstance(config, (str, Path)) else "(mapping given in memory)",
        package_version=_package_version(),
        configuration=configuration,
        variant=variant,
        until=until,
        started=_now(),
        overrides=overrides,
    )
    planned = [
        stage
        for stage in (stages if stages is not None else default_stages(include_propeller=True))
        if stage_index(stage.row.stage) <= last
    ]
    try:
        resolved = _resolve(config, overrides)
        members = _members(resolved, configuration, variant)
    except (GeometryError, ValueError, OSError, tomllib.TOMLDecodeError) as error:
        later = [f"{s.row.stage} {s.row.name}" for s in planned[1:]]
        return _refuse(folder, log, clock, later, _refused(str(error)))
    if stages is None:
        planned = _applicable(planned, resolved, members)
    parameter = parameter_id(resolved)
    log.note(f"parameter id: {parameter}")
    context = RunContext(resolved, configuration, variant, folder, parameter, members)
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    not_run: list[str] = []
    for position, stage in enumerate(planned):
        began = _now()
        ticked = time.perf_counter()
        later = [f"{s.row.stage} {s.row.name}" for s in planned[position + 1 :]]
        try:
            outcome = stage.run(context) if stage.run else _skipped(stage.row)
        except InputValueError as error:
            # An input a stage cannot build is refused, not failed: nothing was measured.
            report = _refused(str(error), parameter=parameter, stage=stage.row.stage, check="input")
            return _refuse(folder, log, clock, later, report)
        except (GeometryError, ValueError, OSError) as error:
            outcome = StageOutcome(
                checks=[check_entry("stage.raised", "failed", 1, 0, note=str(error))]
            )
        if stage.row.stage.startswith(("S11", "S12")):
            for check in outcome.checks:
                if check["status"] == "failed":
                    check["note"] = _failure_note(stage.row.stage, check, members)
        record = stage_record(
            stage.row.stage,
            stage.row.name,
            package_version=_package_version(),
            inputs_id=parameter,
            parameter_id=parameter,
            checks=_with_table_checks(stage.row, outcome),
            findings=outcome.findings,
            warnings=outcome.warnings,
            outputs=outcome.outputs,
            started=began,
            wall_s=round(
                outcome.wall_s if outcome.wall_s is not None else time.perf_counter() - ticked, 3
            ),
        )
        write_record(folder / f"{stage.row.stage}.json", record)
        log.stage(record)
        records.append(record)
        warnings.extend(outcome.warnings)
        not_run.extend(_not_measured(record, stage.run is not None))
        failed = next((c for c in record["checks"] if c["status"] == "failed"), None)
        if failed is not None:
            report = _finish(
                "failed",
                parameter,
                folder,
                records,
                warnings=warnings,
                failure=(stage.row, failed),
                not_run=not_run,
                components=members,
            )
            log.finish(report, later, time.perf_counter() - clock)
            return report
    status = "incomplete" if not_run else "accepted"
    report = _finish(
        status,
        parameter,
        folder,
        records,
        warnings=warnings,
        failure=None,
        not_run=not_run,
        components=members,
    )
    log.finish(report, [], time.perf_counter() - clock)
    return report


def _refuse(
    folder: Path, log: RunLog, clock: float, later: list[str], report: dict[str, Any]
) -> dict[str, Any]:
    """Write the report and the end of the log of a refused run, and return the report."""
    write_record(folder / "report.json", report)
    log.finish(report, later, time.perf_counter() - clock)
    return report


def _failure_note(stage: str, check: dict[str, Any], components: Sequence[str]) -> str:
    """Build a complete diagnostic even when a stage returned only a short note."""
    location = dict(check.get("where") or {})
    if not location.get("component") and not location.get("components"):
        location["components"] = list(components)
    check["where"] = location
    comparison = ">=" if check["bound"] == "min" else "<="
    unit = check.get("unit", "")
    prefix = (
        f"{stage} {check['id']}: measured {check['measured']} {unit}, "
        f"limit {comparison} {check['limit']} {unit}; {json.dumps(location, sort_keys=True)}."
    )
    detail = str(check.get("note") or "")
    if detail.startswith(prefix):
        return detail
    if detail.startswith(f"{check['id']}: measured "):
        _, separator, tail = detail.partition(". ")
        if separator:
            detail = tail
    action = "Check the component geometry and configuration, then rebuild."
    if not detail.rstrip().endswith(action):
        detail += " " + action
    return f"{prefix} {detail.strip()}"


def _finish(
    status: str,
    parameter: str,
    folder: Path,
    records: list[dict[str, Any]],
    *,
    warnings: list[str],
    failure: tuple[StageRow, dict[str, Any]] | None,
    not_run: list[str],
    components: Sequence[str] = (),
) -> dict[str, Any]:
    row, failed = failure or (None, None)
    summary = [
        {
            "stage": r["stage"],
            "name": r["name"],
            "status": r["status"],
            "record": f"{r['stage']}.json",
        }
        for r in records
    ]
    outputs = {k: v for r in records for k, v in r["outputs"].items()}
    message = failed["note"] if failed else ""
    if failed and row and row.stage.startswith(("S11", "S12")):
        message = _failure_note(row.stage, failed, components or list(outputs.get("sources", {})))
    if failed and not message:
        location = failed["where"] or {"components": list(outputs.get("sources", {}))}
        comparison = ">=" if failed["bound"] == "min" else "<="
        message = (
            f"{row.stage if row else ''} {failed['id']}: measured {failed['measured']} "
            f"{failed['unit']}, limit {comparison} {failed['limit']} {failed['unit']}; "
            f"{json.dumps(location, sort_keys=True)}. "
            "Check the component geometry and configuration for this stage, then rebuild."
        )
    report: dict[str, Any] = geometry_report(
        status,
        parameter,
        surface_id=outputs.get("surface_id"),
        build_id=outputs.get("build_id"),
        stage=row.stage if row else None,
        check=failed["id"] if failed else None,
        message=message,
        measured=failed["measured"] if failed else None,
        limit=failed["limit"] if failed else None,
        where=failed["where"] if failed else None,
        stages=summary,
        outputs=outputs,
        warnings=warnings,
        not_run=not_run,
    )
    write_record(folder / "report.json", report)
    return report
