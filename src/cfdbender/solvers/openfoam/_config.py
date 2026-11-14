"""Validate an ESI simpleFoam case and render deterministic ASCII files."""

from __future__ import annotations

import math
import os
import shlex
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from cfdbender.core.flight import agree, row_flow
from cfdbender.core.mesh import MESH_PREFIXES
from cfdbender.core.results import References
from cfdbender.solvers._host._config import number, unknown_keys

#: The keys of the REF and PPROC files the OpenFOAM case reads; any other is skipped.
REFERENCE_KEYS = ("area", "length", "moment_origin")
POST_KEYS = ("history",)


@dataclass(frozen=True)
class Case:
    """Validated dimensional inputs for one unrotated, extruded XY mesh."""

    mesh: str
    speed: float
    nu: float
    turbulence: str
    iterations: int
    target: float
    alpha: float
    references: References
    mesh_converter: str | None = None
    time_integration: str = "steady"
    relaxation: str | None = None
    pimple: str | None = None
    lts: Mapping[str, float] = field(default_factory=dict)


#: Steady SIMPLE under-relaxation written when the setup states no relaxationFactors.
#: p 0.3 with consistent yes and U, nuTilda 0.7 is the stock simpleFoam recipe for a
#: steady RANS airfoil: low enough that a poor initial field does not diverge, high
#: enough that the residual still falls within a few hundred iterations.
DEFAULT_RELAXATION = "fields { p 0.3; } equations { U 0.7; nuTilda 0.7; }"
#: Local time stepping (pimpleFoam, ddtSchemes localEuler) defaults. maxCo is the
#: ceiling of the local Courant number; rDeltaTDampingCoeff bounds how fast the local
#: step may grow, 1 / (1 - 0.2) = 1.25 per iteration, which is the ramp from the first
#: small step up to that ceiling; rDeltaTSmoothingCoeff smooths the step field between
#: neighbouring cells.
LTS_DEFAULTS = {
    "lts_max_co": 5.0,
    "lts_smoothing": 0.1,
    "lts_damping": 0.2,
}
LTS_KEYS = (*LTS_DEFAULTS, "lts_max_delta_t")
TIME_INTEGRATIONS = ("steady", "lts")


def options(
    setup: Mapping[str, object], workspace: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Merge workspace defaults with the setup's process options."""
    table = setup.get("solver", {})
    if not isinstance(table, dict):
        raise ValueError("setup [solver] must be a table")
    result = dict(workspace or {})
    result.update(table)
    allowed = {
        "foam_bashrc",
        "launcher",
        "executable",
        "timeout_s",
        "threads",
        "processes",
        "mpi_launcher",
    }
    if unknown := set(result) - allowed:
        raise ValueError("unknown OpenFOAM solver options: " + ", ".join(sorted(unknown)))
    for key in ("foam_bashrc", "executable"):
        if key in result and (
            not isinstance(result[key], str)
            or not str(result[key])
            or any(ord(c) < 32 for c in str(result[key]))
        ):
            raise ValueError(f"solver.{key} must be a nonempty path without control characters")
    launcher = result.setdefault("launcher", [])
    if not isinstance(launcher, list) or any(
        not isinstance(item, str) or not item or any(ord(c) < 32 for c in item) for item in launcher
    ):
        raise ValueError("solver.launcher must be a list of nonempty strings")
    if type(result.setdefault("threads", 1)) is not int or result["threads"] != 1:
        raise ValueError("solver.threads must be 1 for the OpenFOAM POC")
    processes = result.setdefault("processes", 1)
    if type(processes) is not int or processes < 1:
        raise ValueError("solver.processes must be a positive integer")
    mpi = result.setdefault("mpi_launcher", ["mpirun"])
    if (
        not isinstance(mpi, list)
        or not mpi
        or any(
            not isinstance(item, str) or not item or any(ord(c) < 32 for c in item) for item in mpi
        )
    ):
        raise ValueError("solver.mpi_launcher must be a nonempty list of nonempty strings")
    if number(result.setdefault("timeout_s", 600.0), "timeout_s") <= 0:
        raise ValueError("solver.timeout_s must be positive")
    return result


def config(
    setup: Mapping[str, object],
    reference: Mapping[str, object],
    post: Mapping[str, object],
    condition: Mapping[str, float],
    *,
    geometry: str | None = None,
) -> Case:
    """Refuse unsupported inputs rather than silently ignoring case settings.

    A FLIGHT_CONDITION stating the air (MACH or TASmps with any other keys)
    gives ``speed`` (the resolved velocity) and ``nu`` (viscosity over density,
    REmi solved over the REF length); GEOMETRY gives ``mesh``. A setup value
    that disagrees with the row is refused. A BETA of zero is dropped.
    """
    if condition.get("BETA", 0.0) != 0.0:
        raise ValueError("the OpenFOAM case flies at BETA 0; remove BETA from the row")
    raw_length = reference.get("length")
    kept, state = row_flow(
        {key: value for key, value in condition.items() if key != "BETA"},
        reference_length=number(raw_length, "length") if raw_length is not None else None,
        label="FLIGHT_CONDITION",
        native_mach=False,
    )
    updates: dict[str, object] = {}
    if state is not None:
        updates.update(speed=state.velocity, nu=state.viscosity / state.density)
    if geometry is not None:
        updates["mesh"] = geometry
    setup = agree(setup, updates, "row")
    condition = {key.lower(): value for key, value in kept.items()}
    for table, allowed, label in (
        (
            setup,
            {
                "mesh",
                "mesh_converter",
                "application",
                "speed",
                "nu",
                "turbulence",
                "iterations",
                "residual_target",
                "solver",
                "time_integration",
                "relaxationFactors",
                "PIMPLE",
                *LTS_KEYS,
            },
            "setup file",
        ),
        (condition, {"alpha"}, "FLIGHT_CONDITION"),
    ):
        if unknown := set(table) - allowed:
            raise unknown_keys(label, unknown, allowed)
    if setup.get("application") != "simpleFoam":
        raise ValueError("setup file application must be simpleFoam, the OpenFOAM solver run here")
    mesh = setup.get("mesh")
    if (
        not isinstance(mesh, str)
        or not mesh.startswith(MESH_PREFIXES)
        or not mesh.endswith((".msh", ".cgns", ".ugrid"))
    ):
        raise ValueError("mesh must name inputs/mesh/<name>.msh, .cgns or .ugrid")
    converter = setup.get("mesh_converter")
    if "mesh_converter" in setup or not mesh.endswith(".msh"):
        if (
            not isinstance(converter, str)
            or not converter.strip()
            or any(ord(c) < 32 or ord(c) > 126 for c in converter)
        ):
            raise ValueError("mesh_converter must be a nonempty ASCII command template")
        try:
            for _, field, spec, conversion in string.Formatter().parse(converter):
                if field is not None and (field not in {"mesh", "case"} or spec or conversion):
                    raise ValueError("unsupported placeholder")
        except ValueError as error:
            raise ValueError(
                "mesh_converter supports only {mesh} and {case} placeholders"
            ) from error
    if setup.get("speed") is None:
        raise ValueError("FLIGHT_CONDITION needs MACH or TASmps (or speed in the setup file)")
    if setup.get("nu") is None:
        raise ValueError("FLIGHT_CONDITION needs REmi with a REF length, or nu in the setup file")
    speed = number(setup.get("speed"), "speed", "setup file")
    nu = number(setup.get("nu"), "nu", "setup file")
    target = number(setup.get("residual_target", 1e-5), "residual_target")
    if min(speed, nu, target) <= 0:
        raise ValueError("speed, nu and residual_target must be positive")
    turbulence = setup.get("turbulence")
    if turbulence not in ("SpalartAllmaras", "laminar"):
        raise ValueError("turbulence must be SpalartAllmaras or laminar")
    integration = setup.get("time_integration", "steady")
    if integration not in TIME_INTEGRATIONS:
        raise ValueError('time_integration must be "steady" or "lts"')
    if integration == "steady" and (stated := [k for k in (*LTS_KEYS, "PIMPLE") if k in setup]):
        raise ValueError(
            f'{", ".join(stated)} apply only with time_integration = "lts"; '
            "remove them or select the pseudo-transient route"
        )
    for key in ("relaxationFactors", "PIMPLE"):
        if key in setup:
            text = setup[key]
            if (
                not isinstance(text, str)
                or not text.strip()
                or any((ord(c) < 32 and c not in "\n\t") or ord(c) > 126 for c in text)
            ):
                raise ValueError(f"{key} must be a nonempty ASCII OpenFOAM dictionary body")
    lts = {key: number(setup.get(key, default), key) for key, default in LTS_DEFAULTS.items()}
    if "lts_max_delta_t" in setup:
        lts["lts_max_delta_t"] = number(setup["lts_max_delta_t"], "lts_max_delta_t")
    if lts["lts_max_co"] <= 0 or lts.get("lts_max_delta_t", 1.0) <= 0:
        raise ValueError("lts_max_co and lts_max_delta_t must be positive")
    if not 0 < lts["lts_smoothing"] < 1 or not 0 < lts["lts_damping"] <= 1:
        raise ValueError("lts_smoothing must be in (0, 1) and lts_damping in (0, 1]")
    iterations = setup.get("iterations")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if post.get("history") is not True:
        raise ValueError("post history must be true")
    origin = reference.get("moment_origin")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("moment_origin needs three finite coordinates")
    refs = References(
        number(reference.get("area"), "area", "REF file"),
        number(reference.get("length"), "length"),
        (
            number(origin[0], "moment_origin"),
            number(origin[1], "moment_origin"),
            number(origin[2], "moment_origin"),
        ),
        "OpenFOAM XY wind (Cd, Cl); ESI body -z pitch (CmPitch)",
        (
            "wind drag along alpha",
            "wind lift normal to alpha",
            "body -z pitch (ESI liftDir cross dragDir)",
        ),
    )
    return Case(
        mesh,
        speed,
        nu,
        str(turbulence),
        iterations,
        target,
        number(condition.get("alpha", 0), "alpha"),
        refs,
        cast(str | None, converter),
        str(integration),
        cast(str | None, setup.get("relaxationFactors")),
        cast(str | None, setup.get("PIMPLE")),
        lts,
    )


def _n(value: float) -> str:
    return format(value, ".15g")


def _vector(values: tuple[float, ...]) -> str:
    return "(" + " ".join(_n(v) for v in values) + ")"


def _file(name: str, body: str, kind: str = "dictionary") -> str:
    return (
        f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {kind};\n"
        f"    object {name};\n}}\n" + body + "\n"
    )


def staged_mesh(case: Case) -> str:
    """Return a fixed point-local mesh name with its validated format suffix."""
    return "mesh." + case.mesh.rsplit(".", 1)[1]


def files(case: Case, opts: Mapping[str, object]) -> dict[str, str]:
    """Render case files in fixed order, ending with the sourced Allrun script."""
    c, s = math.cos(math.radians(case.alpha)), math.sin(math.radians(case.alpha))
    drag, lift = _vector((c, s, 0)), _vector((-s, c, 0))
    velocity = _vector((case.speed * c, case.speed * s, 0))
    refs = case.references
    target = _n(case.target)
    turbulent = case.turbulence == "SpalartAllmaras"
    lts = case.time_integration == "lts"
    application = "pimpleFoam" if lts else "simpleFoam"
    relaxation = case.relaxation if case.relaxation is not None else DEFAULT_RELAXATION
    relaxation_block = (
        "relaxationFactors { " + relaxation + " }" if not lts or case.relaxation is not None else ""
    )
    nu_tilda_residual = f"nuTilda {target};" if turbulent else ""
    if lts:
        extra = (
            f"    maxDeltaT {_n(case.lts['lts_max_delta_t'])};\n"
            if "lts_max_delta_t" in case.lts
            else ""
        )
        control = case.pimple or (
            "momentumPredictor yes;\nnOuterCorrectors 1;\nnCorrectors 2;\n"
            "nNonOrthogonalCorrectors 1;\n"
            f"maxCo {_n(case.lts['lts_max_co'])};\n"
            f"rDeltaTSmoothingCoeff {_n(case.lts['lts_smoothing'])};\n"
            f"rDeltaTDampingCoeff {_n(case.lts['lts_damping'])};\n" + extra
        )
        algorithm = "PIMPLE\n{\n" + control + "\n}\n"
        # pimpleFoam solves the Final correctors under their own names.
        solvers = (
            '    "p.*" { solver GAMG; tolerance 1e-8; relTol 0.05; smoother GaussSeidel; }\n'
            '    "(U|nuTilda).*" { solver smoothSolver; smoother symGaussSeidel; '
            "tolerance 1e-9; relTol 0.1; }\n"
        )
        ddt = "localEuler"
        div_u = "Gauss linearUpwind grad(U)"
        div_nu = "Gauss upwind"
    else:
        algorithm = (
            "SIMPLE\n{\n    nNonOrthogonalCorrectors 1;\n    consistent yes;\n    pRefCell 0;\n"
            "    pRefValue 0;\n"
            f"    residualControl {{ p {target}; U {target}; {nu_tilda_residual} }}\n}}\n"
        )
        solvers = (
            "    p { solver GAMG; tolerance 1e-8; relTol 0.05; smoother GaussSeidel; }\n"
            '    "(U|nuTilda)" { solver smoothSolver; smoother symGaussSeidel; '
            "tolerance 1e-9; relTol 0.1; }\n"
        )
        ddt = "steadyState"
        div_u = "bounded Gauss linearUpwind grad(U)"
        div_nu = "bounded Gauss upwind"
    result = {
        "system/controlDict": _file(
            "controlDict",
            f"""application {application};
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {case.iterations};
deltaT 1;
writeControl timeStep;
writeInterval {case.iterations};
purgeWrite 0;
writeFormat ascii;
writePrecision 12;
runTimeModifiable false;
functions
{{
    forceCoeffs
    {{
        type forceCoeffs;
        libs (\"libforces.so\");
        patches (airfoil);
        rho rhoInf;
        rhoInf 1;
        CofR {_vector(refs.moment_origin)};
        liftDir {lift};
        dragDir {drag};
        pitchAxis (0 0 -1);
        magUInf {_n(case.speed)};
        lRef {_n(refs.length)};
        Aref {_n(refs.area)};
        writeControl timeStep;
        writeInterval 1;
        log true;
    }}
}}""",
        ),
        "system/fvSchemes": _file(
            "fvSchemes",
            f"""ddtSchemes {{ default {ddt}; }}
gradSchemes {{ default cellLimited Gauss linear 1; }}
divSchemes
{{
    default none;
    div(phi,U) {div_u};
    div(phi,nuTilda) {div_nu};
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
wallDist {{ method meshWave; }}""",
        ),
        "system/fvSolution": _file(
            "fvSolution",
            "solvers\n{\n" + solvers + "}\n" + algorithm + relaxation_block,
        ),
        "constant/transportProperties": _file(
            "transportProperties",
            (f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {_n(case.nu)};"),
        ),
        "constant/turbulenceProperties": _file(
            "turbulenceProperties",
            (
                "simulationType RAS;\n"
                "RAS { RASModel SpalartAllmaras; turbulence on; printCoeffs on; }"
                if turbulent
                else "simulationType laminar;"
            ),
        ),
    }

    def field(
        name: str, dimensions: str, internal: str, wall: str, far: str, kind: str = "volScalarField"
    ) -> str:
        return _file(
            name,
            f"""dimensions [{dimensions}];
internalField uniform {internal};
boundaryField
{{
    airfoil {{ {wall} }}
    farfield {{ {far} }}
    front {{ type empty; }}
    back {{ type empty; }}
}}""",
            kind,
        )

    result["0/U"] = field(
        "U",
        "0 1 -1 0 0 0 0",
        velocity,
        "type noSlip;",
        f"type freestream; freestreamValue uniform {velocity};",
        "volVectorField",
    )
    result["0/p"] = field(
        "p",
        "0 2 -2 0 0 0 0",
        "0",
        "type zeroGradient;",
        "type freestreamPressure; freestreamValue uniform 0;",
    )
    if turbulent:
        result["0/nut"] = field(
            "nut",
            "0 2 -1 0 0 0 0",
            "0",
            "type nutUSpaldingWallFunction; value uniform 0;",
            "type calculated; value uniform 0;",
        )
        nu_tilda = _n(3 * case.nu)
        result["0/nuTilda"] = field(
            "nuTilda",
            "0 2 -1 0 0 0 0",
            nu_tilda,
            "type fixedValue; value uniform 0;",
            f"type freestream; freestreamValue uniform {nu_tilda};",
        )
    bashrc = opts.get("foam_bashrc") or os.environ.get("CFDBENDER_OPENFOAM_BASHRC")
    source = (
        f"source {shlex.quote(str(bashrc))}\n"
        if bashrc
        else (
            'if [ -n "${CFDBENDER_OPENFOAM_BASHRC:-}" ]; then\n'
            '    source "$CFDBENDER_OPENFOAM_BASHRC"\nfi\n'
        )
    )
    processes = cast(int, opts.get("processes", 1))
    solve = f"{application} > log.simpleFoam 2>&1\n"
    if processes > 1:
        result["system/decomposeParDict"] = _file(
            "decomposeParDict", f"numberOfSubdomains {processes};\nmethod scotch;"
        )
        mpi = shlex.join(cast(list[str], opts.get("mpi_launcher", ["mpirun"])))
        solve = (
            "decomposePar > log.decomposePar 2>&1\n"
            f"{mpi} -np {processes} {application} -parallel > log.simpleFoam 2>&1\n"
            "reconstructPar -latestTime > log.reconstructPar 2>&1\n"
        )
    mesh_import = "gmshToFoam mesh.msh > log.gmshToFoam 2>&1\n"
    if case.mesh_converter is not None:
        command = case.mesh_converter.format(
            mesh=shlex.quote(staged_mesh(case)), case=shlex.quote(".")
        )
        mesh_import = "(\n" + command + "\n) > log.meshConverter 2>&1\n"
    result["Allrun"] = (
        # ESI's etc/bashrc returns nonzero from internal probes; errexit starts after it.
        "#!/usr/bin/env bash\n"
        + source
        + "set -e\nexport OMP_NUM_THREADS=1\n"
        + mesh_import
        + "{\n"
        "    foamDictionary constant/polyMesh/boundary -entry entry0/front/type -set empty\n"
        "    foamDictionary constant/polyMesh/boundary -entry entry0/back/type -set empty\n"
        "    foamDictionary constant/polyMesh/boundary -entry entry0/airfoil/type -set wall\n"
        "} > log.foamDictionary 2>&1\n" + solve
    )
    if not all(text.isascii() for text in result.values()):
        raise ValueError("OpenFOAM translated case must be ASCII, including foam_bashrc")
    return result
