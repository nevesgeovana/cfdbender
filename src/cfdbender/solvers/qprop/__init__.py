"""Plan QPROP 1.22 isolated-propeller points from a blade OBJ or a polars-2D set.

Initialize with ``cfd-workspace init --solver qprop``. The s file names the blade:
``blade = "inputs/geometry/<name>.obj"`` (sliced at ``n_sections`` radial stations,
``spacing`` cosine or uniform, ``cut`` cylinder or plane) or
``blade = "inputs/polars-2D/<set>"`` (a ``blade.csv`` with columns r, chord, beta and
optional airfoil and polar file names inside the set). Also ``n_blades`` (required),
``radius`` (default: the blade tip), ``scale`` (model units to metres),
``solver_2d = "xfoil"`` (the only accepted value), ``axis``, ``hub_center`` and
``rotation`` for the slicing convention, and the XFOIL ``alpha_start``, ``alpha_end``,
``alpha_step``, ``ncrit`` and ``iterations``. Stations without a polar are produced
by XFOIL at their local Reynolds and Mach number; a failed section is retried with
alpha -4 to 10 in steps of 0.5, then reuses the nearest good station, recorded in
``sections.json`` of the point.

The matrix FLIGHT_CONDITION carries V (m/s) and one of RPM or J, optionally BETA07
(target pitch in degrees at 0.7 R) or DBETA (offset in degrees), RHO, MU, A and
N_SECTIONS. Any of them may be the sweep key. Defaults are sea-level standard
density, viscosity and speed of sound.

The [solver] table accepts ``executable`` (QPROP; default CFDBENDER_QPROP_EXE, then
``qprop`` on PATH), ``xfoil_executable``, ``launcher`` (for example
``["wsl", "-d", "Ubuntu", "--"]``), ``timeout_s`` and ``ram_estimate_gib``. The XFOIL
sections run through the XFOIL adapter's execution path and shared host guards.

Collection returns J, CT and CP in the standard T/(rho n^2 D^4) and P/(rho n^3 D^5)
form, CT_QPROP and CP_QPROP as QPROP prints them (tip-speed and disk-area based),
EFFICIENCY, and T, Q and P. The axis is the rotation axis; Q and P are listed as
moments only to fit the neutral contract.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.flight import agree
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import RowInputs, RowRequirements, workspace_path
from cfdbender.core.results import (
    Classification,
    Coefficient,
    ExecutionResult,
    Iteration,
    NeutralResult,
    References,
)
from cfdbender.core.settings import _validate, read_init
from cfdbender.solvers._host import _ram
from cfdbender.solvers._host._config import number
from cfdbender.solvers.qprop import _output
from cfdbender.solvers.qprop._config import Operating, Setup, operating, options, setup
from cfdbender.solvers.qprop._execute import prepare, record, run_qprop, section_fits
from cfdbender.solvers.qprop._prop import Station, render_prop, render_qcon
from cfdbender.solvers.qprop._sections import (
    Blade,
    Planned,
    beta_offset,
    deck,
    plan_sections,
    resolve,
    rotation_rate,
)

SOLVER_NAME = "qprop"
STAGED_CONFIG = "propeller.prop"
EXECUTABLE_KEYS = {"path": "executable", "launcher": "launcher"}
JOB_OUTPUTS = ("stdout.log",)
_OUTPUTS = (
    "propeller.prop",
    "qcon.def",
    "stdout.log",
    "stderr.log",
    "execution.json",
    "sections.json",
)
_MOMENTS = {"Q", "P", "CP", "CP_QPROP"}


def _with_geometry(values: Mapping[str, object], geometry: str | None) -> Mapping[str, object]:
    """Use the row's GEOMETRY as the blade OBJ; a setup blade must agree with it."""
    return values if geometry is None else agree(values, {"blade": geometry}, "GEOMETRY")


def _context(
    point: AdapterPoint,
) -> tuple[Mapping[str, object], Setup, Operating, Blade, float, tuple[Planned, ...]]:
    declaration = read_init(point.root, available_solvers=("qprop",))
    opts = options(point.setup, declaration.solver_options)
    settings = setup(_with_geometry(point.setup, point.geometry))
    op = operating(point.point.condition)
    blade = resolve(
        settings,
        op.n_sections or settings.n_sections,
        lambda name: workspace_path(point.root, name).read_bytes(),
    )
    rpm = rotation_rate(op, blade.radius)
    return opts, settings, op, blade, rpm, plan_sections(blade, op, rpm)


def validate_row(row: MatrixRow, workspace: RowInputs) -> RowRequirements:
    """Validate the setup and every point's conditions, tracking the blade inputs.

    Parameters
    ----------
    row : MatrixRow
        Steady matrix row with scalar conditions or one sweep.
    workspace : RowInputs
        Tracked setup, reference, post and further input reader.
    """
    if row.workflow != "steady":
        raise ValueError("QPROP WORKFLOW must be steady")
    declaration = _validate(
        workspace.read_toml("init.toml"), workspace.root / "init.toml", ("qprop",)
    )
    options(workspace.setup, declaration.solver_options)
    settings = setup(_with_geometry(workspace.setup, workspace.geometry))
    conditions = [dict(row.condition)]
    if row.sweep_key:
        conditions = [dict(row.condition, **{row.sweep_key: value}) for value in row.sweep_values]
    for condition in conditions:
        op = operating(condition)
        blade = resolve(settings, op.n_sections or settings.n_sections, workspace.read_bytes)
        plan_sections(blade, op, rotation_rate(op, blade.radius))
    return RowRequirements(
        (
            "slice or read the blade",
            "run XFOIL for missing polars",
            "write QPROP files",
            "run QPROP",
        ),
        tuple(f"sims/sim_{{pol}}/datapoints/{{point}}/{name}" for name in _OUTPUTS),
    )


def translate(point: AdapterPoint) -> str:
    """Return the section plan, QPROP command, flight file and XFOIL decks as ASCII.

    The propeller file needs the XFOIL fits and is written by execute. Files are
    separated by ``// FILE: <name>`` lines in a fixed order.

    Parameters
    ----------
    point : AdapterPoint
        Digest-checked point.
    """
    point.verify()
    _, settings, op, _, rpm, planned = _context(point)
    sections = ["# index radius chord beta reynolds mach source"]
    sections += [
        f"{p.index} {p.spec.radius:.8f} {p.spec.chord:.8f} {p.spec.beta:.4f} "
        f"{p.reynolds:.0f} {p.mach:.4f} {p.source}"
        for p in planned
    ]
    files = {
        "sections.plan": "\n".join(sections) + "\n",
        "command": f"qprop propeller.prop none {op.speed:.10g} {rpm:.10g}\n",
        "qcon.def": render_qcon(op.rho, op.mu, op.a),
    }
    for p in planned:
        if p.source == "xfoil":
            assert p.spec.shape is not None
            files[f"xfoil/{p.name}/airfoil.dat"] = p.spec.shape
            files[f"xfoil/{p.name}/xfoil.in"] = deck(settings, p)
    return "".join(f"// FILE: {name}\n{body}" for name, body in files.items())


def execute(point: AdapterPoint, *, check_ram: Callable[[], None]) -> ExecutionResult:
    """Make missing polars with XFOIL, write the QPROP files and run QPROP once.

    Parameters
    ----------
    point : AdapterPoint
        Planned point with no existing output folder.
    check_ram : callable
        Coordinator guard, checked before writes and throughout every process.
    """
    folder, opts, op, rpm = _stage(point, check_ram)
    run_qprop(point, folder, op, rpm, opts, check_ram)
    return _read(folder, point)[0]


def stage(point: AdapterPoint) -> None:
    """Stage propeller.prop, qcon.def and sections.json for a cluster job.

    Missing section polars are made here, by the XFOIL step a local run uses,
    so the job only runs qprop propeller.prop none <speed> <rpm> in the point
    folder, with the point's speed and the rpm recorded in sections.json, and
    sends its standard output to stdout.log (the profile's {stdout_log}).

    Parameters
    ----------
    point : AdapterPoint
        Planned point whose folder does not exist yet.
    """
    _stage(point, lambda: None)


def _stage(
    point: AdapterPoint, check_ram: Callable[[], None]
) -> tuple[Path, Mapping[str, object], Operating, float]:
    point.verify()
    opts, settings, op, blade, rpm, planned = _context(point)
    folder = prepare(point)
    check_ram()
    _ram.require_ram(number(opts["ram_estimate_gib"], "ram_estimate_gib"), solver="QPROP")
    folder.mkdir(parents=True, exist_ok=False)
    fits = section_fits(point, folder, settings, planned, opts, check_ram)
    add = beta_offset(blade, op)
    (folder / "sections.json").write_text(
        record(blade, planned, fits, settings.blades, add, rpm), encoding="ascii"
    )
    stations = [
        Station(p.spec.radius, p.spec.chord, p.spec.beta, e["fit"], p.reynolds)  # type: ignore[arg-type]
        for p, e in zip(planned, fits, strict=True)
    ]
    (folder / "propeller.prop").write_text(
        render_prop(
            settings.title,
            settings.blades,
            blade.radius,
            stations,
            beta_add=add,
            re_exponent=settings.re_exponent,
        ),
        encoding="ascii",
    )
    (folder / "qcon.def").write_text(render_qcon(op.rho, op.mu, op.a), encoding="ascii")
    return folder, opts, op, rpm


def classify(result: ExecutionResult) -> Classification:
    """Require exit zero and a parsed operating row; QPROP has no iteration limit class.

    Parameters
    ----------
    result : ExecutionResult
        Process status and the parsed-table acceptance marker.
    """
    if result.exit_code == 0 and result.history and result.history[-1].converged:
        return "converged"
    return "failed"


def _read(
    folder: Path, point: AdapterPoint
) -> tuple[ExecutionResult, dict[str, Coefficient], References]:
    settings = setup(point.setup)
    axis: Literal["x", "y", "z"] = settings.axis  # type: ignore[assignment]
    refs_radius = 1.0
    try:
        # A cluster job leaves no execution.json; the operating table decides.
        execution = folder / "execution.json"
        meta = json.loads(execution.read_bytes()) if execution.is_file() else {"exit_code": 0}
        text = (folder / "stdout.log").read_text(encoding="utf-8", errors="replace")
        refs_radius = float(json.loads((folder / "sections.json").read_bytes())["radius"])
    except (OSError, ValueError, KeyError):
        meta, text = {"exit_code": 1}, ""
    row = _output.parse_table(text)
    failed = "onvergence failed" in text or "not converge" in text.lower()
    history = (Iteration(1, 0.0, True),) if row and not failed else ()
    # The marker is a protocol sentinel: QPROP prints no iteration count or residual.
    coefficients: dict[str, Coefficient] = {}
    if history:
        values = _output.coefficients(
            row, rho=operating(point.point.condition).rho, radius=refs_radius
        )
        coefficients = {
            name: Coefficient(value, axis, "moment" if name in _MOMENTS else "force")
            for name, value in values.items()
        }
    references = References(
        3.141592653589793 * refs_radius**2,
        2.0 * refs_radius,
        tuple(c * settings.scale for c in settings.hub_center),  # type: ignore[arg-type]
        f"propeller axes: thrust along +{axis}, QPROP isolated propeller",
        tuple(  # type: ignore[arg-type]
            f"positive {c}: thrust and rotation axis" if c == axis else f"positive {c}"
            for c in "xyz"
        ),
    )
    result = ExecutionResult(int(meta["exit_code"]), history, 1, _output.version(text))
    return result, coefficients, references


def collect(point: AdapterPoint) -> NeutralResult:
    """Read QPROP's operating table by header names into neutral coefficients.

    Parameters
    ----------
    point : AdapterPoint
        Recorded point; collection reads outputs without executing or writing.
    """
    point.verify()
    result, coefficients, references = _read(
        workspace_path(point.root, point.point.directory), point
    )
    return NeutralResult(
        coefficients, references, result.history, classify(result), result.solver_version
    )
