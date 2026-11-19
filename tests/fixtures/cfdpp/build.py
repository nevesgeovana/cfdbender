"""Builder of every synthetic CFD++ fixture; no fixture in this folder is a file.

Each function returns bytes or text, or writes a folder, from the documented
layout of the native grid files and from our own hypotheses of the output files.
A fixture of an output file encodes a hypothesis until a real file returns from
the licensed machine; the builder then mirrors the real structure, never its
content. Nothing here is a copy of a vendor file.
"""

from __future__ import annotations

import struct
from pathlib import Path

# ---------------------------------------------------------------------------
# Native grid folders (documented layout: 4-byte integers, 8-byte doubles,
# big-endian, no record markers).
# ---------------------------------------------------------------------------

_DOUBLES_PER_NODE_TYPE = {0: 3, 1: 6, 2: 2, 3: 4, 4: 2}


def _pack_nodes(node_type: int, points: list[tuple[float, ...]], endian: str = ">") -> bytes:
    width = _DOUBLES_PER_NODE_TYPE[node_type]
    out = struct.pack(f"{endian}iii", 0, len(points), width)
    for point in points:
        padded = tuple(point) + (0.0,) * (width - len(point))
        out += struct.pack(f"{endian}i{width}d", node_type, *padded)
    return out


def _pack_cells(cells: list[tuple[int, tuple[int, ...]]], endian: str = ">") -> bytes:
    widest = max(1 + len(vertices) for _, vertices in cells)
    out = struct.pack(f"{endian}iii", 0, len(cells), widest)
    for cell_type, vertices in cells:
        out += struct.pack(f"{endian}i{len(vertices)}i", cell_type, *vertices)
    return out


def _pack_faces(faces: list[tuple[int, tuple[int, ...]]], endian: str = ">") -> bytes:
    out = struct.pack(f"{endian}ii", 0, len(faces))
    for selector, vertices in faces:
        out += struct.pack(
            f"{endian}iii{len(vertices)}i", selector, selector, len(vertices), *vertices
        )
    return out


def bc_text(names: list[str], trailing_blank_lines: int = 0) -> str:
    """Return a family-name file: a comment, a header line, a name line and four integers each."""
    lines = ["# synthetic family names", "seq# type modi info"]
    for number, name in enumerate(names, start=1):
        lines += [f"#{name}", f"{number} 0 0 0"]
    return "\n".join(lines) + "\n" + "\n" * trailing_blank_lines


def grid2d_files() -> dict[str, bytes]:
    """A 2x2 quadrilateral grid: selector 1 is the bottom edge, selector 2 the rest."""
    points = [(float(i), float(j)) for j in range(3) for i in range(3)]
    cells = [
        (3, (j * 3 + i, j * 3 + i + 1, (j + 1) * 3 + i + 1, (j + 1) * 3 + i))
        for j in range(2)
        for i in range(2)
    ]
    bottom = [(1, (i, i + 1)) for i in range(2)]
    top = [(2, (6 + i, 6 + i + 1)) for i in range(2)]
    left = [(2, (j * 3, (j + 1) * 3)) for j in range(2)]
    right = [(2, (j * 3 + 2, (j + 1) * 3 + 2)) for j in range(2)]
    return {
        "nodesin.bin": _pack_nodes(2, points),
        "cellsin.bin": _pack_cells(cells),
        "exbcsin.bin": _pack_faces(bottom + top + left + right),
    }


def grid3d_files() -> dict[str, bytes]:
    """One hexahedron: selector 1 bottom, selector 2 four sides, selector 3 top."""
    points = [(float(x), float(y), float(z)) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
    cells = [(0, (0, 1, 3, 2, 4, 5, 7, 6))]
    faces = [
        (1, (0, 1, 3, 2)),
        (2, (0, 1, 5, 4)),
        (2, (1, 3, 7, 5)),
        (2, (3, 2, 6, 7)),
        (2, (2, 0, 4, 6)),
        (3, (4, 5, 7, 6)),
    ]
    return {
        "nodesin.bin": _pack_nodes(0, points),
        "cellsin.bin": _pack_cells(cells),
        "exbcsin.bin": _pack_faces(faces),
    }


def write_grid(root: Path, files: dict[str, bytes], names: list[str] | None) -> Path:
    """Write a grid folder; names None leaves out the family-name file."""
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    if names is not None:
        (root / "mcfd.bc").write_text(bc_text(names), encoding="ascii", newline="\r\n")
    return root


def write_grid2d(root: Path) -> Path:
    return write_grid(root, grid2d_files(), ["airfoil", "farfield"])


def write_grid3d(root: Path) -> Path:
    return write_grid(root, grid3d_files(), ["wing", "farfield", "sym"])


GRID_BAD_CASES = ("truncated", "little_endian", "multi_group", "no_bc", "bad_node_type")


def write_grid_bad(root: Path, case: str) -> Path:
    """Write a defective grid folder; case is one of GRID_BAD_CASES."""
    files = grid2d_files()
    names: list[str] | None = ["airfoil", "farfield"]
    if case == "truncated":
        files["nodesin.bin"] = files["nodesin.bin"][:8]
    elif case == "little_endian":
        points = [(float(i), float(j)) for j in range(3) for i in range(3)]
        files["nodesin.bin"] = _pack_nodes(2, points, "<")
    elif case == "no_bc":
        names = None
    elif case == "bad_node_type":
        files["nodesin.bin"] = _pack_nodes(4, [(0.0, 0.0)] * 9)
    elif case != "multi_group":
        raise ValueError(case)
    folder = write_grid(root, files, names)
    if case == "multi_group":
        (folder / "cgrpsin.bin.1").write_bytes(struct.pack(">4i", 1, 1, 2, 2))
    return folder


# ---------------------------------------------------------------------------
# Output files: residual file, force file and log. Every layout below is a
# hypothesis of the adapter (named in _history.py), not a copy of a solver file.
# ---------------------------------------------------------------------------

KNOWN = {
    "dynamic_pressure": 500.0,
    "area": 2.0,
    "length": 1.0,
    "origin": (0.25, 0.0, 0.0),
    "CL": 0.5,
    "CD": 0.01,
    "CM": -0.02,
}
FORCE_ROWS = ("force_x", "force_y", "force_z", "moment_x", "moment_y", "moment_z")
ALL_ROWS = ("energy_flux", "mass_flux", *FORCE_ROWS)
Rows = dict[str, tuple[float, float]]  # row name to (inviscid, viscous)


def rows(force_inv, force_vis, moment_inv, moment_vis) -> Rows:
    """Rows of one dimensional table; the total column is inviscid plus viscous."""
    values = (*force_inv, *moment_inv), (*force_vis, *moment_vis)
    result: Rows = {"energy_flux": (3.0, 1.0), "mass_flux": (0.0, 0.0)}
    for index, name in enumerate(FORCE_ROWS):
        result[name] = (values[0][index], values[1][index])
    return result


def known_selectors() -> dict[int, Rows]:
    """Two selectors whose sums, about the coordinate origin, give the known coefficients.

    With q = 500, A = 2, L = 1 and the moment centre at x = 0.25, the totals are the
    force (10, 0, 500) and the origin moment (0, -145, 0): CD 0.01, CL 0.5, CM -0.02.
    """
    return {
        1: rows((6, 0, 450), (2, 0, 0), (0, -131, 0), (0, 1, 0)),
        2: rows((1, 0, 50), (1, 0, 0), (0, -15, 0), (0, 0, 0)),
    }


def _table(table_rows: Rows, scale: float) -> list[str]:
    lines = ["total inviscid viscous"]
    for name in ALL_ROWS:
        inv, vis = table_rows[name]
        lines.append(f"{name} {(inv + vis) * scale!r} {inv * scale!r} {vis * scale!r}")
    return lines


def info1_text(steps: dict[int, dict[int, Rows]], scales: tuple[float, ...] = (1.0,) * 10) -> str:
    """Force file text: one scales line, then per step and selector two tables and the areas."""
    lines = ["scales " + " ".join(repr(s) for s in scales)]
    for step, selectors in steps.items():
        lines.append(f"step {step} {float(step)!r} 0.0")
        for selector, table_rows in selectors.items():
            lines.append(f"selector {selector} faces 2")
            lines.append("table nondimensional")
            lines += _table(table_rows, 1e-3)
            lines.append("table dimensional")
            lines += _table(table_rows, 1.0)
            lines.append("areas 1.0 0.0 0.0 1.0")
            lines.append("areamoments 0.0 0.0 0.0")
    return "\n".join(lines) + "\n"


def info1_known_text(first: int = 1, last: int = 3) -> str:
    """Steps first..last, all with the known forces."""
    return info1_text({step: known_selectors() for step in range(first, last + 1)})


def info1_cauchy_text(count: int = 400) -> str:
    """Forces that settle: CL and CD approach their known values, halving the gap each step."""
    qa = KNOWN["dynamic_pressure"] * KNOWN["area"]
    steps = {}
    for step in range(1, count + 1):
        gap = 0.5**step
        cd, cl = KNOWN["CD"] + 0.003 * gap, KNOWN["CL"] + 0.3 * gap
        force = (cd * qa, 0.0, cl * qa)
        moment_y = KNOWN["CM"] * qa * KNOWN["length"] - KNOWN["origin"][0] * force[2]
        steps[step] = {1: rows(force, (0, 0, 0), (0, moment_y, 0), (0, 0, 0))}
    return info1_text(steps)


def info0_text(residuals: list[float], first_step: int = 1) -> str:
    """Residual file text: one line per step, eight columns."""
    lines = []
    for offset, residual in enumerate(residuals):
        step = first_step + offset
        lines.append(f"{step} 0.0 1.0 {residual!r} {residual * 10!r} 1e15 5.0 100.0")
    return "\n".join(lines) + "\n"


def info0_converged_text() -> str:
    """121 steps falling one decade every ten steps: twelve orders."""
    return info0_text([10.0 ** (-0.1 * n) for n in range(121)])


def info0_limit_text() -> str:
    """Two decades down, then a plateau."""
    return info0_text([10.0 ** (-0.1 * min(n, 20)) for n in range(40)])


def info0_slow_text(count: int = 400) -> str:
    """Three decades over the first 150 steps, then still falling slowly."""
    return info0_text([10.0 ** (-min(n, 150) / 50.0) for n in range(count)])


def info0_nan_text() -> str:
    """The third row holds a non-finite residual."""
    values = [1.0, 0.5, float("nan"), 0.1]
    return info0_text(values)


LOG_FAIL_KINDS = ("mesh_match", "mesh_match_parallel", "negative_volume", "cfl_floor")


def log_text(version: str | None = "22.5.U3", redo_events: int = 0, fail: str | None = None) -> str:
    """A log with our own banner, a step line per step and an optional failure line."""
    lines = ["synthetic solver log"]
    if version is not None:
        lines.append(f"version: {version}")
    lines += ["step 1 residual 1.0", "step 2 residual 0.5"]
    lines += [f"redoing step {n} after an unphysical state" for n in range(3, 3 + redo_events)]
    fragments = {
        "mesh_match": "unable to find a face match for a boundary polygon",
        "mesh_match_parallel": "boundary face did not match with mesh on any cpu",
        "negative_volume": "negative volume cell found",
        "cfl_floor": "local cfl fell below the floor; terminating",
    }
    if fail is not None:
        lines.append(fragments[fail])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stand-in executables. The builder writes them as scripts into a temporary
# folder: a solver that parses the generated input with the public parser and
# writes the output files above by a scenario, and a tool that partitions or
# asks the solver to stop. Neither imitates the real solver's behaviour beyond
# the files the adapter reads.
# ---------------------------------------------------------------------------

#: The raw keyword (in the schema's raw section) that selects the stand-in's scenario.
SCENARIO_KEYWORD = "zscenario"
SCENARIOS = ("converge", "limit", "fail", "nan", "exit1", "hang", "none")
STOP_REQUEST = "stop.request"

_SOLVER = '''\
"""Stand-in solver written by the CFD++ fixture builder; see build.py."""
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, {src!r})
spec = importlib.util.spec_from_file_location("cfdpp_fixture_build", {build!r})
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
from cfdbender.solvers.cfdpp._schema import schema_for
from cfdbender.solvers.cfdpp._schema._model import parse

VERSION = {version!r}
schema = schema_for(VERSION, Path({schema!r}))
model = parse(Path("mcfd.inp").read_text(encoding="ascii"), schema)
args = sys.argv[1:]
ncpus = int(args[args.index("-n") + 1]) if "-n" in args else 1
pair = model.pair({scenario!r})
scenario = str(pair.values[0]) if pair is not None else "converge"
steps = int(model.pair({steps!r}).values[0])
restart = model.pair({restart!r})
for name in ("nodesin.bin", "cellsin.bin", "exbcsin.bin"):
    if not Path(name).is_file():
        print("grid file missing: " + name)
        sys.exit(2)
if ncpus > 1 and not Path("mcpusin.bin." + str(ncpus)).is_file():
    print("partition file missing for " + str(ncpus) + " processes")
    sys.exit(3)
if restart is not None and restart.values[0] == 1 and not Path("cdepsout.bin").is_file():
    print("restart solution missing")
    sys.exit(4)


def write(count, plateau=False):
    residuals = [10.0 ** (-0.1 * (min(n, 20) if plateau else n)) for n in range(count)]
    Path("mcfd.info0").write_text(build.info0_text(residuals), encoding="ascii")
    Path("mcfd.info1").write_text(build.info1_known_text(1, count), encoding="ascii")
    Path("cdepsout.bin").write_bytes(b"synthetic solution")


fail = None
code = 0
if scenario == "converge":
    write(min(steps, 121))
elif scenario == "limit":
    write(steps, plateau=True)
elif scenario == "fail":
    write(min(steps, 121))
    fail = "negative_volume"
elif scenario == "nan":
    Path("mcfd.info0").write_text(build.info0_nan_text(), encoding="ascii")
elif scenario == "exit1":
    write(min(steps, 121))
    code = 1
elif scenario == "hang":
    deadline = time.monotonic() + 60
    while not Path({stop!r}).is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    write(10, plateau=True)
sys.stdout.write(build.log_text(version=VERSION, fail=fail))
sys.exit(code)
'''

_TOOL = '''\
"""Stand-in tool written by the CFD++ fixture builder: partition, or ask for a stop."""
import sys
from pathlib import Path

args = sys.argv[1:]
if args and args[0] == "partition":
    if not Path("mcfd.inp").is_file():
        print("no input to partition")
        sys.exit(2)
    Path("mcpusin.bin." + args[1]).write_bytes(bytes(16))
    print("partitioned for " + args[1])
else:
    Path({stop!r}).write_text("stop", encoding="ascii")
'''


def write_standins(folder: Path, schema: Path, version: str, stop_tool: str) -> dict[str, Path]:
    """Write the stand-in solver, the tool and the stop tool; return their paths by role."""
    import cfdbender

    folder.mkdir(parents=True, exist_ok=True)
    solver = folder / "fake_cfdpp.py"
    solver.write_text(
        _SOLVER.format(
            src=str(Path(cfdbender.__file__).resolve().parents[1]),
            build=str(Path(__file__).resolve()),
            version=version,
            schema=str(schema),
            scenario=SCENARIO_KEYWORD,
            steps="tsteps",
            restart="trestart",
            stop=STOP_REQUEST,
        ),
        encoding="ascii",
    )
    tool = folder / "fake_tool.py"
    tool.write_text(_TOOL.format(stop=STOP_REQUEST), encoding="ascii")
    tools_dir = folder / "tools"
    tools_dir.mkdir(exist_ok=True)
    stop = tools_dir / f"{stop_tool}.py"
    stop.write_text(_TOOL.format(stop=STOP_REQUEST), encoding="ascii")
    return {"solver": solver, "tool": tool, "tools_dir": tools_dir}
