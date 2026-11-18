"""The guide `cfd-workspace init` writes to inputs/INPUTS.md: what each input file holds.

Written once per workspace, for the solver the workspace is bound to. The files a user
writes are the matrix, one setup, one reference and one post-processing file; the guide
shows a working example of each, so nothing has to be copied from the test suite.
"""

from __future__ import annotations

#: The CFD solvers whose adapters read a numerics file (inputs/numerics/n<id>.toml).
NUMERICS_SOLVERS = frozenset({"su2", "openfoam", "pyfr"})

HEADER = "POL | RUN | FLIGHT_CONDITION | SWEEP_VALUES | GEOMETRY | SETUP | REF | PPROC | WALLTIME"
#: The header and the cells of the CFD solvers, whose setups may name a numerics file.
HEADER_NUMERICS = (
    "POL | RUN | FLIGHT_CONDITION | SWEEP_VALUES | GEOMETRY | SETUP | NUMERICS | REF | PPROC "
    "| WALLTIME"
)

#: Per solver: the example row's FLIGHT_CONDITION, SWEEP_VALUES and GEOMETRY cells, then the
#: setup (s001.toml), reference (r001.toml) and post (p001.toml) examples.
_EXAMPLES: dict[str, tuple[str, str, str, str, str, str]] = {
    "mock": (
        "MACH:0.2, ALPHA:sweep",
        "0,2",
        "model.dat",
        '[mock]\noutcome = "converged"\niterations = 10\n',
        'area = 1.5\nlength = 1.0\nmoment_origin = [0, 0, 0]\naxis_system = "body"\n'
        'axis_directions = ["aft", "right", "up"]\n',
        "history = true\n",
    ),
    "su2": (
        "MACH:0.5, ALTFT:10000, ALPHA:sweep, BETA:0",
        "0,2",
        "model.su2",
        'iterations = 40\nSOLVER = "EULER"\n',
        "area = 1.0\nlength = 1.0\nmoment_origin = [0, 0, 0]\n",
        "history = true\n",
    ),
    "openfoam": (
        "TASmps:10, REmi:0.6, ALPHA:sweep, BETA:0",
        "0,4",
        "model.msh",
        'application = "simpleFoam"\nturbulence = "laminar"\niterations = 400\n',
        "area = 0.1\nlength = 1.0\nmoment_origin = [0, 0, 0]\n",
        "history = true\n",
    ),
    "pyfr": (
        "MACH:0.2, REmi:0.01, ALPHA:sweep, BETA:0",
        "4",
        "model.msh",
        "order = 3\ntend = 3\nwindow_start = 0\n",
        "",
        "history = true\n",
    ),
    "xfoil": (
        "MACH:0, REmi:1, ALPHA:sweep",
        "0,2",
        "-",
        'airfoil = "NACA 0012"\n',
        "",
        "",
    ),
    "qprop": (
        "TASmps:20, RPM:6000, BETA07:30",
        "-",
        "blade.obj",
        'n_blades = 3\nn_sections = 5\nspacing = "cosine"\ncut = "plane"\nradius = 0.2\n',
        "",
        "",
    ),
}

_MESH_NOTE = {
    "mock": "any file; the mock only records its digest",
    "su2": "an SU2 mesh (.su2)",
    "openfoam": "a Gmsh .msh (format 2.2) the case is built around",
    "pyfr": "a Gmsh .msh (format 4.1)",
    "xfoil": "none: XFOIL reads the airfoil named in the setup; write - in the cell",
    "qprop": "the blade OBJ (or name it in the setup as blade = ...)",
}


def guide_text(solver: str) -> str:
    """Return inputs/INPUTS.md for a solver, or the generic guide for an unlisted one.

    Parameters
    ----------
    solver : str
        Registered solver name the workspace is bound to.
    """
    example = _EXAMPLES.get(solver)
    lines = [
        f"# Inputs of this workspace (solver: {solver})",
        "",
        "Everything the campaign reads lives under `inputs/`. A matrix row names the files",
        "by id: `SETUP 001` reads `inputs/setups/s001.toml`, `REF 001` reads",
        "`inputs/references/r001.toml`, `PPROC 001` reads `inputs/pproc/p001.toml`, and",
        "GEOMETRY names a file in `inputs/mesh/`.",
        "",
        "| Folder | Holds |",
        "|---|---|",
        "| `inputs/matrices/` | `<name>.fs`: one row per simulation, columns separated by `\\|` |",
        "| `inputs/setups/` | `s<id>.toml`: the physical setup of the solver |",
        *(
            ["| `inputs/numerics/` | `n<id>.toml`: the numerical setup, named by NUMERICS |"]
            if solver in NUMERICS_SOLVERS
            else []
        ),
        "| `inputs/references/` | `r<id>.toml`: reference area, length, moment origin |",
        "| `inputs/pproc/` | `p<id>.toml`: what to extract after the run |",
        "| `inputs/mesh/` | meshes and geometry files |",
        "| `inputs/hpc/` | cluster profiles, only to submit instead of running here |",
        "",
    ]
    if example is not None:
        condition, sweep, geometry, setup, reference, post = example
        reference = reference or "# this solver reads no reference keys; the file may be empty\n"
        post = post or "# this solver reads no post-processing keys; the file may be empty\n"
        lines += [
            "## A first matrix, `inputs/matrices/first.fs`",
            "",
            "```text",
            HEADER_NUMERICS if solver in NUMERICS_SOLVERS else HEADER,
            f"7001 | 1 | {condition} | {sweep} | {geometry} | 001 | "
            f"{'- | ' if solver in NUMERICS_SOLVERS else ''}001 | 001 | 2h",
            "```",
            "",
            f"GEOMETRY: {_MESH_NOTE[solver]}.",
            "",
            "FLIGHT_CONDITION is `KEY:value` pairs; the key that varies carries the word",
            "`sweep` and SWEEP_VALUES lists its values. A key this package does not know is",
            "refused with the accepted list.",
            "",
            "## `inputs/setups/s001.toml`",
            "",
            "```toml",
            setup.rstrip(),
            "```",
        ]
        if solver in NUMERICS_SOLVERS:
            lines += [
                "",
                "## `inputs/numerics/n001.toml` (optional)",
                "",
                "The NUMERICS column names a file of numerical keys (schemes, limiters, CFL,",
                "iteration counts, convergence criteria) that several setups can share. `-` or an",
                "empty cell means the setup file carries the numerics. The merge order, lowest to",
                "highest, is: `init.toml` defaults, the numerics file, the setup file, the row's",
                "VAR_NAMES_VALUES. A key stated in both files takes the setup file's value, with",
                "one warning. The file may not hold a `[solver]` table or the mesh. To split",
                "existing setups, run `python -m cfdbender.migrate_setup_numerics WORKSPACE`.",
            ]
        lines += ["", "## `inputs/references/r001.toml`", "", "```toml", reference.rstrip(), "```"]
        lines += ["", "## `inputs/pproc/p001.toml`", "", "```toml", post.rstrip(), "```", ""]
    lines += [
        "## Then",
        "",
        "```text",
        "cfd-matrix plan first --workspace .",
        "cfd-matrix run first --workspace . --local",
        "cfd-matrix collect first --workspace .",
        "```",
        "",
    ]
    return "\n".join(lines)
