# Install

This page installs cfdbender and tells you what each solver needs on the
machine before a campaign can run.

## cfdbender

cfdbender needs Python 3.12 or later. `pip` installs the libraries it uses with
it: numpy, scipy, trimesh, rtree, pydantic, matplotlib, VTK and gmsh. Surface
remeshing uses rtree to find nearby source faces when projecting new nodes.

```console
pip install cfdbender
cfd-workspace init --help
```

The help text ends with the list of solvers. When it prints, the four programs
`cfd-workspace`, `cfd-matrix`, `cfd-geom` and `cfd-mesh` are installed. You can
now follow [First campaign](getting-started.md), which needs no solver.

### The geometry kernel and its license

`cfd-geom` and `cfd-mesh` build and mesh surfaces with gmsh and its
OpenCASCADE kernel. gmsh is distributed under the GNU General Public License,
version 2 or later, with an exception that permits linking it with other
programs. It is installed from its own distribution, as a separate package, and
cfdbender imports it from one module only, `cfdbender/mesh/_gmsh.py`, when a
command that needs the kernel runs. cfdbender itself stays under the MIT
license. Measured with gmsh 4.15.2.

On Linux the gmsh library also needs the system packages `libglu1-mesa
libxcursor1 libxft2 libxinerama1`. Without them a command that needs the kernel
stops and names the library that did not load; the geometry stages that need no
kernel still run.

### Runtime dependency licenses

The [dependency license table](dependency-licences.md) lists every direct runtime
dependency, including gmsh's GPL and linking exception. The offline kit retains
the published wheels with their license files and bundled notices.

FlightStream support is an optional extra:

```console
pip install cfdbender[flightstream]
```

## The solvers

cfdbender does not install a solver. Install the one you use by its own
instructions; the page of each solver gives the steps that were used with
cfdbender. cfdbender then finds the solver in this order: the `[solver]` table
of the setup file, the environment variable, the program name on the `PATH`.

| Solver | Needs on the machine | Key in the setup `[solver]` table | Environment variable | Name looked for on `PATH` |
|---|---|---|---|---|
| [CFD++](cfdpp-workspaces.md) | CFD++ 22.5.U3 and its tools | none; the build entry of `inputs/executables.toml` | none | none |
| [SU2](su2-workspaces.md) | SU2, as the program `SU2_CFD` or as its Python wrapper | `executable`; `pysu2_path` for the wrapper | `CFDBENDER_SU2_EXE`; `CFDBENDER_PYSU2_PATH` | `SU2_CFD` |
| [OpenFOAM](openfoam-workspaces.md) | OpenFOAM and its environment file | `foam_bashrc` | `CFDBENDER_OPENFOAM_BASHRC` | none |
| [PyFR](pyfr-workspaces.md) | PyFR | `executable` | none | `pyfr` |
| [XFOIL](xfoil-workspaces.md) | XFOIL | `executable` | `CFDBENDER_XFOIL_EXE` | `xfoil` |
| [QPROP](qprop-workspaces.md) | QPROP, and XFOIL for the blade section polars | `executable`; `xfoil_executable` | `CFDBENDER_QPROP_EXE`; `CFDBENDER_XFOIL_EXE` | `qprop`; `xfoil` |
| [FlightStream](flightstream-workspaces.md) | see its page | | | |

A solver installed inside WSL is started from Windows through the `launcher`
key of the same table, for example `launcher = ["wsl", "-d", "Ubuntu", "--"]`.

When a campaign uses several builds of a solver, or runs on machines with
different paths, name the builds once in `inputs/executables.toml`; see
[Executables](executables.md).

## Check that the solver is found

`cfd-matrix plan --local` and `cfd-matrix run --local` look for the solver on
this machine; `run` does it before it records any point. Without `--local`, a
plan using an HPC profile normally does not look for a local solver. If the
adapter cannot stage cluster jobs, the plan warns, falls back to local execution
planning and checks the local build and executable. Under `--warning-refusals`,
that warning refuses instead. To check an installation explicitly, type
`cfd-matrix plan <matrix> --local`. A solver that is not found stops the plan,
and the run, before anything is recorded, with the same sentence for every solver:

```text
matrix not run: SU2 executable not found: 'SU2_CFD'; set executable in the [solver] table of the setup file, or declare SOLVER_BUILD in inputs/executables.toml
```

Nothing was written, so the same command works once the solver is installed.
[Troubleshooting](troubleshooting.md) lists the other messages of a run.
