# Install

This page installs cfdbender and tells you what each solver needs on the
machine before a campaign can run.

## cfdbender

cfdbender needs Python 3.12 or later and nothing else.

```console
pip install cfdbender
cfd-workspace init --help
```

The help text ends with the list of solvers. When it prints, the two programs
`cfd-workspace` and `cfd-matrix` are installed. You can now follow
[First campaign](getting-started.md), which needs no solver.

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

cfdbender looks for the solver when a point runs, not when it is planned. To
check an installation, run a matrix with one point. A solver that is not found
is reported on that point:

```text
SU2 executable not found; set solver.executable or CFDBENDER_SU2_EXE
```

[Troubleshooting](troubleshooting.md) lists the other messages of a run.
