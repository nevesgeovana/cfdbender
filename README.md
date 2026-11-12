# cfdbender

cfdbender is an early Python project for CFD workspaces. Version 0.4.2 provides
workspace initialization, named s/r/p matrices, read-only planning, local runs,
neutral result collection, mock and SU2 adapters (the SU2 executable route and
the pysu2 Python wrapper route, run in a child interpreter where the wrapper is
installed), FlightStream workspaces delegated to pyflightstream, ESI OpenFOAM,
XFOIL, PyFR, QPROP, and HPC profiles
with plan/run/collect and --hpc. It includes a draft SRS and documentation.
Grouped HPC jobs (--batch) for SU2 are not implemented.

```console
cfd-workspace init --solver mock --workspace campaign
```

## FlightStream

Install the optional FlightStream support with `pip install cfdbender[flightstream]`.
See [FlightStream workspaces](docs/flightstream-workspaces.md) for initialization
and the delegated commands.

The design keeps the first level independent of any solver.
Each workspace declares one solver in its root `init.toml`.
Supported solvers are SU2, FlightStream, OpenFOAM (ESI), XFOIL, PyFR and QPROP. Fluent and
other solvers are on the roadmap. See the [OpenFOAM](docs/openfoam-workspaces.md),
[XFOIL](docs/xfoil-workspaces.md), [PyFR](docs/pyfr-workspaces.md) and
[QPROP](docs/qprop-workspaces.md) workspace guides.

See the [documentation site](https://nevesgeovana.github.io/cfdbender/) and
the [workspace files](docs/workspace-files.md). This project is licensed under
the MIT license.
