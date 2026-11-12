# cfdbender

cfdbender is a solver-agnostic Python library for CFD workspaces. A workspace
holds named s/r/p matrices, read-only planning, local and HPC runs and neutral
result collection, and the first level of the design never depends on a
particular solver. Each workspace declares one solver in its root `init.toml`.

## Supported solvers

- SU2, through the executable route and the pysu2 wrapper route.
- FlightStream, delegated to pyflightstream.
- OpenFOAM (ESI).
- XFOIL.
- PyFR (high-order flux reconstruction, GPU).
- QPROP (isolated propeller performance).
- A mock solver for trying the workflow without any installation.

Fluent and other solvers are on the roadmap.

## Install

```console
pip install cfdbender
```

FlightStream support is optional: `pip install cfdbender[flightstream]`.

## Minimal example

```console
cfd-workspace init --solver mock --workspace campaign
```

This creates a mock workspace named `campaign` that you can plan, run and
collect with the same commands as any other solver.

## Where to read next

- [Start here](getting-started.md) for the first steps.
- Solver guides: [FlightStream](flightstream-workspaces.md),
  [SU2](su2-workspaces.md), [OpenFOAM](openfoam-workspaces.md),
  [XFOIL](xfoil-workspaces.md), [PyFR](pyfr-workspaces.md) and
  [QPROP](qprop-workspaces.md).
- [Workspace files](workspace-files.md) and
  [workspace settings](workspace-settings.md).
- [Architecture](architecture.md) and the draft
  [requirements](requirements/index.md), which state the status and
  verification method of each requirement.
- Source code and issues on [GitHub](https://github.com/nevesgeovana/cfdbender).
