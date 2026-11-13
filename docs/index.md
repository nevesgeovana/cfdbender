# cfdbender

cfdbender runs CFD campaigns from a table. You write a matrix with one row per
simulation, and the same two commands plan it, run it on your machine or on a
cluster, and collect the results into tables that have the same columns for
every solver. One folder, the workspace, holds the whole campaign, and every
run is recorded, so each result can be traced to the inputs that made it.
Nothing that was computed is overwritten or deleted unless you run a command
that says so.

```console
pip install cfdbender
cfd-workspace init campaign --solver mock
```

The mock solver computes nothing and needs no installation. It is there to
learn the commands before a real solver is involved.

## Solvers

| Solver | Computes |
|---|---|
| [SU2](su2-workspaces.md) | Compressible flow on a volume mesh. |
| [OpenFOAM](openfoam-workspaces.md) | Steady incompressible flow on a volume mesh. |
| [PyFR](pyfr-workspaces.md) | High-order flow simulation with flux reconstruction. |
| [XFOIL](xfoil-workspaces.md) | Polars of a two-dimensional airfoil. |
| [QPROP](qprop-workspaces.md) | Performance of an isolated propeller. |
| [FlightStream](flightstream-workspaces.md) | FlightStream campaigns, handled by pyflightstream. |

## Where to start

- **New to cfdbender.** [Install](install.md) it, then follow
  [First campaign](getting-started.md): one complete campaign with every
  command and every file shown.
- **How a campaign works.** [The workflow](workflow.md) explains the stages in
  order, and the pages beside it cover the workspace, the matrix, the records,
  the tables, cluster jobs and storage.
- **Your solver.** [Solvers, side by side](solvers.md) compares what each
  solver needs, and each solver has a page with its mesh, its setup keys, its
  rule for convergence and a worked example.

[Commands](cli.md) lists every command and option.
[Troubleshooting](troubleshooting.md) explains each refusal by its text. The
[architecture](architecture.md), the [adapter contract](adapter-contract.md)
and the [requirements](requirements/index.md) are for whoever changes cfdbender
or adds a solver. The source code and the issues are on
[GitHub](https://github.com/nevesgeovana/cfdbender).
