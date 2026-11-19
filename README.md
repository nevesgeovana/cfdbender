# cfdbender

cfdbender is an early Python project for CFD workspaces. Version 0.7.0 provides
workspace initialization, named setup, numerics, reference and post-processing
files, read-only planning, local runs, neutral result collection, and HPC
profiles with plan, run and collect. It includes a draft SRS and documentation.

What it offers:

- **Solvers.** Mock and SU2 adapters (the SU2 executable route, and the pysu2
  Python wrapper route run in a child interpreter), ESI OpenFOAM, XFOIL, PyFR,
  QPROP, FlightStream workspaces delegated to pyflightstream, and CFD++
  (incomplete and not yet verified against the solver).
- **Matrix columns.** `SETUP` names the physical setup, `NUMERICS` an optional
  numerics file shared by many setups, and `MESH` the mesh. A header that still
  says `SET` or `GEOMETRY` is read with a notice, and
  `python -m cfdbender.migrate_setup_numerics` moves a workspace.
- **Run types.** The `WORKFLOW` column selects `steady`, `qsteady_rotor`,
  `unsteady` or `unsteady_rotor` for SU2, with the clock, the averaging window
  and the export window of a time-accurate row.
- **Rotors.** `ADVANCE_RATIO`, the rotor block of the reference file,
  `SYMMETRY` `PERIODIC` with `PERIODIC_COPIES`, the rotor table and the
  unsteady polar.
- **Products.** The post-processing file declares sections, a volume section
  and probes; `collect` writes them for every solver that supplies them.
- **Solver-agnostic numerics.** The package `cfdbender.post` holds time
  averages, per-revolution means, sections, volume sampling, a `.vtu` reader and
  far-field numerics.
- **Clusters.** `hpc.<name>` replaces one value of an HPC profile for a row, and
  `cfd-matrix run --batch N` groups rows into N scheduler jobs and writes the
  shell script each runs.
- **Large inputs.** An optional identity mode leaves large meshes unread, and
  planning of large meshes is faster.

## Quick start

```console
pip install cfdbender
cfd-workspace init campaign --solver mock
cd campaign
```

Write `inputs/matrices/first.fs`, with one row per simulation:

```text
POL  | RUN | FLIGHT_CONDITION      | SWEEP_VALUES | REF | SETUP | PPROC
-----|-----|-----------------------|--------------|-----|-------|------
1001 | 1   | MACH:0.2, ALPHA:sweep | 0,2,4        | 001 | 001   | 001
```

Then plan, run and collect it:

```console
cfd-matrix plan first
cfd-matrix run first
cfd-matrix collect first
```

[First campaign](docs/getting-started.md) shows every file. Moving from 0.6 needs
nothing: see [Upgrading from 0.6](docs/upgrading.md).

## FlightStream

Install the optional FlightStream support with `pip install cfdbender[flightstream]`.
See [FlightStream workspaces](docs/flightstream-workspaces.md) for initialization
and the delegated commands.

The design keeps the first level independent of any solver.
Each workspace declares one solver in its root `init.toml`.
Supported solvers are SU2, FlightStream, OpenFOAM (ESI), XFOIL, PyFR and QPROP,
and CFD++, which is incomplete. Fluent and other solvers are on the roadmap. See
the [SU2](docs/su2-workspaces.md), [OpenFOAM](docs/openfoam-workspaces.md),
[XFOIL](docs/xfoil-workspaces.md), [PyFR](docs/pyfr-workspaces.md),
[QPROP](docs/qprop-workspaces.md) and [CFD++](docs/cfdpp-workspaces.md)
workspace guides.

See the [documentation site](https://nevesgeovana.github.io/cfdbender/) and
the [workspace layout](docs/workspace-layout.md). This project is licensed under
the MIT license.
