# Executables

This page shows how to name the solver builds of a campaign once, so that a
matrix row picks its build by name and each machine supplies its own path.

You need this page when a campaign uses more than one build of a solver, or
when the same workspace runs on machines that install the solver in different
places. A campaign with one build on one machine can state the executable in
the setup file instead; the page of your solver gives the key.

## Declare the builds

`inputs/executables.toml` has one table per solver and one line per build:

```toml
# inputs/executables.toml
[su2]
"8.3.0" = "/opt/su2-8.3.0/bin/SU2_CFD"
"8.3.0-mpi" = { path = "/opt/su2-8.3.0/bin/SU2_CFD", launcher = ["mpirun", "-np", "8"] }
```

The table name is the solver name of `init.toml`. The name of a build is yours
to choose; it is what a matrix row writes in its `SOLVER_BUILD` cell. A build
is either a path, or a table with a `path` and an optional `launcher`: the
words that go in front of the executable on the command line.

A relative path that holds a `/` is read from the workspace root. A bare
program name, such as `SU2_CFD`, is looked for on the `PATH` of the machine.

## Give the paths of one machine

`inputs/executables.local.toml`, beside the first file, replaces the paths for
the machine it is on:

```toml
# inputs/executables.local.toml
[su2]
"8.3.0" = "/home/you/su2/8.3.0/bin/SU2_CFD"
```

A local entry replaces the fields it states and keeps the others. It cannot add
a build that `executables.toml` does not declare. `cfd-workspace init` writes
`inputs/.gitignore` so that the local file stays out of version control, while
`executables.toml` is shared with the campaign.

## Pick a build in a row

```text
POL  | RUN | FLIGHT_CONDITION      | SWEEP_VALUES | REF | SET | PPROC | SOLVER_BUILD
-----|-----|-----------------------|--------------|-----|-----|-------|-------------
1001 | 1   | MACH:0.5, ALPHA:sweep | 0,2          | 001 | 001 | 001   | 8.3.0
```

On a run on your machine, a row that names a build takes that build's path and
launcher. They replace the ones the setup file states in its `[solver]` table.

| Solver | The path fills | The launcher fills |
|---|---|---|
| su2 | `executable`, and the run uses the SU2 executable | `mpi_launcher` |
| openfoam | `foam_bashrc`: the build is the environment file that OpenFOAM sources | `launcher` |
| pyfr | `executable` | `launcher` |
| xfoil | `executable` | `launcher` |
| qprop | `executable` | `launcher` |

## The version check

Whenever a row names a build, the version that the solver printed during the
run is compared with it. The build name must start with that version: `8.3.0`
and `8.3.0-mpi` both match a solver that printed `8.3.0`. A point whose solver
printed another version is recorded as failed:

```text
SOLVER_BUILD mismatch: the row names '8.2.0' but the solver reported version '8.3.0'; point SOLVER_BUILD at the executable of that build in inputs/executables.toml, or correct the row
```

A solver that prints no version cannot be checked, and its points stand.

## What happens without these files

| Case | Result |
|---|---|
| The row has no `SOLVER_BUILD`, or its cell is `-` | The setup file's `[solver]` table is used. |
| The workspace has no `inputs/executables.toml` | The setup file's `[solver]` table is used. The version check still applies. |
| The row names a build that `executables.toml` does not declare | `run` refuses before anything is written and lists the declared builds. |

A cluster job does not read these files. It takes its build from the
`[builds]` table of the cluster profile; see [Cluster jobs](cluster-jobs.md).
