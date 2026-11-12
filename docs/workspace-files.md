# Workspace files

Start a campaign with one solver declaration and the empty input library:

```console
cfd-workspace init --solver mock --workspace campaign
```

The positional root is also accepted: `cfd-workspace init campaign --solver mock`.
Without either root spelling, the current directory is used. Naming two different
folders is refused. `cfd-workspace --help` and `cfd-workspace init --help` describe
the command. The offline workspace tests execute these examples.

Initialization creates:

```text
campaign/
  init.toml
  inputs/
    geometries/
    references/       # r<id>.toml
    setups/           # s<id>.toml
    pproc/            # p<id>.toml
    profiles/
    freestreams/
    calibrations/
    hpc/              # h<id>.toml
    fsi/
    matrices/
    management/
  sims/
  post/
  archive/
```

That is the mock solver's layout. A workspace for su2, openfoam, pyfr, xfoil or
qprop gets only the input folders those adapters read: `geometries`, `references`,
`setups`, `pproc`, `hpc` and `matrices`. A matrix GEOMETRY cell names a file of
`inputs/geometries/`; see [matrix columns](matrix-columns.md).

The input folders follow the established campaign layout. They start empty;
input files, matrix expansion, run records, HPC operations and solver execution
are later capabilities. Initialization writes only `init.toml` and directories.
See [the declaration reference](workspace-settings.md) for the root file schema.

Existing folders and files are preserved. A non-empty target is allowed, but an
existing `init.toml` is refused, even if malformed, to protect the workspace's
solver binding. An unknown solver, a target that is a file, or a file occupying a
required folder is also refused. Refusals name the problem on stderr and return
exit code 2; success returns 0 and prints `campaign workspace ready at` with the
absolute root. Parser errors return 2; help returns 0.

## Executables

`inputs/executables.toml` declares, per solver, the builds a matrix row may name
in `SOLVER_BUILD`, and `inputs/executables.local.toml` beside it gives this
machine's paths. `cfd-workspace init` writes `inputs/.gitignore` so the local
file stays out of version control.

```toml
# inputs/executables.toml
[su2]
"8.3.0" = "/opt/su2-8.3.0/bin/SU2_CFD"
"8.3.0-mpi" = { path = "/opt/su2-8.3.0/bin/SU2_CFD", launcher = ["mpirun", "-np", "8"] }
```

```toml
# inputs/executables.local.toml
[su2]
"8.3.0" = "C:/SU2/8.3.0/bin/SU2_CFD.exe"
```

A bare string is the executable path; a table holds `path` and an optional
`launcher` list. A local entry replaces the fields it states and may not add a
build the declared file lacks. A relative path resolves from the workspace root.

On a local run, a row that names `SOLVER_BUILD` takes that build's executable,
overriding the setup's `[solver]` executable, and its launcher when the entry
gives one. The path fills the solver's own option: `executable` for su2 (which
then runs its executable route; the launcher is `mpi_launcher`), pyfr, xfoil
and qprop, and `foam_bashrc` for openfoam, whose build is the environment that
file sources. A build the file does not declare refuses the run before anything
is written. A workspace without `inputs/executables.toml`, or a row without
`SOLVER_BUILD`, keeps the setup's `[solver]` table. A cluster job takes its build
from the HPC profile's `[builds]` table instead.

## Point folders of cluster jobs

`cfd-matrix run` with an HPC profile stages, before it submits, every file the
solver reads into the point folder, exactly as a local run stages them. The job
runs in that folder and needs nothing else from the workspace.

| Solver | Staged in the point folder | `{config_path}` | The job writes |
|---|---|---|---|
| su2 | `su2.cfg`, the mesh (`mesh.su2` or `mesh.cgns`) | `su2.cfg` | `history.csv`, `stdout.log` |
| openfoam | case tree (`system`, `constant`, `0`), `Allrun`, the mesh | `Allrun` | `log.simpleFoam`, `postProcessing/forceCoeffs/0/coefficient.dat` |
| pyfr | `case.ini`, `mesh.msh` | `case.ini` | `forces.csv` |
| qprop | `propeller.prop`, `qcon.def`, `sections.json` (section polars made first) | `propeller.prop` | `stdout.log` |
| xfoil | `xfoil.in`, `airfoil.dat` when the row names one | `xfoil.in` | `polar.txt`, `stdout.log` |

`{config_path}` and `{script_path}` name the file the solver reads.
`{stdout_log}` and `{stderr_log}` name `stdout.log` and `stderr.log` in the point
folder: point the scheduler's output files there, because SU2, XFOIL and QPROP
results are read from the solver's standard output. `cfd-matrix collect` leaves a
submitted point outstanding until the files the job writes exist, then reads it.
A cluster job leaves no `execution.json`; its classification rests on the
numerical evidence alone. A point whose inputs cannot be staged is recorded as
failed with `FAILED_STAGING` and is not submitted.

A mesh is never copied into a point folder. Staging makes a relative symbolic
link, which still resolves after the whole workspace moves to the cluster; on a
Windows host without the symbolic-link privilege it makes a hard link (the same
file on disk). Where neither link can be made, for example across volumes,
staging refuses.

## Running a matrix again

`runs.json` records every point a run starts. Without options, a recorded point
refuses `cfd-matrix run`, so finished evidence is never overwritten.

- `--resume` skips points recorded COMPLETED or SUBMITTED and runs the rest, so a
  matrix that grew a row runs only its new points. A point left RUNNING by a
  local run that was killed runs again when no solver process holds this host's
  process slot. A RUNNING point of a cluster submission, whose job may exist, is
  skipped; name it with `--force-rerun` to redo it.
- `--force-rerun POINT` redoes a recorded point, named `POL/DP-name` or, when
  only one point carries it, `DP-name`. Repeat it for several points. Other
  recorded points are left as they are.
- `--force-rerun-all` redoes every recorded point of the matrix. A SUBMITTED job
  is not cancelled: cancel it in the scheduler first. It cannot be combined with
  `--resume` or `--force-rerun`.

A redone point is archived before it runs, and nothing is deleted: `runs.json`
is copied to `archive/<stamp>/runs.json`, the point folder moves to
`archive/<stamp>/sims/sim_<POL>/datapoints/<point>/`, and the old record joins
the `history` list of `runs.json` with the time and the archive folder.

## Collected tables

`cfd-matrix collect` rebuilds, for each matrix, `post/<matrix>/campaign_summary.json`
and two CSV tables beside it: `campaign_sweep.csv` with one row per recorded
point, and `polars/P<POL>.csv` with one row per point of each simulation, the
`P<sim>` stem pyflightstream gives its polar files. The columns are `POL`,
`point`, the FLIGHT_CONDITION keys in the order the matrix states them,
`status`, `classification`, `iterations` and `residual` of the last history
entry, `wall_time_s` from the record's start and completion times,
`solver_version`, then every coefficient the solver reported. A value a point
does not have, such as the coefficients of a point not yet run, is an empty
cell. The tables are derived files and are rebuilt by every collect.

## Selecting simulations and points

`plan`, `run` and `collect` take `--sims POL [POL ...]` (or `--sims 1001,1002`) and
`--points NAME [NAME ...]`, with points named `DP-name` or `POL/DP-name`, to work
on part of a matrix without editing it. An id or name the matrix does not carry
is refused before anything runs. A selective `collect` reads only the selected
points and keeps the other rows of the summary as they were.

## Datapoint folder names

Each point of a plan collects into `sims/sim_<POL>/datapoints/DP-<point name>`.
The point name is built from the point's FLIGHT_CONDITION exactly as
pyflightstream builds it, so the same operating point has the same folder name
in both libraries: every key of the cell, in the cell's order, written as a
code followed by the value times a scale, rounded and zero padded.

| Key | Code | Scale | Width | Example |
|---|---|---|---|---|
| MACH | M | 1000 | 3 | 0.85 gives `M850` |
| TASmps | V | 10 | 4 | 34 gives `V0340` |
| REmi | RE | 100 | 3 | 5 gives `RE500` |
| ALTFT | ALT | 1 | 5 | 5000 gives `ALT05000` |
| dISA | DT | 10 | 4, signed | 10 gives `DT+100` |
| RHOkgm3, MUPas, ASMPS, TK, PPA | RHO, MU, A, T, PS | 10000, 1e9, 10, 10, 1 | 5, 5, 4, 4, 6 | `RHO12250` |
| ALPHA, BETA | AL, BE | 10 | 4, signed | 2.5 gives `AL+025` |
| ADVANCE_RATIO (or J) | J | 100 | 4, signed | 1.0 gives `J+100` |
| RPM | RPM | 1 | 5 | 800 gives `RPM00800` |
| roll_rate, pitch_rate, yaw_rate | P, Q, R | 10 | 4, signed | `P+025` |

Examples: `DP-M850RE500AL+025` (alpha 2.5 at Mach 0.85, REmi 5),
`DP-M200RE500AL+160`, and `DP-M120RE009AL+000BE+000J+100`.

- A signed field counts its sign in the width. A negative value of an unsigned
  key, or a negative RPM, is refused when the plan is built.
- A key with no code (a native solver key such as QPROP's `V` or `BETA07`) is
  written as its upper case name and the value in general format, for example
  `DP-V20J+100BETA0730`. A row with no FLIGHT_CONDITION is `DP-NOCOND`.
- The order of the keys in the cell is the order in the name, so reordering the
  cell renames the points.
- Two sweep values of one row that round to the same name are refused at plan
  time, naming both values, as pyflightstream refuses them.

Workspaces planned before this naming carry `DP-<20 hex>` folders. A recorded
point keeps its folder and its record: `runs.json` stores each point's
directory, collect reads it from the record, and old records still load. Only
a plan written afterwards uses the new names, so re-planning a matrix in an old
workspace would register its points a second time under the new names; finish
or archive the old campaign first.
