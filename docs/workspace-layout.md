# Workspace layout

This page names every folder and file of a workspace, says who writes it, and
marks the ones that are yours to edit. A workspace is one folder that holds a
whole campaign, and its layout is the same for every solver.

## The tree

```text
campaign/
  init.toml                     # cfd-workspace init: the solver of this workspace
  inputs/                       # yours: everything a run reads
    matrices/<name>.fs          # the run matrix, one row per simulation
    setups/s<id>.toml           # solver setup, named by the SET column
    references/r<id>.toml       # reference values, named by the REF column
    pproc/p<id>.toml            # post-processing options, named by the PPROC column
    mesh/                       # meshes and geometry files, named by the GEOMETRY column
    hpc/h<id>.toml              # cluster profiles (optional)
    executables.toml            # solver builds a row may name (optional)
    executables.local.toml      # this machine's paths for those builds (optional)
    management/<recipe>.toml    # storage recipes (optional)
    sync-workspaces.toml        # other workspaces to sync from (optional)
  sims/                         # cfd-matrix run: one folder per simulation
    sim_<POL>/datapoints/DP-<point name>/
  post/                         # cfd-matrix plan, collect and post
    <matrix>/plan.json
    <matrix>/campaign_summary.json
    <matrix>/polars/FAM-<KIND>-P<POL>.csv
    <matrix>/polars/P<POL>-<GROUP>-<KIND>.csv
    <matrix>/products.json
    <matrix>/post.log
  archive/<stamp>/              # earlier evidence, moved here and never deleted
  runs.json                     # the record of every point that was run
  marks.json                    # verdicts you recorded with mark-failed and mark-converged
  storage_management.json       # the record of every storage command
  logs/                         # what run, collect and post printed, and every warning
```

`cfd-workspace init` creates `init.toml`, `sims/`, `post/`, `archive/`, the
input folders `matrices`, `setups`, `references`, `pproc`, `mesh` and `hpc`
(the mock solver creates a few more), the file `inputs/INPUTS.md`, which says
what each input file of the solver holds, and `inputs/.gitignore`. You create
`inputs/management/` and the optional files when you need them. Every other
file appears when the command beside it first runs.

## What you write

You edit only `inputs/`. Nothing a command writes goes there, and nothing you
write belongs anywhere else.

| File | What it holds | Named by |
|---|---|---|
| `inputs/matrices/<name>.fs` | The run matrix: one row per simulation, with its flight condition and the files it uses. See [matrix columns](matrix-columns.md). | the name you give `cfd-matrix plan <name>` |
| `inputs/setups/s<id>.toml` | How the solver runs: model, numerics, iteration limit. The keys belong to the solver; see its page. | the `SET` cell (`001` and `s001` name `s001.toml`) |
| `inputs/references/r<id>.toml` | Reference area, reference length and moment origin, where the solver takes them from a file. | the `REF` cell |
| `inputs/pproc/p<id>.toml` | What to keep from each run beyond the coefficients. | the `PPROC` cell |
| `inputs/mesh/<file>` | The mesh or geometry file, in the format the solver reads. | the `GEOMETRY` cell |

A matrix row names its mesh by file name, for example `wing.su2`, and the file
is read from `inputs/mesh/`. A stem that names exactly one file there is enough.
The folder names `inputs/geometry/`, `inputs/geometries/` and `inputs/meshes/`
are still read, after `inputs/mesh/`, with one warning that names the folder to
move the files to.
A subfolder `inputs/geometry/<name>/` that holds a `geometry.json` sidecar is
generated geometry, not a legacy mesh folder: a path inside it is used as named,
with no warning.

A mesh is never copied into a simulation folder. Each point folder holds a link
to the file in `inputs/mesh/`, so a large mesh is stored once.

## The solver declaration

`init.toml` binds the workspace to one solver:

```toml
schema_version = 1
solver = "su2"
```

For QPROP the solver is a table, which also holds the settings of its section
polars (see [QPROP](qprop-workspaces.md)):

```toml
schema_version = 1

[solver]
name = "qprop"
section_timeout_s = 60
on_unconverged_section = "warn"
```

Every command reads it first. A workspace keeps its solver for life: `init`
refuses a folder that already holds an `init.toml`, and a collect refuses
records made by another solver. The solver names are lower case, and
`cfd-workspace init --help` lists them.

`init` writes this file. You may add workspace solver settings, `warning_refusals`
and a `[defaults]` table of setup keys. A key other than these, a `schema_version` other than the
integer `1` or a solver name that is not in the list is refused by every
command, with the name of the file and the problem.

## Default setup keys

Use `[defaults]` to avoid repeating the same setup values for every row:

```toml
[defaults]
iterations = 100
cfl = 5

[defaults.solver]
threads = 2
```

The setup file overrides these values, including individual keys in nested tables.
The row's `VAR_NAMES_VALUES` overrides apply last. Explicit row flight conditions
and `GEOMETRY` also replace the corresponding
fallback setup values. Explicit setup values still undergo the adapter's normal
agreement checks with the row. The point records `defaults_omitted`, the default
keys the row replaced, so planning and execution use the same effective setup.
The adapter validates the composed setup, so an unknown default key is refused
just like an unknown setup key. Plan
again after changing defaults: `init.toml` is part of every point's input digest.

## What a run writes

`cfd-matrix run` writes one folder per point:

```text
sims/sim_<POL>/datapoints/DP-<point name>/
```

`<POL>` is the row's `POL` cell and the point name is built from the row's
flight condition, so the same operating point has the same folder name in every
workspace. The folder holds the input deck written for the solver, the link to
the mesh, the solver's own output files, `stdout.log` and `stderr.log`. A run
on this machine also leaves `execution.json`, the record of the process that ran.

`runs.json`, in the workspace root, records every point: its inputs and their
hashes, its status, when it started and finished, and the verdict on its
convergence. `cfd-matrix status` and `cfd-workspace show`, `history` and `log` read this file;
you never edit it.

`.input-digests.cache`, also in the workspace root, lets a large input (above 8 MiB, such as a mesh) be read and hashed once rather than by every command. It holds each such file's size, modification time, change time, SHA-256 and, for an SU2 mesh, its marker names, or, for a Gmsh mesh, its format line and physical names. It is never a planned input and never enters a record, and deleting it loses nothing. A file whose size or timestamps still match is not read again; one rewritten with the same size and timestamps goes unnoticed unless the command runs with `--rehash-inputs`.

## What plan, collect and post write

Everything derived from a matrix is under `post/<matrix>/`:

| File | Written by | Holds |
|---|---|---|
| `plan.json` | `cfd-matrix plan` | The plan receipt: the matrix and every input as planned. A run refuses to start when an input changed after it. |
| `campaign_summary.json` | `collect`, `post` | Every recorded point with its result. |
| `polars/FAM-<KIND>-P<POL>.csv` | `collect`, `post` | One row per point and family: loads (DIM), airframe coefficients (ACOEF), rotor coefficients (RCOEF); see [Results](results.md). |
| `polars/P<POL>-<GROUP>-<KIND>.csv` | `collect`, `post` | One row per point for each group TOTAL, AIRFRAME and each rotor. |
| `products.json` | `collect`, `post` | Every product written, and every one skipped with the reason. |
| `post.log` | `collect`, `post` | The same, as text. |

These files are rebuilt from `runs.json` and the simulation folders each time,
so deleting them loses nothing.

## What is kept

Nothing is deleted without a command that says so. Running a point again moves
its folder and its record to `archive/<stamp>/` first, where `<stamp>` is the
UTC time of the move, and `cfd-workspace history` lists what is there.
`cfd-workspace restore` brings a file back.

## Logs

`run`, `collect` and `post` keep what they print. The commands that only read
write nothing.

| File | Holds |
|---|---|
| `logs/<command>-<stamp>.log` | Every line one call of `run`, `collect` or `post` printed. |
| `logs/activity.log` | One line per call: when it started and how it ended. `cfd-workspace log` lists these with the run events. |
| `logs/warnings.log` | Every warning, also those that `--warning-off` kept out of the console. |

## Version control

`cfd-workspace init` writes `inputs/.gitignore` so that
`executables.local.toml`, which holds paths of one machine, stays out of a
repository. `inputs/` is small text and is worth committing. `sims/` and
`archive/` hold solver output and are usually left out.

A stored digest is reused while the file's size, timestamps and inode are unchanged. The host name, the workspace path and the device number are not checked, so the cache works across the nodes of a shared file system (for example `plan` on one login node and `run` or `collect` on another). A copy of the workspace has new inodes and is hashed again once. All checks follow symbolic links to the target. On POSIX, ctime is metadata change time: restoring mtime after a write normally still invalidates the entry. On Windows, Python ctime is creation time, so an in-place same-size rewrite with restored mtime can remain undetected. Coarse filesystem timestamps can hide same-tick writes on either platform. Use `--rehash-inputs` after such changes. Files less than two seconds past their newest timestamp (including future-dated files) are not persisted. Timestamps are epoch values, independent of time zone.
