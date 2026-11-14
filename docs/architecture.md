# Architecture

cfdbender keeps one campaign folder per solver and reads it the same way for
every solver. The matrix columns, the flight condition, the datapoint folder
names, the run record and the result tables belong to a solver-independent
core. A solver is an adapter: it translates a planned point into the solver's
input, runs it, classifies the outcome and reads the results back. It owns
nothing else.

This page describes the layers, the files each stage owns and the rules a
change must keep. The clauses a solver adapter must satisfy are on the
[adapter contract](adapter-contract.md) page.

## The pipeline

```text
matrix -> plan -> run or submit -> collect -> tables
```

| Stage | Command | Reads | Writes |
| --- | --- | --- | --- |
| init | `cfd-workspace init --solver <name>` | the solver name | `init.toml`, the `inputs/` folders, `sims/`, `post/`, `archive/` |
| plan | `cfd-matrix plan <matrix>` | `init.toml`, the matrix, the setup, reference and post files of each row, the mesh or geometry | the plan receipt `post/<matrix>/plan.json`; no solver file |
| run | `cfd-matrix run <matrix>` | the plan, `runs.json` | one folder per point under `sims/`, one record per point in `runs.json` |
| submit | `cfd-matrix run <matrix> --hpc <profile>` | the plan, the profile in `inputs/hpc/` | the staged point folder, the job descriptor, the record as `SUBMITTED` |
| collect | `cfd-matrix collect [<matrix>]` | `runs.json`, the saved solver outputs | `post/<matrix>/campaign_summary.json`, `campaign_sweep.csv`, `polars/P<POL>.csv`, `products.json`, `post.log` |
| post | `cfd-matrix post [<matrix>]` | `runs.json`, the saved solver outputs; no pickup of submitted jobs | the same products as collect; no record changes |

Planning launches no solver. A run executes every selected point, records a
failed point and continues with the next. Collect launches nothing: it reads
saved outputs and rebuilds every table whole, so a table can always be
regenerated from `runs.json` and the point folders.

## The layers

```text
applications     cfdbender.matrix      cfdbender.workspace
                        |                     |
adapters         cfdbender.solvers.<name>     cfdbender.solvers._host
                        |
core             cfdbender.core

beside the core  cfdbender.hpc   cfdbender.testing   cfdbender.geometry
```

| Layer | Package | Role |
| --- | --- | --- |
| Core | `cfdbender.core` | The matrix reader, the flight condition and atmosphere, the datapoint names, the plan, the run record, the local run loop, the neutral result types and the adapter protocol. |
| Adapters | `cfdbender.solvers.<name>` | One package per solver: `su2`, `openfoam`, `pyfr`, `xfoil`, `qprop`, and `mock`, a licence-free adapter that exercises the contract offline. |
| Host helpers | `cfdbender.solvers._host` | What every adapter shares when it launches a process: the free memory floor, the one-process lease, the mesh link, the timeout and the child process cleanup. |
| Job planner | `cfdbender.hpc` | Renders a job descriptor and a submit command from a profile, and computes walltime. It knows no solver. |
| Applications | `cfdbender.matrix`, `cfdbender.workspace` | The `cfd-matrix` and `cfd-workspace` commands. They read `init.toml`, load the one declared adapter by name and pass it to the core. The two depend on each other in one direction each: `cfdbender.workspace` imports the console and record helpers of `cfdbender.matrix`, and `cfdbender.matrix` imports the planning of `cfdbender.workspace`; neither is imported by the core, the adapters or the job planner. |
| Conformance kit | `cfdbender.testing` | Assertions an adapter's tests call to prove the contract. |
| Geometry | `cfdbender.geometry` | Blade slicing on the standard library alone. |
| Facade | `cfdbender._pyfs` | The only module that imports pyflightstream. |

### The import rule

- The core never imports an adapter. It receives the registered solver names
  as data and an adapter as an argument.
- An adapter never imports another adapter. Shared launch code lives in
  `cfdbender.solvers._host`.
- The job planner imports neither an adapter nor the facade.
- pyflightstream is imported only in `cfdbender._pyfs`.

The rule keeps a user of one solver from paying for another solver's imports
and failures, and it keeps the mock adapter a real proof that the contract is
sufficient: if the core needed to know a solver, the mock could not stand in
for it.

Check the rule with:

```console
python tools/check_architecture.py
```

The checker parses the source without importing it. Absolute and relative
imports, imports inside functions and literal dynamic imports are all read.
It prints each violation with its file and line and exits with code 1; a tree
it cannot measure (no core sources, malformed Python) exits with code 2, so an
unmeasured tree never passes. Computed dynamic imports are outside its
coverage. The offline tests run it against the live package and against
planted violations of each rule.

## The files and who writes them

Each file has one writer. A change that gives a file a second writer breaks
the guarantees listed below the table.

| File | Written by | Holds |
| --- | --- | --- |
| `init.toml` | `cfd-workspace init`, once | The one solver of the workspace and optional solver defaults. |
| `inputs/matrices/<name>.fs` | you | The rows: one simulation each, with its flight condition and the ids of its setup, reference and post files. |
| `inputs/setups/`, `inputs/references/`, `inputs/pproc/` | you | The setup (numerics and solver options), the reference values and axes, and what is saved. |
| `inputs/mesh/` | you | Meshes and geometry files. The earlier folder names `inputs/geometry/`, `inputs/geometries/` and `inputs/meshes/` are still read. |
| `inputs/executables.toml` | you | The solver builds a matrix may name and their executables. |
| `inputs/hpc/<name>.toml` | you | A cluster profile: every descriptor field and the submit command. |
| `post/<matrix>/plan.json` | `cfd-matrix plan` | The digest of the matrix and of every input of every planned point. |
| `runs.json` | the run registry in the core | One record per point: the proposal, the status, the times, the classification, the solver version the solver reported, the digests of inputs and outputs, and the history of archived records. |
| `sims/sim_<POL>/datapoints/DP-<name>/` | the adapter, during a run or a staging | The solver input as written, the solver outputs and the captured output channels. |
| `post/<matrix>/` tables | `cfd-matrix collect` | Derived products, rebuilt on every collect. |
| `archive/<stamp>/` | the run registry | `runs.json` as it was and the folder of each point that was redone. |

### What the core guarantees

- **One solver per workspace.** `init.toml` is created once and read again by
  every stage.
- **The bytes that run are the bytes that were planned.** Planning records
  the SHA-256 of every input it reads. The digests are checked again when a
  point is registered, when it starts and when its results are collected. A
  changed input refuses with the instruction to plan again.
- **Nothing is overwritten.** A point folder or an output that already exists
  refuses the run before anything is written.
- **Nothing is deleted by a run.** Redoing a point moves its folder and its record
  to `archive/<stamp>/`. The management verbs are the exception, and only when
  applied with `--apply`: `delete-sims` removes a simulation's folder and records
  (a `SUBMITTED` or `RUNNING` point needs `--force`; the removed records stay in
  the history), `free-space` removes the files of its recipe, `sync --overwrite`
  replaces a file after archiving the old one, and `restore` replaces the live
  files after archiving them. Without `--apply` each previews and changes nothing.
- **A point is named from its flight condition.** The folder is
  `DP-<point name>`, written the way pyflightstream writes it, so both
  libraries name the same operating point the same way.
- **A run record has four transitions.** `PLANNED` to `RUNNING`, `RUNNING` to
  `COMPLETED`, `RUNNING` to `SUBMITTED` and `SUBMITTED` to `COMPLETED`. A
  finished record is never rewritten by a run. The verbs that correct records
  (`mark-failed`, `mark-converged`, `restore`, `rebuild`, `sync` and `delete-sims`),
  applied with `--apply`, are the exception: each archives the manifest first, and
  `mark-failed` refuses a `RUNNING` point. One writer at a time holds
  `runs.json.lock`.
- **Process success is not convergence.** A point is `converged`,
  `iteration_limit` or `failed` from numerical evidence. Exit code zero alone
  is `failed`.
- **A coefficient carries its axis and its references.** A result without an
  explicit reference area, length, moment origin and axis directions is
  refused.

## Local runs and cluster jobs

A local run and a cluster job read the same files. For a cluster job the
adapter stages the point folder exactly as a local run would, then the profile
submits it. The submit command's return code says only whether the scheduler
accepted the job. The point stays `SUBMITTED` until `cfd-matrix collect` finds
the files the job writes and classifies it from them.

On this machine every adapter launches its solver under the same guards: at
least 0.5 GiB of free memory before and during the run, one solver process per
host, a time limit bounded by the row's `WALLTIME`, and the whole child process
tree stopped on a timeout.

## FlightStream workspaces

FlightStream is not an adapter. A workspace created with
`cfd-workspace init --solver flightstream` is a pyflightstream workspace plus
`init.toml`. In that workspace `cfd-matrix` and `cfd-workspace` forward their
arguments unchanged to pyflightstream's own commands, so every verb, flag,
help text and exit status is pyflightstream's, and the files written are
byte-identical to those pyflightstream writes. The support is optional:
install it with `pip install cfdbender[flightstream]`.

A reader, `cfdbender.solvers.flightstream.read_workspace`, returns the points
of a FlightStream workspace as the same neutral result type the adapters
return, without regenerating any product.

## Adding to cfdbender

- A new solver is a new adapter package. Follow the
  [adapter contract](adapter-contract.md).
- A behaviour every solver should share belongs in the core, with a test that
  runs it for every adapter.
- A change to a file in the table above changes its one writer and never adds
  a second one.
