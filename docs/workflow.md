# The workflow

This page explains the stages of a campaign in the order you do them, and what
each stage reads and writes. The stages and the commands are the same for every
solver.

```console
cfd-workspace init campaign --solver NAME
cd campaign
cfd-matrix plan polar
cfd-matrix run polar
cfd-matrix status
cfd-matrix collect polar
```

[First campaign](getting-started.md) runs these commands on a small example and
shows what each one prints.

## The words

| Word | Meaning |
|---|---|
| workspace | One folder for one campaign and one solver. |
| matrix | A table you write, `inputs/matrices/<name>.fs`, with one row per sim. |
| sim | One row of a matrix, numbered by its `POL` cell. |
| point | One flow condition of a sim. A row with a sweep has one point per sweep value; a row without a sweep has one point. A point is named `POL/DP-<name>`, for example `1001/DP-M200AL+020`. |
| setup | The file that says how the solver runs: `inputs/setups/s<id>.toml`. |
| reference | The file with the reference area, length and moment origin: `inputs/references/r<id>.toml`. |
| pproc | The file that says what to keep from each run: `inputs/pproc/p<id>.toml`. |
| run | One execution of one point. |
| record | What `runs.json` keeps about a run: its inputs, its status, its times and the verdict on its convergence. |

## The stages

| Stage | Command | Reads | Writes |
|---|---|---|---|
| Create | `cfd-workspace init` | the solver name | `init.toml`, the `inputs/` folders, `inputs/INPUTS.md`, `sims/`, `post/`, `archive/` |
| Write the inputs | none: you write them | | the matrix, the setup, reference and pproc files, the mesh |
| Plan | `cfd-matrix plan` | `init.toml`, the matrix and every file its rows name | the plan receipt `post/<matrix>/plan.json` |
| Run | `cfd-matrix run` | the plan receipt, the inputs, `runs.json` | one folder per point under `sims/`, one record per point in `runs.json` |
| Follow | `cfd-matrix status`, `show`, `history`, `log` | `runs.json` | nothing |
| Collect | `cfd-matrix collect` | `runs.json`, the saved solver output of each point | the tables under `post/<matrix>/` |
| Post | `cfd-matrix post` | the same | the same tables |
| Keep or clean | `cfd-workspace archive`, `cfd-matrix free-space`, `delete-sims` | `runs.json`, the folders | `archive/<stamp>/`, `storage_management.json` |

## Create

```console
cfd-workspace init campaign --solver NAME
```

`init` creates the folder `campaign` with its empty input folders and writes
`init.toml`, which names the solver, and `inputs/INPUTS.md`, which says what
each input file of that solver holds. It ends with the next step. A workspace
keeps its solver for life. `cfd-workspace init --help` lists the solver names. To
use two solvers, create two workspaces.

[Workspace layout](workspace-layout.md) names every folder and file.

## Write the inputs

You write four kinds of file, all under `inputs/`:

- the matrix, described in [Matrix columns](matrix-columns.md);
- the setup, reference and pproc files that the rows name, whose keys are on
  the page of your solver (see [Solvers, side by side](solvers.md));
- the mesh or geometry file, in `inputs/mesh/`, in the format your solver reads.

One setup file can serve many rows, and one matrix can name several setups.

## Plan

```console
cfd-matrix plan polar
```

`plan` reads the matrix `inputs/matrices/polar.fs` and every file its rows
name. It prints one line per point and a closing line with the receipt and the
next command; `--verbose` adds, for each point, what a run would do and every
file it would write. It launches nothing. A mistake in an input is refused here:
the message starts with the matrix path and the row, names the file or the cell
at fault, and nothing is written.

A plan that succeeds saves a receipt, `post/polar/plan.json`. The receipt holds
a fingerprint of the matrix and of every input file. `run` compares the inputs
with the receipt and refuses to start when one of them changed. Plan again
after every edit.

## Run

```console
cfd-matrix run polar
```

`run` executes every point of the plan. For each point it creates the folder
`sims/sim_<POL>/datapoints/DP-<name>/`, writes the solver input there, starts
the solver, and records the outcome in `runs.json`.

On your machine the points run one after the other, and a line says when
each starts and when it ends. Before the first point is recorded, `run` checks
that the solver program exists, that no other solver run holds the machine and
that the memory is enough; when one fails the run is refused and nothing is
written. Three rules protect the machine and the results:

- One solver process runs at a time on a machine, also across workspaces. A
  second `run` started while the first is going does not start its solver.
- A point does not start when the free memory of the machine would fall under
  0.5 GiB.
- A point that fails is recorded as failed, shown as `FAILED_EXECUTION` or
  another `FAILED_` word, and the run goes on to the next point. The command
  then ends with exit code 2.

With a cluster profile in `inputs/hpc/`, `run` submits one job per point
instead; `--local` keeps the work on your machine. See
[Cluster jobs](cluster-jobs.md).

A point that is already recorded refuses a second run, so finished work is
never replaced by accident. [Run again, resume, records](records.md) explains
`--resume` and `--force-rerun`.

## Follow

```console
cfd-matrix status
cfd-matrix status --points
cfd-matrix show 1001/DP-M200AL+020
cfd-matrix history 1001
cfd-matrix log
```

These commands only read the records. `status` gives one line per sim, or one
line per point with `--points`. `show` prints the card of a point, and the whole
record with `--verbose`. `history` lists the current and the earlier runs of a
sim or a point. `log` lists what happened in time order. Times are printed in
the time of the machine, to the second.

## Collect

```console
cfd-matrix collect polar
```

`collect` reads the saved solver output of every recorded point and writes the
tables of the matrix under `post/polar/`: one row per point with its flow
condition, its status, the verdict on its convergence and its coefficients.
It launches nothing. [Results and tables](results.md) describes the files and
their columns.

The tables are rebuilt whole each time, so you can collect as often as you
like, also while points are still running.

## Post

```console
cfd-matrix post polar
```

`post` writes the same files as `collect` and prints them the same way. Use
`collect` while cluster jobs are outstanding, because `collect --watch` can wait
for them. Use `post` to rebuild the tables of a campaign that has finished, with
no solver and no pickup.

## Keep or clean

Nothing is deleted unless you run a command that says so. Running a point again
moves its earlier folder and record to `archive/<stamp>/`.
[Storage and sync](storage.md) shows how to see the space a campaign uses, how
to free it, and how to delete a sim.

## Part of a matrix

`plan`, `run`, `collect` and `post` take `--sims` and `--points` to work on some
rows or points without editing the matrix:

```console
cfd-matrix run polar --sims 1001 1002
cfd-matrix collect polar --points 1001/DP-M200AL+020
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Done. |
| 1 | `collect` or `post`: at least one point failed. `diff`: the two runs differ. |
| 2 | Refused, or a `run` in which a point failed. The message says what is at fault. |
| 3 | `collect` or `post`: no point failed and cluster jobs are still outstanding. |

[Commands](cli.md) lists every command with every option.
