# Command line

cfdbender has two commands. `cfd-workspace` creates a workspace and
`cfd-matrix` does everything else: it plans a matrix, runs it, follows it and
collects its results. Both work the same way for every solver, because they
read the workspace and its records, never the solver.

```console
cfd-workspace init campaign --solver su2
cd campaign
cfd-matrix plan polar
cfd-matrix run polar
cfd-matrix status
cfd-matrix collect polar
```

`polar` is the matrix `inputs/matrices/polar.fs`. Every command works on the
current folder; `--workspace PATH` names another one.

## Three rules that hold for every command

1. **The commands that correct records or manage storage preview first.** Each prints
   what it would do and changes nothing until you repeat it with `--apply`.
2. **Nothing is overwritten.** Before a record or a result is replaced, the old
   one is moved to `archive/<stamp>/`.
3. **Exit code 0 means done, 2 means refused.** A refusal names the file and the
   cell or key at fault and writes nothing. `collect` and `post` also return 1
   when a point failed and 3 when submitted work is still outstanding, and
   `diff` returns 1 when the two runs differ.

## The commands by stage

### Create

| Command | Does |
|---|---|
| `cfd-workspace init [ROOT] --solver NAME` | Creates the workspace folders and `init.toml`. Refuses a folder that is already a workspace. |

### Plan

| Command | Does |
|---|---|
| `cfd-matrix plan MATRIX` | Reads the matrix and every file it names, lists each point with the files the run would write, and saves the plan receipt `post/<matrix>/plan.json`. Launches nothing. |

Plan again after every edit of an input: `run` refuses a matrix whose inputs
changed since its receipt was written.

### Run

| Command | Does |
|---|---|
| `cfd-matrix run MATRIX` | Runs every planned point, on this machine or as cluster jobs when the workspace has an HPC profile. |
| `cfd-matrix run MATRIX --local` | Runs on this machine even when a profile exists. |
| `cfd-matrix run MATRIX --resume` | Skips the points already recorded and runs the rest. |
| `cfd-matrix run MATRIX --force-rerun POINT` | Archives one recorded point and runs it again. `--force-rerun-all` does so for every point. |

Without one of the last three options, a point that is already recorded
refuses the run, so finished evidence is never replaced by accident.

### Follow

These commands only read, and take `--json` or `--csv` to print rows for a program.

| Command | Prints |
|---|---|
| `cfd-matrix status` | Each simulation with the states of its points; `--points` lists each point, `--failed` keeps the simulations with a failed point. |
| `cfd-matrix show TARGET` | One simulation, point or run record in full. |
| `cfd-matrix history TARGET` | The current and the earlier records of a simulation or point, with their archives. |
| `cfd-matrix log` | The recorded events in time order; `--problems` keeps the failures, `--since 2h` the recent ones. |
| `cfd-matrix trace PRODUCT` | The runs, inputs and hashes behind a product file. |
| `cfd-matrix diff RUN_A RUN_B` | Two run records, field by field. |

### Collect and post

| Command | Does |
|---|---|
| `cfd-matrix collect [MATRIX]` | Reads the saved results, picks up finished cluster jobs and rebuilds the tables under `post/<matrix>/`. `--watch` repeats until no job is outstanding. |
| `cfd-matrix post [MATRIX]` | Rebuilds the same tables from the records and the saved outputs alone. |

### Correct the records

The `cfd-matrix` commands here preview unless `--apply` is given.

| Command | Does |
|---|---|
| `cfd-matrix mark-failed --sims POL` | Records your verdict that points failed, whatever the solver reported. |
| `cfd-matrix mark-converged --sims POL --reason TEXT` | Records your verdict that completed points converged. |
| `cfd-matrix restore KIND` | Brings the run records, the storage record, or a matrix's products or plan back from `archive/<stamp>/`. |
| `cfd-matrix rebuild` | Writes records for simulation folders that have none. |
| `cfd-workspace archive SIM_ID` | Moves a simulation's folder and records to `archive/<stamp>/`, so it can run again in a clean folder. It acts at once, without a preview. |

### Manage storage

| Command | Does |
|---|---|
| `cfd-matrix space-in-use` | Sizes by folder, by simulation and by file extension. |
| `cfd-matrix free-space RECIPE` | Deletes the files that `inputs/management/<RECIPE>.toml` names, except those a record needs. Preview unless `--apply`. |
| `cfd-matrix delete-sims --sims POL` | Deletes simulation folders and retires their records. Preview unless `--apply`. |
| `cfd-matrix sync LEVEL` | Brings records, products or simulation files from the workspaces named in `inputs/sync-workspaces.toml`. Preview unless `--apply`. |

## Choosing part of a matrix

`plan`, `run`, `collect` and `post` take `--sims` and `--points` to work on part
of a matrix without editing it:

```console
cfd-matrix run polar --sims 1001 1002
cfd-matrix run polar --points 1001/DP-M850RE500AL+025
```

`--sims` takes `POL` values, separated by spaces or commas. `--points` takes a
point name, with or without its `POL/` prefix. A name the matrix does not carry
is refused before anything runs.

## How much a command prints

| Option | Effect |
|---|---|
| `--log-level quiet` | Errors and the final summary only. |
| `--log-level normal` | Adds one line per point. This is the default. |
| `--log-level verbose`, `--verbose` | Adds details, such as the path of the plan receipt. |
| `--warning-off` (before the command name) | Keeps warnings out of the console and writes them to `logs/warnings.log`. |

## Every option

`cfd-workspace --help`, `cfd-matrix --help` and `cfd-matrix COMMAND --help`
print every option of a command with its meaning.

In a FlightStream workspace both commands hand over to pyflightstream, and the
options are the ones its own documentation lists; see
[FlightStream workspaces](flightstream-workspaces.md).
