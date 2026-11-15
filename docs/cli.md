# Commands

This page lists every command and option of the two programs. `cfd-workspace`
creates a workspace and `cfd-matrix` does everything else: it plans a matrix,
runs it, follows it and collects its results. Both work the same way for every
solver, because they read the workspace and its records, never the solver.

```console
cfd-workspace init campaign --solver NAME
cd campaign
cfd-matrix plan polar
cfd-matrix run polar
cfd-matrix status
cfd-matrix collect polar
```

`NAME` is one of the solvers that `cfd-workspace init --help` lists. `polar` is
the matrix `inputs/matrices/polar.fs`. Every command works on the current
folder; `--workspace PATH` names another one. [The workflow](workflow.md)
explains what each stage reads and writes.

## Three rules that hold for every command

1. **The commands that correct records or manage storage preview first.** Each
   prints what it would do and changes nothing until you repeat it with
   `--apply`. The one exception is `cfd-workspace archive`, which acts at once.
2. **Nothing is overwritten.** Before a record or a point folder is replaced,
   the old one is moved to `archive/<stamp>/`.
3. **Exit code 0 means done, 2 means refused or failed.** A refusal names what
   is at fault and writes nothing. `collect` and `post` also return 1 when a
   point failed and 3 when submitted work is still outstanding, and `diff`
   returns 1 when the two runs differ.

## Choosing part of a matrix

`plan`, `run`, `collect` and `post` take `--sims` and `--points` to work on part
of a matrix without editing it:

```console
cfd-matrix run polar --sims 1001 1002
cfd-matrix run polar --points 1001/DP-M200AL+020
```

`--sims` takes `POL` values, separated by spaces or commas. `--points` takes a
point name, with or without its `POL/` prefix. A name the matrix does not carry
is refused before anything runs.

## Output for a program

The commands of the Follow stage print tables for a reader. Add `--json` or
`--csv` to get the same rows for a program.

The sections below are written from the programs themselves, so they hold
exactly the arguments and options that `--help` prints.

<!-- BEGIN GENERATED COMMAND REFERENCE -->

## Options before the command name

These go between the program name and the command, for example
`cfd-matrix --warning-off run polar`.

| Option | Meaning |
|---|---|
| `cfd-workspace --warning-off` | Hide console warnings; they are kept in logs/warnings.log |
| `cfd-matrix --warning-off` | Hide console warnings; they are kept in logs/warnings.log |

## Create

### `cfd-workspace init`

Create the full campaign tree (inputs library, sims, post, archive).

```text
cfd-workspace init [ROOT] [options]
```

| Argument or option | Meaning |
|---|---|
| `[ROOT]` | Campaign root to create or complete (default: the current directory) |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory); the positional root names the same folder |
| `--solver NAME` | Solver for this workspace, one of: flightstream, mock, openfoam, pyfr, qprop, su2, xfoil |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

## Plan

### `cfd-matrix plan`

Print proposed work and write the plan receipt post/&lt;matrix&gt;/plan.json.

```text
cfd-matrix plan MATRIX [options]
```

| Argument or option | Meaning |
|---|---|
| `MATRIX` | Matrix name in inputs/matrices/ (.fs is optional) |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--sims POL [POL ...]` | Only these simulations, by POL, space or comma separated (--sims 1001 1002 or --sims 1001,1002); an id the matrix lacks is refused |
| `--points NAME [NAME ...]` | Only these points, named DP-name or POL/DP-name, within --sims when given; a name no point carries is refused |
| `--hpc NAME` | The HPC profile inputs/hpc/&lt;NAME&gt;.toml to use when the workspace holds several |
| `--local` | Keep work on THIS machine instead of submitting it |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

## Run

### `cfd-matrix run`

Run every active point; on this machine a row's SOLVER_BUILD picks its executable from inputs/executables.toml and executables.local.toml.

```text
cfd-matrix run MATRIX [options]
```

| Argument or option | Meaning |
|---|---|
| `MATRIX` | Matrix name in inputs/matrices/ (.fs is optional) |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--one-run-at-a-time` | Reserve this machine for one guarded local run at a time |
| `--resume` | Skip points already recorded COMPLETED or SUBMITTED and run the rest, so a grown matrix runs only its new points; a point left RUNNING by a crashed local run runs again when no solver process holds this host's slot. Without it a recorded point refuses the run |
| `--force-rerun POINT` | Redo this recorded point, named POL/DP-name or DP-name; repeat for several. runs.json is copied to archive/&lt;stamp&gt;/runs.json, the point folder moves to archive/&lt;stamp&gt;/ and its record joins the runs.json history; nothing is deleted |
| `--force-rerun-all` | Redo every recorded point of the matrix, archived as --force-rerun archives; a SUBMITTED job is not cancelled, cancel it in the scheduler first |
| `--sims POL [POL ...]` | Only these simulations, by POL, space or comma separated (--sims 1001 1002 or --sims 1001,1002); an id the matrix lacks is refused |
| `--points NAME [NAME ...]` | Only these points, named DP-name or POL/DP-name, within --sims when given; a name no point carries is refused |
| `--hpc NAME` | The HPC profile inputs/hpc/&lt;NAME&gt;.toml to use when the workspace holds several |
| `--local` | Keep work on THIS machine instead of submitting it |
| `--no-local` | The opposite of --local: submit when a profile exists (the default) |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

## Follow

### `cfd-matrix status`

List every recorded sim per matrix with its states, filterable.

```text
cfd-matrix status [options]
```

| Argument or option | Meaning |
|---|---|
| `--sims POL [POL ...]` | Only these simulations, by POL, space or comma separated (--sims 1001 1002 or --sims 1001,1002); an id the matrix lacks is refused |
| `--matrix MATRIX` | Only this recorded matrix |
| `--status WORD` | Only sims with a point in this status or classification (repeatable; a trailing * matches every word it begins) |
| `--failed` | Only sims with a failed point |
| `--points` | One row per point |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--json` | Print a JSON document of the rows |
| `--csv` | Print CSV; nested cells are JSON |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix show`

Print one sim, point or run record in full.

```text
cfd-matrix show TARGET [options]
```

| Argument or option | Meaning |
|---|---|
| `TARGET` | A sim (POL), a point id (POL/DP-name) or a run id |
| `--point POINT` | Only the run of this point name within the target |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--json` | Print a JSON document of the rows |
| `--csv` | Print CSV; nested cells are JSON |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix history`

Print the history of a sim or point: reruns, archives and deletions.

```text
cfd-matrix history TARGET [options]
```

| Argument or option | Meaning |
|---|---|
| `TARGET` | A sim (POL) or a point id (POL/DP-name) |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--json` | Print a JSON document of the rows |
| `--csv` | Print CSV; nested cells are JSON |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix log`

List the recorded events of runs, storage calls or a post log.

```text
cfd-matrix log [options]
```

| Argument or option | Meaning |
|---|---|
| `--post MATRIX` | The saved post log of a matrix |
| `--storage` | List the recorded storage calls |
| `--sims POL [POL ...]` | Only these simulations, by POL, space or comma separated (--sims 1001 1002 or --sims 1001,1002); an id the matrix lacks is refused |
| `--run RUN` | Only events naming this run id |
| `--stage STAGE` | Only this stage (storage: command) |
| `--since SINCE` | Events since an ISO time or duration, such as 2h or 7d |
| `--problems` | Only problems and failed outcomes |
| `--open` | Runs started without a finish in the records; never asks a scheduler |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--json` | Print a JSON document of the rows |
| `--csv` | Print CSV; nested cells are JSON |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix trace`

Follow a product file to the run, inputs and hashes that made it.

```text
cfd-matrix trace PRODUCT [options]
```

| Argument or option | Meaning |
|---|---|
| `PRODUCT` | A product path in the workspace, such as post/&lt;matrix&gt;/... |
| `--run RUN` | Trace this run id instead of looking the product up |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--json` | Print a JSON document of the rows |
| `--csv` | Print CSV; nested cells are JSON |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix diff`

Compare two runs field by field; exit 1 when they differ.

```text
cfd-matrix diff RUN_A RUN_B [options]
```

| Argument or option | Meaning |
|---|---|
| `RUN_A` | A run id (see history) |
| `RUN_B` | A run id (see history) |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--json` | Print a JSON document of the rows |
| `--csv` | Print CSV; nested cells are JSON |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

## Collect and post

### `cfd-matrix collect`

Read saved results and rebuild post/&lt;matrix&gt;/campaign_summary.json, campaign_sweep.csv and polars/P&lt;POL&gt;.csv.

```text
cfd-matrix collect [MATRIX] [options]
```

| Argument or option | Meaning |
|---|---|
| `[MATRIX]` | Recorded matrix name (default: all) |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--sims POL [POL ...]` | Only these simulations, by POL, space or comma separated (--sims 1001 1002 or --sims 1001,1002); an id the matrix lacks is refused |
| `--points NAME [NAME ...]` | Only these points, named DP-name or POL/DP-name, within --sims when given; a name no point carries is refused |
| `--hpc NAME` | The HPC profile inputs/hpc/&lt;NAME&gt;.toml to use when the workspace holds several |
| `--local` | ==SUPPRESS== |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--archive` | Keep a copy of each replaced summary, table and polar under archive/&lt;stamp&gt;/; without it they are overwritten in place |
| `--watch` | Keep sweeping until no submitted point is outstanding, instead of once |
| `--interval INTERVAL` | Seconds between the two observations that decide an output is settled (size and modification time stable); without it or --watch no settle check is made |
| `--watch-interval WATCH_INTERVAL` | Seconds between sweeps under --watch (default: 60) |
| `--rounds ROUNDS` | Stop a watch after this many sweeps, whatever is still outstanding |

### `cfd-matrix post`

Rebuild every product under post/&lt;matrix&gt;/ from runs.json and the saved outputs, without running anything.

```text
cfd-matrix post [MATRIX] [options]
```

| Argument or option | Meaning |
|---|---|
| `[MATRIX]` | Recorded matrix name (default: all) |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--sims POL [POL ...]` | Only these simulations, by POL, space or comma separated (--sims 1001 1002 or --sims 1001,1002); an id the matrix lacks is refused |
| `--points NAME [NAME ...]` | Only these points |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--archive` | Keep a copy of each replaced product under archive/&lt;stamp&gt;/ |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

## Correct the records

### `cfd-matrix mark-failed`

Mark points FAILED with a reason, whatever they ended in (preview unless --apply).

```text
cfd-matrix mark-failed [options]
```

| Argument or option | Meaning |
|---|---|
| `--sims POL [POL ...]` | The simulations to mark, by POL |
| `--reason REASON` | Why, recorded as given |
| `--points NAME [NAME ...]` | Only these points of the sims |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--apply` | Change files; without it the command previews and changes nothing |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix mark-converged`

Record the verdict that completed points converged (preview unless --apply).

```text
cfd-matrix mark-converged [options]
```

| Argument or option | Meaning |
|---|---|
| `--sims POL [POL ...]` | The simulations to mark, by POL |
| `--reason REASON` | Why, recorded as given |
| `--points NAME [NAME ...]` | Only these points of the sims |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--apply` | Change files; without it the command previews and changes nothing |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix restore`

Restore archived inputs or results from archive/&lt;stamp&gt; (preview unless --apply).

```text
cfd-matrix restore {runs,storage,products,plan} [options]
```

| Argument or option | Meaning |
|---|---|
| `{runs,storage,products,plan}` | The file to restore |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--stamp STAMP` | The archive stamp to restore from (default: the latest) |
| `--matrix MATRIX` | For products and plan: the matrix whose file to restore |
| `--apply` | Change files; without it the command previews and changes nothing |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix rebuild`

Rebuild run records from the sim folders on disk (preview unless --apply).

```text
cfd-matrix rebuild [options]
```

| Argument or option | Meaning |
|---|---|
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--out NAME` | Write the rebuilt records to this manifest in the workspace root |
| `--all-sims` | Rebuild every sim folder on disk, recorded or not |
| `--sims POL [POL ...]` | Rebuild these simulations |
| `--matrix MATRIX` | The matrix the simulations ran from |
| `--apply` | Change files; without it the command previews and changes nothing |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-workspace archive`

Moves sims/sim_&lt;sim_id&gt;/ of a recorded simulation into archive/&lt;stamp&gt;/ and moves its records from runs.json to the manifest history, so the simulation can be run again into a clean folder while its evidence stays. Refuses a simulation the manifest does not record. It does not read the status of the runs: do not archive a simulation whose run is still going. It acts at once, with no preview, as pyflightstream's archive does; look first with cfd-matrix status --points --sims &lt;POL&gt;.

```text
cfd-workspace archive [ROOT] SIM_ID [options]
```

| Argument or option | Meaning |
|---|---|
| `[ROOT]` | The campaign root, an alias of --workspace |
| `SIM_ID` | The simulation to archive, as the manifest records it (sims/sim_&lt;sim_id&gt;/) |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

## Manage storage

### `cfd-matrix space-in-use`

Report disk space by folder, by sim and by extension.

```text
cfd-matrix space-in-use [options]
```

| Argument or option | Meaning |
|---|---|
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--top TOP` | Rows shown per grouping (default: 15) |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix free-space`

Run a storage recipe inputs/management/&lt;recipe&gt;.toml (preview unless --apply).

```text
cfd-matrix free-space [RECIPE] [options]
```

| Argument or option | Meaning |
|---|---|
| `[RECIPE]` | The recipe id: a file stem in inputs/management/ (with --list alone: list them) |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--apply` | Change files; without it the command previews and changes nothing |
| `--list` | Print every path a step touches, with its size and what happens to it |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix delete-sims`

Delete simulations, their folders and their records (preview unless --apply).

```text
cfd-matrix delete-sims [options]
```

| Argument or option | Meaning |
|---|---|
| `--sims POL [POL ...]` | The simulations to delete |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--matrix-products {points-only,regenerate}` | What happens to a matrix product that also holds the deleted points |
| `--force` | Delete a sim whatever the status of its records; without it a SUBMITTED point refuses |
| `--apply` | Change files; without it the command previews and changes nothing |
| `--archive` | Keep replaced matrix products when regenerating |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

### `cfd-matrix sync`

Bring records and results from the workspaces of inputs/sync-workspaces.toml.

```text
cfd-matrix sync {runs,post,all} [options]
```

| Argument or option | Meaning |
|---|---|
| `{runs,post,all}` | What to bring; each level adds to the one before |
| `--workspace WORKSPACE` | Campaign workspace (default: the current directory) |
| `--from SOURCE` | One workspace name from sync-workspaces.toml (default: every one) |
| `--apply` | Change files; without it the command previews and changes nothing |
| `--prefer-other` | On a record conflict, take the other record (levels post and all need --overwrite) |
| `--overwrite` | On a file conflict, archive main's copy and take the other's |
| `--restore` | With --apply, remind to rebuild the records of sim folders no record carries |
| `--include-archives` | Also bring the files under folders named archive |
| `--runs NAME` | Another run manifest in the workspace root (default: runs.json) |
| `--verbose` | Print details too; the same as --log-level verbose |
| `--log-level {quiet,normal,verbose}` | Console verbosity: quiet prints errors and the final summary, normal adds progress and per-point lines (default), verbose adds details |

<!-- END GENERATED COMMAND REFERENCE -->

## FlightStream workspaces

In a FlightStream workspace the two programs take the commands and options of
the FlightStream support instead of the ones above; see
[FlightStream](flightstream-workspaces.md).
