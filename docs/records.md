# Run again, resume, records

This page shows how to run a matrix a second time, how to redo a point, and how
to read and correct the records that every run leaves.

`runs.json`, in the workspace root, holds one record per point: its inputs, its
status, its times and the verdict on its convergence. The commands on this page
read it or change it. You never edit it by hand.

Writers hold `runs.json.lock`, which names their process. A live writer or a
lock with no readable process identity refuses another writer. If the named
process is gone, the next writer takes over under an exclusive operating-system
lease and warns. You do not need to remove a departed writer's lock by hand.

The run manifest now uses schema version 2. Reading a valid version 1 manifest
from 0.5.x validates it, copies its exact bytes to `archive/<stamp>/runs.json`,
then atomically upgrades it under the writer lock. History and recorded evidence
are preserved. An invalid or future manifest is refused without rewriting it.
The first read therefore needs write access to the workspace. Restoring an old
archive also reads version 1 and writes version 2.

## The status of a record

| Status | Meaning |
|---|---|
| `PLANNED` | `run` registered the point and has not started it yet. |
| `RUNNING` | The solver is running the point on this machine, or the run was interrupted before the point ended. |
| `SUBMITTED` | The point was sent to a cluster and its result has not been collected. |
| `COMPLETED` | The point ended. Its verdict says how: `converged`, `iteration_limit` or `failed`. |

A point that ended without a usable result is recorded `COMPLETED` with the
verdict `failed`, and the record keeps the reason. The commands show such a
point by the word of its reason, `FAILED_EXECUTION` or another `FAILED_` word,
so that it is not read as a success. [Results and tables](results.md) explains
the verdicts.

## Restart from a saved solution

Use a row's `RESTART` entry to start new points from a recorded solution. To
continue each selected point from its own last solution instead:

```console
cfd-matrix run polar --local --restart
cfd-matrix run polar --local --restart --sims 1001
```

Every selected source must have saved solution state. A source from another point
must be completed. `run --local --restart` can also continue a point left
`RUNNING` by an interrupted local run, after you confirm its process has stopped.
Sources and local executables are checked before archiving. Add `--one-run-at-a-time`
to reserve the machine against other guarded local runs before archiving. The previous folders
and records are archived, then the solution files are copied into new point
folders. Existing evidence is preserved. This option cannot be combined with
`--resume` or the force-rerun options. `--resume` selects points to run and does
not request saved solution state.

### Starting a sliding-mesh row from its own frozen-rotor solution

`RESTART` = `MRF` is not a source point. It runs a steady stage on the row's own two-zone mesh in
the same point folder and job, and starts the time-accurate stage from its solution; the folder
holds `precursor.cfg`, `pre_zone_<i>.cfg`, `run_mrf.sh`, `precursor.log`, `precursor.csv` and the
staged `solution_<zone>_<level>.dat` files beside `su2.cfg`. The record of the point states
`precursor` (true) and, in its provenance, the iterations and the stop level. A row with `-`
records nothing new. The point is judged on the time-accurate stage as before (`su2.csv`); the
steady stage's history is evidence only. A failed steady stage ends the job with status 91 to 94
and a `PRECURSOR_FAILED` line in the point's `stderr.log`; the point is then recorded failed.
`--force-rerun-all` writes the stage files again with the new batch. See
[SU2 workspaces](su2-workspaces.md#run-types).

### A source that runs earlier in the same job

With `--batch N`, a source that is not completed yet is accepted when it is in the same job as the
point it feeds: the job runs the rows in sequence, so the solution exists when the dependent
starts. `cfd-matrix plan --batch N` lists each such pair. The job script reads the source's
`job.exit` and, only when it is 0, copies the solution into the dependent's folder with `cp`.
The job script needs only a POSIX shell: the source files and the names the solver reads are
written into it at submission, and no Python is called on the compute node. The staging needs
the solver's restart file to exist and not be empty, not collected products. When `sha256sum`
exists on the node, the script writes the digests of the staged files to
`restart-staged.sha256` in the dependent's folder; nothing reads that file. `su2.cfg`
is never edited: it was written at submission from the source's rendered configuration.
If the source did not end with status 0 (or its `job.exit` is missing, for a killed job), the
dependent is not run: its `job.exit` is 97 and its `stderr.log` holds one line saying why. If the
staging fails, `job.exit` is 98 and `stderr.log` holds one line naming the restart file that is
missing or empty. The job goes on with the next run. `collect` marks such a run failed with
that reason; a run the script ended with a failure and left no solver output is never left
outstanding. A source later in the job is moved ahead of its dependents (runs of a job share
no state, so no result changes). Points that feed each other are refused. This holds for SU2
with a steady source whose rendering fixes the restart file: `OUTPUT_FILES`, when stated,
holds `RESTART`, `WRT_BINARY_RESTART` is not `NO`, and `RESTART_FILENAME` is a plain name
ending in `.dat` (default `restart_flow.dat`); otherwise the run is refused naming the key.
A source that is neither completed nor in the same job (no `--batch`, another job, or left
out by `--points` or `--sims`) is refused as before. With `--force-rerun-all` both rows run
again in the job, fed from each other's new attempt.

A two-zone (sliding-mesh) point cannot take part in a `RESTART`, as the source or as the
dependent: it is refused at planning and submission; run the sliding point cold.

SU2 copies its configured `RESTART_FILENAME` (default `restart_flow.dat` or
`restart_flow.csv`) to `restart-input.dat` or `.csv` and sets `RESTART_SOL` and
`SOLUTION_FILENAME`. `READ_BINARY_RESTART` matches the saved binary or ASCII
output format. OpenFOAM copies every field of the latest positive time directory,
requires `U` and `p`, sets `startFrom latestTime` and adds the requested
iterations to the saved time. PyFR copies its most recently written `.pyfrs`
output to `restart.pyfrs` together with its saved native `mesh.pyfrm`, and runs
`pyfr restart` without reimporting the mesh. A different source mesh is refused.
Its writer saves a solution
every `tend` time units; a restarted run adds that duration to the saved time.
PyFR keeps its existing force window relative to the new start time.

The point fields `restart_from`, `restart_directory` and `restart_sha256` in
runs.json name the source point, its saved folder and the SHA-256 of each copied
solution input. A source whose solution is absent or changes before staging is
refused with the source point named. Other adapters warn and skip a restart;
`--warning-refusals` makes the skip a refusal.

For grouped forces, the adapter returns `NeutralResult.force_groups`; collect
places it at `points[].result.force_groups` in the campaign summary. Each
requested group and `TOTAL` carries coefficient values with explicit axes and
kinds, using the result's reference values. The per-group values are read from
the retained solver history files; no group CSV is written.

## Running a matrix again

A point that has a record refuses a second run:

```console
cfd-matrix run polar
```

```text
matrix not run: point '1001/DP-M200AL+000' already recorded COMPLETED; use --resume to run only the points not yet recorded, or --force-rerun POINT to redo one
```

Nothing ran and nothing changed. Choose one of the three options below.

### Run only what is new: `--resume`

```console
cfd-matrix run polar --resume
```

```text
  1001/DP-M200AL+000  skipped: COMPLETED
  1001/DP-M200AL+020  skipped: COMPLETED
  1001/DP-M200AL+040  skipped: COMPLETED
```

`--resume` skips every point recorded `COMPLETED` or `SUBMITTED`, whatever its
verdict, and runs the rest. Use it after adding rows or sweep values to a
matrix, and after a run that was interrupted. Plan the grown matrix first: the
points already recorded stay valid, because a point is stale when a file it read
(its setup, reference, pproc or geometry file) changed, never because the matrix
gained a row. `collect` and `post` read the old points as before. A skipped point whose verdict is
`failed` says how to run it again:

```text
  1001/DP-M200AL+040  skipped: COMPLETED failed; redo it with --force-rerun 1001/DP-M200AL+040
```

Fixing the cause of a failure and typing `--resume` therefore does not run the
failed point; name it with `--force-rerun`, or every failed point with
`--force-rerun-all`.

A `RUNNING` point is skipped because its local process or cluster job may still
exist. After confirming it has stopped, name it with `--force-rerun` to redo it.

### Redo one point: `--force-rerun`

```console
cfd-matrix run polar --force-rerun 1001/DP-M200AL+020
```

```text
archived previous evidence: /home/you/campaign/archive/20261113T135750Z
  1001/DP-M200AL+000  skipped: COMPLETED, not named by --force-rerun
  1001/DP-M200AL+040  skipped: COMPLETED, not named by --force-rerun
[1/1] 1001/DP-M200AL+020  running
[1/1] 1001/DP-M200AL+020  COMPLETED converged in 0 s
```

The point is archived before it runs again, and nothing is deleted:

- `runs.json` is copied to `archive/<stamp>/runs.json`;
- the point folder moves to `archive/<stamp>/sims/sim_<POL>/datapoints/<point>/`;
- the earlier record joins the history of the point.

`<stamp>` is the date and time of the move, in UTC. Repeat the option to redo
several points. The short name `DP-M200AL+020` is enough when only one sim of
the matrix has a point of that name.

### Redo every point: `--force-rerun-all`

```console
cfd-matrix run polar --force-rerun-all
```

Every selected point is run again whatever its recorded state: not yet run,
`SUBMITTED`, `RUNNING`, `COMPLETED`, failed, with or without a job on record. A recorded
point is archived first as `--force-rerun` archives it, so its `stdout.log`, `history.csv`,
`restart` folder and `job.exit` stay together in `archive/<stamp>/` and the new attempt
starts in an empty folder. A point that was `SUBMITTED` or `RUNNING` prints one warning
line: a job of the earlier submission may still be running, and cancelling it is your
action, because the library does not talk to the scheduler. The option works with
`--points`, `--sims` and `--batch` (plan the batch again first: it gets a new folder,
and the folder of the earlier batch stays where it is).
Only two things still refuse. The validity of the inputs and of the row refuses a point, for
example a missing mesh or an input changed after planning. A point folder or output that
exists with no record is data the registry does not know, so it refuses with
`existing output; refusing overwrite`; move or delete it yourself and run again. The option cannot be combined with
`--resume`, and cannot be combined with `--force-rerun`.

## Reading the records

These commands change nothing. Each takes `--json` or `--csv` to print its rows
for a program.

```console
cfd-matrix status --points
```

```text
SIM   POINT       MATRIX  STATUS     WAS  ITER  RESIDUAL  WALL  BUILD   WHERE
1001  M200AL+000  polar   COMPLETED       2     0.01      0.03  mock-1  local
1001  M200AL+020  polar   COMPLETED       2     0.01      0.03  mock-1  local
1001  M200AL+040  polar   COMPLETED       2     0.01      0.03  mock-1  local
3 datapoint(s): COMPLETED
plan/polar: made from the matrix on disk
post/polar: no post yet
```

| Command | Prints |
|---|---|
| `cfd-matrix status` | One row per polar, the points of one simulation of one matrix, as `pyfs-matrix status` prints it: `SIM`, `POLAR` (the name of its points with the swept field written `+sweep`), `SWEEP` (what varies), `POINTS` (recorded out of planned, `3/4`; only the recorded count when the matrix has no plan receipt), `MATRIX` and `STATUS`: the one word when every point shares it, otherwise the count of each (`COMPLETED 2, FAILED_MARKED 1, planned 1`). A point the plan receipt `post/<matrix>/plan.json` names and no record carries reads `planned`, in lower case because no recorded status is written so. Then the totals, and for each matrix one line on its plan (`made from the matrix on disk`, or `made from another revision of the matrix; plan it again`) and one on its post (when it was written, whether it is complete, how many recorded runs it does not hold yet). `--points` gives one row per point: `SIM POINT MATRIX STATUS WAS ITER RESIDUAL WALL BUILD WHERE`, with a trailing `ERROR` column when a shown point has one; `WAS` is what a marked point was before you marked it. `--failed` keeps the polars with a failed point. `--status WORD` keeps the polars with a point in that status or verdict, repeatable, and refuses a word no status takes; `FAILED_*` matches every word that starts with `FAILED_`. `--sims`, `--matrix`, `--json` (one document with the rows and the freshness) and `--csv` select and print the same rows. A sim that holds no record and no planned point is named on standard error; the exit code is 1 only when every sim named holds nothing. |
| `cfd-workspace show 1001/DP-M200AL+020` | The card of one point: status and verdict, solver and version, flight condition, iterations and coefficients once collected, folder, times and error if any. `show 1001` lists the points of the sim, one line each. `--verbose` prints the whole record: inputs, fingerprints, every field that has a value. |
| `cfd-workspace history 1001` | The current and the earlier records of a sim or a point, with the archive folder of each earlier one. |
| `cfd-workspace log` | What happened, in time order: the commands and, for each point, `registered`, `started` and `finished`. `--problems` keeps the failures, `--since 2h` the last two hours, `--open` the runs that started and have no end. |
| `cfd-workspace diff RUN_A RUN_B` | Two records, field by field. The exit code is 1 when they differ. The fingerprints and output paths, which differ whenever two points do, are folded into one line; `--verbose` lists them. |
| `cfd-workspace trace post/polar/polars/P1001-TOTAL-ACOEF.csv` | The runs behind a collected file, with the number of input files each was made from; `--verbose` lists the fingerprints. |

An empty answer keeps its heading and totals and says why: `no recorded sim yet: run a matrix first`, `no
failed point`, or `no recorded point matches the selection`. The times are
printed in the time of this machine, to the second; the records keep the full
UTC time, which `--json` prints.

An earlier record is named by its point and its archive stamp, as `history`
prints it:

```console
cfd-workspace history 1001/DP-M200AL+020
```

```text
run                                  kind      when                 status     classification  archive
1001/DP-M200AL+020                   current   2026-11-13 10:57:50  COMPLETED  converged
1001/DP-M200AL+020@20261113T135750Z  archived  2026-11-13 10:57:50  COMPLETED  converged       archive/20261113T135750Z
```

## Recording your own verdict

The verdict of a record comes from what the solver wrote. When you have looked
at a point and judge it differently, record your verdict. Both commands show
what they would do and change nothing until you add `--apply`.

```console
cfd-workspace mark-failed --sims 1001 --points DP-M200AL+040 --reason "residual stalled"
```

```text
would mark 1001/DP-M200AL+040 failed (was COMPLETED converged)
preview only; add --apply to change the records
```

```console
cfd-workspace mark-failed --sims 1001 --points DP-M200AL+040 --reason "residual stalled" --apply
```

```text
marking 1001/DP-M200AL+040 failed (was COMPLETED converged)
marked 1 point(s) failed; the manifest before is in /home/you/campaign/archive/20261113T140213Z
```

Without `--points`, every point of the named sims is marked. A point that is
`RUNNING` is refused with `stop the run first`: a live run would seal it over
the mark. After a mark, `status` and the `error` of the point in the collected
summary read `FAILED_MARKED: <your reason>`; `collect` keeps the reason and
the result keeps the class the solver's own evidence gives.

`cfd-workspace mark-converged --sims POL --reason TEXT` records that completed
points converged, for example a point that reached its iteration limit with
steady forces. It needs a reason. `collect` then reads the point as before: its
row carries your verdict in `classification`, and the result keeps the class the
solver's own evidence gives. It refuses a point that has not completed and
a point that was marked failed:

```text
refused 1001/DP-M200AL+040: marked failed
nothing marked
```

Your verdicts are kept in `marks.json` with their reason and their time, and
`runs.json` as it was before the change is copied to `archive/<stamp>/`.

## Collecting over a file you edited: `--overwrite`

`collect` and `post` replace a table only when it is the file the manifest
(`post/<matrix>/products.json`) recorded. A file you edited by hand, or one another tool left
under `post/<matrix>/`, stops the command, and the message names the file:

```text
<workspace>/post/full/polars/P7001-TOTAL-DIM.csv: unrecorded or edited output; refusing overwrite; move it aside, or run the command again with --overwrite to replace it
```

`cfd-matrix collect --overwrite` and `cfd-matrix post --overwrite` replace every such file
and go on through every point: nothing stops in the middle. The manifest is then written from
the new files, so the recorded digest of a replaced file is that of the new bytes, and the
next collect without the flag is clean. A linked file is still refused. `--archive` keeps a
copy of what was replaced under `archive/<stamp>/`.

## Bringing a file back

```console
cfd-workspace restore runs
```

```text
would restore runs from archive/20261113T140213Z/runs.json
preview only; add --apply to restore
```

`restore` takes what to bring back from the latest archive folder that holds
it, or from the one you name with `--stamp`:

| Word | Restores |
|---|---|
| `runs` | `runs.json` |
| `storage` | `storage_management.json` |
| `products` | the collected files of the matrix named with `--matrix` |
| `plan` | the plan receipt of the matrix named with `--matrix` |

## Records for folders that have none

`cfd-workspace rebuild --matrix NAME` writes records for sim folders that are on
disk and have no record. It reads the plan of the matrix to know which points to
expect. It previews unless `--apply` is given.

## Clearing a sim to run it again

```console
cfd-workspace archive 1001
```

```text
archived sim 1001 (3 point(s)) to archive/20261113T140557Z; sims/sim_1001/ is removed
```

`archive` moves the whole folder of a sim to `archive/<stamp>/` and its records
to the history, so the sim can run again in a clean folder. Unlike the commands
above it acts at once, without a preview, as `archive` does in pyflightstream.
To see what it would move, look first with `cfd-matrix status --points --sims
1001`. It does not look at the status of the runs: do not archive a sim that is
still running.

## How a point is named

The folder of a point is `sims/sim_<POL>/datapoints/DP-<name>/`. The name is
built from the `FLIGHT_CONDITION` of the row: each key of the cell, in the
order of the cell, becomes a short code followed by its value, multiplied by a
scale, rounded and padded with zeros. The same flow condition has the same name
in every workspace.

| Key | Code | Scale | Width | Example |
|---|---|---|---|---|
| `MACH` | `M` | 1000 | 3 | 0.85 gives `M850` |
| `TASmps` | `V` | 10 | 4 | 34 m/s gives `V0340` |
| `REmi` | `RE` | 100 | 3 | 5 million gives `RE500` |
| `ALTFT` | `ALT` | 1 | 5 | 5000 ft gives `ALT05000` |
| `dISA` | `DT` | 10 | 4, signed | 10 K gives `DT+100` |
| `RHOkgm3`, `MUPas`, `ASMPS`, `TK`, `PPA` | `RHO`, `MU`, `A`, `T`, `PS` | 10000, 1e9, 10, 10, 1 | 5, 5, 4, 4, 6 | 1.225 kg/m3 gives `RHO12250` |
| `ALPHA`, `BETA` | `AL`, `BE` | 10 | 4, signed | 2.5 degrees gives `AL+025` |
| `ADVANCE_RATIO` or `J` | `J` | 100 | 4, signed | 1.0 gives `J+100` |
| `RPM` | `RPM` | 1 | 5 | 800 rev/min gives `RPM00800` |
| `roll_rate`, `pitch_rate`, `yaw_rate` | `P`, `Q`, `R` | 10 | 4, signed | 2.5 gives `P+025` |

`DP-M850RE500AL+025` is Mach 0.85, a Reynolds number of 5 million and alpha
2.5 degrees.

- A signed field counts its sign in the width.
- A key without a code, such as a key that only one solver reads, is written as
  its name in upper case followed by its value. A row with no
  `FLIGHT_CONDITION` is `DP-NOCOND`.
- The order of the keys in the cell is the order in the name, so reordering the
  cell renames the points.
- Two sweep values of one row that round to the same name are refused by
  `plan`, with both values in the message.

A workspace made with an early version can hold folders named `DP-` followed
by 20 letters and digits. Their records keep working. Planning such a matrix
again registers its points under the new names, so finish or archive the old
campaign first.

## Management path safety

Management destinations must stay inside the workspace without passing through a
symbolic link or Windows junction. Sync skips linked source folders; restore
refuses linked source trees.

Restore refuses symbolic links and junctions anywhere in an archived product tree before replacing live products.

Marks apply to the execution they judged. A forced rerun keeps the old mark in the journal but does not inherit its verdict.

The grown-matrix digest exception applies to matrix inputs. A tracked geometry file with the .fs extension is still checked for changed bytes.
