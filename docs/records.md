# Run again, resume, records

This page shows how to run a matrix a second time, how to redo a point, and how
to read and correct the records that every run leaves.

`runs.json`, in the workspace root, holds one record per point: its inputs, its
status, its times and the verdict on its convergence. The commands on this page
read it or change it. You never edit it by hand.

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

Every recorded point of the matrix is archived and run again. A job that is
still `SUBMITTED` on a cluster is not cancelled: cancel it in the scheduler
first. This option cannot be combined with the other two.

## Reading the records

These commands change nothing. Each takes `--json` or `--csv` to print its rows
for a program.

```console
cfd-matrix status --points
```

```text
sim   matrix  point               status     classification  error
1001  polar   1001/DP-M200AL+000  COMPLETED  converged
1001  polar   1001/DP-M200AL+020  COMPLETED  converged
1001  polar   1001/DP-M200AL+040  COMPLETED  converged
```

| Command | Prints |
|---|---|
| `cfd-matrix status` | One line per sim with the count of its points in each status. `--points` gives one line per point. `--failed` keeps the sims with a failed point. `--status WORD` keeps the sims with a point in that status or verdict; `fail*` matches every word that starts with `fail`. |
| `cfd-matrix show 1001/DP-M200AL+020` | The card of one point: status and verdict, solver and version, flight condition, iterations and coefficients once collected, folder, times and error if any. `show 1001` lists the points of the sim, one line each. `--verbose` prints the whole record: inputs, fingerprints, every field that has a value. |
| `cfd-matrix history 1001` | The current and the earlier records of a sim or a point, with the archive folder of each earlier one. |
| `cfd-matrix log` | What happened, in time order: the commands and, for each point, `registered`, `started` and `finished`. `--problems` keeps the failures, `--since 2h` the last two hours, `--open` the runs that started and have no end. |
| `cfd-matrix diff RUN_A RUN_B` | Two records, field by field. The exit code is 1 when they differ. The fingerprints and output paths, which differ whenever two points do, are folded into one line; `--verbose` lists them. |
| `cfd-matrix trace post/polar/campaign_sweep.csv` | The runs behind a collected file, with the number of input files each was made from; `--verbose` lists the fingerprints. |

An empty answer says why: `no recorded sim yet: run a matrix first`, `no
failed point`, or `no recorded point matches the selection`. The times are
printed in the time of this machine, to the second; the records keep the full
UTC time, which `--json` prints.

An earlier record is named by its point and its archive stamp, as `history`
prints it:

```console
cfd-matrix history 1001/DP-M200AL+020
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
cfd-matrix mark-failed --sims 1001 --points DP-M200AL+040 --reason "residual stalled"
```

```text
would mark 1001/DP-M200AL+040 failed (was COMPLETED converged)
preview only; add --apply to change the records
```

```console
cfd-matrix mark-failed --sims 1001 --points DP-M200AL+040 --reason "residual stalled" --apply
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

`cfd-matrix mark-converged --sims POL --reason TEXT` records that completed
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

## Bringing a file back

```console
cfd-matrix restore runs
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

`cfd-matrix rebuild --matrix NAME` writes records for sim folders that are on
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
