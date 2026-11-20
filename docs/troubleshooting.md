# Troubleshooting

This page explains the messages that stop a command, by their text. Search the
page for a few words of the message you see. In each entry, the parts that
change from case to case are replaced by `...` or shown with example values.

A refusal changes nothing: correct the cause and run the same command again.
The exit code of a refusal is 2.

## Creating a workspace

```text
.../campaign/init.toml already exists; choose a new workspace root
```

The folder is already a workspace. A workspace keeps its solver, so `init`
never writes over it. Create the new workspace in another folder.

```text
--solver fluent: unknown solver 'fluent'; available solvers: flightstream, mock, openfoam, pyfr, qprop, su2, xfoil
```

`--solver` takes one of the listed names. A name in capitals, such as `SU2`, is
accepted as `su2`.

```text
.../plain is not a cfdbender workspace (no init.toml); run cfd-workspace init or pass --workspace
```

A `cfd-matrix` command was started in a folder that is not a workspace. Every
command refuses it in these words, also `status`, `log` and `history`. Change to
the workspace folder, or name it with `--workspace PATH`.

An empty answer from `cfd-matrix status` keeps its heading and totals and says why: `no recorded sim yet: run a
matrix first`, or `no failed point` for `--failed`. `cfd-matrix plan` and `run`
say `no active row: every row of polar.fs has RUN = 0` when there is nothing to
plan.

## Planning

```text
no matrix polar in inputs/matrices/ (found: full) and no polar.fs in the workspace root
```

The name after `plan` is the file name of the matrix without `.fs`. The message
lists the matrices it found. Check the spelling and the folder.

```text
inputs/matrices/typo.fs: unsupported columns: FLIGHT_CONDITON (closest known column: FLIGHT_CONDITION)
```

A column name in the header is misspelt or unknown. The message proposes the
closest name, and says `missing columns: ...` first when a required column is
absent too. [Matrix columns](matrix-columns.md) lists them.

```text
inputs/matrices/polar.fs: columns are separated by |, not by commas (POL | RUN | SETUP | ...)
```

A matrix uses `|` between columns, not commas or tabs.

```text
inputs/matrices/polar.fs, data row 1 (POL 2001): SETUP s002 names inputs/setups/s002.toml, which does not exist
```

The row names a setup, reference or pproc file that does not exist. Here the
`SETUP` cell is `002` and `inputs/setups/s002.toml` is missing; `REF` and `PPROC`
are worded the same. Every refusal of a row starts like this, with the path of
the matrix, the row counted from the first row under the header, and its `POL`.

```text
inputs/matrices/polar.fs, data row 1 (POL 6001): only one FLIGHT_CONDITION key may sweep
```

A row can sweep one variable. Write one row per value of the second variable.

```text
inputs/matrices/polar.fs, data row 1 (POL 4001): POL 4001: sweep values 0.0 and 0.01 of ALPHA round to the same point name M200AL+000; a datapoint folder is named from its flight condition, so give the values distinct names (for example further apart)
```

Two sweep values are closer than the name of a point can tell apart. For
`ALPHA` the name keeps one decimal. [Run again, resume, records](records.md)
gives the precision of each key.

```text
--sims 9999: no simulation with that POL
```

`--sims` takes `POL` values of the matrix you named.

```text
no active row: every row of polar.fs has RUN = 0
```

This is not a refusal: every row of the matrix has `RUN` set to `0`, so there
is nothing to plan.

```text
inputs/matrices/polar.fs, data row 1 (POL 1001): inputs/setups/s001.toml needs iterations
```

A message that starts with the matrix path and `data row N (POL P):` and then
names a file and a key comes from the rules of your solver. It says `needs`
when the key is absent, names the file the key belongs to (`inputs/setups/`,
`inputs/references/` or `inputs/pproc/`, or `FLIGHT_CONDITION`), and for an
unknown key proposes the closest known one. The page of the solver lists the
keys it reads and the values it accepts.

```text
POL 1001: 'area' of inputs/references/r001.toml is skipped; XFOIL does not read it (reads only length of this file)
```

A key that a file holds and the solver never reads is not an error: the plan
warns once and goes on without it.

```text
SOLVER_BUILD 8.3.0 is carried but no inputs/executables.toml declares it; a local run uses the executable the setup names
```

A row names a build, and the workspace declares no builds. `plan` warns; when
`inputs/executables.toml` exists and lacks the build, `plan` refuses with the
sentence that `run` would give. See [Executables](executables.md).

## Running

```text
matrix not run: no plan for this matrix, and a run needs one. Run `cfd-matrix plan <matrix>` first: ...
```

Run `cfd-matrix plan` on the matrix, then `run` again.

```text
matrix not run: input changed since it was planned: inputs/setups/s001.toml (point 1001/DP-M200AL+000). Plan it again.
```

A file was edited after the last `plan`. Plan again so that the check of the
inputs covers the file as it is now, then run.

```text
matrix not run: the matrix changed since it was planned: ...
```

The same, for the matrix file itself.

```text
matrix not run: point '1001/DP-M200AL+000' already recorded COMPLETED; use --resume to run only the points not yet recorded, or --force-rerun POINT to redo one
```

The point has run before, and a finished point is never replaced by accident.
Add `--resume` to run only the points that have no record, or
`--force-rerun 1001/DP-M200AL+000` to archive this point and run it again. See
[Run again, resume, records](records.md).

```text
  1001/DP-M200AL+000  skipped: COMPLETED
  1001/DP-M200AL+040  skipped: COMPLETED failed; redo it with --force-rerun 1001/DP-M200AL+040
```

Under `--resume`, a point that already completed is skipped, also when its
verdict is `failed`; the line of a failed point says how to run it again.

```text
SOLVER_BUILD '1.0' is not declared under [su2] in inputs/executables.toml (declared: 9.9); add it with its executable path
```

The row names a build that the workspace does not declare. Add the build to
`inputs/executables.toml` or correct the `SOLVER_BUILD` cell. `plan` gives this
refusal as well as `run`. See [Executables](executables.md).

```text
matrix not run: SU2 executable not found: 'SU2_CFD'; set executable in the [solver] table of the setup file, or declare SOLVER_BUILD in inputs/executables.toml
```

`plan --local` and `run --local` check the machine before they record
anything: the solver program (every solver gives this one sentence) and the free memory.
When one of them fails the run is refused, no record is
written, and the same command works once the cause is gone.

```text
matrix not run: another cfdbender solver run holds this machine (one run at a time); wait for it to end, then run again
```

This refusal occurs only when the one-run-at-a-time guard is enabled and another
guarded run holds the lease. Wait for it to end. The guard is off by default;
enable it with `cfd-matrix run MATRIX --local --one-run-at-a-time`, or add
`one_run_at_a_time = true` under `[solver]` in `init.toml`.

```text
matrix not run: free RAM ... GiB minus estimate ... GiB would breach the 0.5 GiB floor; refusing ...
```

The machine does not have enough free memory to start the point and keep
0.5 GiB free. Close other programs or run on a larger machine.

```text
matrix run with failures
```

At least one point was recorded as failed; the lines above it give the reason
for each. The other points ran. The next section covers the reasons.

## A point recorded as failed

`cfd-matrix status --failed --points` lists the failed points with their
error.

```text
FAILED_EXECUTION: SOLVER_BUILD mismatch: the row names '8.2.0' but the solver reported version '8.3.0'; ...
```

The executable that ran is not the build the row names. See
[Executables](executables.md).

```text
FAILED_EXECUTION: exit code 1; numerical evidence failed
```

The solver ended with an error. Read `stdout.log` and `stderr.log` in the
folder of the point, `sims/sim_<POL>/datapoints/DP-<name>/`.

```text
FAILED_INCOMPLETE_OUTPUT: missing declared output
```

The solver ended and did not write a file it should have. The same two logs
say why.

In each of these cases the point is recorded `COMPLETED` with the verdict
`failed`, and the console shows it by the word of its reason, here
`FAILED_EXECUTION`. A solver that is missing or busy is not one of these cases:
`run` refuses before it records anything. After you
correct the cause, run it again with
`cfd-matrix run MATRIX --force-rerun POINT`, or with `--force-rerun-all` when
every point failed for the same reason.

## Collecting

```text
collected 2, failed 1, outstanding 0
```

with exit code 1: one recorded point is failed. The tables are written and
hold every point. This is the normal result of a campaign with a failed point.

```text
2 recorded points are not in polar.fs any more (1001/DP-M200AL+000, ...); retire them with cfd-workspace delete-sims or cfd-workspace archive
```

The matrix was edited after a run so that points renamed or removed there still
have records. They are reported apart and are not counted among the failures.

Exit code 3 means that cluster jobs have not finished; see
[Cluster jobs](cluster-jobs.md).

## Cluster profiles

```text
.../inputs/hpc holds 2 profiles (h001.toml, h002.toml) and nothing says which cluster this machine is. ...
```

Name the profile with `--hpc h001`, or add `--local` to run on this machine.

## Records and storage

```text
refused 1001/DP-M200AL+040: marked failed
```

`mark-converged` does not change a point that you marked failed.

```text
the products of polar hold the deleted points; choose --matrix-products points-only or regenerate
```

See [Storage and sync](storage.md).

```text
sync needs inputs/sync-workspaces.toml: [workspaces] name = "path"
```

See [Storage and sync](storage.md).

## Warnings

```text
inputs/geometries/ is a legacy mesh folder name; move its files to inputs/mesh/
```

The mesh was found in a folder with an earlier name. It is read, and the
warning repeats until you move the files to `inputs/mesh/`.

Every warning is also written to `logs/warnings.log`. `--warning-off`, placed
before the command name, keeps warnings out of the console.
