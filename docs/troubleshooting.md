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
.../init.toml: unknown solver 'fluent'; available solvers: flightstream, mock, openfoam, pyfr, qprop, su2, xfoil
```

`--solver` takes one of the listed names, in lower case.

```text
.../init.toml: cannot read: [Errno 2] No such file or directory: ...; available solvers: ...
```

A `cfd-matrix` command was started in a folder that is not a workspace. Change
to the workspace folder, or name it with `--workspace PATH`.

An empty table from `cfd-matrix status` or `cfd-matrix log`, with a header and
no row, has the same two causes: the folder is not a workspace, or nothing has
run in it yet.

## Planning

```text
matrix 'polar': no inputs/matrices/polar.fs in inputs/matrices/ and no polar.fs in the workspace root
```

The name after `plan` is the file name of the matrix without `.fs`. Check the
spelling and the folder.

```text
...typo.fs: missing columns: (none); unsupported columns: FLIGHT_CONDITON (closest known column: FLIGHT_CONDITION)
```

A column name in the header is misspelt or unknown. The message proposes the
closest name. [Matrix columns](matrix-columns.md) lists them.

```text
...: missing columns: POL, PPROC, REF, RUN, SET; unsupported columns: ...
```

When every column is reported missing, the header is not separated by `|`.
A matrix uses `|` between columns, not commas or tabs.

```text
data row 1, POL 2001: inputs/setups/s002.toml: cannot read: [Errno 2] No such file or directory: ...
```

The row names a setup, reference or pproc file that does not exist. Here the
`SET` cell is `002` and `inputs/setups/s002.toml` is missing. Rows are counted
from the first row under the header.

```text
... data row 1, POL '6001': only one FLIGHT_CONDITION key may sweep
```

A row can sweep one variable. Write one row per value of the second variable.

```text
data row 1, POL 4001: POL 4001: sweep values 0.0 and 0.01 of ALPHA round to the same point name M200AL+000; a datapoint folder is named from its flight condition, so give the values distinct names (for example further apart)
```

Two sweep values are closer than the name of a point can tell apart. For
`ALPHA` the name keeps one decimal. [Run again, resume, records](records.md)
gives the precision of each key.

```text
--sims 9999: no simulation with that POL
```

`--sims` takes `POL` values of the matrix you named.

```text
0 point(s)
```

This is not a refusal: every row of the matrix has `RUN` set to `0`, so there
is nothing to plan.

A message that starts with `data row N, POL P:` and then names a key of the
setup or a flight condition comes from the rules of your solver. The page of
the solver lists the keys it reads and the values it accepts.

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
matrix not run: point '1001/DP-M200AL+000' already recorded COMPLETED
```

The point has run before, and a finished point is never replaced by accident.
Add `--resume` to run only the points that have no record, or
`--force-rerun 1001/DP-M200AL+000` to archive this point and run it again. See
[Run again, resume, records](records.md).

```text
  1001/DP-M200AL+000  skipped: COMPLETED
```

Under `--resume`, a point that already completed is skipped, also when its
verdict is `failed`. To run a failed point again, name it with `--force-rerun`.

```text
matrix not run: SOLVER_BUILD '1.0' is not declared under [su2] in inputs/executables.toml (declared: 9.9); add it with its executable path
```

The row names a build that the workspace does not declare. Add the build to
`inputs/executables.toml` or correct the `SOLVER_BUILD` cell. See
[Executables](executables.md).

```text
matrix run with failures
```

At least one point was recorded as failed; the lines above it give the reason
for each. The other points ran. The next section covers the reasons.

## A point recorded as failed

`cfd-matrix status --failed --points` lists the failed points with their
error.

```text
FAILED_EXECUTION: SU2 executable not found; set solver.executable or CFDBENDER_SU2_EXE
```

The solver was not found on this machine. Give its path in the `[solver]`
table of the setup file, in the environment variable that the message names, or
through `SOLVER_BUILD`. [Install](install.md) has the table of keys per
solver. For some solvers the message is that of the operating system instead,
for example `exit code 127` or a file not found error.

```text
FAILED_EXECUTION: ... is already running; only one process at a time
```

Another solver run is going on this machine, from this workspace or another.
One solver process runs at a time on a machine. Wait for it to end.

```text
FAILED_EXECUTION: free RAM ... GiB minus estimate ... GiB would breach the 0.5 GiB floor; refusing ...
```

The machine does not have enough free memory to start the point and keep
0.5 GiB free. Close other programs or run on a larger machine.

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

In each of these cases the point is recorded as `COMPLETED failed`. After you
correct the cause, run it again with
`cfd-matrix run MATRIX --force-rerun POINT`, or with `--force-rerun-all` when
every point failed for the same reason.

## Collecting

```text
collected 2, failed 1, outstanding 0
```

with exit code 1: one recorded point is failed. The tables are written and
hold every point. This is the normal result of a campaign with a failed point.

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
