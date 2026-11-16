# First campaign

This page takes you through one complete campaign: three points of an angle of
attack sweep, planned, run and collected into a table. It uses the mock solver,
which is part of cfdbender and computes nothing, so you need no solver
installed. Every command works the same way with a real solver.

You need cfdbender installed; see [Install](install.md). The outputs below are
those of a workspace created in `/home/you`.

## 1. Create the workspace

```console
cfd-workspace init campaign --solver mock
cd campaign
```

```text
campaign workspace ready at /home/you/campaign
  init.toml (solver: mock)
  inputs/mesh/
  inputs/references/
  inputs/setups/
  inputs/pproc/
  inputs/profiles/
  inputs/freestreams/
  inputs/calibrations/
  inputs/hpc/
  inputs/fsi/
  inputs/matrices/
  inputs/management/
  inputs/INPUTS.md (what each input file holds)
  inputs/.gitignore (keeps this machine's executable paths out of git)
  sims/
  post/
  archive/
next: write inputs/matrices/<name>.fs, then cfd-matrix plan <name>
```

A workspace is one folder for one campaign and one solver. You write files in
`inputs/` and nowhere else. The commands write `sims/`, `post/` and `archive/`.
This campaign uses four of the input folders: `matrices`, `setups`,
`references` and `pproc`. The mock solver lists more folders than the real
solvers do, which list only the ones they read. When `init` is run on a folder
that already holds files, it adds the line `existing folders and files were
kept untouched` before the `next:` line.

`inputs/INPUTS.md` says, for the solver of the workspace, what each input file
holds and which keys it reads. Open it when you do not know what to write.

## 2. Write the four input files

A campaign is described by a matrix and the three files each of its rows names.

`inputs/matrices/polar.fs`:

```text
POL  | RUN | DESCRIPTION | FLIGHT_CONDITION      | SWEEP_VALUES | REF | SET | PPROC
-----|-----|-------------|-----------------------|--------------|-----|-----|------
1001 | 1   | First polar | MACH:0.2, ALPHA:sweep | 0,2,4        | 001 | 001 | 001
```

The matrix is a table with `|` between its columns. Each row is one simulation,
called a sim. `POL` is the number you give it. `RUN` is `1` for a row to run
and `0` for a row to keep without running. `FLIGHT_CONDITION` states the flow:
here Mach 0.2 and an angle of attack, in degrees, that takes each value of
`SWEEP_VALUES` in turn. Each value is one point, so this row has three points.
`SET`, `REF` and `PPROC` name the three files below by their number.

Save every input file as UTF-8 without a byte order mark. A TOML file with one
is refused as invalid TOML. Windows PowerShell 5.1 `Set-Content -Encoding UTF8`
and some editors add the mark; PowerShell 7 and most editors do not.

`inputs/setups/s001.toml`:

```toml
[mock]
outcome = "converged"
iterations = 10
```

The setup says how the solver runs. Its keys belong to the solver. The mock
takes the outcome it should report and an iteration limit.

`inputs/references/r001.toml`:

```toml
area = 1.5
length = 1.0
moment_origin = [0, 0, 0]
axis_system = "body"
axis_directions = ["aft", "right", "up"]
```

The reference file holds the reference area in m2, the reference length in m
and the point about which moments are taken, in m.

`inputs/pproc/p001.toml`:

```toml
history = true
```

The pproc file says what to keep from each run beyond the coefficients.

## 3. Plan

```console
cfd-matrix plan polar
```

```text
matrix: inputs/matrices/polar.fs
solver: mock
3 point(s)
  1001/DP-M200AL+000: SET=s001 REF=r001 PPROC=p001 WALLTIME=NA  ALPHA=0, MACH=0.2
  1001/DP-M200AL+020: SET=s001 REF=r001 PPROC=p001 WALLTIME=NA  ALPHA=2, MACH=0.2
  1001/DP-M200AL+040: SET=s001 REF=r001 PPROC=p001 WALLTIME=NA  ALPHA=4, MACH=0.2
3 point(s) planned; receipt post/polar/plan.json; next: cfd-matrix run polar
```

`plan` reads the matrix and every file it names, and prints one line per point
with the three files it uses and its flight condition. It launches nothing. A
mistake in an input is refused here, with the matrix path, the row and the file
at fault, before any solver time is spent. Add `--verbose` to see, for each
point, the labels of its row, what a run would do and every file it would
write.

Each point has a name built from its flight condition: `DP-M200AL+020` is Mach
0.200 and alpha +2.0 degrees. With its `POL` in front, `1001/DP-M200AL+020`,
the name identifies the point in every command.

`plan` also saves a receipt of what it read, `post/polar/plan.json`. Plan again
after every change to an input: `run` refuses a matrix whose inputs changed
since its receipt was written.

## 4. Run

```console
cfd-matrix run polar
```

```text
[1/3] 1001/DP-M200AL+000  running
[1/3] 1001/DP-M200AL+000  COMPLETED converged in 0 s
[2/3] 1001/DP-M200AL+020  running
[2/3] 1001/DP-M200AL+020  COMPLETED converged in 0 s
[3/3] 1001/DP-M200AL+040  running
[3/3] 1001/DP-M200AL+040  COMPLETED converged in 0 s
```

`run` executes each point in its own folder,
`sims/sim_1001/datapoints/DP-M200AL+000/` for the first one, and records it in
`runs.json`. A line says when a point starts and another when it ends, with the
time it took. The first word of the second line is the status of the run:
`COMPLETED` means the solver process ended. The second word is the verdict on
the numbers: `converged`, `iteration_limit` or `failed`. A completed run is not
always a converged one; [Results and tables](results.md) explains the three
verdicts. A point that ended in failure reads `FAILED_EXECUTION` or another
`FAILED_` word instead of `COMPLETED`, so a failure is not mistaken for a
success. With a real solver, `run` first checks that the solver can start on this machine and
refuses before it records anything when it cannot; see
[Troubleshooting](troubleshooting.md).

## 5. Follow

```console
cfd-matrix status
```

```text
sim   matrix  points  status       failed
1001  polar   3       COMPLETED=3  0
```

```console
cfd-matrix status --points
```

```text
sim   matrix  point               status     classification  error
1001  polar   1001/DP-M200AL+000  COMPLETED  converged
1001  polar   1001/DP-M200AL+020  COMPLETED  converged
1001  polar   1001/DP-M200AL+040  COMPLETED  converged
```

`status` reads the records, so it works at any time, also while a long run is
going in another terminal. `cfd-workspace show 1001/DP-M200AL+020` prints the card
of one point: its status and verdict, its iterations and coefficients once
collected, and its folder.

## 6. Collect

```console
cfd-matrix collect polar
```

```text
wrote post/polar/campaign_summary.json
wrote post/polar/polars/FAM-DIM-P1001.csv
wrote post/polar/polars/FAM-ACOEF-P1001.csv
wrote post/polar/polars/P1001-TOTAL-DIM.csv
wrote post/polar/polars/P1001-TOTAL-ACOEF.csv
wrote post/polar/polars/P1001-AIRFRAME-DIM.csv
wrote post/polar/polars/P1001-AIRFRAME-ACOEF.csv
collected 3, failed 0, outstanding 0
```

`collect` reads the saved output of every recorded point and writes the tables
of the matrix under `post/polar/`. `polars/FAM-ACOEF-P1001.csv` has one row per point
and family. Its first and last columns for this campaign, with the coefficients the mock
reports under its own names:

```text
POL,POINT_ID,POINT,RUN_TYPE,ALPHA,...,CX,CY,CZ,Cl,Cm,Cn,MESH_FACES
1001,1,DP-M200AL+000,steady,0.0,...,0.01,0.02,0.03,0.04,0.05,0.06,NA
1001,2,DP-M200AL+020,steady,2.0,...,0.01,0.02,0.03,0.04,0.05,0.06,NA
1001,3,DP-M200AL+040,steady,4.0,...,0.01,0.02,0.03,0.04,0.05,0.06,NA
```

The `...` stands for the air state columns, left out here to fit the page. The
mock reports the same numbers for every point. `NA` marks a value a point does
not have. `CL`, `CD` and `CM` are the same three coefficients on every solver,
filled where the solver reports them and `NA` where it does not; the columns
after them keep the solver's own names, here `CX` to `Cn`. A script that reads
`CL` works with any solver. `collect` also prints `wrote` before each file as a
path relative to the workspace.

## What you have now

```text
campaign/
  init.toml
  inputs/                      the four files you wrote, and INPUTS.md
  sims/sim_1001/datapoints/    one folder per point, with the solver files
  post/polar/                  the plan receipt and the collected tables
  runs.json                    the record of every point
  logs/                        what run and collect printed
```

## Where to go next

- [The workflow](workflow.md) explains each stage and what it reads and writes.
- [Solvers, side by side](solvers.md) shows what changes with a real solver:
  the mesh, the setup keys and how convergence is judged.
- [Run again, resume, records](records.md) covers a second run, a failed point
  and the history of a point.
- [Troubleshooting](troubleshooting.md) explains each refusal by its text.
