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
  inputs/geometries/
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
  sims/
  post/
  archive/
existing folders and files were kept untouched
```

A workspace is one folder for one campaign and one solver. You write files in
`inputs/` and nowhere else. The commands write `sims/`, `post/` and `archive/`.
This campaign uses four of the input folders: `matrices`, `setups`,
`references` and `pproc`.

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
  1001/DP-M200AL+000: SET=s001 REF=r001 PPROC=p001 WALLTIME=NA
    DESCRIPTION=First polar
    ALPHA=0, MACH=0.2
    would: write mock ASCII input; run mock; read mock history
    output: sims/sim_1001/datapoints/DP-M200AL+000/mock.in
    output: sims/sim_1001/datapoints/DP-M200AL+000/history.csv
    output: sims/sim_1001/datapoints/DP-M200AL+000/execution.json
  1001/DP-M200AL+020: SET=s001 REF=r001 PPROC=p001 WALLTIME=NA
    DESCRIPTION=First polar
    ALPHA=2, MACH=0.2
    would: write mock ASCII input; run mock; read mock history
    output: sims/sim_1001/datapoints/DP-M200AL+020/mock.in
    output: sims/sim_1001/datapoints/DP-M200AL+020/history.csv
    output: sims/sim_1001/datapoints/DP-M200AL+020/execution.json
  1001/DP-M200AL+040: SET=s001 REF=r001 PPROC=p001 WALLTIME=NA
    DESCRIPTION=First polar
    ALPHA=4, MACH=0.2
    would: write mock ASCII input; run mock; read mock history
    output: sims/sim_1001/datapoints/DP-M200AL+040/mock.in
    output: sims/sim_1001/datapoints/DP-M200AL+040/history.csv
    output: sims/sim_1001/datapoints/DP-M200AL+040/execution.json
```

`plan` reads the matrix and every file it names, and lists each point with the
files a run would write. It launches nothing. A mistake in an input is refused
here, with the row and the file at fault, before any solver time is spent.

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
  1001/DP-M200AL+000  COMPLETED converged
  1001/DP-M200AL+020  COMPLETED converged
  1001/DP-M200AL+040  COMPLETED converged
```

`run` executes each point in its own folder,
`sims/sim_1001/datapoints/DP-M200AL+000/` for the first one, and records it in
`runs.json`. The first word is the status of the run: `COMPLETED` means the
solver process ended. The second word is the verdict on the numbers:
`converged`, `iteration_limit` or `failed`. A completed run is not always a
converged one; [Results and tables](results.md) explains the three verdicts.

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
going in another terminal.

## 6. Collect

```console
cfd-matrix collect polar
```

```text
collected summary: /home/you/campaign/post/polar/campaign_summary.json
collected table: /home/you/campaign/post/polar/campaign_sweep.csv
collected table: /home/you/campaign/post/polar/polars/P1001.csv
collected 3, failed 0, outstanding 0
```

`collect` reads the saved output of every recorded point and writes the tables
of the matrix under `post/polar/`. `campaign_sweep.csv` has one row per point.
Its first columns for this campaign, with the coefficients the mock reports:

```text
POL,point,ALPHA,BETA,MACH,...,status,classification,iterations,residual,wall_time_s,solver_version,CX,CY,CZ,Cl,Cm,Cn
1001,DP-M200AL+000,0.0,NA,0.2,...,COMPLETED,converged,2,0.01,0.031809,mock-1,0.01,0.02,0.03,0.04,0.05,0.06
1001,DP-M200AL+020,2.0,NA,0.2,...,COMPLETED,converged,2,0.01,0.024971,mock-1,0.01,0.02,0.03,0.04,0.05,0.06
1001,DP-M200AL+040,4.0,NA,0.2,...,COMPLETED,converged,2,0.01,0.034441,mock-1,0.01,0.02,0.03,0.04,0.05,0.06
```

The `...` stands for the air state columns, left out here to fit the page. The
mock reports the same numbers for every point. `NA` marks a value a point does
not have.

## What you have now

```text
campaign/
  init.toml
  inputs/                      the four files you wrote
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
