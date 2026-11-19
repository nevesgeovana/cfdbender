# CFD++

The CFD++ adapter is incomplete and not yet verified against the solver: it plans rows, writes the solver input and checks native grid folders offline, and no run, residual history or force reading has been compared with a licensed installation.

This page lists only what is specific to CFD++.

## What works offline and what does not exist yet

| Works offline, without the solver | Does not exist yet |
|---|---|
| Planning a matrix of rows for the supported version, 22.5.U3, with every missing setup key named at once | A run of the solver through the adapter that has been compared with a licensed installation |
| Writing the solver input `mcfd.inp` from the setup file, and refusing a combination without evidence | A `verified` status for any entry of the interface database, and output file layouts in the database |
| Checking a native grid folder: the three binary grid files, the family-name file, one cell group, partition files | Conversion of a mesh from another format; convert it once with the solver's own converter |
| The `cfd-workspace setups --template subsonic` template, whose numbers are placeholders to check against your case | Unsteady workflows, more than one cell group, versions other than 22.5.U3 |
| Staging a point folder with a job script, linking the grid files rather than copying them | Reading of residual histories and forces that has been checked against solver output |
| | The adapter is not yet run through the shared adapter conformance tests |

The residual and force readers exist, and their file layouts are hypotheses. A result they produce is not evidence until it has been compared with a solver run.

## Setup

`cfd-workspace init --solver cfdpp` creates the workspace. The setup file states the case and nothing else: `iterations`, `convergence_orders`, `cfl`, `cfl_max`, `cfl_ramp_steps`, `turbulence` with the `viscosity_ratio` or `turbulence_intensity` the model needs, the tables `[physics]`, `[fluid]` and `[numerics]`, and one `[boundaries.<family>]` table for each family of the grid. No key has a default: a missing key is refused, and `cfd-workspace setups --template subsonic` writes a complete file to start from. Raw solver keywords go in the one `[keywords]` table; `cfd-matrix plan --verbose` lists each with its evidence class.

The setup file has no `[solver]` table. The solver executables are declared per build in `inputs/executables.toml`, with the fields `serial`, `mpi`, `mpi_launcher`, `partitioner` and `tools_dir`; see [Executables](executables.md).

## Matrix

`SOLVER_BUILD` is required and selects the version of the interface database. `MESH` names a native grid folder in `inputs/mesh/`. `NUMERICS` is not read by this adapter: a filled cell is warned once and ignored.

```text
POL  | RUN | DESCRIPTION | FLIGHT_CONDITION                          | SWEEP_VALUES | MESH       | REF | SETUP | PPROC | NCPUS | WALLTIME | SOLVER_BUILD
-----|-----|-------------|-------------------------------------------|--------------|------------|-----|-------|-------|-------|----------|-------------
1001 | 1   | Polar       | MACH:0.15, REmi:6, TK:300, ALPHA:sweep    | 0,10         | naca.cfdpp | 001 | 001   | 001   | 1     | 1h       | 22.5.U3
```

NCPUS 1 runs the serial executable; a larger value runs the MPI launcher and executable, with a partition step when the grid folder holds no partition file.

## Large grids

A grid file larger than the large-input threshold is read once, in the pass that digests it, and its facts (dimension, counts, selectors) are remembered with its digest. With `--no-read-large-inputs`, the environment variable `CFDBENDER_NO_READ_LARGE_INPUTS` or the init key `large_input_check = "identity"`, no grid file is opened: each is recorded by size and modification time, planning warns that the grid facts were not checked, and writing the input is refused, naming each grid file whose facts are missing, until the facts are known. One plan with the check on is enough: it remembers the facts with the digest, and later plans with the check off reuse them and open nothing.

## Capabilities

The adapter declares `cluster_staging`, `restart` and `boundary_groups`. It does not declare `numerics_file`. See [Solvers side by side](solvers.md).
