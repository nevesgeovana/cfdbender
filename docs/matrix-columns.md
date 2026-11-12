# Matrix columns

A run matrix is a pipe-delimited text file in `inputs/matrices/<name>.fs`. Every
solver reads the same columns with the same meanings, which are those of a
pyflightstream matrix, so one layout serves a FlightStream campaign and an SU2,
OpenFOAM, PyFR, XFOIL or QPROP campaign. Cells may be padded with spaces, lines
starting with `#` are comments, and a line of dashes under the header is a rule.

```text
POL  | HIDDEN | RUN | AIRCRAFT | CONFIGURATION | DESCRIPTION       | FLIGHT_CONDITION                                  | SWEEP_VALUES | GEOMETRY     | REF | SET | PPROC | SYMMETRY | SYMMETRY_LOADS | NCPUS | WALLTIME | SOLVER_BUILD | WORKFLOW | VAR_NAMES_VALUES
-----|--------|-----|----------|---------------|-------------------|---------------------------------------------------|--------------|--------------|-----|-----|-------|----------|----------------|-------|----------|--------------|----------|-----------------
5001 |   0    |  1  | WB       | cruise        | Polar near design | MACH:0.85, REmi:5, TK:310.928, ALPHA:sweep, BETA:0 | 2.5,2.75     | wb.su2       | 001 | 001 | 001   | -        | -              | 192   | 4h       | 8.3.0        | -        | iterations: 2000
```

POL, RUN, SET, REF and PPROC are required; every other column is optional and
may be left out of the header. **A cell holding `-` states nothing.** A column
the layout does not know is refused before anything is planned, naming the
closest known column.

| Column | Meaning |
|---|---|
| `POL` | Point of interest; the simulation folder is `sims/sim_<POL>/`. Unique per matrix. |
| `HIDDEN` | `0` or `1`: whether a solver that opens a window shows it. It never removes a row; `RUN` decides that. The library solvers run without a window, so the value is carried and listed only. |
| `RUN` | `1` plans and runs the row, `0` keeps it in the file without running it. |
| `AIRCRAFT` | Your name for the vehicle. Labels only; it reaches the plan listing and the campaign summary. |
| `CONFIGURATION` | Your name for what is in the flow. Labels only, as `AIRCRAFT`. |
| `DESCRIPTION` | Free text for the row. Labels only, as `AIRCRAFT`. |
| `FLIGHT_CONDITION` | The flow state and attitude as `KEY:value` pairs; see below. |
| `SWEEP_VALUES` | The comma-separated values of the one `FLIGHT_CONDITION` key written `sweep`. |
| `GEOMETRY` | The mesh or geometry file: a file name in `inputs/geometries/`, a stem naming exactly one file there, or a workspace-relative path. The setup may then omit its mesh key; one it states must name the same file. |
| `REF` | Reference file `inputs/references/r<id>.toml`; `001` and `r001` are the same. |
| `SET` | Setup file `inputs/setups/s<id>.toml`. |
| `PPROC` | Post-processing file `inputs/pproc/p<id>.toml`. |
| `SYMMETRY`, `SYMMETRY_LOADS` | Accepted for layout parity and must be `-`: the library solvers take symmetry from the mesh markers and the setup. |
| `NCPUS` | Processor count for the scheduler, `{ncpus}` in an HPC profile. Where it is `-`, a setup `ncpus` answers. |
| `WALLTIME` | Wall clock with its unit: `90s`, `240m`, `4h`, `1d`. `{walltime}` in an HPC profile. |
| `SOLVER_BUILD` | Solver version the row runs, such as `8.3.0`. A local run takes that build's executable from `inputs/executables.toml` (see [workspace files](workspace-files.md#executables)); the HPC profile's `[builds]` table maps it to `{solver_version_alias}`. A pyflightstream matrix names this column `FS_BUILD`, which is accepted with the same meaning; a header holding both is refused. |
| `WORKFLOW` | Run type; `-` means `steady`, the only type the library adapters run. |
| `VAR_NAMES_VALUES` | Per-row overrides of setup keys; see below. |

## FLIGHT_CONDITION

The cell holds comma-separated `KEY:value` pairs. Keys match case-insensitively,
a key written twice is refused, and exactly one key may carry the word `sweep`,
whose values come from `SWEEP_VALUES`.

The air is stated with keys that carry their unit in their name:

| Key | Unit | Role |
|---|---|---|
| `MACH` | - | velocity, through the speed of sound at the state's temperature |
| `TASmps` | m/s | velocity (true airspeed) |
| `REmi` | millions | density, solved from Re = rho V L / mu with L the REF file's `length` |
| `ALTFT` | ft | pressure and temperature of the ISO 2533 atmosphere (-6561 ft to 65616 ft) |
| `dISA` | K, as an offset | temperature offset from the standard value at the same pressure |
| `RHOkgm3` | kg/m3 | pins the density |
| `MUPas` | Pa s | pins the viscosity |
| `ASMPS` | m/s | pins the speed of sound |
| `TK` | K | pins the static temperature |
| `PPA` | Pa | pins the static pressure |

State exactly one of `MACH` or `TASmps`. An absent altitude is sea level and an
absent `dISA` is zero. The viscosity follows Sutherland's law at the state's
temperature. Without `REmi` or `RHOkgm3`, the density is the ideal-gas density of
the altitude's pressure (or the `PPA` pin), and the Reynolds number is an outcome.
Whenever the density is solved or pinned, the pressure is `p = rho R T` unless
`PPA` pins it. `REmi` beside `RHOkgm3` is refused.

The attitude keys are `ALPHA` and `BETA` in degrees, and `ADVANCE_RATIO`, `RPM`,
`roll_rate`, `pitch_rate` and `yaw_rate` for solvers that read them. Other keys a
solver documents (for example `ORDER` for PyFR or `J` for QPROP) are kept in upper case.

For SU2, PyFR and XFOIL, a row that states only `MACH` and angles, such as
`MACH:0.3, ALPHA:sweep`, keeps the flow state of its setup file, as matrices
written before these keys did. OpenFOAM and QPROP resolve a bare `MACH` at sea
level like any other air statement.

### What each solver takes from the resolved state

| Solver | Row state becomes | GEOMETRY becomes |
|---|---|---|
| SU2 | `MACH_NUMBER`, `FREESTREAM_TEMPERATURE`, `FREESTREAM_PRESSURE`, `FREESTREAM_DENSITY`, `REYNOLDS_NUMBER`, `REYNOLDS_LENGTH` (see the SU2 page) | `MESH_FILENAME` |
| OpenFOAM | `speed` (the velocity) and `nu` (viscosity over density) | `mesh` |
| PyFR | `mach` and `reynolds` | `mesh` |
| XFOIL | `mach` and `reynolds`; a bare `MACH` is the setup's `mach` | `airfoil_file` |
| QPROP | `V`, `RHO`, `MU` and `A`; `ADVANCE_RATIO` is `J` | `blade` (an OBJ) |

For PyFR and XFOIL, `REmi` gives the Reynolds number directly, also at `MACH:0`;
other air keys give it over the REF file's `length`. OpenFOAM, PyFR and XFOIL run at `BETA` 0
and QPROP in axial flow, so a nonzero `BETA` (or `ALPHA` for QPROP) is refused
and a zero one is accepted. A value the setup also states must agree with the
row's, except that a PyFR row stating only `MACH` and angles replaces the setup's
`mach`, as before.

## VAR_NAMES_VALUES

The cell holds `KEY: value` pairs separated by ` / `, as in pyflightstream:

```text
iterations: 2000 / CFL_NUMBER: 5.0 / MARKER_FAR: ["farfield"] / solver.threads: 4
```

Each key replaces the key of the same name in the row's setup file. A dotted key
replaces a key inside a table of the setup (`solver.threads` is `threads` in the
`[solver]` table). A value is read as a TOML value when it is one (numbers,
`true`, quoted strings, `[...]` lists) and as text otherwise. A key stated twice
is refused, and so is a key that is a matrix column, such as `NCPUS`: state it in
its column. Override the key under the name the setup uses: a setup stating
`iterations` is overridden by `iterations`, not by the SU2 name `ITER`.

## Where the values go

The plan listing (`cfd-matrix plan <name>`) prints AIRCRAFT, CONFIGURATION,
DESCRIPTION, GEOMETRY, NCPUS, SOLVER_BUILD, HIDDEN and each variable under the
point. The run record keeps them with the point, and the campaign summary written
by `cfd-matrix collect` carries AIRCRAFT, CONFIGURATION and DESCRIPTION beside each
result. A value the matrix states and the s/r/p files also state must agree, or
the row is refused naming both.
