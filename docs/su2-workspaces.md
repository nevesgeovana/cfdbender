# SU2


You run SU2 cases through cfdbender, steady or time accurate (see [Run types](#run-types)): it writes `su2.cfg`, links your mesh, runs SU2 and reads the residual history and the force and moment coefficients. This page lists only what is specific to SU2.

## Install

Install SU2 so that `SU2_CFD` exists on the machine. Install it with CGNS support if you run `.cgns` meshes. cfdbender runs SU2 by one of two routes, set by `route` in the `[solver]` table of the setup file:

| `route` | What runs |
|---|---|
| `"executable"` | The `SU2_CFD` program, started as `SU2_CFD su2.cfg` in the point folder. |
| `"pysu2"` | The SU2 Python wrapper, in a child process. |

Without `route`, cfdbender prefers `pysu2`, unless the setup states `executable`: a stated executable chooses the executable route, with no search for the wrapper and no warning. When nothing is stated and the module is absent it warns `pysu2 is unavailable: <reason>; falling back to the executable route` and runs the executable. With `route = "pysu2"` and no wrapper, it refuses:

> pysu2 is unavailable: `<reason>`. Build SU2 with the Python wrapper (-Denable-pywrapper=true) on Linux or WSL using a compatible Python; the Windows interpreter cannot import that Linux build. Set solver.pysu2_path or CFDBENDER_PYSU2_PATH to its install directory. The executable route is available: choose solver.route='executable'.

cfdbender finds the files in this order:

| What | Setup key | Environment variable | Last resort |
|---|---|---|---|
| `SU2_CFD` | `executable` in `[solver]` | `CFDBENDER_SU2_EXE` | `SU2_CFD` on the PATH |
| Python wrapper directory | `pysu2_path` in `[solver]` | `CFDBENDER_PYSU2_PATH` | the interpreter's own path |

A relative path resolves from the workspace root. If no executable is found, `run --local` refuses before it records any point, with `SU2 executable not found: 'SU2_CFD'; set executable in the [solver] table of the setup file, or declare SOLVER_BUILD in inputs/executables.toml`; the same command works once the program is installed.

A matrix row can also pick a build by name; see [Executables](executables.md).

## Mesh

SU2 reads a native `.su2` mesh or a CGNS `.cgns` mesh. Put the file in `inputs/mesh/` and name it in the matrix `MESH` column (a file name or a stem), or in the setup key `mesh`. A `mesh` key in the setup and a `MESH` cell must name the same file. Each point folder holds the link as `mesh.su2` (with `MESH_FORMAT= SU2`) or `mesh.cgns` (with `MESH_FORMAT= CGNS`); see [Workspace layout](workspace-layout.md) for how a mesh is linked.

Any other suffix is refused: `mesh must name a workspace-relative .su2 or .cgns file`.

Marker names come from the setup file, never from the mesh. Match the names and the boundary conditions to your grid.

## Setup file

### Setup and numerics

The `SETUP` column (named `SET` before) names the physical setup
`inputs/setups/s<id>.toml`; the optional `NUMERICS` column names the numerical
keys in `inputs/numerics/n<id>.toml`: the schemes, limiters, gradient method,
time discretization, CFL, linear solver, multigrid, iteration counts and
convergence criteria. The merge order and the rules are in
[Setup and numerics](matrix-columns.md#setup-and-numerics). A matrix without the
column plans exactly as before.

To move an existing workspace to the new layout, run the migration script. It
needs only Python 3.11 or later and imports nothing from cfdbender, so it also
runs beside an older install; the same file ships as
`python -m cfdbender.migrate_setup_numerics` and as `tools/migrate_setup_numerics.py`.
The default is a dry run:

```text
python migrate_setup_numerics.py campaign            # prints what would change
python migrate_setup_numerics.py campaign --apply    # archives, then changes
cfd-matrix plan study --workspace campaign --verbose
```

With `--apply` the script first copies every file it changes to
`archive/migration-setup-numerics-<n>/`, renames the header `SET` to `SETUP` and
`GEOMETRY` to `MESH`, inserts `NUMERICS` after the setup column, and moves the numerical top-level keys of each setup
(with the comment lines directly above each key) to `inputs/numerics/`. Setups
whose numerical keys are identical share one numerics file. Keys it does not
list stay in the setup. Run on a migrated workspace it changes nothing. For a
solver other than SU2 it only renames the columns and adds an empty `NUMERICS`
column. The solver input planned before and after is byte-identical.

### Boundary force groups

Use a setup `[groups]` table to keep forces for parts of a model separately:

```toml
[groups]
main = ["wing"]
combined = ["wing", "tail"]
```

The markers must exist in the mesh. An unknown marker is refused with the
mesh's marker names. SU2 monitors the requested markers in addition to the
markers already monitored, and writes `AERO_COEFF_SURF` into its existing
history. Collect sums each group's native coefficients about the same origin
and axes and retains SU2's total as `TOTAL`. Groups may overlap. A group name
cannot be `TOTAL`, and a marker cannot be repeated within one group.

The result is in `NeutralResult.force_groups` and in
`points[].result.force_groups` of `post/<matrix>/campaign_summary.json`.
No group CSV is written. See [results](results.md#boundary-force-sets).


`inputs/setups/s001.toml` holds numerics, markers and the free stream. cfdbender writes every key into `su2.cfg` as `KEY= value`. Neutral names below map to SU2 keys. Any key written in capitals, such as `SOLVER` or `MARKER_FAR`, passes through unchanged, and a list is written as `( a, b )`. Any other setup key is refused as `<file>: unknown key '<key>' (closest known key: <neutral name>); known keys: ...`, naming the setup file it was written in. Unread reference and pproc keys are warned about and skipped. A key given twice, in any of the setup, reference and pproc files or the row, is refused as `conflict: <key> is supplied more than once`.

| Key | SU2 key | Unit | Default |
|---|---|---|---|
| `mesh` | `MESH_FILENAME` | - | none (or `MESH`) |
| `iterations` | `ITER` | - | required, integer above 0 |
| `mach` | `MACH_NUMBER` | - | none |
| `alpha` or `angle_of_attack` | `AOA` | degrees | none |
| `beta` or `sideslip` | `SIDESLIP_ANGLE` | degrees | none |
| `freestream_temperature` | `FREESTREAM_TEMPERATURE` | K | none |
| `freestream_pressure` | `FREESTREAM_PRESSURE` | Pa | none |
| `freestream_density` | `FREESTREAM_DENSITY` | kg/m^3 | none |
| `freestream_velocity` | `FREESTREAM_VELOCITY` | m/s, three components | none |
| `reynolds`, `reynolds_length` | `REYNOLDS_NUMBER`, `REYNOLDS_LENGTH` | -, m | none |
| `convergence_field` | `CONV_FIELD` | - | SU2's own |
| `convergence_residual` | `CONV_RESIDUAL_MINVAL` | log10 of the residual | -12 with `FREESTREAM_PRESS_EQ_ONE` (the default scaling of compressible decks), else SU2's own (-8) |
| `output_frequency` | `OUTPUT_WRT_FREQ` | iterations | SU2's own |
| `cfl` | `CFL_NUMBER` | - | a ramp from 5 to 25, see below |
| `cfl_max`, `cfl_growth` | the maximum and the up factor of `CFL_ADAPT_PARAM` | - | 25 and 1.2, see below |
| `mesh_link` | none | - | `true`; `false` is refused |

Temperature, pressure, density and velocity use SU2's units, SI by default. `mesh_link = true` changes nothing; `mesh_link = false` is refused with `mesh_link = false asks for a mesh copy; meshes are always linked, never copied: remove mesh_link`.

cfdbender also writes `MESH_FORMAT` to match the mesh, `CONV_FILENAME= history` and `TABULAR_FORMAT= CSV`, because the history must be a CSV file. If you state one of these with another value, it refuses with `conflict: <key> must be <value>`. The history must contain at least one `rms[...]` residual column and one coefficient column; cfdbender writes `HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )` when the setup states none, and adds `AERO_COEFF_SURF` (`AERO_COEFF_SURF[<zone>]` on a multizone point) to every point's list, so the history holds the load coefficients of each monitored wall marker (`CL(wing)`, `CFx(wing)`; one block of coefficient columns per marker, about ten, so a mesh with a few dozen wall markers adds a few hundred columns to a history row, and the cost matters only beyond some hundreds of markers; it adds no solver work, SU2 already integrates every monitored marker), and a `HISTORY_OUTPUT` of the setup that lacks one of these groups is extended with it and a warning, because SU2's own default history holds neither residuals nor coefficients.

### Gas constants

cfdbender also writes `GAS_CONSTANT= 287.05287` and `GAMMA_VALUE= 1.4`, the constants of its own standard atmosphere, so the free-stream speed SU2 computes equals the one the other solvers receive. It writes `FLUID_MODEL= IDEAL_GAS` with them, because SU2's default fluid model, `STANDARD_AIR`, replaces both constants with its own (287.058 and 1.4) and would ignore them. State `GAS_CONSTANT` or `GAMMA_VALUE` in the setup to use your own value; yours wins.

If the setup states `FLUID_MODEL`, yours wins and cfdbender writes neither constant. When the gas SU2 then uses differs from the core one (for example `STANDARD_AIR`, or `IDEAL_GAS` without your own constants, which falls back to 287.058), cfdbender warns that SU2's free stream differs from the other solvers'. The incompressible solvers (`SOLVER = "INC_..."`) get the two constants but no `FLUID_MODEL`, because their fluid models have other names.

### Spalart-Allmaras variant

When the setup states `KIND_TURB_MODEL = "SA"` and no `SA_OPTIONS`, cfdbender writes `SA_OPTIONS= ( NEGATIVE, WITHFT2 )`, the standard SA-neg of the NASA turbulence modeling resource. SU2's own default for SA is SA-noft2, which limits every nu-tilde update to 99 percent of the local value and cancels updates smaller than 1e-10; on the TMR NACA 0012 that held the turbulence residual from falling. SU2 8 refuses `NEGATIVE` without `WITHFT2` unless `EXPERIMENTAL` is added, because SA-neg is defined with the ft2 term. State `SA_OPTIONS` to choose another variant, for example `SA_OPTIONS = ["NEGATIVE", "EXPERIMENTAL"]` (SA-neg without ft2, as in SU2's own NACA 0012 case); yours replaces the default entirely.

### Wall markers

SU2 reads `MARKER_HEATFLUX` as pairs of a marker name and its heat flux. A list of names alone, such as `MARKER_HEATFLUX = ["airfoil"]`, is written with the adiabatic value for each name: `MARKER_HEATFLUX= ( airfoil, 0.0 )`. A list that holds any number is written as stated.

SU2 sums the force coefficients only over the markers in `MARKER_MONITORING`, and writes surface rows only for the markers in `MARKER_PLOTTING`: without `MARKER_PLOTTING`, a requested `SURFACE_CSV` holds a header and no rows. When the setup states neither, cfdbender writes both with the wall markers it finds in `MARKER_EULER`, `MARKER_HEATFLUX` and `MARKER_ISOTHERMAL`, in that order, for example `MARKER_PLOTTING= ( airfoil )` and `MARKER_MONITORING= ( airfoil )`. State either key to choose your own list; yours wins, and the other key keeps the default.

### CFL ramp

The CFL number is the size of the pseudo-time step relative to the stability limit: a larger CFL converges faster until it is too large and the residual stalls or the run diverges. When the setup states no CFL key, cfdbender writes a ramp that starts gentle and grows as the residual falls:

```
CFL_NUMBER= 5
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.2, 5, 25 )
```

`CFL_ADAPT_PARAM` is (factor down, factor up, CFL minimum, CFL maximum). The start 5 is safe on a poor initial field. The maximum 25 is conservative. The stalls once measured on the TMR NACA 0012 at CFL 50 or more ran with SU2's dimensional scaling; with the free-stream scaling cfdbender now writes, SU2's own NACA 0012 deck runs at CFL 1000. A larger CFL often converges much faster, but no larger default is yet proven both convergent and accurate there with these numerics, so state one in the setup when you need it. The factor 1.2 raises the CFL 20 percent per iteration while the residual falls, and 0.5 halves it when the residual grows. The minimum 5 never lets the ramp fall below the start.

To change it, state `cfl` (or `CFL_NUMBER`), `CFL_ADAPT` or `CFL_ADAPT_PARAM` in `s<id>.toml`, for example `CFL_ADAPT_PARAM = [0.5, 1.2, 5, 50]`. Any one of these three keys switches the default off completely: cfdbender then writes only the CFL keys you stated and merges nothing, so a lone `cfl = 10` is a fixed CFL 10 and a lone `CFL_ADAPT_PARAM` without `CFL_ADAPT = true` is written as stated. Your keys always win.

To keep the ramp and move only its ceiling or its speed, state `cfl_max` or `cfl_growth`. cfdbender then writes the default ramp with those values in place, starting at `cfl` (5 if not stated), which is also the ramp's minimum. For example `cfl_max = 1000` writes `CFL_NUMBER= 5`, `CFL_ADAPT= YES` and `CFL_ADAPT_PARAM= ( 0.5, 1.2, 5, 1000 )`, and `cfl = 10` with `cfl_growth = 1.5` and `cfl_max = 500` writes `CFL_ADAPT_PARAM= ( 0.5, 1.5, 10, 500 )`. `cfl_growth` must be above 1 and `cfl_max` at least the start. With either key, `CFL_ADAPT_PARAM` is refused (`conflict: cfl_max and cfl_growth set CFL_ADAPT_PARAM; state either, not both`), and so is `CFL_ADAPT = false`.

**`[solver]` table**

| Key | Meaning | Unit | Default |
|---|---|---|---|
| `route` | `"pysu2"` or `"executable"` | - | pysu2 if available |
| `executable` | path to `SU2_CFD` | - | see Install |
| `pysu2_path` | directory holding the Python wrapper | - | see Install |
| `threads` | OpenMP threads, passed as `OMP_NUM_THREADS` and recorded in `execution.json`; behind a launcher script executable (`.cmd`, `.bat`, `.sh`, `.ps1`), which sets its own, `execution.json` records `"unknown"` | - | 1 |
| `timeout_s` | time after which cfdbender stops the run | s | 600 |
| `ram_estimate_gib` | memory the run is expected to need | GiB | 0.25 |
| `mpi_launcher` | tokens that start MPI, see below | - | `[]` (serial) |

The same `[solver]` keys can be set for the whole workspace in `init.toml`; the setup file overrides them.

**MPI launch.** On the executable route, `mpi_launcher` goes in front of the executable. Tokens are passed as arguments without shell expansion.

```toml
[solver]
route = "executable"
executable = "/opt/su2/bin/SU2_CFD"
mpi_launcher = ["mpirun", "-np", "192"]
```

This starts `mpirun -np 192 /opt/su2/bin/SU2_CFD su2.cfg` in the point folder. Under a scheduler allocation, `mpi_launcher = ["srun"]` uses the allocation's task count. The `pysu2` route always runs serially. For a scheduler job, see [Cluster jobs](cluster-jobs.md).

**Reference file** `inputs/references/r001.toml` holds `area` (m^2), `length` (m) and `moment_origin` (three coordinates, m). They become `REF_AREA`, `REF_LENGTH` and `REF_ORIGIN_MOMENT_X`, `_Y`, `_Z`, and cfdbender refuses a point without all of them. `length` is also the length that `REmi` is measured against.

**Pproc file** `inputs/pproc/p001.toml` must hold `history = true`; otherwise `pproc history must be true`. It may also hold `output_frequency`.

**Flight condition.** The row can carry the flow state, so the setup holds only the numerics. For each point SU2 receives:

| Matrix | SU2 key |
|---|---|
| resolved Mach number | `MACH_NUMBER` |
| `ALPHA`, `BETA` | `AOA`, `SIDESLIP_ANGLE` |
| resolved temperature, pressure, density | `FREESTREAM_TEMPERATURE`, and `FREESTREAM_PRESSURE` and `FREESTREAM_DENSITY` where SU2 reads them (below) |
| `REmi` x 1e6 and the reference file's `length` | `REYNOLDS_NUMBER`, `REYNOLDS_LENGTH` |
| `MUPas` | `VISCOSITY_MODEL= CONSTANT_VISCOSITY`, `MU_CONSTANT` |
| `MESH` | `MESH_FILENAME` |

A row that states only `MACH` and angles keeps the setup's flow state. When neither the row nor the setup states a Mach number, `plan` warns that SU2 will run with its own default free stream. When the air is stated without `REmi`, the Reynolds number written is the outcome of the resolved density, velocity and viscosity over the reference file's `length`. A value that the setup also states must agree with the row, or the point is refused: `conflict: <key> is <value> in the s/r/p files and <value> from the matrix FLIGHT_CONDITION; state it in one place`. `ASMPS` is refused unless it equals the speed of sound SU2 derives from the temperature: `ASMPS pins a speed of sound SU2 cannot hold: SU2 derives it from the temperature; pin TK instead`. See [Matrix columns](matrix-columns.md) for every column.

The deck states no free-stream value that SU2 recomputes and ignores. Under SU2's default `FREESTREAM_OPTION`, `TEMPERATURE_FS`, the density follows from the pressure and the temperature, so the resolved density is written only with `FREESTREAM_OPTION = "DENSITY_FS"`. A compressible viscous run (`SOLVER` `NAVIER_STOKES` or `RANS`) with a Reynolds number, under SU2's default `INIT_OPTION`, `REYNOLDS`, takes the density from the Reynolds number and recomputes the pressure from it, so the resolved pressure is not written either; `INIT_OPTION = "TD_CONDITIONS"` keeps it. An Euler run gets the pressure. A value you state in the setup is always written. A compressible deck (`SOLVER` `EULER`, `NAVIER_STOKES` or `RANS`) without `REF_DIMENSIONALIZATION` gets `REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE`, as SU2's own NACA 0012 case states: SU2's default, `DIMENSIONAL`, leaves the energy rows of the implicit system about 1e5 times the density rows, and on the TMR NACA 0012 at CFL 1000 SU2's official deck then stalls at rms[Rho] -4.4 instead of converging in 688 iterations. The coefficients do not change; the residual levels in `history.csv` are then relative to the free stream, so a `CONV_RESIDUAL_MINVAL` chosen for a dimensional run means a different level. State `REF_DIMENSIONALIZATION = "DIMENSIONAL"` to keep SU2's default.

### Run types

The matrix column `WORKFLOW` selects the run type with pyflightstream's names: `steady` (also `-`), `qsteady_rotor`, `unsteady` and `unsteady_rotor`. Each run type carries the SU2 keys that follow from it, so the setup states only what is particular to the case; a key the setup states itself wins over a carried one. Any other value, `LEGACY` included, is refused.

| `WORKFLOW` | What it solves | What cfdbender writes |
|---|---|---|
| `steady` | a steady case | nothing beyond the rules of this page: an existing workspace plans the same `su2.cfg` |
| `qsteady_rotor` | a rotor solved steady in the frame that turns with it (blades held still, free stream turning) | `GRID_MOVEMENT= ROTATING_FRAME`, `ROTATION_RATE`, `MOTION_ORIGIN`, `MACH_MOTION`, and `MARKER_PERIODIC` of a sector |
| `unsteady` | time accurate, nothing turns | `TIME_DOMAIN`, `TIME_MARCHING`, `TIME_STEP`, `TIME_ITER`, `INNER_ITER`, `WINDOW_START_ITER` |
| `unsteady_rotor` | time accurate, the rotor turning | the time keys and the rotation: the whole mesh in rigid motion, or a sliding-mesh multizone layout |

Row variables of another run type are warned and skipped, as pyflightstream does (for example `DELTA_THETA` on a `steady` row: "DELTA_THETA is not registered for run type 'steady'. It is read by unsteady, unsteady_rotor."). `CLOCK_MOTION`, `MOTIONS` and `PASSAGE_POSITIONS` are not offered yet: a row turns the one rotor of its reference file, and the clockings of a whole wheel are not solved; each is warned and skipped.

#### The rotor declaration

The reference file declares the rotor as a pyflightstream reference file does, a table with `kind = "rotor"`; the table name is the rotor's alias:

```toml
area = 1.0
length = 1.0
moment_origin = [0.0, 0.0, 0.0]

[ROTOR]
kind = "rotor"
x_m = 0.0                  # the hub in the mesh frame, m (default 0)
y_m = 0.0
z_m = 0.0
axis = "X"                 # the shaft: thrust is counted positive along it
rpm_sign = -1              # 1: right-handed about the shaft; -1: left-handed
diameter_m = 0.40741537
families_general = ["S"]
families_blades = ["Blade1", "Blade6", "Blade5", "Blade4", "Blade3", "Blade2"]
blade1 = { azimuth_deg = 0.0, zero = "Z" }
```

`axis` is a letter or, for a shaft installed at an angle, `axis = { shaft = "X", vector_wrt_global = [0.996, 0, 0.087] }`: the vector is normalized and must lie within 45 degrees of the positive axis the letter names, as pyflightstream requires. `blade1` places blade one at step zero, `azimuth_deg` measured from the axis `zero` (one of the two axes that are not the shaft, with an optional sign) right-handed about the shaft; blade `k` of `B` sits `(k - 1) 360 / B` further on. Without `blade1` the datum is `zero = "X"`, which a shaft `X` refuses, as in pyflightstream. `alias`, when stated, equals the table name. The blade count is the length of `families_blades`. One rotor only, and the SU2 adapter reads the shaft `X` only (the free stream runs along +x).

What differs from the pyflightstream reference file: the reference lengths are `area`, `length` and `moment_origin`, not `area_m2`, `chord_m` and `moment_point`.

#### The rotor speed

The row states the speed in `FLIGHT_CONDITION` as `ADVANCE_RATIO` (a value or the swept variable) or as `RPM`, a magnitude whose hand is `rpm_sign`:

```text
POL  | RUN | SETUP | REF | PPROC | FLIGHT_CONDITION                                  | SWEEP_VALUES                       | MESH          | WORKFLOW
-----|-----|-------|-----|-------|---------------------------------------------------|------------------------------------|---------------|---------
1001 | 1   | 001   | 001 | 001   | MACH:0.0529, ALPHA:0, BETA:0, ADVANCE_RATIO:sweep | 0.30,0.35,0.45,0.50,0.55,0.60,0.70 | tud_xprop.su2 | qsteady_rotor
```

`J = V / (n D)`: `V` is the free-stream speed of the point (`MACH_NUMBER` times `sqrt(GAMMA R T)` of the deck), `n` the speed in revolutions per second and `D` the rotor's `diameter_m`. cfdbender writes `ROTATION_RATE` as `rpm_sign 2 pi n` along the shaft in rad/s, `MOTION_ORIGIN` as the hub and `MACH_MOTION` as the free-stream Mach number unless the setup states it (a hover, with no free stream, takes the tip Mach number). For `V = 18.0017 m/s`, `D = 0.40741537 m`, `J = 0.4116` and `rpm_sign = -1` it writes about `-674.44 0 0`. Each `J` is its own point, named like pyflightstream's: `DP-M053AL+000BE+000J+041`.

What is refused: an advance ratio or a speed that is not positive, both `ADVANCE_RATIO` and `RPM`, no rotor declaration, a `ROTATION_RATE` stated as well (two sources of one quantity), a `MOTION_ORIGIN` that differs from the hub. A `qsteady_rotor` row with a nonzero `ALPHA` or `BETA` is refused: an angle makes the inflow vary around the disc, which a steady rotating frame does not hold; solve it as `unsteady_rotor`.

A `steady` row whose setup states `GRID_MOVEMENT = "ROTATING_FRAME"` with an `ADVANCE_RATIO` still plans as before, to the same `su2.cfg` as `qsteady_rotor`; with a nonzero angle it warns and continues.

#### Periodic sector

The matrix column `SYMMETRY` says what was meshed: `NONE` (the default) or `PERIODIC` (SU2 reads no `MIRROR`; a mirror plane is a marker of the setup). With `PERIODIC` the row variable `PERIODIC_COPIES` is required, a positive whole number. `SYMMETRY_LOADS` (`true` or `false`, or `-` for the setup's `symmetry_loads`, else `true`) says whether the loads are the whole rotor's or the sector's:

```text
POL  | RUN | SETUP | REF | PPROC | SYMMETRY | SYMMETRY_LOADS | WORKFLOW      | VAR_NAMES_VALUES
-----|-----|-------|-----|-------|----------|----------------|---------------|-----------------
1001 | 1   | 001   | 001 | 001   | PERIODIC | true           | qsteady_rotor | hpc: h004 / PERIODIC_COPIES: 6
```

On a `qsteady_rotor` row the setup may name its periodic pair, `periodic_markers = ["left", "right"]` (a provisional name), listed so that the first, turned right-handed about the shaft by `360 / PERIODIC_COPIES` degrees, lands on the second; cfdbender then writes `MARKER_PERIODIC`. A `MARKER_PERIODIC` the setup states itself wins. These warnings check the row against the setup: `MARKER_PERIODIC` turns by an angle whose `360 / angle` differs from `PERIODIC_COPIES`; `SYMMETRY` is `PERIODIC` and no `MARKER_PERIODIC` is written; the setup states `MARKER_PERIODIC` beside a rotor while `SYMMETRY` is not `PERIODIC`.

#### Time-accurate rows

A row of `unsteady` or `unsteady_rotor` states its clock and its averaging window; cfdbender supplies neither:

- the clock, `DELTA_THETA` (degrees per step) and `REVOLUTIONS` against the rotor speed, or `DELTA_TIME` (s) and `TIME_ITERATIONS` (steps): `TIME_STEP = DELTA_THETA / (6 abs(rpm))` and `TIME_ITER = REVOLUTIONS 360 / DELTA_THETA`, a whole number of steps;
- the window, `LAST_REVS_AVG` on a row that turns a rotor or `LAST_ITERS_AVG`, never both.

```text
POL  | RUN | SETUP | REF | PPROC | FLIGHT_CONDITION         | MESH       | WORKFLOW       | VAR_NAMES_VALUES
-----|-----|-------|-----|-------|--------------------------|------------|----------------|-----------------
2001 | 1   | 002   | 002 | 001   | MACH:0.1, ALPHA:5, RPM:972 | wllk.su2 | unsteady_rotor | DELTA_THETA: 1.5 / REVOLUTIONS: 5 / LAST_REVS_AVG: 2.5
```

That row writes `TIME_STEP= 0.000257201646090535`, `TIME_ITER= 1200` and `WINDOW_START_ITER= 600`.

An `unsteady` row, nothing turning, states the clock in seconds and the window in steps:

```text
POL  | RUN | SETUP | REF | PPROC | FLIGHT_CONDITION  | MESH     | WORKFLOW | VAR_NAMES_VALUES
-----|-----|-------|-----|-------|-------------------|----------|----------|-----------------
3001 | 1   | 003   | 001 | 001   | MACH:0.1, ALPHA:4 | wing.su2 | unsteady | DELTA_TIME: 0.00025 / TIME_ITERATIONS: 480 / LAST_ITERS_AVG: 100
```

It writes `TIME_STEP= 0.00025`, `TIME_ITER= 480` and `WINDOW_START_ITER= 380`: the window is steps 381 to 480. SU2's time iteration `k` (0-based) is cfdbender's step `k + 1`; `INNER_ITER` is the setup's `iterations`. `TIME_MARCHING` is `DUAL_TIME_STEPPING-2ND_ORDER` unless the setup states another.

SU2 8.6.0 refuses `WRT_RESTART_OVERWRITE`, `WRT_SURFACE_OVERWRITE` and `WRT_VOLUME_OVERWRITE` set to `NO` for a time-accurate run (it numbers each time step's files itself), so a time-accurate row whose setup states one of them `NO` is refused at `plan` naming the key, as other native values SU2 cannot run are; `YES`, SU2's default, passes, and cfdbender writes none of them.

A time-accurate row whose `RESTART` names a steady point starts from that point's solution. SU2 8.6.0 starts a dual-time run from a solution only as a restart at a time iteration `RESTART_ITER`, reading the files of the iterations before it (`<SOLUTION_FILENAME stem>_<k>`, five digits: `RESTART_ITER - 2` and `RESTART_ITER - 1` under second-order dual time stepping, `RESTART_ITER - 1` otherwise), and it refuses `RESTART_ITER= 0`: a second-order run cannot start from a single steady file without that offset. cfdbender therefore stages the steady solution under those names (`restart-input_00000.dat` and `restart-input_00001.dat`, one state repeated) and writes `RESTART_SOL= YES`, `SOLUTION_FILENAME= restart-input.dat` and `RESTART_ITER= 2` (1 under first-order marching) itself, and counts every step in the row's own numbers: step `s` is SU2's time iteration `s - 1 + RESTART_ITER`, `TIME_ITER` is the row's steps plus `RESTART_ITER`, and `WINDOW_START_ITER` and the export window are shifted the same way. So the point marches every step the row asks for: `DELTA_THETA: 5 / REVOLUTIONS: 1` writes `TIME_ITER= 74`, SU2 marches time iterations 2 to 73, and the history carries steps 1 to 72, a complete revolution. The surface and volume files keep SU2's numbering (`surface_flow_00073.vtu` is step 72); the products and the tables are in steps. On a rigid motion, SU2 turns the mesh by one time step from the steady solution's position before step 1, so step `s` is at `s DELTA_THETA` from it (inferred from SU2's source, `CVolumetricMovement::Rigid_Rotation`). SU2 prints `WARNING: The restart file does not contain grid velocities, these will be set to zero.` at this start, because a steady solution holds none.

An `unsteady_rotor` setup turns the whole mesh (`GRID_MOVEMENT= RIGID_MOTION`) unless it states a sliding interface (provisional names):

```toml
[sliding]
zones = 2                         # the zones of the mesh file
rotating_zone = 1                 # the zone that turns
interface = ["X_BOXD1", "X_BOX1"] # [stationary side, rotating side]
```

cfdbender then writes `MULTIZONE= YES`, `CONFIG_LIST= ( zone_0.cfg, zone_1.cfg )`, the interface pair as `MARKER_ZONE_INTERFACE` and `MARKER_FLUID_INTERFACE`, `KIND_INTERPOLATION= NEAREST_NEIGHBOR`, `OUTER_ITER` from `iterations` with `INNER_ITER= 1`, per-zone history groups, and one file per zone beside `su2.cfg`: the turning zone's states `GRID_MOVEMENT= RIGID_MOTION`, `MOTION_ORIGIN`, `ROTATION_RATE` and `MACH_MOTION`, the others `GRID_MOVEMENT= NONE`.

`NEAREST_NEIGHBOR`, SU2's own default, is the interpolation of the interface unless the setup states `KIND_INTERPOLATION`, whose value is kept; `WEIGHTED_AVERAGE` was seen to diverge at the first step on SU2 8.3.0 and 8.6.0.

SU2 writes a multizone run's history under the name of the master configuration, `su2.csv`, whatever `CONV_FILENAME` states, with the residual and load columns of each zone suffixed by the zone (`rms[Rho][0]`, `CD[1]`). cfdbender reads that file, takes the point's loads (`CL`, `CD`, `CFx` to `CMz`, the rotor quantities and the time average) from the turning zone, records the residual of every zone, and sums the per-surface columns into the groups of the PPROC file (see "Plot groups" below). The surface and volume files of a sliding-mesh point are one per zone and step, `surface_flow_<zone>_<iteration>.vtu` and `flow_<zone>_<iteration>.vtu` (the zone, counted from 0, before SU2's five-digit time iteration; the names of SU2 8.3.0's two-zone runs), and the declared products are read from them: see "Products on a sliding-mesh point" below.

A sliding-mesh `unsteady_rotor` row may start from a steady frozen-rotor solution of its own mesh: write `MRF` in its `RESTART` cell (`-` stays a cold start). The point then runs two solver stages in its one folder and one job. Stage 1, `precursor.cfg`, is the same two-zone mesh at the same condition (the same angle of attack, Mach number and rotation rate) solved steady, with the turning zone in `GRID_MOVEMENT= ROTATING_FRAME` (`pre_zone_0.cfg` and its siblings carry the same `ROTATION_RATE`, `MOTION_ORIGIN` and `MACH_MOTION` as the rigid-motion zone files) and the other zones fixed. Stage 2 is the time-accurate run described above, with `RESTART_SOL= YES`, `RESTART_ITER= 2` (1 under first-order marching), `SOLUTION_FILENAME= solution.dat` and `TIME_ITER` the steps asked plus `RESTART_ITER`. The job copies each zone's stage-1 restart file to `solution_<zone>_00000.dat` and `solution_<zone>_00001.dat` (the first only under first-order marching) with `cp`, so stage 2 reads the iterations before `RESTART_ITER` as a restart. Both stages come from one rendering of the setup: stage 1 carries the point's own keys (solver, turbulence model, numerics, markers, interface) except those only a time-accurate or restarted run reads (`TIME_*`, `WINDOW_*`, `RESTART_*`, `SOLUTION_FILENAME`, the convergence keys and the output frequencies), with `TIME_DOMAIN= NO`, `INNER_ITER= 1`, `OUTER_ITER` the precursor iterations, and its own file names (`precursor_restart.dat`, `precursor_flow`, `precursor_surface`; its history is `precursor.csv`, named after its master file, and is never read as the point's history).

The setup states the stage in an optional table (provisional names):

```toml
[precursor]
iterations = 300      # outer iterations of stage 1, default 300
converge_to = -9      # optional: stop earlier at this density residual of the turning zone
```

Without `converge_to` the iteration count is exact. With it, `iterations` is the cap, and a stage that reaches the cap without meeting the level is not a failure: the point's `stdout.log` gets one `PRECURSOR_WARNING` line and stage 2 runs. SU2 8.6.0 ignores a convergence field without a zone index in a multizone run, so cfdbender writes `CONV_FIELD= RMS_DENSITY[<turning zone>]`.

The clock and the windows keep their meaning. SU2 turns the mesh by one step before the first marched iteration whatever `RESTART_ITER` is, so step `s` is the mesh-file azimuth plus `s DELTA_THETA`; `RESTART_ITER` is a file-name and `TIME_ITER` offset and nothing geometric. `TIME_ITER`, `WINDOW_START_ITER` and the export window carry the offset exactly as for a point started from a saved solution; the history is counted in the row's own steps (`Time_Iter` 2 is step 1), the averaging window is counted back from the last marched step, and N steps asked are N steps marched. A row with `MRF` and the same row with `-` differ in `RESTART_SOL`, `RESTART_ITER`, `SOLUTION_FILENAME`, `TIME_ITER`, `WINDOW_START_ITER` and the output frequency that follows `TIME_ITER`, and nothing else; a row with `-` renders as before.

The job runs the stages through one launcher. A point with `MRF` has `run_mrf.sh` in its folder, and the command of its job is `sh run_mrf.sh <the command the row would run>`: the script calls that command once per stage with the stage's configuration file as its last argument (`precursor.cfg`, then `su2.cfg`, then `su2_segment2.cfg` when the point has an export window), as the command already receives `su2.cfg`. A launcher or an `hpc.application_id` wrapper script that passes its arguments on to `SU2_CFD` needs no change; see [cluster jobs](cluster-jobs.md#a-point-that-starts-from-a-frozen-rotor-solution). The script is POSIX shell, `cp` and `grep`; no Python runs on the node. A stage 1 that fails is never followed by stage 2, and the point never falls back to a cold start. The point's `job.exit` says which check failed: 91 (the command ended with a non-zero status, or did not run the configuration file it was given, so no precursor ran), 92 (`precursor.csv` holds NaN), 93 (a restart file of stage 1 is missing or empty), 94 (a copy failed); one line `PRECURSOR_FAILED: ...` is in the point's `stderr.log`. The statuses 97 and 98 belong to a RESTART source staged inside the same job and are not used here. A stage 2 that fails returns its own status. A local `run` does the same checks and the same copies in Python and records the same statuses in `execution.json`.

`MRF` on a row that is not time-accurate (`steady`, `qsteady_rotor`) is warned and skipped, because such a row is solved steady already and the entry changes no result; `--warning-refusals` makes the skip a refusal. On a time-accurate row it is refused at `plan` when the row is not `unsteady_rotor` or the setup has no `[sliding]` table (a cold start is the `-` cell, and skipping would start the row cold, not as asked), when `KIND_INTERPOLATION` is `WEIGHTED_AVERAGE` (a steady run with it diverges at the first iteration), when the restart format is ASCII, and when `[precursor]` states a bad value. A `RESTART` naming a point on a sliding row stays refused. `plan` prints one line for a point with the stage: `precursor: steady frozen-rotor stage first, 300 iterations`.

The reported result of a time-accurate point is the mean over the window of each history column (equal weights, one row per time step, the last row of each step), and its rotor row is formed from the mean force and moment. A history that does not hold every step of the window leaves the point out with the reason. `post/<matrix>/polars/P<sim>_<matrix>_uns_avg.csv` holds one row per point: `POL`, `FIRST_STEP`, `LAST_STEP`, `STEPS`, the condition block, `XMOM`, `YMOM`, `ZMOM`, the averaged history columns, then `RPM_<alias>` and `DIAMETER_<alias>`. The point's result also holds one row per complete revolution with each column's mean and its drift from the previous revolution, and a drift of the last revolution above `[per_revolution] drift_limit_pct` of the PPROC file (default 1 per cent) warns, never blocks. The column names are those of a real SU2 8.6.0 time-accurate history: `Time_Iter` counts the time steps and the loads are `CFx` to `CMz`; with `AERO_COEFF_SURF` in `HISTORY_OUTPUT` the loads of one marker are `CFx(blade)`, the marker in round brackets. A multizone history appends the zone index in square brackets, `CFx[1]` and `CFx[1](blade)`, as SU2 forms them in its multizone output; no multizone time-accurate history has been run. `solvers/su2/_time_history.py` keeps these names in one table.

#### Export window

A time-accurate row may state one of pyflightstream's `EXPORT_UNSTEADY_LAST_REV`, `EXPORT_UNSTEADY_LAST_ITER`, `EXPORT_UNSTEADY_AFTER_REV` or `EXPORT_UNSTEADY_AFTER_ITER`: the surface and volume files are written at every step from the window's first step to the last. SU2 writes its files at one frequency for a whole run, so such a point runs in two segments of one job: `su2.cfg` marches to the step before the window writing restart files only, `su2_segment2.cfg` restarts there (`RESTART_SOL= YES`, `RESTART_ITER`) and writes every step to the end, and `run_segments.sh` runs the two in order (`sh run_segments.sh mpirun -np 64 SU2_CFD`). Verified on a real SU2 8.6.0 run: segment two at `RESTART_ITER= R` reads `<SOLUTION_FILENAME stem>_<R-2>` and `_<R-1>` (`restart_flow_00058.dat` and `restart_flow_00059.dat` for `R = 60`), which segment one wrote under its `RESTART_FILENAME`, so `su2_segment2.cfg` states `SOLUTION_FILENAME` as segment one's `RESTART_FILENAME` (SU2's default `restart` when the setup states none) and `READ_BINARY_RESTART= YES`; the two segments write `history_<first time iteration>.csv` each (or `history.csv` for a cold first segment), and the history across the hand-over agrees with an uninterrupted run to the run-to-run noise of the solver (about 1e-5 in `CFx` at the first step after it). `RESTART_ITER` of segment two is the window's first step minus 1 plus the start offset of a point started from a steady solution. A window that begins so early that segment one would march fewer steps than the restart reads (two under second-order dual time, so a window from step 2) runs as one segment writing every step.

What a time-accurate point writes to disk. SU2 8.6.0 writes each file of `OUTPUT_FILES` at every time iteration that is a multiple of its `OUTPUT_WRT_FREQ` (one frequency per file, in the same order) and always at its last time iteration, and under second-order dual time stepping it also writes a restart file one iteration before each multiple and before the last, so a continuation has its two previous states (measured); without `OUTPUT_WRT_FREQ` a transient run writes every file at every step. Following pyflightstream, whose unsteady point with no `EXPORT_UNSTEADY_*` key exports its end-of-run files only, a row with no export window gets `OUTPUT_WRT_FREQ` equal to `TIME_ITER` for every file, so the volume and surface files are written at the last step and the restart file at the last two (and, on a cold start, at step 1 too, because every frequency divides time iteration 0); a setup that states `OUTPUT_WRT_FREQ`, or the PPROC `output_frequency`, keeps its own. With an export window, segment one writes restart files only, with its frequency equal to its `TIME_ITER`, which leaves exactly the two segment two reads, and segment two writes the volume and surface files at every step of the window and its restart files at its last two steps. SU2 also refuses a restart whose `WINDOW_START_ITER` is below `RESTART_ITER` (`WINDOW_START_ITER must be larger than or equal to: RESTART_ITER!`, measured), so when the averaging window begins before the export window segment two states `WINDOW_START_ITER` equal to its `RESTART_ITER`: SU2's own `tavg[...]` columns then average segment two only, while cfdbender's average reads the joined history over the whole averaging window. SU2 has no key that keeps fewer files than that, and pyflightstream has no key to keep or remove superseded restart files, so cfdbender deletes nothing. Measured sizes per written step on a 476,445-point mesh (8,626 blade points): `flow_<k>.vtu` 88.1 MiB, `surface_flow_<k>.vtu` 1.1 MiB, `restart_<k>.dat` 43.6 MiB; a row with an export window of `n` steps writes about `n` times 89 MiB plus four restart files, where a run writing every file at every step wrote about 133 MiB per step (about 9.8 GB for 70 steps). The export window and the averaging window are independent: either may begin first. Two keys, a value that is not positive, a window longer than the run and a threshold beyond the run are refused; a revolutions form on a row with no rotor clock warns and is skipped.

#### Turbulence model

The row variable `TURBULENCE_MODEL` (a provisional name: pyflightstream has none) names the model, `SA` or `SST`, and SU2 writes `KIND_TURB_MODEL` (with the SA-neg options below for `SA`). Every model is fully turbulent: cfdbender never writes a transition model. A setup that states its own `KIND_TURB_MODEL` wins, with one warning when the two differ; another name is refused, listing `SA` and `SST`.

### Rotor coefficients

When the reference file declares a rotor and the point turns it, `collect` adds a `rotor` mapping to the point's result: the force and moment the adapter supplies, the condition and the rotor row. The rotor table is pyflightstream's `post/<matrix>/polars/P<sim>-<alias>_rotor.csv`, one row per point, with its columns in its order: `POL`, `ROTOR`, the condition block (`ALPHA`, `BETA`, `MACH`, `RE`, `VINF`, `VREF`, `ALT`, `RHO`, `TEMP`, `MU`, `J`, `J_CLOCK`, `RPM_CLOCK`, `SREF`, `CREF`), then `RPM_<alias>`, `DIAMETER_<alias>`, `J_<alias>`, `CT_<alias>`, `CQ_<alias>`, `CP_<alias>`, `ETA_<alias>`, `CN_<alias>`, `CS_<alias>`, `CMN_<alias>` and `CMS_<alias>`. Absent, and named under `absent_columns` in `products.json`: `BREF` (SU2 has no reference span), and `ETAW`, `MTIP`, `MHEL` and `MADV`, which cfdbender does not compute yet. For a `SYMMETRY = PERIODIC` row the polar tables keep the sector's own coefficients, as SU2 computed them; the whole-rotor loads are in the rotor table.

SU2's coefficients are of the monitored surface. The adapter turns them into a force and a moment, and cfdbender forms every rotor coefficient from those vectors, with pyflightstream's definitions and signs. With `q = 0.5 rho V_q^2` the dynamic pressure SU2 divides by (`V_q` is `MACH_MOTION` times the speed of sound), `A` and `L` the reference area and length, `r` the moment origin, `o` the hub and `a` the unit shaft:

```text
F    = q A (CFx, CFy, CFz)                    the monitored force
M_o  = q A L (CMx, CMy, CMz) + (r - o) x F    its moment about the hub
F_w  = sum_j R_j F        M_w = sum_j R_j M_o     R_j: turn by 360 j / m degrees about the shaft
T    = F_w . a            Q = M_w . a         thrust and torque along the shaft
n    = rpm_sign rev/s
J    = V / (abs(n) D)
CT   = T / (rho n^2 D^4)    CQ = Q / (rho n^2 D^5)    CP = 2 pi CQ sign(n)    ETA = J CT / CP
CN, CS = F_w . N, F_w . S over rho n^2 D^4    CMN, CMS = M_w . N, M_w . S over rho n^2 D^5
```

`N` is the part of the mesh frame's up (`+z`) square to the shaft and `S = N x a`; the four in-plane coefficients are `NA` for a shaft along up. With `SYMMETRY = PERIODIC` and `SYMMETRY_LOADS` true the whole-rotor loads are the vector sum over `m = PERIODIC_COPIES` copies, never a multiplication; with `SYMMETRY_LOADS` false they are the sector's. Thrust is counted along the declared shaft, so with `axis = "X"` and the flow along +x a pulling propeller has a negative `CT`, as in pyflightstream; `ETA` is `NA` where `CP` is zero or the free stream is. The density `rho` is `FREESTREAM_DENSITY` under `FREESTREAM_OPTION = "DENSITY_FS"`; for a viscous deck that SU2 initializes from `REYNOLDS_NUMBER` it is `Re mu / (V L)`; otherwise `p / (R T)`.

The same quantities hold for a `steady` point whose rotation you typed by hand (`GRID_MOVEMENT` and `ROTATION_RATE` in the setup), as long as the reference file declares the rotor.

### Post-processing products

The PPROC file declares sections, a volume cut and probes with pyflightstream's tables, keys, defaults and refusals: `[sections]` with `[[sections.distributions]]`, one `[volume_section]`, and `[[probes]]` entries. A malformed table is refused at `plan` with pyflightstream's message. When the PPROC file declares one, SU2 is asked for the file it is read from, and only then: `OUTPUT_FILES` gains `SURFACE_PARAVIEW` for sections and `PARAVIEW` for a volume cut or probes (a list the setup states keeps its entries). `collect` and `post` then write, under `post/<matrix>/`:

| File | From | Holds |
|---|---|---|
| `sections/P<POL>/<point>_cp_<name>.csv` | `surface_flow.vtu` | One row per section and chordwise station: `POL, STEP, time_s, FAMILY, PLANE, ROTOR, AZIMUTH`, the condition block, `SECTION` and the Cp columns. |
| `sections/P<POL>/<point>_sloads_<name>.csv` | `surface_flow.vtu` | One row per section: the same leading columns, `Offset, Chord, X_QC, Z_QC, Fx, Fz, Moment`, and with `integrate = true` `Strip_length, Fx_int, Fz_int, My_int`; then the pressure and friction parts (below). |
| `fields/P<POL>/<point>_vsec.vtk` or `.dat` | `flow.vtu` | The velocity interpolated at the laid points of the cut, with its `.provenance.json`. |
| `probes/P<POL>/<point>_probes.csv` | `flow.vtu` | `POL, PROBE, X, Y, Z, FRAME, STEP`, the condition block, then the sampled parameters; a sampled `MACH` is `MACH_LOCAL`. |

`<name>` is the entry's `families` word, as pyflightstream names it. A section is cut by cfdbender from the surface file and its loads integrated there, the same way for every solver; a volume cut and probes are interpolated inside the cells of the volume file at the exact declared positions. A probe point that lies in no cell keeps its row, and its value cells hold `NA`; a volume cut drops such a point. The frames a declaration may name are built from the reference file: `REFERENCE`, `MRP` (at the moment origin), and for the declared rotor `<ALIAS>_SMRP` (the hub frame: the shaft letter along the installed shaft, the `blade1.zero` letter toward that direction projected onto the disc), `<ALIAS>_BMRP`, `<ALIAS>_RMRP`, `<ALIAS>_RMRP<k>`, `SMRP` and `RMRP`. A probe entry with no `frame` is in the rotor's hub frame.

```toml
history = true

[sections]
count = 20

[[sections.distributions]]
families = "blade"
frame = "REFERENCE"
planes = ["XZ"]       # the offset runs along y, the blade span
integrate = true

[volume_section]
shape = "circle"
frame = "REFERENCE"
plane = "YZ"
offset_m = 0.05       # downstream of the blade
radii_m = [0.03, 0.2]
points = [8, 36]

[[probes]]
frame = "REFERENCE"
points = 3
parameters = ["VX", "VY", "VZ", "MACH", "CP_FREE"]

[[probes.lines]]
start = [0.05, 0.08, 0.0]
end = [0.05, 0.16, 0.0]
```

On a `steady` or `qsteady_rotor` point each product is one instant, and `STEP` of a sections file is the solver iteration of the files. On a time-accurate point with an export window, the sections files and a probes entry of `kind = "instantaneous"` hold every exported step in order (`products.json` says `"kind": "history"`), and the volume cut is one file per step, `_step_<STEP>` before the extension; an entry of `kind = "default"` samples the last step. Without an export window a time-accurate point writes its last step, one instant. A turning frame (`<ALIAS>_RMRP`, `<ALIAS>_RMRP<k>`) follows blade one's azimuth at each step, and `AZIMUTH` states where the block's blade is.

What differs from pyflightstream, and why:

- Each product file of a point is in a folder of its polar, `P<POL>/`, because a cfdbender point name does not carry its polar.
- Columns SU2 cannot fill are absent, never empty, and `products.json` names each with its reason: the boundary-layer columns `Theta`, `Delta*`, `Delta`, `H` of the Cp file and the `BL_*` probe parameters.
- The plot parameters SU2 cannot fill are `CDI`, `CDO` (no induced or profile drag split), `CDP` and `CDV` (no pressure or viscous drag column in its history); they are absent from the unsteady polar with their reason in `products.json`. `[plots]` applies to time-accurate points, as in the sister library; the families of a group are declared in the post file's `[plots.families]` (a key this library adds) because a SU2 mesh carries wall markers, not families; groups in a rotor's own frame and the eighteen axis coefficients of the sister library's unsteady polar are not produced.
- SU2's surface file does not say which cell is of which marker, so a distribution cuts all of `MARKER_PLOTTING` or none of it; plot only the markers you cut.
- A field solver with a wall shear field offers more than a panel solver: in its sectional-loads file `Fx`, `Fz` and `Moment` (and with `integrate = true` `Fx_int`, `Fz_int`, `My_int`) are the totals, pressure plus friction, and the two parts follow them as extra columns, `Fx_pres, Fz_pres, Moment_pres, Fx_fric, Fz_fric, Moment_fric` and with `integrate = true` `Fx_pres_int, Fz_pres_int, My_pres_int, Fx_fric_int, Fz_fric_int, My_fric_int`, with the same axes, moment point, signs and strip rule. SU2's wall shear is `Skin_Friction_Coefficient` times the dynamic pressure of the Cp reference. Without a shear field the load columns are the pressure loads, as pyflightstream writes them, and the part columns are absent and named in `products.json`.
- Provisional readings: the Cp columns refer to the free stream (`p_inf` and `0.5 rho V^2` of the deck), also in a rotating frame; `Cp_ref` and `CP_REF` are SU2's own `Pressure_Coefficient`; `CF` is the magnitude of `Skin_Friction_Coefficient`; the winding normal of a surface cell is taken to point into the fluid; the velocity is absolute.
- Not offered, warned and skipped: `include_symmetry = true` (no symmetry copy of a section is made), `attach = "body"`, a probe entry's `field_formats`, `reusable_inflow`, `lattices`, `grids` and `points_file`, `families = "each"`, and the frames `LOCAL_AXIS`, `BLADE_AXIS` and those of a `[[frames]]` table. The periodic copies of a sector are not sectioned.

### Plot groups: the loads of parts of the model

A time-accurate point's PPROC file may declare `[plots]` with pyflightstream's keys, defaults and refusals: `parameters` (default `CL`, `CDI`, `CDO`, `CD`, `FX`, `FY`, `FZ`, `MX`, `MY`, `MZ`; `CDP` and `CDV` only when named) and `[[plots.groups]]` entries with a `name`, a `frame` (default `MRP`) and `families` (`"all"`, `"each"`, or a word or list of words). A group is named by its `name`, and its plot columns are `<parameter>_<group>`: `CL_MRP_TOTAL`, `FX_MRP_W`. A name carries `{family}` exactly when the entry is `families = "each"`, one group per family. `plan` refuses a malformed table with the sister library's message.

**A family** is a named set of wall markers, declared in the post file in one more table, `[plots.families]` (the one key this library adds to pyflightstream's `[plots]`, because the sister library takes its families from the geometry): `<family> = ["<marker>", ...]`. No marker is in two families, and a family naming a marker the point's history holds no column for is refused with the markers it has. A wall marker no family claims is a family under its own name, so the names of `"each"` are the declared families, then these markers. A word of a group's `families` is `"all"` (every monitored marker, summed by the history's own total), a family name, a wall marker, an alias of the reference file's `[aliases]` or the rotor's name; a list sums the families and markers it names. A group in a rotor's own frame (`SMRP`, `RMRP`, `LOCAL_AXIS`) is not supplied: it is skipped with the reason. No group is added for the global frame beyond pyflightstream's rule: when no group of the `MRP` frame plots the six load components, `MRP_TOTAL` over every marker is added with them.

```toml
[plots]
parameters = ["CL", "CD", "FX", "FY", "FZ", "MX", "MY", "MZ"]

[plots.families]          # an aircraft with a propeller: the airframe in parts, the propeller apart
Wing     = ["wing", "wing_te_base", "wing_tip"]
Body     = ["fuselage", "nose", "fuselage_base"]
Nacelle  = ["nacelle", "nacelle_base"]
Spinner  = ["spinner"]
Blade1   = ["blade_1"]
Blade2   = ["blade_2"]

[[plots.groups]]
name = "MRP_TOTAL"        # CL_MRP_TOTAL: every wall of every zone
frame = "MRP"
families = "all"
[[plots.groups]]
name = "MRP_{family}"     # CL_MRP_Wing, CL_MRP_Body, CL_MRP_Blade1, ...
frame = "MRP"
families = "each"
[[plots.groups]]
name = "MRP_AIRFRAME"     # a group composed of several families
frame = "MRP"
families = ["Wing", "Body", "Nacelle"]
```

Where the numbers come from. Every SU2 point's configuration asks for SU2's per-surface history group (`AERO_COEFF_SURF`, and `AERO_COEFF_SURF[<zone>]` on a multizone point), so the history holds `CL(wing)`, `CFx(wing)` and the rest for every monitored wall marker (`CFx[1](blade_1)`, the zone before the marker, on a multizone point). A group's value is the sum of its markers' coefficients: SU2 forms every one with the same reference area, length and moment origin, the point's. `CL` and `CD` are the wind-axis coefficients; `FX` to `FZ` are forces in N and `MX` to `MZ` moments in N m about the reference moment origin, in SU2's body axes (`q A` and `q A L` times the coefficient, with the dynamic pressure SU2 scales by). `CDI`, `CDO`, `CDP` and `CDV` are not columns: SU2 does not split drag, so each is absent from the table, not `NA`, with its reason in `products.json` under `skipped` (`polars/P<sim>_<matrix>_uns_avg.csv#plots:CDI,point=<id>`). A group whose markers carry exactly zero load in every row (a wall that is not monitored in that zone) is absent the same way. The columns come from the same rows and window as every other column: the window means are in `polars/P<sim>_<matrix>_uns_avg.csv`, ahead of the history's own columns, and each revolution's means and drifts are in the point's per-revolution table. The point's `result.plots` names the columns, what is absent and what was skipped.

A multizone point whose post file declares no `[plots]` gets the two groups its per-zone totals can fill, with no per-surface data: `MRP_TOTAL` (the sum of the zones, so the airplane with its propeller) and `ROTOR_<alias>` (the turning zone, the name the sister library gives a rotor's own group); the airframe alone is their difference. A point with one zone and no `[plots]` gets no plot column, so its tables are unchanged. The turning zone is the one whose zone file states a grid motion, and the setup's `[sliding] rotating_zone` must name it: a disagreement is refused.

To read the propeller's effect on the airframe, run the aircraft with the propeller and without it, two rows of one matrix, and subtract `CL_MRP_AIRFRAME` of the second from the first; the same group names make the columns line up.

A new grouping needs only a new post, never a new plan or a new run. Edit the PPROC file and run `cfd-matrix post <matrix> [--points ...]` (or `collect`): both read the saved history again, with the edited PPROC file accepted in place of the planned one (a run never accepts it), and replace `post/<matrix>/` in place; `--archive` keeps a copy of each replaced table under `archive/<stamp>/`.

### Products on a sliding-mesh point

A multizone point's products are written with the same files, names and columns as a single-zone point's, from the per-zone files; no `ZONE` column is needed, because a probe or a station is one row whichever zone supplies it.

- A probe point or a volume-cut point is located in the zone volumes in zone order and takes the value of the first zone whose volume holds it. A point in no zone is reported as a single-zone miss is: its probe row is `NA`, the cut leaves it out, and `products.json` counts it under `points_not_found`. Only the fields every zone's file holds are sampled, so a column is never filled from some zones alone. A step is used only when the volume file of every zone is present; the missing ones are named in the manifest with the reason.
- A section distribution cuts the surface file of the zone that holds its markers. The zone of each marker is read from the point folder's `mesh.su2` (`IZONE=` opens a zone, each `MARKER_TAG=` after it belongs to it); without that file the entry is skipped with the reason. SU2's surface file does not say which cell is of which marker, so an entry cuts all of the markers of a zone or none, as on a single-zone point; markers that span both zones cut both files, joined in one surface, and the loads of both bodies are summed at each station. `sources` in the manifest lists the files used.
- On a time-accurate point `STEP`, `time_s` and the export window are those of the single-zone path: the steps of the window, one file per zone each, the last step without a window.
- The frame of the sampled coordinates: the stationary zone's coordinates are fixed. SU2 writes the grid of a zone in rigid motion as it is at the time iteration of the file, moved, so the rotating zone's coordinates in a file of step `s` are inertial (space-fixed) coordinates at the instant of that step, and a probe or a section is placed at inertial positions as on a single-zone rigid-motion point. This follows SU2's moving-grid output; it has not been checked against a two-zone VTU file of a real run.

A product whose file, field, family or frame is missing is left out with the reason in `products.json` and `post.log`, and never stops another product or the polar.

## What converged means

cfdbender reads `history.csv` and the SU2 log (`stdout.log` and `stderr.log`). The history gives the residuals and coefficients. The log gives the verdict, because a clean exit says nothing about numerics.

A time-accurate point reads the history of every configuration it ran, under the name SU2 8.6.0 gives it: `history.csv` for a cold start and `history_<RESTART_ITER>.csv`, five digits, for a restarted run (`history_00002.csv`, `history_00060.csv`), which writes no `history.csv`. The two segments of an export window are joined in step order into one history; a gap or an overlap between them is refused naming both files, and a missing history names every file looked for. Its iterations are the row's steps counted from 1, and the limit is the row's last step.

| Result | Rule |
|---|---|
| `converged` | The log holds a line that reads `Convergence criteria satisfied` (`All` before it and `were` before `satisfied` are accepted) or `SU2 has successfully converged`, and the exit code is 0. |
| `iteration_limit` | No such line, and the last history iteration reached `ITER`. |
| `failed` | A nonzero exit code, no readable history, or a history that stops before `ITER` without a convergence line. |

A history is unreadable when it lacks `Inner_Iter`, a residual column or a coefficient column, holds a non-finite value, or its iterations do not increase. The residual of an iteration is the residual SU2 converges on: the `rms[...]` column of `CONV_FIELD` (`rms[Rho]` for SU2's default `RMS_DENSITY`, `rms[RhoE]` for `RMS_ENERGY`, `rms[nu]` for `RMS_NU_TILDE`), or the largest of them when `CONV_FIELD` names several, converted from log10 to a linear value. The result records the column under `residual_field`, for example `"residual_field": "rms[Rho]"`, and the `residual` column of the `collect` tables is that value at the last iteration. A `CONV_FIELD` that names no residual column, such as a Cauchy criterion on `DRAG`, records the largest `rms[...]` column, named `largest rms`. Iteration numbers start at 1.

cfdbender reports the coefficients SU2 wrote, with the axis system SU2 uses. SU2 normalizes them with the reference area, length and moment origin of the reference file.

| Name | Meaning | Axes |
|---|---|---|
| `CL`, `CD`, `CSF` | lift, drag, side force | wind: z, x, y |
| `CFx`, `CFy`, `CFz` | force | body x, y, z |
| `CMx`, `CMy`, `CMz` | moment | body x, y, z |

SU2's `CL`, `CD` and `CSF` are in its wind axes and `CFx`, `CFy`, `CFz`, `CMx`, `CMy`, `CMz` in its body axes, and the result keeps each under its own name. The result also records the route that ran. In the tables of `collect` the neutral columns `CL`, `CD` and `CM` are filled from `CL`, `CD` and `CMy`, in the standard frame (nose-up positive). A two-dimensional case lies in the x-y plane, so its pitching moment is `CMz` and `CMy` is zero: when SU2 prints `Two dimensional problem.`, or the native mesh states `NDIME= 2`, `CM` is `-CMz` (SU2's `CMz` is counterclockwise, which is nose-down for a flow along +x with y up), and the result records `dimensions`; see [Results and tables](results.md).

## Worked example

A 2D airfoil with an Euler solution at Mach 0.3 and sea level, at two angles of attack. You supply your own mesh as `inputs/mesh/airfoil.su2`, with markers `airfoil` and `farfield`.

```console
cfd-workspace init campaign --solver su2
cd campaign
```

`inputs/setups/s001.toml`:

```toml
iterations = 40
convergence_field = "RMS_DENSITY"
convergence_residual = -14
SOLVER = "EULER"
MARKER_EULER = ["airfoil"]
MARKER_FAR = ["farfield"]
MARKER_MONITORING = ["airfoil"]
HISTORY_OUTPUT = ["ITER", "RMS_RES", "AERO_COEFF"]

[solver]
route = "executable"
executable = "/opt/su2/bin/SU2_CFD"
```

`inputs/references/r001.toml`:

```toml
area = 1.0
length = 1.0
moment_origin = [0.25, 0.0, 0.0]
```

`inputs/pproc/p001.toml`:

```toml
history = true
```

`inputs/matrices/airfoil.fs`:

```text
POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES|MESH
---|---|-----|---|-----|----------------|------------|----
1001|1|001  |001|001|MACH:0.3, ALTFT:0, ALPHA:sweep|0,2|airfoil.su2
```

```console
cfd-matrix plan airfoil
cfd-matrix run airfoil
cfd-matrix collect airfoil
```
