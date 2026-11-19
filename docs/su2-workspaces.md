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
time discretisation, CFL, linear solver, multigrid, iteration counts and
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

cfdbender also writes `MESH_FORMAT` to match the mesh, `CONV_FILENAME= history` and `TABULAR_FORMAT= CSV`, because the history must be a CSV file. If you state one of these with another value, it refuses with `conflict: <key> must be <value>`. The history must contain at least one `rms[...]` residual column and one coefficient column; cfdbender writes `HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )` when the setup states none, and a `HISTORY_OUTPUT` of the setup that lacks one of these groups is extended with it and a warning, because SU2's own default history holds neither residuals nor coefficients.

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
| `threads` | OpenMP threads, passed as `OMP_NUM_THREADS` | - | 1 |
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

`axis` is a letter or, for a shaft installed at an angle, `axis = { shaft = "X", vector_wrt_global = [0.996, 0, 0.087] }`: the vector is normalised and must lie within 45 degrees of the positive axis the letter names, as pyflightstream requires. `blade1` places blade one at step zero, `azimuth_deg` measured from the axis `zero` (one of the two axes that are not the shaft, with an optional sign) right-handed about the shaft; blade `k` of `B` sits `(k - 1) 360 / B` further on. Without `blade1` the datum is `zero = "X"`, which a shaft `X` refuses, as in pyflightstream. `alias`, when stated, equals the table name. The blade count is the length of `families_blades`. One rotor only, and the SU2 adapter reads the shaft `X` only (the free stream runs along +x).

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

That row writes `TIME_STEP= 0.000257201646090535`, `TIME_ITER= 1200` and `WINDOW_START_ITER= 600`. SU2's time iteration `k` (0-based) is cfdbender's step `k + 1`; `INNER_ITER` is the setup's `iterations`. `TIME_MARCHING` is `DUAL_TIME_STEPPING-2ND_ORDER` unless the setup states another.

An `unsteady_rotor` setup turns the whole mesh (`GRID_MOVEMENT= RIGID_MOTION`) unless it states a sliding interface (provisional names):

```toml
[sliding]
zones = 2                         # the zones of the mesh file
rotating_zone = 1                 # the zone that turns
interface = ["X_BOXD1", "X_BOX1"] # [stationary side, rotating side]
```

cfdbender then writes `MULTIZONE= YES`, `CONFIG_LIST= ( zone_0.cfg, zone_1.cfg )`, the interface pair as `MARKER_ZONE_INTERFACE` and `MARKER_FLUID_INTERFACE`, `KIND_INTERPOLATION= WEIGHTED_AVERAGE`, `OUTER_ITER` from `iterations` with `INNER_ITER= 1`, per-zone history groups, and one file per zone beside `su2.cfg`: the turning zone's states `GRID_MOVEMENT= RIGID_MOTION`, `MOTION_ORIGIN`, `ROTATION_RATE` and `MACH_MOTION`, the others `GRID_MOVEMENT= NONE`.

The reported result of a time-accurate point is the mean over the window of each history column (equal weights, one row per time step, the last row of each step), and its rotor row is formed from the mean force and moment. A history that does not hold every step of the window leaves the point out with the reason. `post/<matrix>/polars/P<sim>_<matrix>_uns_avg.csv` holds one row per point: `POL`, `FIRST_STEP`, `LAST_STEP`, `STEPS`, the condition block, `XMOM`, `YMOM`, `ZMOM`, the averaged history columns, then `RPM_<alias>` and `DIAMETER_<alias>`. The point's result also holds one row per complete revolution with each column's mean and its drift from the previous revolution, and a drift of the last revolution above `[per_revolution] drift_limit_pct` of the PPROC file (default 1 per cent) warns, never blocks. To verify on a real time-accurate run: the column names SU2 gives a time-accurate and a multizone history (`solvers/su2/_time_history.py` keeps them in one table).

#### Export window

A time-accurate row may state one of pyflightstream's `EXPORT_UNSTEADY_LAST_REV`, `EXPORT_UNSTEADY_LAST_ITER`, `EXPORT_UNSTEADY_AFTER_REV` or `EXPORT_UNSTEADY_AFTER_ITER`: the surface and volume files are written at every step from the window's first step to the last. SU2 writes its files at one frequency for a whole run, so such a point runs in two segments of one job: `su2.cfg` marches to the step before the window writing restart files only, `su2_segment2.cfg` restarts there (`RESTART_SOL= YES`, `RESTART_ITER`) and writes every step to the end, and `run_segments.sh` runs the two in order (`sh run_segments.sh mpirun -np 64 SU2_CFD`). That an SU2 time-accurate restart continues cleanly is to be verified on a real run. Two keys, a value that is not positive, a window longer than the run, a threshold beyond the run, and a window that begins after the averaging window are refused; a revolutions form on a row with no rotor clock warns and is skipped.

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

`N` is the part of the mesh frame's up (`+z`) square to the shaft and `S = N x a`; the four in-plane coefficients are `NA` for a shaft along up. With `SYMMETRY = PERIODIC` and `SYMMETRY_LOADS` true the whole-rotor loads are the vector sum over `m = PERIODIC_COPIES` copies, never a multiplication; with `SYMMETRY_LOADS` false they are the sector's. Thrust is counted along the declared shaft, so with `axis = "X"` and the flow along +x a pulling propeller has a negative `CT`, as in pyflightstream; `ETA` is `NA` where `CP` is zero or the free stream is. The density `rho` is `FREESTREAM_DENSITY` under `FREESTREAM_OPTION = "DENSITY_FS"`; for a viscous deck that SU2 initialises from `REYNOLDS_NUMBER` it is `Re mu / (V L)`; otherwise `p / (R T)`.

The same quantities hold for a `steady` point whose rotation you typed by hand (`GRID_MOVEMENT` and `ROTATION_RATE` in the setup), as long as the reference file declares the rotor.

## What converged means

cfdbender reads `history.csv` and the SU2 log (`stdout.log` and `stderr.log`). The history gives the residuals and coefficients. The log gives the verdict, because a clean exit says nothing about numerics.

| Result | Rule |
|---|---|
| `converged` | The log holds a line that reads `Convergence criteria satisfied` (`All` before it and `were` before `satisfied` are accepted) or `SU2 has successfully converged`, and the exit code is 0. |
| `iteration_limit` | No such line, and the last history iteration reached `ITER`. |
| `failed` | A nonzero exit code, no readable history, or a history that stops before `ITER` without a convergence line. |

A history is unreadable when it lacks `Inner_Iter`, a residual column or a coefficient column, holds a non-finite value, or its iterations do not increase. The residual of an iteration is the residual SU2 converges on: the `rms[...]` column of `CONV_FIELD` (`rms[Rho]` for SU2's default `RMS_DENSITY`, `rms[RhoE]` for `RMS_ENERGY`, `rms[nu]` for `RMS_NU_TILDE`), or the largest of them when `CONV_FIELD` names several, converted from log10 to a linear value. The result records the column under `residual_field`, for example `"residual_field": "rms[Rho]"`, and the `residual` column of the `collect` tables is that value at the last iteration. A `CONV_FIELD` that names no residual column, such as a Cauchy criterion on `DRAG`, records the largest `rms[...]` column, named `largest rms`. Iteration numbers start at 1.

cfdbender reports the coefficients SU2 wrote, with the axis system SU2 uses. SU2 normalises them with the reference area, length and moment origin of the reference file.

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
