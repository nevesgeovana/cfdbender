# SU2


You run steady SU2 cases through cfdbender: it writes `su2.cfg`, links your mesh, runs SU2 and reads the residual history and the force and moment coefficients. This page lists only what is specific to SU2.

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

SU2 reads a native `.su2` mesh or a CGNS `.cgns` mesh. Put the file in `inputs/mesh/` and name it in the matrix `GEOMETRY` column (a file name or a stem), or in the setup key `mesh`. A `mesh` key in the setup and a `GEOMETRY` cell must name the same file. Each point folder holds the link as `mesh.su2` (with `MESH_FORMAT= SU2`) or `mesh.cgns` (with `MESH_FORMAT= CGNS`); see [Workspace layout](workspace-layout.md) for how a mesh is linked.

Any other suffix is refused: `mesh must name a workspace-relative .su2 or .cgns file`.

Marker names come from the setup file, never from the mesh. Match the names and the boundary conditions to your grid.

## Setup file

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
| `mesh` | `MESH_FILENAME` | - | none (or `GEOMETRY`) |
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
| `GEOMETRY` | `MESH_FILENAME` |

A row that states only `MACH` and angles keeps the setup's flow state. When neither the row nor the setup states a Mach number, `plan` warns that SU2 will run with its own default free stream. When the air is stated without `REmi`, the Reynolds number written is the outcome of the resolved density, velocity and viscosity over the reference file's `length`. A value that the setup also states must agree with the row, or the point is refused: `conflict: <key> is <value> in the s/r/p files and <value> from the matrix FLIGHT_CONDITION; state it in one place`. `ASMPS` is refused unless it equals the speed of sound SU2 derives from the temperature: `ASMPS pins a speed of sound SU2 cannot hold: SU2 derives it from the temperature; pin TK instead`. See [Matrix columns](matrix-columns.md) for every column.

The deck states no free-stream value that SU2 recomputes and ignores. Under SU2's default `FREESTREAM_OPTION`, `TEMPERATURE_FS`, the density follows from the pressure and the temperature, so the resolved density is written only with `FREESTREAM_OPTION = "DENSITY_FS"`. A compressible viscous run (`SOLVER` `NAVIER_STOKES` or `RANS`) with a Reynolds number, under SU2's default `INIT_OPTION`, `REYNOLDS`, takes the density from the Reynolds number and recomputes the pressure from it, so the resolved pressure is not written either; `INIT_OPTION = "TD_CONDITIONS"` keeps it. An Euler run gets the pressure. A value you state in the setup is always written. A compressible deck (`SOLVER` `EULER`, `NAVIER_STOKES` or `RANS`) without `REF_DIMENSIONALIZATION` gets `REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE`, as SU2's own NACA 0012 case states: SU2's default, `DIMENSIONAL`, leaves the energy rows of the implicit system about 1e5 times the density rows, and on the TMR NACA 0012 at CFL 1000 SU2's official deck then stalls at rms[Rho] -4.4 instead of converging in 688 iterations. The coefficients do not change; the residual levels in `history.csv` are then relative to the free stream, so a `CONV_RESIDUAL_MINVAL` chosen for a dimensional run means a different level. State `REF_DIMENSIONALIZATION = "DIMENSIONAL"` to keep SU2's default.

### Propeller in a steady rotating frame

A propeller in axial flow can be run steady in a frame that turns with the blade (a moving reference frame). The setup declares the frame with SU2's own keys, the reference file describes the rotor, and the matrix row states the advance ratio `J`. cfdbender then writes the rotation of that point, so a sweep over `J` is one row.

The setup declares the rotating frame; this is the switch (a later release may offer it as a run type):

```toml
GRID_MOVEMENT = "ROTATING_FRAME"
freestream_temperature = 288.15
```

The reference file holds one rotor block, in the vocabulary of pyflightstream's reference rotor:

```toml
area = 1.0
length = 1.0
moment_origin = [0.0, 0.0, 0.0]

[rotors.PROP]
diameter_m = 0.4064        # rotor diameter D in m, the one the advance ratio uses
blades = 6                 # blades of the whole rotor
monitored_blades = 1       # blades inside the monitored surface; default: blades
origin = [0.0, 0.0, 0.0]   # hub centre in the mesh frame, m
shaft = [-1.0, 0.0, 0.0]   # direction of positive thrust (any nonzero length)
rpm_sign = 1               # 1: turns right-handed about the shaft; -1: left-handed
```

`diameter_m` must be positive, `blades` and `monitored_blades` are positive integers (`monitored_blades` cannot exceed `blades`), `origin` and `shaft` are three finite numbers and `shaft` is not the zero vector. One rotor only: a reference file with two blocks is refused with a message that multi-rotor rotating-frame runs are not supported yet. A reference file without the block behaves exactly as before.

The matrix row states `ADVANCE_RATIO` as a value or as the swept variable:

```text
POL  | FLIGHT_CONDITION                              | SWEEP_VALUES                       | ...
1001 | MACH:0.0529, ALPHA:0, BETA:0, ADVANCE_RATIO:sweep | 0.30,0.35,0.45,0.50,0.55,0.60,0.70 | ...
```

The definition is the one pyflightstream uses, `J = V / (n D)`: `V` is the free-stream speed of the point (the written `MACH_NUMBER` times the speed of sound `sqrt(GAMMA R T)` of the deck's ideal gas, with the deck's `FREESTREAM_TEMPERATURE`, else SU2's 288.15 K), `n` the rotation speed in revolutions per second and `D` the block's `diameter_m`. For each point cfdbender writes:

| Key | Value |
|---|---|
| `ROTATION_RATE` | the vector `rpm_sign * 2 pi n * shaft / abs(shaft)` in rad/s, with `n = V / (J D)` |
| `MOTION_ORIGIN` | the rotor block's `origin` |
| `MACH_MOTION` | the free-stream Mach number, unless the setup states `MACH_MOTION`: the setup wins |

The sign convention: the shaft is the direction of positive thrust, and `rpm_sign = 1` turns right-handed about it. A rotor that thrusts toward minus x, `shaft = [-1, 0, 0]` with `rpm_sign = 1`, therefore gets a negative x rotation rate. For `V = 18.0017 m/s`, `D = 0.40741537 m` and `J = 0.4116` it writes a `ROTATION_RATE` of about `-674.44 0 0` (printed to 15 digits), the rate of the case this was built from; `rpm_sign = -1` writes `674.44 0 0`.

Each `J` is its own point. The point name carries it like pyflightstream: `J` and the advance ratio times 100 with its sign, for example `DP-M053AL+000BE+000J+030`; the condition is in the record, the summary and the `J` column of the polar files. Two values of `J` that differ by less than 0.01 write the same name, and planning refuses them naming both points.

What is refused, each with its own message:

- `ADVANCE_RATIO` zero or negative, or a speed of zero (`MACH:0`);
- no rotor block in the reference file;
- a setup without `GRID_MOVEMENT = "ROTATING_FRAME"`;
- `ROTATION_RATE` stated as well, in the setup or in `VAR_NAMES_VALUES`: two sources of one quantity, and a silent choice would change the physics;
- a `MOTION_ORIGIN` in the setup that differs from the rotor's `origin`.

A steady rotating frame assumes axial flow: a nonzero `ALPHA` or `BETA` with `ADVANCE_RATIO` warns and the run continues. `RPM` is not read from `FLIGHT_CONDITION` by SU2; state `ADVANCE_RATIO`.

#### Rotor coefficients

When the reference file has a rotor block and the point is a rotating frame, `collect` adds a `rotor` mapping to the point's result and writes `post/<matrix>/rotor.csv`, one row per such point. It is a table of its own beside the polar files, so the documented columns of those files do not change; a matrix without a rotor writes none.

SU2's coefficients are of the monitored surface only. cfdbender cannot know from the configuration whether that surface is one blade of a periodic sector or the whole rotor, so the block says: `monitored_blades` is the number of blades inside the monitored surface (`1` for a one-blade sector, the default `blades` for the whole rotor), and the loads are scaled by `blades / monitored_blades`. With `q = 0.5 rho V_q^2` the dynamic pressure SU2 divides by (`V_q` is `MACH_MOTION` times the speed of sound), `A` and `L` the reference area and length, `r` the moment origin, `o` the rotor origin and `k = blades / monitored_blades`:

```text
F   = q A (CFx, CFy, CFz)
M_o = q A L (CMx, CMy, CMz) + (r - o) x F
T   = k F . shaft_hat                 thrust, positive along the shaft
Q   = -k M_o . omega_hat              torque the shaft supplies about the rotation axis
n   = abs(ROTATION_RATE) / (2 pi)     rev/s
J   = V / (n D)
CT  = T / (rho n^2 D^4)    CQ = Q / (rho n^2 D^5)    CP = 2 pi CQ    eta = J CT / CP
```

`omega_hat` is the unit vector of `ROTATION_RATE`; `Q` is positive for a propeller that absorbs power. The density `rho` is `FREESTREAM_DENSITY` under `FREESTREAM_OPTION = "DENSITY_FS"`; for a viscous deck that SU2 initialises from `REYNOLDS_NUMBER` it is `Re mu / (V L)`; otherwise `p / (R T)` with SU2's 101325 Pa when the deck states no pressure. The columns of `rotor.csv` are `POL`, `POINT`, `ROTOR`, `BLADES`, `MONITORED_BLADES`, `D`, `VINF`, `RHO`, `QINF`, `RPM`, `N`, `J`, `T`, `Q`, `CT`, `CQ`, `CP` and `ETA`; a value that does not exist (no density, `CP` zero) is `NA`. The in-plane forces and the moments about the other axes are not scaled by `k`: they do not add up over blades.

The same quantities hold for a point whose rotation you typed by hand (`GRID_MOVEMENT` and `ROTATION_RATE` in the setup), as long as the reference file has the rotor block.

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
POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES|GEOMETRY
---|---|---|---|-----|----------------|------------|--------
1001|1|001|001|001|MACH:0.3, ALTFT:0, ALPHA:sweep|0,2|airfoil.su2
```

```console
cfd-matrix plan airfoil
cfd-matrix run airfoil
cfd-matrix collect airfoil
```
