# QPROP

You compute the performance of an isolated propeller with QPROP through cfdbender, from a blade geometry and two-dimensional section polars. It plans one QPROP run per operating point, makes any missing section polar with XFOIL at each section's own Reynolds and Mach number, and collects thrust, torque, power and the coefficients in two normalisations. This page lists only what is specific to QPROP.

## Install

Install cfdbender on Windows and build QPROP and XFOIL in WSL. Download and unpack the [MIT QPROP source](https://web.mit.edu/drela/Public/web/qprop/) to `~/src/Qprop`. QPROP is double-precision Fortran 77, so build it with gfortran and the flags `-O2 -fdefault-real-8 -std=legacy`: in the supplied `bin/Makefile`, set the compiler to `gfortran`, set its flags to those three options, and run `make qprop`. The executable is `~/src/Qprop/bin/qprop`. Build XFOIL as described in [XFOIL](xfoil-workspaces.md).

| What | Setup key (`[solver]`) | Environment variable | Last resort |
|---|---|---|---|
| QPROP | `executable` | `CFDBENDER_QPROP_EXE` | `qprop` on the PATH |
| XFOIL, for section polars | `xfoil_executable` | `CFDBENDER_XFOIL_EXE` | `xfoil` on the PATH |

With a `launcher` (for example `["wsl", "-d", "Ubuntu", "--"]`), cfdbender passes both executable paths unchanged to the guest system. Give absolute Linux paths; `~` is not expanded. The launcher applies to QPROP and XFOIL alike. Without a launcher, a missing QPROP refuses `run --local` before any point is recorded: `QPROP executable not found: 'qprop'; set executable in the [solver] table of the setup file, or declare SOLVER_BUILD in inputs/executables.toml`. A missing XFOIL says the same with `XFOIL` and `xfoil_executable`.

A matrix row can also pick a build by name; see [Executables](executables.md).

## Mesh

QPROP takes a blade, not a mesh. The setup key `blade` names either one OBJ in `inputs/mesh/` or one set in `inputs/polars-2D/`. The matrix `GEOMETRY` column names the OBJ and becomes `blade`, so a row that names the blade only in `GEOMETRY` runs; a `blade` in the setup must name the same file. See [Workspace layout](workspace-layout.md) for how a mesh is linked.

```text
campaign/
  init.toml                  the solver and its section settings
  inputs/
    mesh/blade.obj           a 3D blade, in metres (or set scale)
    polars-2D/<set>/         an existing set of section cuts and polars
    setups/s001.toml
    matrices/study.fs
```

**Blade OBJ.** cfdbender slices the blade at `n_sections` radial stations (`spacing = "cosine"` or `"uniform"`) and measures radius, chord, pitch and the section shape. The OBJ follows this convention; each item is a setup key with the default shown.

| Key | Default | Meaning |
|---|---|---|
| `axis` | `"x"` | rotation axis, `"x"`, `"y"` or `"z"`; thrust points along its positive direction |
| `hub_center` | `[0, 0, 0]` | a point on the axis, in model units |
| `rotation` | `"ccw_from_front"` | counter-clockwise seen from +axis looking aft; or `"cw_from_front"`. Used by `cut = "cylinder"`; a `cut = "plane"` does not read it: the plane cut measures the motion from the blade itself |
| `cut` | `"cylinder"` | the cylinder r = const, unrolled; or `"plane"`, the plane normal to the blade's radial direction, following its skew |
| `scale` | `1` | model length unit in metres |
| `radius` | blade tip | tip radius in m |
| `mesh_frame` | `["aft", "right", "up"]` | directions of the OBJ's +x, +y and +z before conversion |

An OBJ built in another axis-aligned frame declares it at the top level of the setup:

```toml
mesh_frame = ["forward", "left", "up"]
blade = "inputs/mesh/blade.obj"
n_blades = 3
n_sections = 12
```

The declaration accepts `aft`, `forward`, `right`, `left`, `up` and `down`, one per
axis, forming a right-handed set. The blade is converted in memory to x aft, y right,
z up before either planning or slicing reads it; faces, groups and the source file are
kept. `axis` and `hub_center` refer to that expected frame. `sections.json` records the
declaration, conversion matrix and both byte digests, and `products.json` lists these
under `frames.mesh.points` by full point id. A polars-2D set accepts no mesh declaration.

The blade is the largest connected mesh component, so an OBJ may hold one blade or a full rotor. The radial direction runs from the axis to the blade centroid. The leading edge is the section point farthest ahead along the rotation direction; the trailing edge is the section point farthest from it; the chord is that distance. Beta is the angle in degrees between the chord line and the rotation plane, positive when the leading edge lies ahead along the thrust axis (nose up into the oncoming relative flow). The section shape is normalised to unit chord for XFOIL.

**Polars-2D set.** A set holds `blade.csv` with the columns `r`, `chord` and `beta` (m, m, degrees, strictly ascending radius) and optional `airfoil` and `polar` columns naming files inside the set. A station with a `polar` file uses it directly (an XFOIL polar with columns named alpha, CL and CD; the polar of an SU2 or OpenFOAM alpha sweep is such a file, see [2D polars](polars-2d.md)). A station with only an `airfoil` file gets its polar from XFOIL. Each station needs one of the two. `n_sections` may be omitted; if given it must equal the number of stations.

**Section polars.** For each section, the polar fit gives CL0, CL_a (per radian), CLmin, CLmax, CD0, CD2u, CD2l, CLCD0 and the reference Reynolds number, as QPROP's per-station input. The lift line uses alpha between -4 and 6 degrees. When XFOIL fails for a section, the point retries with alpha -4 to 10 in steps of 0.5 degrees. If that fails as well, the station reuses the nearest successful station's parameters. If no station succeeds, the point fails. `run` prints a line when each section starts and another when it ends, with the time it took, so a long point is never silent:

```text
  section 3/5 (r = 0.12): XFOIL started
  section 3/5 (r = 0.12): XFOIL converged in 4 s
```

XFOIL may not converge at some angles of a sweep. A polar that lacks some of the requested angles ends the section as `did not converge`, and the workspace says what to do, in `init.toml`:

```toml
schema_version = 1

[solver]
name = "qprop"
section_timeout_s = 60
on_unconverged_section = "warn"
```

`section_timeout_s` is the time one XFOIL section polar may run, in seconds; a value in the `[solver]` table of a setup file overrides it. `on_unconverged_section = "warn"` drops the angles XFOIL did not converge at, warns `section 5/10 (r = 0.1): XFOIL did not converge at alpha 14; those angles are dropped from this section's polar`, and fits the section on the rest. `"fail"` fails the point instead, with `did not converge at alpha 14; on_unconverged_section is 'fail' (set it to 'warn' in init.toml to run on the rest)`. Each point folder holds the XFOIL decks and results under `xfoil/sec_NN`, and `sections.json`, which records each station's radius, chord, pitch, Reynolds and Mach number, fitted parameters and their source: `xfoil`, `xfoil_narrow`, `table` or `reused`.

## Setup file

`inputs/setups/s001.toml`:

| Key | Meaning | Unit | Default |
|---|---|---|---|
| `blade` | OBJ or polars-2D set, see Mesh | - | required (or `GEOMETRY`) |
| `mesh_frame` | OBJ axes before conversion, see Mesh | - | `["aft", "right", "up"]` |
| `n_blades` | number of blades, integer from 1 | - | required |
| `n_sections` | radial stations of an OBJ; the matrix key `N_SECTIONS` overrides it | - | required for an OBJ, unless the row states it |
| `spacing`, `cut`, `axis`, `hub_center`, `rotation`, `scale`, `radius` | slicing convention, see Mesh | see Mesh | see Mesh |
| `solver_2d` | program for section polars; only `"xfoil"` | - | `"xfoil"` |
| `alpha_start`, `alpha_end`, `alpha_step` | XFOIL sweep for each section | degrees | -8, 16, 1 |
| `ncrit` | XFOIL transition criterion | - | 9 |
| `iterations` | XFOIL iteration limit | - | 150 |
| `re_exponent` | Reynolds exponent written to the propeller file | - | -0.5 |
| `title` | first line of the propeller file, one line of ASCII | - | `cfdbender QPROP` |

`alpha_end` must exceed `alpha_start`, and `alpha_step` must be above 0. Any other `solver_2d` is refused: `solver_2d must be one of: xfoil (got <value>)`. Any other key is refused as `<file>: unknown key '<key>' (closest known key: <name>); known keys: ...`.

**`[solver]` table**

| Key | Meaning | Unit | Default |
|---|---|---|---|
| `executable` | path to QPROP | - | see Install |
| `xfoil_executable` | path to XFOIL | - | see Install |
| `launcher` | tokens placed before both executables | - | `[]` |
| `timeout_s` | time after which cfdbender stops QPROP | s | 600 |
| `section_timeout_s` | time after which cfdbender stops one XFOIL section polar; also settable in `init.toml` | s | 60 |
| `on_unconverged_section` | `"warn"` or `"fail"`, see Section polars; also settable in `init.toml` | - | `"warn"` |
| `ram_estimate_gib` | memory the run is expected to need | GiB | 0.25 |

**Reference and pproc files.** QPROP takes its references from the blade and the operating point, so `inputs/references/r001.toml` and `inputs/pproc/p001.toml` may be empty. A key in either file is skipped with a warning.

**Flight condition.** The operating point is the row's `FLIGHT_CONDITION`; any key may be the sweep key.

| Key | Unit | Meaning |
|---|---|---|
| `V` | m/s | flight speed (required) |
| `RPM` or `J` | rev/min or - | rotation rate; `J = V / (n D)` with `D = 2 R` (give one) |
| `BETA07` or `DBETA` | degrees | target pitch at 0.7 R, or a plain pitch offset (give at most one) |
| `RHO`, `MU`, `A` | kg/m^3, kg/(m s), m/s | density, viscosity, speed of sound; sea-level standard if omitted (1.225, 1.789e-5, 340.3) |
| `N_SECTIONS` | - | overrides `n_sections` for the point |

`ADVANCE_RATIO` is another name for `J`. `MACH` or `TASmps`, with the other air keys of [Matrix columns](matrix-columns.md), resolve to `V`, `RHO`, `MU` and `A`; a value you also state must agree. The propeller runs in axial flow, so `ALPHA` and `BETA` must be 0 or absent: `QPROP runs in axial flow; ALPHA must be 0 or absent`. Other refusals name the rule: `V (m/s) is required`, `specify RPM or J`, `give only one of RPM and J`, `give BETA07 or DBETA, not both`, `J needs V greater than zero; use RPM for a static point`. A key outside the table is refused as `unsupported QPROP condition: <keys>; accepted: V, RPM, J, BETA07, DBETA, RHO, MU, A, N_SECTIONS`.

For a scheduler job, see [Cluster jobs](cluster-jobs.md).

## What converged means

cfdbender reads QPROP's printed operating table from `stdout.log` by header names, so column order does not matter. A point is `converged` when the exit code is 0, the table holds a numeric operating row, and the output does not report that the solution did not converge. Otherwise it is `failed`. QPROP has no `iteration_limit` here. The history has one entry, iteration 1 and residual 0, which marks that the row was accepted; it is not a measured iteration.

The axis of each value is the rotation axis. The references are the disk area (pi R^2), the length 2 R and the hub centre.

| Name | Meaning |
|---|---|
| `J` | V / (n D) |
| `CT`, `CP` | T / (rho n^2 D^4) and P / (rho n^3 D^5) |
| `CT_QPROP`, `CP_QPROP` | as QPROP prints them, normalised by the dynamic pressure at the tip speed and the disk area |
| `EFFICIENCY` | propeller efficiency as QPROP prints it (`effprop`) |
| `EFFICIENCY_OVERALL` | overall efficiency including the motor, as QPROP prints it (`eff`) |
| `T`, `Q`, `P` | thrust (N), shaft torque (N m), shaft power (W) |

Here `n` is the rotation rate in revolutions per second and `D = 2 R`. In the polar files the propeller is the family `BLADES` inside rotor 0, the group `PROPELLER` (`GROUP_ID` 10) and the group `TOTAL`; there is no `AIRFRAME` group. Its shaft points forward, so the rotor frame R0 is the body frame: `FAM-RCOEF-P<POL>.csv` states `T`, `Q`, `P` (kW), `CT`, `CQ`, `CP`, `ETA` and the disc loadings from the printed thrust and torque, and `CFZ`, `CFY`, `CMZ`, `CMY` are `NA` because QPROP prints no in-plane load; see [Results and tables](results.md). `RPM0` is the row's `RPM`, or `60 V / (J D)` when the row states `J`. In ACOEF the thrust is a negative `CDB`, and QPROP's own columns follow the frames. `Q`, `P`, `CP` and `CP_QPROP` are listed as moments in the campaign summary only to fit the common result format. A QPROP result is a blade-element estimate with the fitted two-dimensional polars, not a validation of the propeller.

## Worked example

A three-blade propeller at 20 m/s, with advance ratios 0.6, 0.8 and 1.0, and a pitch of 30 degrees at 0.7 R. You supply your own blade as `inputs/mesh/blade.obj`, in metres, rotating about x.

```console
cfd-workspace init campaign --solver qprop
cd campaign
```

`inputs/setups/s001.toml`:

```toml
blade = "inputs/mesh/blade.obj"
n_blades = 3
n_sections = 12
spacing = "cosine"
cut = "cylinder"
solver_2d = "xfoil"

[solver]
launcher = ["wsl", "-d", "Ubuntu", "--"]
executable = "/home/your-user/src/Qprop/bin/qprop"
xfoil_executable = "/home/your-user/src/Xfoil/bin/xfoil"
```

`inputs/references/r001.toml` and `inputs/pproc/p001.toml` are empty files. `inputs/matrices/study.fs`:

```text
POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES
---|---|---|---|-----|----------------|------------
1001|1|001|001|001|V:20, J:sweep, BETA07:30, RHO:1.2, MU:1.8e-5, A:340|0.6,0.8,1.0
```

```console
cfd-matrix plan study
cfd-matrix run study
cfd-matrix collect study
```

Cluster staging requires a supplied polar for every section. If a section needs XFOIL, staging refuses before writing or submitting; provide the missing polars or run locally.
