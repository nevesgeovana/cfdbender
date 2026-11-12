# QPROP workspaces

QPROP computes the performance of an isolated propeller from blade geometry and
two-dimensional section polars. cfdbender plans one QPROP run per operating
point, makes any missing section polar with XFOIL at each section's own
Reynolds and Mach number, and collects thrust, torque, power and the
coefficients in two normalisations. Install cfdbender on Windows and supply
QPROP and XFOIL separately in WSL.

## Install in WSL

Download and unpack the [MIT QPROP source](https://web.mit.edu/drela/Public/web/qprop/)
to `~/src/Qprop`. QPROP is double-precision Fortran 77, so build it with
gfortran and the flags `-O2 -fdefault-real-8 -std=legacy`: in the supplied
`bin/Makefile`, set the compiler to `gfortran`, set its flags to those three
options, and run `make qprop`. The resulting executable is
`~/src/Qprop/bin/qprop`. Build XFOIL as described in
[XFOIL workspaces](xfoil-workspaces.md). The launcher passes each executable
path verbatim, so give absolute Linux paths below; `~` is not expanded.

## Inputs

```text
campaign/
  init.toml                  solver = "qprop"
  inputs/
    geometry/blade.obj       a 3D blade, in metres (or set scale)
    polars-2D/<set>/         an existing set of section cuts and polars
    setups/s001.toml
    matrices/study.fs
```

The `blade` key of the s file points at either one OBJ under `inputs/geometry`
or one set under `inputs/polars-2D`.

**Blade OBJ.** cfdbender slices the blade at `n_sections` radial stations
(`spacing = "cosine"` or `"uniform"`) and measures radius, chord, pitch and the
section shape. The OBJ follows this convention; each item is a setup key with
the default shown.

| Key | Default | Meaning |
| --- | --- | --- |
| `axis` | `"x"` | rotation axis; thrust points along its positive direction |
| `hub_center` | `[0, 0, 0]` | a point on the axis |
| `rotation` | `"ccw_from_front"` | counter-clockwise seen from +axis looking aft; or `"cw_from_front"` |
| `cut` | `"cylinder"` | the cylinder r = const, unrolled; or `"plane"` normal to the radial direction |
| `scale` | `1` | model length unit in metres |
| `radius` | blade tip | tip radius in metres |

The blade is the largest connected mesh component, so an OBJ may hold one blade
or a full rotor. The radial direction runs from the axis to the blade centroid.
The leading edge is the section point farthest ahead along the rotation
direction; the trailing edge is the section point farthest from it; the chord is
that distance. Beta is the angle in degrees between the chord line and the
rotation plane, positive when the leading edge lies ahead along the thrust axis
(nose up into the oncoming relative flow). The section shape is normalised to
unit chord for XFOIL. On a synthetic blade with known chord and pitch laws the
slicer recovers the chord to 0.6 percent and beta to 0.1 degree on the unrolled
cylinder, and far more closely on the plane cut.

**Polars-2D set.** A set holds `blade.csv` with the columns `r`, `chord` and
`beta` (metres, metres, degrees, ascending radius) and optional `airfoil` and
`polar` columns naming files inside the set. A station with a `polar` file uses
it directly (an XFOIL polar with alpha, CL and CD in the first three columns); a
station with only an `airfoil` file gets its polar from XFOIL. `n_sections` may
be omitted; if given it must equal the number of stations.

## Setup file

`campaign/inputs/setups/s001.toml`:

```toml
blade = "inputs/geometry/blade.obj"
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

`n_blades` is required. `solver_2d` accepts only `"xfoil"`; any other value is
refused with a message naming the accepted values. Optional keys are `radius`,
`scale`, `title`, `alpha_start`, `alpha_end`, `alpha_step` (default -8, 16, 1
degrees), `ncrit` (9), `iterations` (150) and `re_exponent` (-0.5). The
`[solver]` table also accepts `timeout_s` and `ram_estimate_gib`. Without an
explicit executable, `CFDBENDER_QPROP_EXE` and then `qprop` on PATH are used.
The r and p files may be empty.

## Matrix

Operating points are `FLIGHT_CONDITION` keys; any may be the sweep key.

| Key | Unit | Meaning |
| --- | --- | --- |
| `V` | m/s | flight speed (required) |
| `RPM` or `J` | rev/min or - | rotation rate; `J = V / (n D)` with `D = 2 R` (give one) |
| `BETA07` or `DBETA` | degrees | target pitch at 0.7 R, or a plain pitch offset (give at most one) |
| `RHO`, `MU`, `A` | kg/m^3, kg/(m s), m/s | density, viscosity, speed of sound; sea-level standard if omitted |
| `N_SECTIONS` | - | overrides `n_sections` for the point |

```text
POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES
---|---|---|---|-----|----------------|------------
1001|1|001|001|001|V:20,J:sweep,BETA07:30,RHO:1.2,MU:1.8e-5,A:340|0.6,0.8,1.0
```

```console
cfd-matrix plan study --workspace campaign --local
cfd-matrix run study --workspace campaign --local
cfd-matrix collect study --workspace campaign
```

Planning reads inputs and slices the blade without writing point folders;
translating the same point twice gives byte-identical text. Each point folder
holds the XFOIL decks and results under `xfoil/sec_NN`, `propeller.prop`,
`qcon.def`, QPROP's `stdout.log`, `stderr.log`, `execution.json`, and
`sections.json`, which records each station's radius, chord, pitch, Reynolds and
Mach number, fitted parameters and where they came from: `xfoil`,
`xfoil_narrow`, `table`, or `reused`.

## Section polars

For each section, the polar fit gives CL0, CL_a (per radian), CLmin, CLmax, CD0,
CD2u, CD2l, CLCD0 and the reference Reynolds number, as QPROP's per-station
input. The lift line uses alpha between -4 and 6 degrees. When XFOIL fails for a
section the point retries with alpha -4 to 10 in steps of 0.5; if that fails as
well the station reuses the nearest successful station's parameters and
`sections.json` says so. If no station succeeds the point fails.

## Results

Collection returns, with the rotation axis as the axis of each value:

| Name | Meaning |
| --- | --- |
| `J` | V / (n D) |
| `CT`, `CP` | T / (rho n^2 D^4) and P / (rho n^3 D^5) |
| `CT_QPROP`, `CP_QPROP` | as QPROP prints them, normalised by dynamic pressure at the tip speed and the disk area |
| `EFFICIENCY` | T V / P |
| `T`, `Q`, `P` | thrust (N), shaft torque (N m), shaft power (W) |

QPROP's table is read by header names, so column order does not matter. A point
is `converged` only with exit code zero and a parsed operating row; QPROP has no
`iteration_limit` class here. The history's single iteration is an acceptance
marker, not a measured iteration. Q and P are listed as moments only to fit the
neutral result contract. A QPROP result is a blade-element estimate with the
fitted two-dimensional polars, not a validation of the propeller.
