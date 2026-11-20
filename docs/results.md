# Results and tables

A point without a readable result is named in the product skip log and emits
a warning with its point id and reason. `--warning-refusals` (before the verb),
or `warning_refusals = true` in init.toml, makes that product skip a refusal.

## Boundary force sets

For SU2 and OpenFOAM, a setup `[groups]` table requests one coefficient set per
marker group. Collect's neutral result has a `force_groups` mapping, including
`TOTAL`. Each entry maps coefficient names to the same value, axis, kind and
optional frame fields used by the total `coefficients` mapping. Every set uses
the result's `references`. For example, `result.force_groups["main"]["CL"]`
is SU2's lift coefficient for the markers in `main`; OpenFOAM calls it `Cl`.
The campaign summary stores this at `points[].result.force_groups`.

`TOTAL` is the solver's complete monitored force set. It is not the sum of the
named groups, which may overlap or omit a boundary. These boundary force sets stay in
the neutral result and campaign summary; they do not add CSV files. The family-based
TOTAL and AIRFRAME group CSV files described below are still written.
Group validation reads SU2 marker tags, Gmsh physical boundary names or HDF5
CGNS boundary nodes (the latter needs h5py). A mesh with no readable named
markers refuses a grouped-force request.

This page describes the files that `collect` and `post` write, the columns of
the tables, and what the verdicts `converged`, `iteration_limit` and `failed`
mean.

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

Without a matrix name, `collect` does this for every matrix that has records.
`post` prints the same `wrote` lines and ends with `post read 3, failed 0,
outstanding 0`. Use `collect` to read the results of points that were
submitted to a cluster; use `post` to rebuild the tables of points that
already ran here, with no solver and no pickup.

When a matrix was edited after a run, so that some recorded points are no
longer in it, `collect` says so on its own line and does not count them among
the failures:

```text
2 recorded points are not in polar.fs any more (1001/DP-M200AL+000, 1001/DP-M200AL+020); retire them with cfd-workspace delete-sims or cfd-workspace archive
```

## The files

Everything is under `post/<matrix>/`.

| File | Holds |
|---|---|
| `polars/FAM-DIM-P<POL>.csv` | One row per point and family: the force and the moment in newtons, in the geometry frame. |
| `polars/FAM-ACOEF-P<POL>.csv` | One row per point and family: the airframe coefficients in the frames W, S, B and R<id>, then the solver's own coefficients. |
| `polars/FAM-RCOEF-P<POL>.csv` | One row per point and rotor family: the rotor coefficients. Only for a solver that states a rotor (QPROP). |
| `polars/P<POL>-<GROUP>-<KIND>.csv` | One row per point for each group (`TOTAL`, `AIRFRAME`, each rotor) and kind (`DIM`, `ACOEF`, `RCOEF`). |
| `polars/P<POL>.txt` | A 2D SU2 or OpenFOAM ALPHA sweep in the XFOIL polar layout, one row per point converged by the polar rule; see [2D polars](polars-2d.md). |
| `polars/P<POL>-<ALIAS>_rotor.csv` | pyflightstream's rotor table: one row per point that turns the rotor `<ALIAS>`, for a solver that computes it (SU2 rotor run types); see [SU2](su2-workspaces.md#rotor-coefficients). |
| `polars/P<POL>_<matrix>_uns_avg.csv` | pyflightstream's unsteady polar: one row per time-accurate point, the mean over its averaging window; see [SU2](su2-workspaces.md#time-accurate-rows). |
| `sections/P<POL>/<point>_sections.csv` | The sectional distribution of a point, for a solver that prints one (QPROP's radial table). |
| `sections/P<POL>/<point>_cp_<name>.csv`, `sections/P<POL>/<point>_sloads_<name>.csv` | The Cp and the sectional loads of each `[[sections.distributions]]` entry of the PPROC file; see [SU2](su2-workspaces.md#post-processing-products). |
| `fields/P<POL>/<point>_vsec.vtk` (or `.dat`) | The `[volume_section]` of the PPROC file, with its `.provenance.json`. |
| `fields/P<POL>/<point>_vsec_<name>.vtk` (or `.dat`) | Each cut of `[[volume_sections]]` (provisional name), with its `.provenance.json`; `products.json` lists it with its `cut` name. |
| `probes/P<POL>/<point>_probes.csv` | The `[[probes]]` of the PPROC file. |
| `campaign_summary.json` | Every recorded point with its record and its full result: each coefficient with its axis, the reference values, and the iteration history. |
| `products.json` | Every file written (with its kind, and its polar and group for a polar file) and every product skipped, with the reason; `complete` is true only when every point has a result. It also states the units, frames, quantities and signs of every column (`kinds`), the run block (`run_block`), the frames with the rotor angles used (`frames`), the families, the groups and their members, the run type of each polar, and any earlier-layout file left in place (`migrated`). |
| `post.log` | The same as `products.json`, as text. |

These files are rebuilt whole by every `collect` and `post` from `runs.json`
and the point folders. Deleting them loses nothing. `--archive` keeps a copy of
each replaced file under `archive/<stamp>/`.

## The verdicts

A solver process that ends without an error has not proved anything about the
numbers. Each point therefore gets a verdict from the output it wrote:

| Verdict | Meaning |
|---|---|
| `converged` | The solver output shows convergence, by the rule of that solver. |
| `iteration_limit` | The run ended at its iteration limit without showing convergence. The coefficients are there; whether to use them is your judgment. |
| `failed` | The run gave no usable result. The record keeps the reason in its `error` field. |

The rule for `converged` differs by solver and is stated on the page of each
one, under "What converged means"; [Solvers, side by side](solvers.md) compares
them.

The `error` of a failed point starts with one of these words:

| Word | Meaning |
|---|---|
| `FAILED_EXECUTION` | The solver could not be started, ended with an error, or wrote output that shows no valid result. The text after the word says which. |
| `FAILED_INCOMPLETE_OUTPUT` | The solver ended and a file it should have written is missing. |
| `FAILED_STAGING` | The files of a cluster job could not be prepared, and the job was not submitted. |
| `FAILED_MARKED` | You marked the point failed; see [Run again, resume, records](records.md). |

`cfd-matrix status --failed --points` lists the failed points with their error.

## The exit code of collect

| Code | Meaning |
|---|---|
| 0 | Every recorded point was read and none failed. |
| 1 | At least one point failed. The tables are written and hold every point. |
| 2 | Refused. The message says what is at fault. |
| 3 | No point failed, and cluster jobs are still outstanding. |

These are pyflightstream's codes, so a script written for one library reads the other. A point that failed is also in the tables, with `NA` where it has no value.

`post` follows pyflightstream's `post` the same way: it exits 0 once the tables are written, even when points failed (the count is on its last line), 3 when points are still outstanding and so left unread, and 2 when it is refused. Use `collect` when the exit code must say that a point failed.

## The polar files

Every polar `<POL>` of the matrix has one file set: `FAM-DIM`, `FAM-ACOEF` and, for a rotor
solver, `FAM-RCOEF` with one row per point and family, and `P<POL>-<GROUP>-<KIND>.csv` with
one row per point for each group. A polar file holds every point of its polar; the point's
condition is in its run block, so no Mach number or sweep name enters a file name.

### Families and groups

A solver states its integrated loads, so each point has one family: `BODY` (`FAMILY_ID` 1)
for an airframe solver and `BLADES` (`FAMILY_ID` 1, inside rotor 0) for QPROP's propeller.

| Group | `GROUP_ID` | Members |
|---|---|---|
| `TOTAL` | 0 | every family |
| `AIRFRAME` | 1 | every family not inside a rotor |
| one per rotor, named by its alias (`PROPELLER` for QPROP) | 10 + the rotor id | the families inside that rotor |

A group with no member is not written: `products.json` names it under `skipped`. A solver
without a rotor writes no rotor group and no RCOEF file; QPROP writes no `AIRFRAME` group.
A group's values come from the sum of its members' loads in newtons, never from summing
coefficients. An RCOEF group file exists only for a group whose members sit inside one rotor.

### The run block

Every polar file opens with the same columns, in this order. A column that does not apply
is `NA`, never absent.

| Columns | Unit | Holds |
|---|---|---|
| `POL`, `POINT_ID`, `POINT` | | The sim, the 1-based position of the point in its polar, and the point name. |
| other `FLIGHT_CONDITION` keys | as written | One column per key that is not a block column, in the order written. |
| `RUN_TYPE` | | `steady`, or `unsteady` for a time-accurate solver (PyFR). |
| `ALPHA`, `BETA` | degrees | Angle of attack and of sideslip, from the row. |
| `MACH`, `RE` | , millions | Mach number and Reynolds number over the reference length. |
| `VINF`, `VREF` | m/s | Flow speed, and the speed the coefficients are taken against (`VINF`). |
| `RPM<id>`, `J<id>`, `MTIP<id>`, `MHEL<id>` | rev/min, | Per rotor: its speed, the advance ratio it ran at `VINF / (abs(n) D<id>)`, and its tip and helical Mach numbers (`NA`: no solver states them yet). |
| `J`, `J_CLOCK`, `RPM_CLOCK` | , , rev/min | The advance ratio the row requested and the rotor speed it states. |
| `THRUST` | N | A thrust condition the row states. |
| `ALT` | ft | `ALTFT` of the row, or 0 where the air state is the standard atmosphere at sea level. |
| `RHO`, `TEMP`, `MU` | kg/m3, K, Pa s | Density, temperature and viscosity of the resolved air state; `MU` in scientific notation. |
| `SREF`, `CREF`, `BREF` | m2, m, m | The reference area and length the solver used; `BREF` is `NA`, no solver states a span. |
| `XMOM`, `YMOM`, `ZMOM` | m | The moment reference point, in the geometry frame. |
| `D<id>`, `ADISC<id>`, `PITCH<id>`, `TOE<id>` | m, m2, deg, deg | Per rotor: diameter, disc area, and the pitch and toe of its frame. |
| `FIRST_STEP` to `WINDOW_PASSAGES` | | The averaging window; `NA` on a steady row. |

### DIM

`FAMILY` (or `GROUP`), `FAMILY_ID` (or `GROUP_ID`), then `FX_MRP`, `FY_MRP`, `FZ_MRP` in N and
`MX_MRP`, `MY_MRP`, `MZ_MRP` in N m: the force and the moment about the moment reference point
in the geometry frame (x aft, y right, z up), then `MESH_FACES`. An airframe solver's
coefficients times `q SREF` (forces) and `q SREF CREF` (moments), `q = 0.5 RHO VINF^2`; a
component the solver does not state is `NA` (a solver stating lift and drag in the wind frame
states no side force). Without a dynamic pressure the loads are `NA` and ACOEF is still formed
from the coefficients.

### ACOEF

The identity, then for each frame in the order `W`, `S`, `B`, `R0`, `R1`, ... the six columns
`CD<f>`, `CY<f>`, `CL<f>`, `Cl<f>`, `Cm<f>`, `Cn<f>`, then the solver's own coefficients under
the names the solver gives them, then `MESH_FACES`. A solver coefficient spelled as a run block
column (QPROP's `J`) is stated there, not twice.

### RCOEF

The identity and `ROTOR_ID`, then in the rotor frame `T` (N), `Q` (N m), `P` (kW), `CT`, `CQ`,
`CP`, `CFZ`, `CFY`, `CMZ`, `CMY`, `ETA`, the disc loadings `TC`, `YC`, `ZC`, `QC`, `NC`, `MC`
and the airframe loadings `TCA` to `MCA`, then `CT<f>`, `CQ<f>`, `ETA<f>` for every frame.

### Sections

SU2 and OpenFOAM family and group tables also carry `solver_version`, followed by
`rule_verdict`, `rule_iteration` and `rule_drop_orders` after their load columns.
These state the per-point [polar convergence rule](polars-2d.md), independently of the
adapter's verdict in `campaign_summary.json`. Unconverged points keep their CSV rows;
only the XFOIL-format `.txt` polar omits them.

`sections/P<POL>/<point>_sections.csv` holds a point's sectional distribution: the run block, then
`FAMILY`, `FAMILY_ID`, `STEP`, `PLANE`, `ROTOR_ID`, `AZIMUTH`, then the solver's own columns in
the order it prints them. Every family the distribution cuts is in the one file of its point,
keyed by `FAMILY` and `FAMILY_ID`. The polar folder keeps equal point names in different
polars distinct. QPROP writes its radial table (one row per blade station,
`PLANE` `radial`, `ROTOR_ID` 0; `radius` and `chord` in m, `beta` in degrees); a QPROP point
whose output holds no radial row is named under `skipped`. A solver that prints no
distribution writes no `_sections.csv` file. The Cp and sectional-loads files of the
sections a PPROC file declares are written, for every solver that supplies them, by
`collect` and `post`; where the solver supplies a wall shear field their `Fx`, `Fz` and
`Moment` columns are the totals, pressure plus friction, and the parts follow in their own
columns.

Three rules make the tables safe to read with any tool:

- A value that a point does not have is `NA`, for example the coefficients of a
  point that has not run.
- No cell holds a comma, a double quote or a line break, and no cell is quoted.
- A point is never left out. A point that failed or could not be read keeps its
  row, with `NA` where it has no value.

## The frames

The geometry frame is x aft (downstream), y to the right, z up. Every other frame is
forward, right, down:

- **B** (body): half a turn about y from the geometry frame.
- **S** (stability): the body turned by `-alpha_s` about y.
- **W** (wind): the stability frame turned by `beta_w` about z.
- **R<id>** (rotor): the body turned by the pitch and toe of rotor `<id>`'s shaft, with zero
  roll; x runs along the shaft in the sense that makes thrust positive forward (up for a
  vertical lift rotor).

For a QPROP blade OBJ, the setup's `mesh_frame` declares axes that differ from the
expected mesh frame; see the [worked declaration](qprop-workspaces.md#mesh). Conversion
runs before planning and slicing, preserves the source, and is recorded per point in
`products.json` under `frames.mesh.points`. The other solvers take meshes in their
documented expected frame. The Python helper `cfdbender.core.mesh_frame.convert_mesh_text`
also converts an OBJ before it is supplied to another consumer; it refuses conversion
of other formats.

`alpha_s = atan2(w, u)` and `beta_w = asin(v)` are read off the unit velocity of the aircraft
in body axes, `(cos a cos b, cos a sin b, sin a)`, so they equal `ALPHA` and `BETA` whenever
either is zero. In every frame:

- **`CD<f>`** is `-F_x / (q SREF)`, positive aft; **`CL<f>`** is `-F_z / (q SREF)`, positive
  up; **`CY<f>`** is `F_y / (q SREF)`, positive right.
- **`Cl<f>`**, **`Cm<f>`**, **`Cn<f>`** are the moments about the moment reference point,
  right-hand about the frame's x, y and z, over `q SREF BREF`, `q SREF CREF` and `q SREF BREF`.
  `Cm` is positive nose up.

A rotor declared with its axis pointing aft is turned forward: its thrust, torque and
efficiency are counted against the declared axis, so its `RPM<id>` is signed about the
declared axis and its `CT` is positive for a forward thrust.

How each solver's own values become the geometry-frame loads:

| Solver | Native columns | Loads in the geometry frame |
|---|---|---|
| SU2 | `CL`, `CD` (wind), `CMy`; two dimensions `CMz` | force `-CD x_W - CL z_W`; `CM = CMy`, or `-CMz` in two dimensions (counterclockwise is nose-down) |
| OpenFOAM | `Cl`, `Cd`, `CmPitch` | as SU2 |
| PyFR | `Cl`, `Cd` | as SU2; no moment |
| XFOIL | `CL`, `CD`, `CM` | as SU2 |
| FlightStream | `CDB`, `CYB`, `CLB` (aft, right, up), `CMB25` | the force as stated; the pitching moment about y |
| mock | `CX`, `CZ`, `Cm` (drag, lift, pitching moment) | as SU2 |
| QPROP | `T`, `Q` | thrust and torque along a forward shaft; no in-plane load |

A two-dimensional case is stated in the x-z plane of the geometry frame. The neutral `CL`
and `CD` of earlier releases are `CLW` and `CDW`, and `CM` is `CmW`.

## The rotor coefficients

`n` is the rate in revolutions per second signed about the rotor frame's x, `D` the diameter,
`A = pi D^2 / 4`, `q = 0.5 RHO VINF^2` and `J = VINF / (abs(n) D)`:

- `T = F_x`, `Y = F_y`, `Z = -F_z` (thrust forward, side force right, normal force up) and
  `Q = M_x`, `M = M_y`, `N = M_z` about the hub, all in the rotor frame.
- `P = 2 pi n Q / 1000` in kW; `CT = T / (rho n^2 D^4)`; `CQ = Q / (rho n^2 D^5)`;
  `CP = 2 pi CQ sign(n)`; `CFZ`, `CFY` are `Z`, `Y` over `rho n^2 D^4`; `CMZ`, `CMY` are `N`,
  `M` over `rho n^2 D^5`; `ETA = J CT / CP`.
- `TC`, `YC`, `ZC` are `T`, `Y`, `Z` over `q A`; `QC`, `NC`, `MC` are `Q`, `N`, `M` over
  `q A D`; `TCA`, `YCA`, `ZCA` over `q SREF`; `QCA`, `NCA` over `q SREF BREF`; `MCA` over
  `q SREF CREF`.
- `CT<f>`, `CQ<f>`, `ETA<f>` are the same along the x of frame f; `CTR0 = CT` for rotor 0.

A positive `CQ` is a torque vector along the rotor frame's x, a rotor turning right-handed
about its thrust direction. A static point (`VINF = 0`) states no `ETA`, no `ETA<f>` and none
of `TC` to `MCA`. A rotor at rest or of unknown speed states no coefficient over `n^2` and no
`ETA`; `P` is 0 at rest and `NA` when the speed is unknown.
An unknown, nonfinite or nonpositive diameter gives `NA` for `D<id>`, `ADISC<id>`,
`J<id>` and every coefficient that needs the diameter. The measured thrust and torque
remain available, as do airframe loadings whose references are known. Zero rotor speed
gives zero shaft power even when no torque was measured.

QPROP's propeller flies along its shaft, so its rotor frame R0 is the body frame. Its own
columns (`CT_QPROP`, `CP_QPROP`, `EFFICIENCY`, `EFFICIENCY_OVERALL`, `T`, `Q`, `P` in W) follow
in ACOEF; `CT`, `CQ` and `CP` of RCOEF equal QPROP's `CT`, `CQ` and `CP`, and its in-plane
coefficients are `NA`, because QPROP prints no in-plane load.

The rotor table of an SU2 rotor run type, `polars/P<POL>-<ALIAS>_rotor.csv`, counts thrust
and torque along the shaft the reference file declares, so the sign of its `CT` and `CQ`
follows that axis; see [SU2](su2-workspaces.md#rotor-coefficients).

`RE` is in millions: a Reynolds number of 500000 is `0.5`.

## The earlier layout

A post writes only the current tree. A file of the earlier layout already under
`post/<matrix>/` (`polars/P<POL>.csv`, or the retired `campaign_sweep.csv`) is left
untouched: `collect` and `post` name it once per kind in one warning, and `products.json`
lists it under `migrated` with the file that replaces it (`polars/FAM-ACOEF-P<POL>.csv`,
or each polar's `P<POL>-TOTAL-ACOEF.csv` for the sweep table, which was removed: the TOTAL
group file holds every point of its polar, and several polars join by concatenating them;
`cfd-workspace trace` refuses the retired name). Delete it when nothing reads it any more. With
`--warning-refusals` the warning is a refusal, so a campaign that must not mix layouts stops.
The same setting also refuses legacy mesh-folder warnings and an automatic fallback
from the SU2 Python wrapper to the executable route. An explicitly selected executable
route needs no fallback and gives no warning.

A point without a readable result is listed under `skipped` with its reason and gives
a warning, or refuses under `--warning-refusals`. An empty family group is a product
that does not apply and remains a named skip without a warning.

The tree keeps two folders for an unsteady point, `time_history/` (its history, one row per
step) and `reductions/` (its reductions); `products.json` states them under `tree`. No solver
of this release writes them.

## Part of a matrix

`collect` and `post` take `--sims` and `--points`. Only the chosen points are
read again; the other rows of the summary stay as they were.
