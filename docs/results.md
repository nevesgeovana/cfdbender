# Results and tables

This page describes the files that `collect` and `post` write, the columns of
the tables, and what the verdicts `converged`, `iteration_limit` and `failed`
mean.

```console
cfd-matrix collect polar
```

```text
wrote post/polar/campaign_summary.json
wrote post/polar/campaign_sweep.csv
wrote post/polar/polars/P1001.csv
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
2 recorded points are not in polar.fs any more (1001/DP-M200AL+000, 1001/DP-M200AL+020); retire them with cfd-matrix delete-sims or cfd-workspace archive
```

## The files

Everything is under `post/<matrix>/`.

| File | Holds |
|---|---|
| `campaign_sweep.csv` | One row per recorded point of the matrix. |
| `polars/P<POL>.csv` | The same columns, one file per sim. |
| `campaign_summary.json` | Every recorded point with its record and its full result: each coefficient with its axis, the reference values, and the iteration history. |
| `products.json` | Every file written and every point skipped, with the reason. `complete` is true only when nothing was skipped. |
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

## The columns

Every table of a matrix has the same header, in this order:

| Columns | Unit | Holds |
|---|---|---|
| `POL`, `point` | | The sim and the name of the point. |
| `ALPHA`, `BETA` | degrees | Angle of attack and of sideslip, from the row. |
| `MACH` | | Mach number, from the row or resolved from the air state. |
| `RE` | millions | Reynolds number over the reference length. |
| `VINF` | m/s | Flow speed. |
| `ALT` | ft | Altitude: `ALTFT` of the row, or 0 where the air state is the standard atmosphere at sea level. |
| `RHO`, `TEMP`, `MU` | kg/m3, K, Pa s | Density, static temperature and viscosity of the resolved air state. |
| `J`, `RPM_CLOCK` | , rev/min | Advance ratio and rotation speed, when the row states them. |
| `VREF`, `J_CLOCK`, `BREF` | | Kept in the header so that the tables of every campaign line up; filled only when the row states a key of that name. |
| `SREF`, `CREF` | m2, m | Reference area and length that the solver used. |
| other `FLIGHT_CONDITION` keys | | One column per key that is not in the list above. |
| `status`, `classification` | | The status of the record and the verdict. |
| `iterations`, `residual` | | The last entry of the iteration history. |
| `wall_time_s` | s | Time from the start to the end of the run. |
| `solver_version` | | The version the solver printed. |
| `CL`, `CD`, `CM` | | The lift, drag and pitching-moment coefficients, the same on every solver; see below. |
| coefficients | | One column per coefficient the solver reported, under the name the solver gives it, after the three above. |

The air state columns are resolved from the `FLIGHT_CONDITION` of the row by
the same rules that prepare the solver input, so the table and the solver
state the same air; see [Matrix columns](matrix-columns.md).

Three rules make the tables safe to read with any tool:

- A value that a point does not have is `NA`, for example the coefficients of a
  point that has not run.
- No cell holds a comma, a double quote or a line break, and no cell is quoted.
- A point is never left out. A point that failed or could not be read keeps its
  row, with `NA` where it has no value.

The columns after `CM` carry the solver's own names, so they differ between
solvers and stay exactly as the solver states them. `CL`, `CD` and `CM` have the same
names **and the same meaning** on every solver: cfdbender transforms each solver's
native axes into the one standard frame below, so a script that reads them works on
every solver. A solver that reports no such coefficient writes `NA`. The page of each
solver lists its own names, their axes and their reference values.

## The standard frame of `CL`, `CD` and `CM`

The geometry frame is x aft (downstream), y to the right, z up, the frame
pyflightstream's loads export states. A two-dimensional case lies in the x-y plane
with y up.

- **Freestream direction:** `V = (cos a cos b, sin b, sin a cos b)` for the row's `ALPHA`
  `a` and `BETA` `b`. In two dimensions `V = (cos a, sin a)`.
- **`CD`** is the force along `V`. **`CL`** is the force along `(-sin a, 0, cos a)`
  (two dimensions: `(-sin a, cos a)`), normal to `V` and in the plane of symmetry. Both are
  normalised by the dynamic pressure and the row's reference area. Positive `ALPHA`
  gives positive lift on a lifting body.
- **`CM`** is the pitching moment about the axis through the row's moment reference point
  that is perpendicular to the plane of symmetry (y; in two dimensions the axis normal to the
  plane), **positive nose-up**, normalised by the dynamic pressure, the reference area and
  the reference length. It is not turned by `BETA`.

This is pyflightstream's geometry frame and its wind axes (drag opposes the stream, lift
opposes the wind-axis z), with one stated difference: pyflightstream turns sideslip before
incidence for FlightStream (its angles differ from `ALPHA` and `BETA` at second order),
while cfdbender uses the textbook order above for every solver, so the two agree whenever
`ALPHA` or `BETA` is zero.

| Solver | Native column, axes | Transform to the standard frame | Sign |
|---|---|---|---|
| SU2 | `CL`, `CD`: wind axes from `AOA` and `SIDESLIP_ANGLE` | none: SU2's wind axes are the standard ones | `CL` up, `CD` along the stream |
| SU2, three dimensions | `CMy`: body y | none (the mesh follows SU2's convention: x streamwise, y spanwise, z up) | nose-up positive |
| SU2, two dimensions (`NDIME= 2`, or the line SU2 prints) | `CMz`: body z, counterclockwise in the x-y plane | `CM = -CMz` | counterclockwise is nose-down for a flow along +x with y up, so the sign flips |
| OpenFOAM | `Cl`, `Cd`: along `liftDir (-sin a, cos a, 0)` and `dragDir (cos a, sin a, 0)` as the adapter writes them | none | up, along the stream |
| OpenFOAM | `CmPitch` about `pitchAxis (0 0 -1)` (the adapter writes lift cross drag) | none | positive about -z is nose-up |
| PyFR | raw force components `px, py, vx, vy` | `Cl = 2 (Fy cos a - Fx sin a)`, `Cd = 2 (Fx cos a + Fy sin a)` by `ALPHA`, in the adapter | up, along the stream; no moment, `NA` |
| XFOIL | `CL`, `CD` wind axes; `CM` about the quarter chord | none | nose-up positive |
| FlightStream | `CDB`, `CYB`, `CLB`: body axes (forward, right, down), force on the body (aft, right, up) = `(CDB, CYB, CLB)` | `CD = F . V`, `CL = F . (-sin a, 0, cos a)` with `ALPHA`, `BETA` | up, along the stream |
| FlightStream | `CMB25`: body y | none | nose-up positive |
| mock | `CZ`, `CX`, `Cm` | none: the stand-in states its values in the standard frame | as stated |
| QPROP | none | `NA` for `CL`, `CD`, `CM`; see the rotor frame | |

## The rotor frame

A rotor or propeller has its own standard frame, and its coefficients are the columns
`CT`, `CQ`, `CP`, `CN`, `CY`, `J` and `EFFICIENCY`, written beside the solver's own columns
by every rotor-capable adapter (QPROP now). `CL`, `CD` and `CM` of a propeller are `NA`.

- **Axes:** x along the thrust direction, positive forward; z positive up (the normal force
  direction); y = z cross x, completing a right-handed frame.
- **Normalisation:** `n` in revolutions per second, `D` the diameter, `rho` the air density.
- `J = V / (n D)`, with `V` the speed along the thrust axis.
- `CT = T / (rho n^2 D^4)`: positive when the thrust points forward (+x).
- `CQ = Q / (rho n^2 D^5)`: the shaft torque about +x, signed by the right-hand rule. A
  positive `CQ` is a torque vector along +x, a rotor turning right-handed about +x
  (counterclockwise to an observer behind the rotor looking forward). Reversing the rotation
  reverses the torque and so the sign of `CQ`; `CT`, `CP` and the efficiency keep their meaning.
- `CP = P / (rho n^3 D^5)`: the shaft power. `CP = 2 pi CQ` for the positive rotation sense,
  and `CP = 2 pi CQ sign(rotation)` in general.
- `CN = N / (rho n^2 D^4)`: the in-plane normal force, positive up (+z). `CY = Y / (rho n^2 D^4)`:
  the side force, along +y.
- `EFFICIENCY = J CT / CP`, as the solver states it.

pyflightstream agrees on `J`, `CT`, `CQ`, `CP` and the in-plane normal force: the same
normalisations, `CP = 2 pi CQ sign(RPM)`, a signed `CQ`, and a normal force along up (+z of the
loads frame). It differs in two names and one frame choice: its side force is `CS` where
cfdbender's is `CY`; its columns carry a rotor alias suffix (`CT_<alias>`); and its thrust axis
is the reference's rotor axis in the loads frame (x aft), where cfdbender's rotor x is the
thrust direction itself.

| Solver | `CT` | `CQ` | `CP` | `CN`, `CY` | `J` | `EFFICIENCY` |
|---|---|---|---|---|---|---|
| QPROP | from the printed `T` (positive forward), `T / (rho n^2 D^4)` | from the printed `Q` (shaft torque, positive for a right-handed rotation about the thrust axis), `Q / (rho n^2 D^5)` | from the printed `Pshaft` | `NA`: an isolated propeller in axial flow has no in-plane force | `V / (n D)` | QPROP's `effprop` |

QPROP's own columns follow, unchanged: `CT_QPROP` and `CP_QPROP` (its tip-speed
normalisation), `EFFICIENCY_OVERALL`, `T`, `Q` and `P`. Entries such as the efficiency,
the advance ratio and the power use the coefficient record shape (a value, an axis and a
kind of `force` or `moment`) by convention, so that every solver's result has one format.
Read their meaning from this page, not from the `kind` field.

`J` occupies its existing fixed-context column once, including rotor tables.

`RE` is in millions: a Reynolds number of 500000 is `0.5`.

## Part of a matrix

`collect` and `post` take `--sims` and `--points`. Only the chosen points are
read again; the other rows of the summary stay as they were.
