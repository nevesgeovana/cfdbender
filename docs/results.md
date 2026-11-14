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
solvers. `CL`, `CD` and `CM` have the same names on every solver: they are filled
from the solver's own columns by the table below, and a script that reads them
works on every solver. **They are aliases of each solver's native coefficients, not
one signed physical quantity.** Each keeps its solver's axes, sign and reference
area, length and moment origin, so two solvers' values are comparable only after
the reference values and the axis of the table below are matched. A solver that
reports no such coefficient writes `NA`. The page of each solver lists its own
names, their axes and their reference values.

| Solver | `CL` | `CD` | `CM` |
|---|---|---|---|
| SU2 | `CL`, wind z | `CD`, wind x | `CMy`, body y (pitch of a three-dimensional case); `CMz`, body z, when SU2 states a two-dimensional problem |
| OpenFOAM | `Cl`, along (-sin alpha, cos alpha, 0) | `Cd`, along (cos alpha, sin alpha, 0) | `CmPitch`, ESI's pitch axis (body minus z), sign as OpenFOAM prints it |
| PyFR | `Cl`, wind y | `Cd`, wind x | `NA`, PyFR reports no moment |
| XFOIL | `CL`, wind z | `CD`, wind x | `CM`, nose-up about the quarter chord |
| QPROP | `NA` | `NA` | `NA`, a propeller has thrust and power instead; see its page |
| FlightStream | `CLB` | `CDB` | `CMB25` |
| mock | `CZ` | `CX` | `Cm` |

QPROP's efficiency, advance ratio and power entries are not forces or moments, yet
they use the coefficient record shape (a value, an axis and a kind of `force` or
`moment`) by convention, so that every solver's result has one format. Read their
meaning from the QPROP page, not from the `kind` field.

`RE` is in millions: a Reynolds number of 500000 is `0.5`.

## Part of a matrix

`collect` and `post` take `--sims` and `--points`. Only the chosen points are
read again; the other rows of the summary stay as they were.
