# Results and tables

This page describes the files that `collect` and `post` write, the columns of
the tables, and what the verdicts `converged`, `iteration_limit` and `failed`
mean.

```console
cfd-matrix collect polar
```

```text
collected summary: /home/you/campaign/post/polar/campaign_summary.json
collected table: /home/you/campaign/post/polar/campaign_sweep.csv
collected table: /home/you/campaign/post/polar/polars/P1001.csv
collected 3, failed 0, outstanding 0
```

Without a matrix name, `collect` does this for every matrix that has records.

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
| `ALT` | ft | Altitude, when the row states `ALTFT`. |
| `RHO`, `TEMP`, `MU` | kg/m3, K, Pa s | Density, static temperature and viscosity of the resolved air state. |
| `J`, `RPM_CLOCK` | , rev/min | Advance ratio and rotation speed, when the row states them. |
| `VREF`, `J_CLOCK`, `BREF` | | Kept in the header so that the tables of every campaign line up; filled only when the row states a key of that name. |
| `SREF`, `CREF` | m2, m | Reference area and length that the solver used. |
| other `FLIGHT_CONDITION` keys | | One column per key that is not in the list above. |
| `status`, `classification` | | The status of the record and the verdict. |
| `iterations`, `residual` | | The last entry of the iteration history. |
| `wall_time_s` | s | Time from the start to the end of the run. |
| `solver_version` | | The version the solver printed. |
| coefficients | | One column per coefficient the solver reported, under the name the solver gives it. |

The air state columns are resolved from the `FLIGHT_CONDITION` of the row by
the same rules that prepare the solver input, so the table and the solver
state the same air; see [Matrix columns](matrix-columns.md).

Three rules make the tables safe to read with any tool:

- A value that a point does not have is `NA`, for example the coefficients of a
  point that has not run.
- No cell holds a comma, a double quote or a line break, and no cell is quoted.
- A point is never left out. A point that failed or could not be read keeps its
  row, with `NA` where it has no value.

The coefficient names are those of the solver, so they differ between
solvers: one reports `CL` and `CD`, another `Cl` and `Cd`. The page of each
solver lists its names, their axes and their reference values.

## Part of a matrix

`collect` and `post` take `--sims` and `--points`. Only the chosen points are
read again; the other rows of the summary stay as they were.
