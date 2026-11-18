# 2D polars

A matrix row that sweeps `ALPHA` over a two-dimensional SU2 or OpenFOAM case gives a
section polar: `collect` and `post` write `post/<matrix>/polars/P<POL>.txt` in the layout
of an XFOIL polar file, beside `polars/FAM-ACOEF-P<POL>.csv` and the group tables.
cfdbender's XFOIL reader and
the QPROP section reader both read it unchanged, so the file can feed a QPROP blade.

## The matrix row

One polar is one row: `FLIGHT_CONDITION` holds the Mach number, the Reynolds number and
`ALPHA:sweep`, `SWEEP_VALUES` lists the angles, and `GEOMETRY` names the 2D mesh
(`<name>.su2` for SU2, `<name>.msh` for OpenFOAM). The recommended sweep is -8 to 16
degrees in steps of 1, 25 points, listed in ascending order (the matrix takes no range
syntax):

```text
POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES|GEOMETRY
---|---|-----|---|-----|----------------|------------|--------
1001|1|001  |001|001|MACH:0.15, REmi:6, ALPHA:sweep|-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16|naca0012.su2
```

The reference file states the unit chord and the quarter-chord moment origin, the moment
an XFOIL polar uses:

```toml
area = 1.0
length = 1.0
moment_origin = [0.25, 0.0, 0.0]
```

Every point starts cold, from the free stream: no point restarts from the previous angle,
so the points are independent, a point that fails cannot spoil the next one, and the order
of `SWEEP_VALUES` changes no result. A polar for QPROP is one row per station Reynolds
number.

## SU2 setup

The SU2 adapter's defaults are not tuned for a polar. This setup states in full a RANS
deck with the JST scheme (sensor coefficients 0.5 and 0.005, no reconstruction) and a CFL
number that starts at 5, grows by a factor 1.05 per iteration up to 1e5, and on a residual
rise is halved but never falls below 5. It is written in the setup, not taken from the
adapter, so every value is visible in the workspace.

`inputs/setups/s001.toml`:

```toml
iterations = 20000
SOLVER = "RANS"
KIND_TURB_MODEL = "SA"
SA_OPTIONS = ["NEGATIVE", "WITHFT2"]
CONV_NUM_METHOD_FLOW = "JST"
JST_SENSOR_COEFF = [0.5, 0.005]
MUSCL_FLOW = "NO"
MUSCL_TURB = "NO"
TIME_DISCRE_FLOW = "EULER_IMPLICIT"
REF_DIMENSIONALIZATION = "FREESTREAM_PRESS_EQ_ONE"
CFL_NUMBER = 5.0
CFL_ADAPT = "YES"
CFL_ADAPT_PARAM = [0.5, 1.05, 5.0, 100000.0, 1.0, 0.0]
LINEAR_SOLVER = "FGMRES"
LINEAR_SOLVER_PREC = "ILU"
LINEAR_SOLVER_ERROR = 1e-10
LINEAR_SOLVER_ITER = 50
MARKER_HEATFLUX = ["airfoil"]
MARKER_FAR = ["farfield"]
MARKER_MONITORING = ["airfoil"]
MARKER_PLOTTING = ["airfoil"]
CONV_FIELD = "RMS_DENSITY"
CONV_RESIDUAL_MINVAL = -12
HISTORY_OUTPUT = ["ITER", "RMS_RES", "AERO_COEFF"]
OUTPUT_FILES = ["RESTART", "SURFACE_CSV"]
```

`CFL_ADAPT_PARAM` lists the factor down (0.5), the factor up (1.05), the floor (5), the
ceiling (1e5) and SU2's two adaptation thresholds. The mesh carries the markers `airfoil`
and `farfield`; see [SU2](su2-workspaces.md).

## OpenFOAM setup

The OpenFOAM adapter's steady simpleFoam case with the Spalart-Allmaras model, with the
residual target of the convergence rule below and room for slow points:

```toml
application = "simpleFoam"
turbulence = "SpalartAllmaras"
iterations = 20000
residual_target = 1e-6
```

OpenFOAM is incompressible here: the Mach number sets only the flow speed, and the polar's
header says so. See [OpenFOAM](openfoam-workspaces.md).

## Convergence of each point

Each point is judged by one rule, from the files it already wrote. A point is converged
when either holds:

- **R**, the residual reaches its target: SU2's `rms[Rho]` (a log10 value) reaches -12 on
  some row of `history.csv`; for OpenFOAM, the initial residual of every solved field
  reaches 1e-6 on one iteration of the solver log (the first solve of each field in that
  iteration, the one OpenFOAM's residual control tests);
- **C**, the forces have settled: on the same row, CL and CD each change by less than 1e-6
  in relative terms, averaged over the last 250 iterations (SU2's Cauchy measure), and the
  residual lies at least 3 orders of magnitude below its first value. For OpenFOAM the
  forces come from `forceCoeffs` and every field must have dropped 3 orders.

For OpenFOAM, a row missing the initial residual of any field seen in the log cannot
establish either criterion.
Non-finite residuals cannot establish either criterion, and OpenFOAM residuals must be
nonnegative to reach R.
An OpenFOAM residual iteration without a matching force sample can establish R, but it
cannot establish C until the missing force changes have left the Cauchy window.
OpenFOAM's `Time` values identify log and force samples, including fractional values;
the verdict records that value in `rule_iteration`.
If a coefficient is zero, its relative change is undefined for that iteration. The
Cauchy test can pass again once that iteration leaves the window.

The criterion that holds first decides. The drop required by C keeps a solution that
froze at the start from passing; a solution that settled into a two-state cycle whose
forces repeat to seven digits can still pass C, so the residual level is written beside
each verdict.

The rule is applied beside the solver's own classification, which it does not change: an
SU2 point that met C stops only at `iterations`, so its `classification` stays
`iteration_limit` while its `rule_verdict` is `C`. Every SU2 and OpenFOAM table carries the
verdict in three columns after `solver_version`:

| Column | Holds |
|---|---|
| `rule_verdict` | `R`, `C` or `none`; `NA` for a point without a result. |
| `rule_iteration` | The iteration at which the deciding criterion first held. |
| `rule_drop_orders` | The residual drop below its first value at the last row, in orders of magnitude (the smallest over the fields for OpenFOAM). |

## The polar file

`post/<matrix>/polars/P<POL>.txt` holds a header block, the column line and its dashed
rule, then one row per converged point in ascending alpha:

```text
 
       cfdbender 0.6.0 polar from SU2 8.6.0
 
 Calculated polar for: naca0012.su2
 
 1 1 Reynolds number fixed          Mach number fixed
 
 xtrf =   0.000 (top)        0.000 (bottom)   fully turbulent
 Mach =   0.150     Re =     6.000 e 6     turbulence model SA
 
 mesh sha256 9f2c...
 CDp not measured: the column holds 0.00000
 rows: the points converged by the rule (R, residual target; C, Cauchy on the forces)
 left out, not converged: 16.000
 
   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr
  ------ -------- --------- --------- -------- -------- --------
```

The rows follow the rule, one per converged point.

- `CL`, `CD` and `CM` are the neutral coefficients of the tables: lift and drag in the wind
  axes and the pitching moment about the reference origin, nose-up positive. For a 2D SU2
  case `CM` is minus SU2's `CMz`; for OpenFOAM it is `CmPitch`.
- `Top_Xtr` and `Bot_Xtr` are 0.0000: the RANS flow is fully turbulent, transition at the
  leading edge.
- `CDp` is 0.00000 and the header says it is not measured: neither adapter reads a split of
  the drag into pressure and friction.
- The widths are XFOIL's: alpha 3 decimals, CL 4, CD and CDp 5, CM and the transition
  columns 4. The full precision stays in `polars/FAM-ACOEF-P<POL>.csv`.

A point that meets neither criterion, or has no readable result, is left out of the polar
with one warning that names it and its measured state; it keeps its row, with its verdict,
in the family and group CSV tables, and the header lists the angles left out. With `--warning-refusals`
(or `warning_refusals = true` in `init.toml`) the warning is a refusal and no polar is
written. A polar with fewer than three converged points between -4 and 6 degrees is
written with a warning, because QPROP fits its lift slope there.

A row that sweeps another key, or sweeps `ALPHA` on a three-dimensional SU2 mesh, gets no
polar file.

## Feeding QPROP

Copy the polar into a section set and name it in the `polar` column of its `blade.csv`:

```text
inputs/polars-2D/<set>/P1001.txt
```

See [QPROP](qprop-workspaces.md) for the blade table.
