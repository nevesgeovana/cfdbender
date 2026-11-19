# Upgrading from 0.6

The new input layout is optional for a workspace made with cfdbender 0.6.
Review the result and validation changes below, including the required
`blade1` declaration for an X-axis rotor, before reusing a workspace.

## What you do not have to do

- **Splitting setup and numerics is optional.** A matrix without a `NUMERICS`
  column, or with `-` or an empty cell in it, keeps its numerical settings in
  the setup file. The new column alone does not change the SU2 solver input.
- **The old column names are still read.** The columns were renamed: `SET` is
  now `SETUP` and `GEOMETRY` is now `MESH`. A matrix whose header still says
  `SET` or `GEOMETRY` is read under the new name with one notice per matrix file
  and command. A header that holds both the old and the new name of a column is
  refused. The notice stays a notice under `--warning-refusals`, and the values
  are read exactly as before.

## Moving a workspace to the new layout

The migration script renames the headers, adds the `NUMERICS` column and, for
SU2, moves the numerical keys of each setup into shared numerics files:

```console
python -m cfdbender.migrate_setup_numerics campaign           # a dry run
python -m cfdbender.migrate_setup_numerics campaign --apply   # archive, then change
```

It first copies every file it changes to
`archive/migration-setup-numerics-<n>/`. A second run changes nothing, and the
planned solver input before and after is byte-identical. The same file ships
as `tools/migrate_setup_numerics.py`, which needs only Python 3.11 or later.
See [Setup and numerics](su2-workspaces.md#setup-and-numerics).

## What is new

| You can now | Where |
|---|---|
| Name a numerics file in the `NUMERICS` column, shared by many setups | [Matrix columns](matrix-columns.md#setup-and-numerics) |
| Select a run type in the `WORKFLOW` column (SU2) and state its clock, averaging window and export window | [SU2 run types](su2-workspaces.md#run-types) |
| Solve a propeller from `ADVANCE_RATIO` and a rotor block, or one periodic sector with `PERIODIC_COPIES` | [SU2 run types](su2-workspaces.md#run-types) |
| Declare sections, a volume cut and probes in the post file | [Post-processing products](su2-workspaces.md#post-processing-products) |
| Replace one value of an HPC profile for a row, and group rows into jobs with `--batch N` | [Cluster jobs](cluster-jobs.md) |
| Leave large inputs unread, and plan large meshes faster | [Workspace layout](workspace-layout.md#large-inputs) |
| Plan a CFD++ campaign (incomplete, not yet verified against the solver) | [CFD++](cfdpp-workspaces.md) |
| Write an adapter that declares the products it supplies | [Adapter contract](adapter-contract.md#product-declarations) |

## What changes in the results

- **A rotor has its own table.** A rotor's table is
  `post/<matrix>/polars/P<sim>-<alias>_rotor.csv`, with its columns in
  pyflightstream's order. Columns that
  nothing fills are absent and named in `products.json`.
- **The sign of the thrust follows the declared axis.** Thrust and torque are
  counted along the shaft the reference file declares. With `axis = "X"` and the
  flow along +x, a pulling propeller has a negative `CT`. Read the sign against
  the declared axis, not against the flow.
- **A rotor with `axis = "X"` needs `blade1`.** A rotor block with that axis and
  no `blade1` is refused. Add `blade1 = { azimuth_deg = 0.0, zero = "Z" }` (or
  `"Y"`).
- **Sectional loads are totals where the solver supplies wall shear.** In a
  sectional-loads file `Fx`, `Fz` and `Moment` are then the pressure plus the
  friction, and the two parts follow in their own columns (`Fx_pres`,
  `Fx_fric` and the others). Without a shear field the file holds the pressure
  loads. SU2 supplies it.
- **SU2 reads `SYMMETRY` and `SYMMETRY_LOADS`.** The other solvers still need
  `-` in both.
- **Time-accurate rows report a mean.** The result of an `unsteady` or
  `unsteady_rotor` point is the mean over its averaging window, and the unsteady
  polar `polars/P<sim>_<matrix>_uns_avg.csv` holds one row per point.
- **A double quote in `VAR_NAMES_VALUES` protects a `/`.** A cell without double
  quotes is split exactly as before.
