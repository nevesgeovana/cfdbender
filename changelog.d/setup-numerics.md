## Added

- Matrix column `NUMERICS` names a numerical setup file `inputs/numerics/n<id>.toml` for the SU2, OpenFOAM and PyFR adapters. The setup a solver reads is merged from the `init.toml` defaults, the numerics file, the setup file and the row's `VAR_NAMES_VALUES`, in that order of precedence; a key stated in both files takes the setup file's value with one warning. The numerics file is a planned input (digested, refused when changed after planning) and the plan and run record name its id. A numerics file may not hold a `[solver]` table or the mesh.
- A new adapter capability, `numerics_file`, declared by SU2, OpenFOAM and PyFR. A filled NUMERICS cell for a solver without it warns once and is ignored.
- `cfd-workspace init` creates `inputs/numerics/` for the three CFD solvers and documents the file in `inputs/INPUTS.md`; `cfd-matrix plan --verbose` shows which layer each setup key came from for a row with a numerics file.
- `python -m cfdbender.migrate_setup_numerics WORKSPACE [--apply]` (also `tools/migrate_setup_numerics.py`, standalone, standard library only) moves an existing workspace to the new columns and splits the numerical keys of SU2 setups into shared numerics files, after archiving every file it changes. The default is a dry run, and a second run changes nothing.

## Changed

- The matrix column `SET` is renamed `SETUP`, and `cfd-matrix plan` lists `SETUP=`. The canonical order is `SETUP`, `NUMERICS`, `REF`, `PPROC`. A matrix whose header still says `SET` is read as `SETUP` with one warning per matrix file and command; a header with both is refused.

## Migration

- Run the migration script on each workspace; the solver input planned before and after is byte-identical. A matrix without a NUMERICS column, or with `-` or an empty cell, plans exactly as before.
