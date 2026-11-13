# Commands

`cfd-workspace` creates and archives; `cfd-matrix` plans, runs, collects and manages the
records of a matrix. The names and meanings follow pyflightstream's `pyfs-workspace` and
`pyfs-matrix`, and they work the same for every solver, because they read `runs.json`
and the folders of the workspace and never the solver.

A **sim** is a POL. A **run** is a recorded point, named by its point id
(`POL/DP-name`); an earlier record of it, archived or deleted, is named
`POL/DP-name@<archive folder>` (see `history`).

Every command that changes files previews and changes nothing unless `--apply` is given,
and a change to `runs.json` copies it to `archive/<stamp>/` first. Nothing is deleted
without saying so.

## Options every command takes

| Option | Meaning |
|---|---|
| `--warning-off` (before the command) | Hide console warnings from the console only; every warning is kept in `logs/warnings.log` either way. |
| `--verbose`, `--log-level quiet\|normal\|verbose` | How much the command says: quiet prints errors and the final summary, normal adds the per-point lines, verbose adds details. |
| `--runs NAME` | Read another run manifest in the workspace root instead of `runs.json`. |
| `--json`, `--csv` | On the read-only commands: print the rows for a program. A nested cell is JSON. |

## Plan, run, collect

- `cfd-matrix plan`, `run` and `collect` as before. `run --no-local` is the opposite of `--local`.
- `collect --archive` keeps a copy of each summary, table and polar the collect replaces
  under `archive/<stamp>/`.
- `collect --watch` sweeps again until no submitted point is outstanding;
  `--watch-interval` seconds separate the sweeps (default 60), `--rounds N` stops after N
  sweeps, and `--interval` seconds separate the two observations that decide a job output
  is settled (size and modification time stable; 2 under `--watch`).

## Read the records

| Command | Prints |
|---|---|
| `status [--sims] [--matrix] [--status WORD] [--failed] [--points]` | Each sim of each matrix with its states, or each point. A trailing `*` in a status word matches every word it begins. |
| `show TARGET [--point NAME]` | One sim, point or run record in full. |
| `history TARGET` | The current and earlier records of a sim or point, with their archives. |
| `log [--post MATRIX \| --storage] [--sims] [--run] [--stage] [--since] [--problems] [--open]` | The recorded events of runs, the storage calls, or a matrix's saved post log. `--since` takes an ISO time or a duration such as `2h`. |
| `trace PRODUCT [--run ID]` | The run, inputs and hashes that made a product file. |
| `diff RUN_A RUN_B` | Two runs field by field; exit 1 when they differ. |

## Change the records

| Command | Does |
|---|---|
| `mark-failed --sims POL [--points] [--reason]` | Records the verdict that points failed, whatever they ended in. |
| `mark-converged --sims POL --reason TEXT [--points]` | Records the verdict that completed points converged; refuses a point that has not completed or was marked failed. Verdicts are kept in `marks.json`. |
| `restore KIND [--stamp] [--matrix]` | Brings the manifest (`runs`), the storage journal (`storage`), or a matrix's `products` or `plan` back from `archive/<stamp>/`. |
| `rebuild [--matrix] [--sims] [--all-sims] [--out NAME]` | Writes records from the sim folders on disk, by the matrix's plan. |
| `cfd-workspace archive SIM_ID` | Moves a recorded sim's folder into `archive/<stamp>/` and its records to the manifest history. |

## Storage

Each call is recorded in `storage_management.json`.

| Command | Does |
|---|---|
| `space-in-use [--top N]` | Sizes by folder, by sim and by extension. |
| `free-space RECIPE [--list]` | Runs `inputs/management/<RECIPE>.toml`: `[[delete_extensions]]` with `extensions = [".vtu"]` deletes those files under `sims/` except the ones a record names. |
| `delete-sims --sims POL [--matrix-products points-only\|regenerate] [--force] [--archive]` | Deletes the sim folders and their records (kept in the manifest history). A matrix product that holds the deleted points needs `--matrix-products`. |
| `sync LEVEL [--from NAME] [--prefer-other] [--overwrite] [--include-archives] [--restore]` | Brings records (`runs`), products (`post`) and sim files (`all`) from the workspaces of `inputs/sync-workspaces.toml` (`[workspaces]` name = "path"). Main wins a conflict unless `--prefer-other` or `--overwrite`. |
