## Added

- `cfd-matrix` commands `status`, `show`, `log`, `trace`, `history` and `diff` (read only; `--json` and `--csv`), `mark-failed`, `mark-converged`, `restore`, `rebuild`, `space-in-use`, `free-space`, `delete-sims` and `sync`, and `cfd-workspace archive`, with the names and meanings of pyflightstream's commands. They work for every solver over `runs.json`. See the new Commands page.
- Options `--warning-off`, `--verbose`, `--log-level`, `--runs`, `run --no-local`, and `collect --archive`, `--watch`, `--interval`, `--watch-interval` and `--rounds`.
- `RunRegistry(root, runs=NAME)` reads another manifest; `RunRegistry.replace_records` and `restore_from` change records under the writer lock with the manifest archived first.

## Changed

- Warnings of a command are held and printed together at the end under `Warnings (n)`, and `--warning-off` keeps them in `logs/warnings.log` instead of the console.
- `cfd-matrix post [matrix]` rebuilds every product from the records without running anything; `plan` writes the plan receipt and `run` refuses a missing or stale one.
