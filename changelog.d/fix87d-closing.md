## Added

- The SU2 adapter declares its native passthrough (`NATIVE_PASSTHROUGH`): upper-case native option names of the setup file are forwarded to SU2; every other unknown key is refused.
- `cfdbender.core.longpath`: extended-length file operations on Windows, so `delete-sims`, `space-in-use`, `sync`, `restore`, `archive` and the `--force-rerun` move work on point trees deeper than 260 characters.

## Changed

- SU2 convergence counts only when the solver prints its convergence message after the last iteration line of its log.
- A setup without `HISTORY_OUTPUT` gets `ITER, RMS_RES, AERO_COEFF`, the groups the adapter reads; a `HISTORY_OUTPUT` that lacks one is extended with a warning.
- A matrix that gains a row after points ran keeps its finished points valid: staleness is decided by the files each point read, not by the matrix file.
- `SOLVER_BUILD` is checked when the run starts for SU2 (executable route) and PyFR, whose executables state their version; the run is refused before any point starts.
- The neutral `CL`, `CD` and `CM` are defined in one standard frame for every solver (lift normal to the freestream and drag along it, from `ALPHA` and `BETA`; pitching moment nose-up positive): SU2's two-dimensional `CM` is `-CMz`, and FlightStream's body-axis forces are rotated into the wind axes. Each solver's own columns are unchanged beside them.
- Rotor-capable adapters (QPROP) write the rotor-frame columns `CT`, `CQ`, `CP`, `CN`, `CY`, `J` and `EFFICIENCY` (x along the thrust, z up, y completing the right-handed frame; `CQ` is the shaft torque about the thrust axis); QPROP gains the own column `CQ`.
- QPROP's staged config is `propeller.prop`, the file the solver reads.
- `post` exits 0 once the tables are written, even when points failed, as pyflightstream's `post` does; `collect` keeps 1 for failures and 3 for outstanding work.
- `mark-failed` refuses a `RUNNING` point (stop the run first); `delete-sims` needs `--force` for a `RUNNING` point and refuses a sim another manifest still records; a marked-failed point keeps the mark's reason in `collect`, `status` and the summary; `sync --prefer-other` at level `post` or `all` refuses without `--overwrite`.

## Fixed

- The contract and architecture pages state their exceptions: declared outputs, the native passthrough, the applied management verbs, the per-solver axes behind the neutral `CL`, `CD` and `CM`, and the post stage.
