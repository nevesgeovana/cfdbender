## Added

- `cfd-workspace init` writes `inputs/INPUTS.md`: what each input file holds for the chosen solver, with an example matrix row and setup, reference and post files that plan as written.
- `cfd-workspace init --solver qprop` writes `section_timeout_s = 60` and `on_unconverged_section = "warn"` under `[solver]` in `init.toml`. A section whose XFOIL polar lacks requested angles drops them with a warning naming the section and the angles (`warn`), or fails the point (`fail`); a setup `[solver]` value overrides.
- `cfd-matrix run` prints a line when each point starts and ends, and one line per QPROP section.
- A FLIGHT_CONDITION key outside the grammar (such as a raw `RE`) is refused with the accepted keys and the closest known key; a solver's own keys (QPROP `V`, `RPM`, `J`) stay accepted.

## Changed

- A missing executable, a busy machine-wide slot or the RAM floor refuses `run --local` before any point is recorded (`matrix not run: ...`, exit 2), so the same command works once the cause is gone. Every solver words a missing executable alike and names the key to set.
- A failed point reads `FAILED_EXECUTION` (or `FAILED`) at the console instead of `COMPLETED`; the records are unchanged. Console times print as local time to the second.
- `--resume` says `skipped: COMPLETED failed; redo it with --force-rerun <point>`, and the already-recorded refusal says what to type.
- A setup path that names an existing legacy mesh file is used even when `inputs/mesh/` holds a same-named file; one warning says both exist.
- `cfd-workspace init --solver mock` creates `inputs/mesh/`.
- QPROP staging for a cluster runs no process: it writes `sections.plan`, `qcon.def` and, when every polar is given, `propeller.prop` and `sections.json`. `STAGED_CONFIG` is `sections.plan`.

## Fixed

- QPROP takes the blade from the row GEOMETRY alone on run and collect.
- The mock refuses an unknown setup key and collects without `execution.json`; PyFR classifies exit 0 with an empty history as failed; XFOIL reads its polar by column name.
- QPROP collection reports EFFICIENCY as the printed effprop and adds EFFICIENCY_OVERALL (the printed eff), instead of recomputing J CT / CP.
