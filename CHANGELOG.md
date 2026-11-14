# Changelog

All notable changes to cfdbender. The format follows Keep a Changelog;
versions follow SemVer and are independent of solver versions.

## [Unreleased]

### Added

- The architecture page describes the layers, the pipeline, the files each stage writes and what the core guarantees.
- A new adapter contract page states, as numbered clauses, what a solver adapter must satisfy, and how `cfdbender.testing` proves it.
- `CONTRIBUTING.md` lists the rules a change must respect and what checks each one.
- `cfd-matrix` commands `status`, `show`, `log`, `trace`, `history` and `diff` (read only; `--json` and `--csv`), `mark-failed`, `mark-converged`, `restore`, `rebuild`, `space-in-use`, `free-space`, `delete-sims` and `sync`, and `cfd-workspace archive`, with the names and meanings of pyflightstream's commands. They work for every solver over `runs.json`. See the new Commands page.
- Options `--warning-off`, `--verbose`, `--log-level`, `--runs`, `run --no-local`, and `collect --archive`, `--watch`, `--interval`, `--watch-interval` and `--rounds`.
- `RunRegistry(root, runs=NAME)` reads another manifest; `RunRegistry.replace_records` and `restore_from` change records under the writer lock with the manifest archived first.
- `cfd-workspace init` writes `inputs/INPUTS.md`: what each input file holds for the chosen solver, with an example matrix row and setup, reference and post files that plan as written.
- `cfd-workspace init --solver qprop` writes `section_timeout_s = 60` and `on_unconverged_section = "warn"` under `[solver]` in `init.toml`. A section whose XFOIL polar lacks requested angles drops them with a warning naming the section and the angles (`warn`), or fails the point (`fail`); a setup `[solver]` value overrides.
- `cfd-matrix run` prints a line when each point starts and ends, and one line per QPROP section.
- A FLIGHT_CONDITION key outside the grammar (such as a raw `RE`) is refused with the accepted keys and the closest known key; a solver's own keys (QPROP `V`, `RPM`, `J`) stay accepted.
- The SU2 adapter declares its native passthrough (`NATIVE_PASSTHROUGH`): upper-case native option names of the setup file are forwarded to SU2; every other unknown key is refused.
- `cfdbender.core.longpath`: extended-length file operations on Windows, so `delete-sims`, `space-in-use`, `sync`, `restore`, `archive` and the `--force-rerun` move work on point trees deeper than 260 characters.
- A matrix may sit in the workspace root as `<name>.fs` as well as in `inputs/matrices/`; a name found in both is refused as ambiguous.
- The SET, REF and PPROC cells may name an artifact (`sxbbase`, `rwing`, `pforces`: the letter prefix plus a name) as well as number it (`001`).
- `run`, `collect` and `post` write a live log `logs/<command>-<stamp>.log` and start and end events to `logs/activity.log` and `logs/activity.log.jsonl`; the `log` verb lists the events (stage `command`).
- Every completed run record carries `solver_version`, the version the solver reported; PyFR now asks its executable (`--version`) instead of always reporting `unknown`, and a rebuilt record takes it from the collected result.
- A local run whose row states `SOLVER_BUILD` is recorded failed, with a clear message, when the version the solver printed differs from it (`8.3.0-mpi` is an alias of `8.3.0`); an unreadable version cannot be checked and is not refused.
- A plan receipt, `post/<matrix>/plan.json`: the matrix digest and every planned point's input digests, written by `cfdbender.matrix.receipt.write_receipt` and checked by `require_receipt`, which refuses a missing or stale receipt with a message naming what changed.
- `collect` also writes `post/<matrix>/products.json`, listing each product written and each skipped with its reason and a `complete` flag, and `post/<matrix>/post.log`; `cfdbender.matrix.post.post_workspace` rebuilds them from runs.json and the saved outputs without running a solver.

### Changed

- Warnings of a command are held and printed together at the end under `Warnings (n)`, and `--warning-off` keeps them in `logs/warnings.log` instead of the console.
- `cfd-matrix post [matrix]` rebuilds every product from the records without running anything; `plan` writes the plan receipt and `run` refuses a missing or stale one.
- `cfd-workspace init` creates `inputs/mesh/` for every library solver (su2, openfoam, pyfr, xfoil, qprop), in place of `inputs/geometries/`. One mesh resolver reads `inputs/mesh/` first, then the older `inputs/geometry/`, `inputs/geometries/` and `inputs/meshes/`, for the GEOMETRY column and for every mesh path a setup names.
- The collected CSV tables (`campaign_sweep.csv`, `polars/P<POL>.csv`) carry the fixed flight-condition and reference columns of pyflightstream after `POL` and `point`, write `NA` for a missing value instead of an empty cell, and quote no cell, so `numpy.genfromtxt` reads them.
- The documentation site has five tabs: Start here, Workflow, Solvers, Commands and Reference. The common workflow is told once (workflow, workspace layout, executables, records, results, cluster jobs, storage, troubleshooting) and each solver page states what differs, under the same five headings: install, mesh, setup file, what converged means, worked example.
- The Commands page lists every command and option of `cfd-workspace` and `cfd-matrix`, written from the argument parsers by `python tools/build_cli_page.py`; a test fails when a command or an option is missing from the page.
- First campaign is a complete campaign on the mock solver, with every file and command shown.
- `cfd-matrix plan --local` fails with the one missing-executable sentence when the solver executable does not exist on this machine, as `run --local` does; a plan without `--local`, made for a cluster, does not look for it.
- A missing executable, a busy machine-wide slot or the RAM floor refuses `run --local` before any point is recorded (`matrix not run: ...`, exit 2), so the same command works once the cause is gone. Every solver words a missing executable alike and names the key to set.
- A failed point reads `FAILED_EXECUTION` (or `FAILED`) at the console instead of `COMPLETED`; the records are unchanged. Console times print as local time to the second.
- `--resume` says `skipped: COMPLETED failed; redo it with --force-rerun <point>`, and the already-recorded refusal says what to type.
- A setup path that names an existing legacy mesh file is used even when `inputs/mesh/` holds a same-named file; one warning says both exist.
- `cfd-workspace init --solver mock` creates `inputs/mesh/`.
- QPROP staging for a cluster runs no process: it writes `sections.plan`, `qcon.def` and, when every polar is given, `propeller.prop` and `sections.json`. `STAGED_CONFIG` is `sections.plan`.
- SU2 convergence counts only when the solver prints its convergence message after the last iteration line of its log.
- A setup without `HISTORY_OUTPUT` gets `ITER, RMS_RES, AERO_COEFF`, the groups the adapter reads; a `HISTORY_OUTPUT` that lacks one is extended with a warning.
- A matrix that gains a row after points ran keeps its finished points valid: staleness is decided by the files each point read, not by the matrix file.
- `SOLVER_BUILD` is checked when the run starts for SU2 (executable route) and PyFR, whose executables state their version; the run is refused before any point starts.
- The neutral `CL`, `CD` and `CM` are defined in one standard frame for every solver (lift normal to the freestream and drag along it, from `ALPHA` and `BETA`; pitching moment nose-up positive): SU2's two-dimensional `CM` is `-CMz`, and FlightStream's body-axis forces are rotated into the wind axes. Each solver's own columns are unchanged beside them.
- Rotor-capable adapters (QPROP) write the rotor-frame columns `CT`, `CQ`, `CP`, `CN`, `CY`, `J` and `EFFICIENCY` (x along the thrust, z up, y completing the right-handed frame; `CQ` is the shaft torque about the thrust axis); QPROP gains the own column `CQ`.
- QPROP's staged config is `propeller.prop`, the file the solver reads.
- `post` exits 0 once the tables are written, even when points failed, as pyflightstream's `post` does; `collect` keeps 1 for failures and 3 for outstanding work.
- `mark-failed` refuses a `RUNNING` point (stop the run first); `delete-sims` needs `--force` for a `RUNNING` point and refuses a sim another manifest still records; a marked-failed point keeps the mark's reason in `collect`, `status` and the summary; `sync --prefer-other` at level `post` or `all` refuses without `--overwrite`.
- Every warning is now written to `logs/warnings.log`; `--warning-off` only silences the console.
- The 0.5 GiB free-RAM floor and the host memory measurement have one home, `cfdbender.core.run` (`free_ram_gib`, `check_available_ram`); the adapters' `require_ram` imports it, so the two guards cannot drift.
- An entry that does not apply to a row is now warned and skipped instead of silently ignored or refused: a GEOMETRY cell for a solver that does not read geometry (it was refused), and REF or PPROC keys that XFOIL and QPROP never read. The warning names the row, the file, the entry and why.

### Fixed

- QPROP takes the blade from the row GEOMETRY alone on run and collect.
- The mock refuses an unknown setup key and collects without `execution.json`; PyFR classifies exit 0 with an empty history as failed; XFOIL reads its polar by column name.
- QPROP collection reports EFFICIENCY as the printed effprop and adds EFFICIENCY_OVERALL (the printed eff), instead of recomputing J CT / CP.
- The contract and architecture pages state their exceptions: declared outputs, the native passthrough, the applied management verbs, the per-solver axes behind the neutral `CL`, `CD` and `CM`, and the post stage.
- QPROP writes its staged `propeller.prop`, `qcon.def` and `sections.json` with LF line ends on Windows, as on Linux.

### Removed

- The pages Workspace files, Workspace settings and the earlier Commands page; their content is on the new pages.
- `cfd-matrix rebuild --inputs-from` is gone: it only fed pyflightstream's comparison of the executed script with the rendered one, which has no solver-neutral counterpart. It was accepted and then refused; the parser now rejects it.

### Migration

- A workspace that keeps its meshes in `inputs/geometry/`, `inputs/geometries/` or `inputs/meshes/` still plans and runs; the first use of an older name prints one warning. Move the files to `inputs/mesh/` to silence it. When a file exists in both `inputs/mesh/` and an older folder, `inputs/mesh/` wins.

## [0.4.2] - 2026-11-12

- Release workflow installs the docs extra for the wheel tests and can run as a dry run without publishing.

## [0.4.1] - 2026-11-12

- Release workflow installs libGLU for gmsh before testing the wheel, so the package reaches PyPI.
- Planning and the run registry judge an output by its folder and name, not by a mesh link target.

### Added

- `cfd-matrix run --resume` skips recorded COMPLETED and SUBMITTED points and
  runs the rest, including a point a killed local run left RUNNING once no
  solver process holds the host's slot. `--force-rerun POINT` (repeatable) and
  `--force-rerun-all` redo recorded points after archiving them:
  `archive/<stamp>/` receives a copy of `runs.json` and the point folders, and
  `runs.json` keeps the old records in its `history` list.
- `cfd-matrix plan`, `run` and `collect` take `--sims POL[,POL]` and
  `--points NAME` to work on part of a matrix; an unknown id is refused before
  anything runs.
- `cfd-matrix collect` writes `post/<matrix>/campaign_sweep.csv` and
  `post/<matrix>/polars/P<POL>.csv` beside the campaign summary: one row per
  point with its condition, status, classification, iterations, residual, wall
  time, solver version and every coefficient.
- `inputs/executables.toml` and its unversioned overlay
  `inputs/executables.local.toml` map each solver build to an executable and an
  optional launcher. A local run of a row naming SOLVER_BUILD uses that build's
  executable over the setup's `[solver]` executable and refuses a build the file
  does not declare. `cfd-workspace init` writes `inputs/.gitignore` for the
  local file. Cluster jobs keep taking their build from the HPC profile.
- Matrices read the pyflightstream column set with the same names and
  meanings: HIDDEN, AIRCRAFT, CONFIGURATION, DESCRIPTION, GEOMETRY, SYMMETRY,
  SYMMETRY_LOADS, NCPUS, SOLVER_BUILD (FS_BUILD accepted), WORKFLOW and
  VAR_NAMES_VALUES, with space-padded cells and `-` for an unstated cell. See
  the matrix columns page.
- FLIGHT_CONDITION reads the pyflightstream keys (MACH, TASmps, REmi, ALTFT,
  dISA and the RHOkgm3, MUPas, ASMPS, TK and PPA pins) and resolves them into
  one flow state with the ISO 2533 atmosphere. SU2 writes the resolved Mach,
  temperature, pressure, density and Reynolds number.
- GEOMETRY names the mesh of an SU2 row; the setup no longer needs one.
- OpenFOAM, PyFR, XFOIL and QPROP take their free stream from the resolved
  FLIGHT_CONDITION (speed and nu; mach and reynolds; V, RHO, MU and A) and
  their mesh, airfoil or blade from GEOMETRY.
- VAR_NAMES_VALUES overrides setup keys per row.
- NCPUS, WALLTIME and SOLVER_BUILD fill `{ncpus}`, `{walltime}` and
  `{solver_version_alias}` of an HPC profile; `{solver_type}` defaults to the
  solver name.
- The plan listing and the campaign summary carry AIRCRAFT, CONFIGURATION and
  DESCRIPTION.

### Changed

- A submitted point is self-contained: before the submit command runs, the
  point folder receives every file the solver reads, as a local run stages them
  (SU2 `su2.cfg` and the mesh, the OpenFOAM case tree and `Allrun`, PyFR
  `case.ini` and `mesh.msh`, QPROP `propeller.prop`, `qcon.def` and
  `sections.json`, XFOIL `xfoil.in`). `{config_path}` names that file instead
  of `solver.in`. `{stdout_log}` and `{stderr_log}` let a profile send the job's
  channels to the point folder, and collect completes a submitted point from
  the files the job writes.
- Meshes are never copied into a point folder: every solver stages a relative
  symbolic link, or a hard link on a Windows host without the symbolic-link
  privilege, and refuses when neither can be made. `mesh_link = false` is
  refused.
- Datapoint folders are named like pyflightstream's, `DP-M850RE500AL+025`, from the
  point's FLIGHT_CONDITION in the cell's order, instead of `DP-<20 hex>`. Two sweep
  values that round to one name are refused at plan time. Recorded points keep
  their folders and old `runs.json` records still load. See the workspace files
  page.
- `cfd-workspace init` for a library solver creates only the input folders it
  reads: geometries, references, setups, pproc, hpc and matrices.

### Migration

- HPC profiles that read `solver.in` through a fixed name read the solver's
  own file now (`su2.cfg` for SU2); profiles using `{config_path}` need no edit.
  Points submitted before this change hold only `solver.in` and no mesh: plan
  and submit them again.
- Matrices and setups written for 0.4.0 plan as before. A row stating only
  MACH and angles keeps the setup's free stream. A setup value that the matrix
  also states must now agree with it.

## [0.4.0] - 2026-11-12

### Added

- PyFR 3.1 workspaces plan two-dimensional compressible Navier-Stokes cases
  from Mach, Reynolds and alpha, run mesh import and the solver through an
  optional WSL launcher, and collect window-averaged Cl and Cd with their
  references. Convergence means stationary window means; see the PyFR guide.
- QPROP 1.22 workspaces for an isolated propeller. A blade comes from a 3D OBJ,
  sliced at a chosen number of radial stations, or from a polars-2D set of
  section cuts and polars. Missing section polars are produced by XFOIL at each
  section's local Reynolds and Mach number, retried with a narrower alpha range
  and then reused from the nearest good section, recorded in `sections.json`.
  Collection returns J, CT and CP in QPROP's and the standard normalisation,
  efficiency, thrust, torque and power.
- A solver-neutral blade slicing module recovers radius, chord, pitch and the
  unit-chord section shape from a blade mesh.

### Changed

- The XFOIL adapter exposes `run_script`, its guarded execution path for a
  keystroke script, so other adapters can run XFOIL without a second copy.
  XFOIL workspace behaviour is unchanged.

## [0.3.0] - 2026-11-12

### Added

- XFOIL workspaces plan deterministic scripts for NACA airfoils or tracked
  coordinates and run independent alpha points. Collection preserves printed
  polar coefficients, axes and unit-chord, quarter-chord references.
- Serial ESI OpenFOAM `simpleFoam` workspaces translate tracked gmsh meshes
  and alpha conditions into cases and collect force coefficients with their
  references. The kit includes a repeatable NACA 0012 mesh generator.
- Both adapters support Windows-to-WSL launchers, process safeguards and
  offline stand-ins. Solver guides and requirement rows document their use.

### Fixed

- XFOIL rejects unknown setup keys and invalid or missing polar evidence.
- OpenFOAM requires matching final force and SIMPLE evidence, retains the ESI
  body minus z pitch direction, and rejects incomplete or fatal output.
- Shared solver host helpers keep execution safeguards solver-neutral, and the
  architecture guard rejects imports between sibling adapters.

### Migration

- Select `xfoil` or `openfoam` in a new workspace and supply the documented
  s/r/p inputs. XFOIL classifies missing requested polar rows as failed and
  reports an acceptance marker rather than internal iteration history.
- OpenFOAM CmPitch retains ESI's sign about body minus z. The mesh generator
  produces a reference example, not a mesh-independence claim.

## [0.2.0] - 2026-11-11

### Added

- FlightStream workspaces and commands delegate to pyflightstream, including
  planning, runs, collection, post-processing, grouped batches and HPC options.
  The optional `flightstream` extra installs the upstream dependency. The SRS
  and documentation describe the interface and verification status.
- A read-only FlightStream neutral reader maps recorded statuses and final
  evidence, retaining coefficient axes, references and upstream values. Submitted
  points remain unclassified; missing or ambiguous completed evidence is refused.
- The SU2 pysu2 route runs in a child interpreter with the 8.6 serial driver,
  execution safeguards and route evidence. Configure its path in solver settings
  or `CFDBENDER_PYSU2_PATH`; automatic selection falls back if unavailable.
- A 1,280-cell reference mesh and an opt-in route comparison tool support SU2
  acceptance checks. Public readiness permits this specific fixture.
- The architecture guard refuses core and HPC imports of the FlightStream facade.
  Offline tests cover delegation parity and comparator mutants.

### Changed

- The development gmsh dependency is pinned to 4.15.2 for mesh reproduction.

### Fixed

- Configured SU2 executable paths resolve directly, including Python scripts on
  Windows when `.PY` is absent from `PATHEXT`.

### Migration

- Install `cfdbender[flightstream]` before using FlightStream workspaces. SU2 and
  mock campaigns retain their command paths. The licensed solver is separate.
- An available pysu2 route now runs instead of refusing as a stub. Explicit
  unavailable selection still refuses; automatic selection can fall back.
- The FlightStream reader maps walltime and missing-log statuses to failed and
  leaves submitted points unclassified until upstream records an outcome.
- The SU2 route comparison needs an installed wheel and both SU2 routes in the
  same interpreter environment; existing inputs need no migration.

## [0.1.0] - 2026-11-10

### Added

- Solver-independent workspace initialization through `cfd-workspace init
  --solver mock|su2`, one solver per `init.toml`, the campaign folder tree,
  and refusals for unknown solvers and existing declarations.
- Named pipe-delimited `.fs` matrices with s/r/p TOML inputs, flight conditions,
  sweeps, walltimes and stable point identities. `cfd-matrix plan` reads inputs
  and prints proposed actions and outputs without writing or running a solver.
  Invalid cells, duplicate identities and overlapping output claims are refused.
- A locked, atomic run registry with input/output SHA-256 provenance and
  PLANNED, RUNNING, SUBMITTED and COMPLETED states. Finished evidence is sealed;
  input drift and output collisions refuse before execution.
- A shared adapter contract and conformance suite. The licence-free mock and
  SU2 adapters expose explicit coefficient axes, references, iteration history
  and classifications `converged`, `iteration_limit` and `failed`.
- `cfd-matrix run --local` translates inputs, executes locally with RAM and
  process guards, and records numerical outcomes. `collect` reads saved evidence
  and writes neutral campaign summaries; process success alone is not convergence.
- SU2 executable execution with s/r/p translation, tracked mesh staging,
  choosable routes, captured channels, timeout and history parsing. The default
  route prefers pysu2; when it cannot import, selection falls back with a warning
  to the executable. Explicit `route = "executable"` selects native execution.
  Importable pysu2 currently refuses execution with a clear pending-support message.
- HPC profiles in `inputs/hpc/h<id>.toml`, descriptor rendering and submit argv,
  walltime limits and exact build/application translations. `plan`, `run` and
  `collect` accept `--hpc NAME` and `--local`. Planning submits nothing; submission
  acceptance stays SUBMITTED until saved numerical outputs can be collected.
  Submission timeout retains unknown acceptance without automatic resubmission.
- Draft SRS v0, requirement-driven documentation, MIT licence, citation metadata,
  offline CI, an architecture guard and repository/history readiness checks.
- `python tools/build_kit.py --out OUTPUT` builds a deterministic wheel and one
  unzip-and-go offline kit. Its first prompt creates a folder-local venv,
  installs offline, finds the site's separately supplied SU2 executable and
  runs/collects a generated 2D smoke case. Its HPC profile contains fake values.

### Changed

- The package version is 0.1.0. There are no runtime dependencies; the pure Python
  wheel installs offline with Python 3.12 or newer. SU2 itself is supplied separately.

### Limitations

- The pysu2 route implementation is not implemented. The available route selector
  is a documented fallback stub, not a working wrapper driver.
- Grouped HPC jobs (`--batch`) are not implemented in the CLI. Pure grouped budget
  planning exists, but no grouped-job execution is claimed.
- FlightStream, OpenFOAM and Fluent solvers are not implemented.
- A fake submit client proves the offline HPC chain, not real cluster execution.
  SU2 8.3.0 cluster compatibility and native Windows pysu2 remain unverified.

### Migration

- Declare exactly one registered solver in `init.toml`. Workspace defaults may use
  `[solver]` with `name`, route and executable; the s file's `[solver]` overrides them.
- All solver inputs live in s/r/p files. SU2 keys may pass through in uppercase;
  invalid physical values, duplicate assignments and misspelled options refuse.
- One HPC profile selects itself and submits by default. Use `--local` for the
  local smoke run. Collect exit 3 means outstanding outputs; exit 1 means failure.
- Registry readers must recognize SUBMITTED and submission evidence. Adapters
  implement validation, translation, execution, classification and collection.
- Contributors run `python tools/check_architecture.py` when extending boundaries.
- Kit builders need build, setuptools>=77 and gmsh in their build interpreter.
  Kit users need Python>=3.12 and SU2_CFD. Use direct venv CLI paths and --local.
