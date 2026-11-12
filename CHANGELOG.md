# Changelog

All notable changes to cfdbender. The format follows Keep a Changelog;
versions follow SemVer and are independent of solver versions.

## [Unreleased]

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
