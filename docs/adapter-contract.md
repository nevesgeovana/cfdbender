# Adapter contract

A solver adapter is a Python package, `cfdbender.solvers.<name>`, that
translates a planned point into one solver's input, runs the solver, says how
the run ended and reads the results back. Everything else (the matrix, the
flight condition, the point names, the run record, the tables) belongs to the
core and is the same for every solver. The [architecture](architecture.md)
page shows where an adapter sits.

This page states each clause an adapter must satisfy. A clause is numbered so
that a test, a review or a bug report can name it. `cfdbender.testing` holds
the assertions that prove the clauses; the last section shows how to call
them.

The words used here: **must** is a clause every adapter satisfies; **may** is
an optional behaviour an adapter declares by providing it. The **point
folder** is `sims/sim_<POL>/datapoints/DP-<name>/`. The **s, r and p files**
are the row's setup, reference and post files.

## The five functions

An adapter exposes five functions at its package boundary. The core calls
them in this order and never anything else.

```python
def validate_row(row: MatrixRow, workspace: RowInputs) -> RowRequirements: ...
def translate(point: AdapterPoint) -> str: ...
def execute(point: AdapterPoint, *, check_ram: Callable[[], None]) -> ExecutionResult: ...
def classify(result: ExecutionResult) -> Classification: ...
def collect(point: AdapterPoint) -> NeutralResult: ...
```

| Function | When | It may | It must not |
| --- | --- | --- | --- |
| `validate_row` | planning, once per row | read inputs through `workspace`; refuse | write a file; launch a process |
| `translate` | before a run or a staging | return the solver input as text | write; depend on the clock or the machine |
| `execute` | a local run, once per point | create the point folder, write, run the solver once | touch anything outside the point folder; overwrite |
| `classify` | after `execute`, and inside `collect` | judge numerical evidence | call a run converged from its exit code |
| `collect` | `cfd-matrix collect` | read the saved outputs | write; run anything |

The types come from `cfdbender.core.matrix` (`MatrixRow`),
`cfdbender.core.plan` (`RowInputs`, `RowRequirements`),
`cfdbender.core.adapter` (`AdapterPoint`, `SolverAdapter`) and
`cfdbender.core.results` (`ExecutionResult`, `NeutralResult`, `Coefficient`,
`References`, `Iteration`, `Classification`).

## Registration and boundaries

**C-01.** An adapter is registered by one entry, name to package, in
`cfdbender.solvers`. It is loaded by that exact name; an unknown name is
refused with the list of available solvers.

**C-02.** The package exposes the five functions above with those signatures.
An adapter is a module, not a class: the functions are module attributes.

**C-03.** An adapter imports the core, `cfdbender.geometry`,
`cfdbender.solvers._host` and its own package. It never imports another
adapter's package and never imports pyflightstream. An adapter that needs
another solver's run (QPROP needs XFOIL polars) asks the registry for it by
name and calls a function that adapter exports for that purpose.

**C-04.** An adapter never imports `cfdbender.matrix`, `cfdbender.workspace`
or `cfdbender.hpc`. The applications call the adapter, never the reverse.

**C-05.** Importing an adapter imports no solver binding and launches
nothing. A solver's Python binding is imported only inside the child process
that runs it.

## Planning: `validate_row`

**C-06.** `validate_row` only reads. It writes no file, creates no folder and
launches no process.

**C-07.** Every input byte is read through the tracked reader
(`workspace.read_bytes`, `workspace.read_toml`, and the prepared
`workspace.setup`, `workspace.reference`, `workspace.post`). The reader
records the SHA-256 of each file, and the core checks those digests again
before the point runs and before it is collected. A file opened any other way
is invisible to that check. A path outside the workspace is refused.

**C-08.** An invalid input is refused with a `ValueError` whose message names
the key or value at fault and, where a choice exists, the accepted values.
The core adds the data row and the POL.

**C-09.** A key of the s file that the adapter does not know is refused at
planning, never ignored: the refusal names the file, the key as written and the
closest known key. A key of the r or p file that the adapter does not read is
not refused: it is warned about once, naming the row, the file and the key, and
skipped (`RowInputs.warn_unread`).

**C-10.** `validate_row` returns at least one action (a short phrase the plan
listing prints after "would:" under `--verbose`) and every output file the point will write, as
templates under `sims/sim_{pol}/datapoints/{point}/`. An adapter that declares
no action or no output is refused by the core, as is an output outside
`sims/` or `post/`. After a run that is not failed every declared output
exists; the core records a point with a missing declared output as failed.

**C-11.** The adapter accepts the full shared matrix header. It reads the
parsed row it is given and never reads the matrix file itself.

**C-12.** An adapter that takes a mesh or a geometry file accepts it from the
`GEOMETRY` column and reads it through the tracked reader. Where the s file
may also name it, the two must agree or the row is refused.

**C-13.** An adapter does not spell a mesh folder name. It asks the core's
mesh resolver, which reads `inputs/mesh/` first and then the earlier folder
names.

**C-14.** The row's `FLIGHT_CONDITION` is the authority on the air and the
attitude. The adapter resolves it with `cfdbender.core.flight` and never with
an atmosphere of its own. A value that the s, r or p file also states must
agree with the row or the row is refused.

**C-15.** `VAR_NAMES_VALUES` overrides arrive already applied in
`workspace.setup` and in `point.setup`. The adapter does not apply them
again.

**C-16.** A `WORKFLOW` the adapter does not implement is refused at planning,
naming the workflow it accepts.

**C-17.** Every point of a sweep is validated at planning, not only the first
value, so that no point fails on input a plan could have refused.

## Translation and staging: `translate`, `stage`

**C-18.** `translate` returns ASCII text and returns the same bytes every time
for the same planned point. It writes nothing.

**C-19.** `translate` verifies the point first (`point.verify()`): the
`init.toml` solver and every planned input digest. A changed input refuses
with the instruction to plan again.

**C-20.** A local run and a cluster job read identical solver inputs. One
staging routine writes them for both.

## Execution: `execute`

**C-21.** `execute` creates, writes and changes files only inside the point
folder.

**C-22.** `execute` refuses when the point folder or any declared output
already exists. A second execution leaves every byte of the first unchanged.

**C-23.** `execute` calls `check_ram` before it writes and repeatedly while
the solver runs, and uses the host memory guard, which refuses below 0.5 GiB
of free memory. When the guard trips during a run the child process tree is
stopped and reaped.

**C-24.** `execute` holds the host process lease while the solver runs: one
solver process per machine, whichever the solver.

**C-25.** The run is bounded: the adapter's `timeout_s` option, itself bounded
by the row's `WALLTIME`. On a timeout the whole child process tree is stopped,
the captured channels are kept and the point is `failed`.

**C-26.** The solver is launched without a shell and without a console
window. Its standard output and standard error are captured to `stdout.log`
and `stderr.log` in the point folder.

**C-27.** A local run writes `execution.json` in the point folder with at
least the process exit code. `collect` must also work without
that file, because a cluster job does not write it.

**C-28.** An expected failure (a missing executable, a refused input, a solver
that exits with an error) is raised as `ValueError` or `OSError`, or returned
as failed evidence. The core records the point as failed and runs the next
one. An adapter does not catch other exceptions to present a programming
error as a solver failure.

**C-29.** The executable is chosen in this order: the build named by the
row's `SOLVER_BUILD` (through `inputs/executables.toml`), the s file's
`[solver]` table, then the environment variable and the `PATH` lookup the
adapter documents. Launcher tokens (for example `mpirun -np 8`, or a
WSL prefix) precede the executable verbatim. A missing executable refuses
before anything is written.

**C-30.** A mesh is linked into the point folder, never copied: a relative
symbolic link, or a hard link where the host cannot make symbolic links. The
linked bytes are checked against the planned digest.

## Classification: `classify`

**C-31.** `classify` returns exactly one of `converged`, `iteration_limit` or
`failed`.

**C-32.** Process success is not convergence. Exit code zero with no
numerical evidence is `failed`. A non-zero exit code is never `converged`.

**C-33.** `converged` requires affirmative numerical evidence, and the
adapter states in its documentation what that evidence is:

| Solver | A point is `converged` when |
| --- | --- |
| SU2 | the solver prints its convergence message after the last iteration line of its log (a message followed by more iterations does not count) |
| OpenFOAM | the force output and the log end on the same iteration, and on it SIMPLE reports convergence or every required initial residual is below the residual target |
| PyFR | no value is non-finite and the two halves of the averaging window agree within the row's tolerance |
| XFOIL | the polar holds a row at the requested angle of attack |
| QPROP | the operating table holds a parsed row and the output reports no convergence failure |

**C-34.** `iteration_limit` means the solver used its whole iteration or time
budget without the evidence of C-33. A solver that has no such budget (XFOIL,
QPROP) never returns it.

**C-35.** `classify` depends only on the `ExecutionResult` it is given.
`collect` rebuilds that evidence from the saved files and reaches the same
class as the one recorded when the point ran.

## Collection and results: `collect`

**C-36.** `collect` only reads. It verifies the point, reads the saved
outputs, and neither writes nor launches anything.

**C-37.** The `NeutralResult` carries, for every coefficient, a finite value,
its axis (`x`, `y` or `z`) and its kind (`force` or `moment`); the reference
area and length (positive, finite), the moment origin (three coordinates), the
name of the axis system and the direction of positive x, y and z; the
iteration history; the classification; and the solver version.

**C-38.** A coefficient keeps the name, the sign and the normalisation the
solver prints. The adapter does not rotate a frame or renormalise a
coefficient unless it documents it and reports both. Where a solver reports
in two frames, each coefficient states its frame. The tables of `collect` add
the neutral columns `CL`, `CD` and `CM` beside the solver's own names; which
of its names an adapter gives them is stated in one table, `core/neutral.py`,
and a solver with no such coefficient writes `NA`.

**C-39.** The references in the result are the ones the solver used: read
from the r file, or read back from the solver's own input, or, for a solver
with fixed references, stated by the adapter:

| Solver | Reference area and length | Moment origin |
| --- | --- | --- |
| SU2 | read back from the configuration the solver used | read back from the same configuration |
| OpenFOAM | the r file | the r file |
| PyFR | unit chord; dynamic pressure 0.5 | no moment is reported |
| XFOIL | unit chord and unit span | the quarter chord |
| QPROP | the disk area and the diameter | the hub centre |

**C-40.** The history counts iterations from one and carries a linear,
non-negative residual; a solver that prints a logarithm or counts from zero is
converted by its reader. A solver that reports no iterations (XFOIL, QPROP)
writes one marker, iteration 1 with residual 0, and its documentation says
that the marker is not a measurement.

**C-41.** An output table is read by its column names, never by column
position.

**C-42.** A `failed` result may carry no coefficient. Any other result
carries at least one.

**C-43.** An adapter may add solver-specific evidence to the result it
returns (SU2 adds the route it ran, PyFR the averaging window). It never
removes or renames a common field.

**C-44.** `solver_version` is the version the solver printed in its own
output, or the text `unknown` when the solver prints none.

## Cluster jobs: optional

An adapter declares these behaviours by providing the attribute.

| Attribute | Meaning |
| --- | --- |
| `stage(point)` | Write every file a job reads into a new point folder, without running the solver. |
| `STAGED_CONFIG` | The file in the point folder that the job's solver reads. |
| `JOB_OUTPUTS` | The declared outputs a cluster job writes. |
| `SOLVER_TYPE` | The solver's name as a cluster profile spells it, when it differs from the registered name. |
| `EXECUTABLE_KEYS` | The adapter's own `[solver]` option names for the executable path and the launcher. |
| `BUILD_OPTIONS` | Fixed `[solver]` options that apply when a build is bound. |

**C-45.** An adapter that provides `stage` also provides `STAGED_CONFIG`.

**C-46.** `stage` refuses an existing point folder and runs no solver. A
staging failure is raised as `ValueError` or `OSError`; the point is recorded
as failed and nothing is submitted.

**C-47.** `JOB_OUTPUTS` names the files, among the declared outputs, that a
cluster job writes. `collect` reads a submitted point once those files exist
and classifies it from them alone.

**C-48.** An adapter never submits a job and never names a scheduler option.
The descriptor and the submit command belong to the profile in `inputs/hpc/`.

**C-49.** An adapter whose `[solver]` options are not named `executable` and
`launcher` provides `EXECUTABLE_KEYS`, so that a row's `SOLVER_BUILD` selects
its executable like every other adapter's.

## What an adapter leaves to the core

**C-50.** The adapter never names a point folder. It receives
`point.point.directory`, which the core derives from the flight condition.

**C-51.** The adapter never reads or writes `runs.json`, the plan receipt or
anything under `post/`.

**C-52.** A point folder may be moved to `archive/` when the point is redone.
Whatever the adapter leaves in it (a mesh link above all) must still resolve
after the move, which is why links are relative.

**C-53.** The adapter writes no campaign table. The names of the coefficients
it returns become column names, so a name holds no comma, quote or line
break.

**C-54.** `cfd-workspace init --solver <name>` creates a workspace the adapter
can plan in: every input folder it reads exists.

## Physical meaning

**C-55.** Neutral quantities are in SI units: metres, square metres, kelvin,
pascals, kilograms per cubic metre, pascal seconds, metres per second. Angles
are in degrees. A matrix key carries its unit in its name (`ALTFT` is feet,
`REmi` is millions) and the core converts it; an adapter that hands a value
to a solver in another unit converts it explicitly.

**C-56.** The Reynolds number of a point is based on the reference length of
the r file, or on the fixed length the adapter states (unit chord for XFOIL
and PyFR). The adapter's documentation states the direction of positive
`ALPHA` and `BETA` in its axis system.

## Proving an adapter

`cfdbender.testing` provides one assertion per group of clauses. Each takes
the adapter and a freshly prepared case; none imports or launches a solver by
itself, so the adapter's tests supply a stand-in executable.

| Assertion | Clauses |
| --- | --- |
| `assert_reads_init(adapter, point)` | C-19 |
| `assert_validation(adapter, row, workspace, problem)` | C-08, C-16 |
| `assert_deterministic(adapter, point)` | C-18 |
| `assert_classifications(adapter, cases)` | C-23, C-31, C-35 |
| `assert_exit_evidence(adapter, point)` | C-32 |
| `assert_results(adapter, point)` | C-37 |
| `assert_no_overwrite(adapter, point)` | C-22 |
| `assert_execution_isolation(adapter, point)` | C-21 |
| `assert_run_record(adapter, plan)` | C-10 |
| `assert_finished_no_overwrite(adapter, plan)` | C-22 |
| `assert_collected_results(adapter, plan)` | C-36, C-37 |

```python
from cfdbender import testing
from cfdbender.core.adapter import load_point
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

adapter = load_solver("mysolver")
plan = plan_workspace(root, "study")
point = load_point(root, plan.points[0], solver="mysolver")
testing.assert_deterministic(adapter, point)
```

## Adding a solver

1. Create `src/cfdbender/solvers/<name>/` with the five functions and a module
   docstring that says what the s, r and p files hold, how the executable is
   chosen, what `converged` means and which references and axes the result
   carries.
2. Register the name in `cfdbender.solvers`.
3. Add the name to the solvers that get the short input layout in
   `cfdbender.core.workspace`, unless it needs the full one.
4. Reuse `cfdbender.solvers._host` for the memory guard, the process lease,
   the timeout, the process cleanup and the mesh link.
5. Write a stand-in executable for the offline tests and add the adapter to
   the conformance suite and to the all-solvers matrix test.
6. Add `docs/<name>-workspaces.md` and its navigation entry.
7. Run the gates listed in `CONTRIBUTING.md`.
