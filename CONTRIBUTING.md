# Contributing to cfdbender

Write a failing test for the intended behavior first, then implement it. Install
the development dependencies with `pip install -e ".[dev]"` and run:

```console
pytest
ruff check .
python tools/check_public_text.py
ruff format --check .
mypy src
python tools/check_architecture.py
```

Build the documentation site with `pip install -e ".[docs]"` and
`properdocs build --strict`.

Do not commit solver outputs or private geometry. Keep contributions original
or compatible with the MIT license.

## Rules a change must respect

The design behind these rules is on the
[architecture page](docs/architecture.md). Each rule names what checks it. A
rule marked *convention* has no automatic check yet: a reviewer checks it.

| Rule | Checked by |
| --- | --- |
| **Test first, and prove the test.** Write the test before the code. Before believing it, revert the fix and see the test fail. | convention |
| **Keep the import rule.** The core imports no adapter; an adapter imports no other adapter; pyflightstream is imported only in `cfdbender._pyfs`. | `python tools/check_architecture.py` |
| **Do not import upward.** The core and the adapters do not import `cfdbender.matrix`, `cfdbender.workspace` or `cfdbender.hpc`; `cfdbender.hpc` and `cfdbender.geometry` import nothing else of cfdbender. | convention |
| **One writer per file.** `runs.json` is written only by the run registry in `cfdbender.core.registry`, under its lock and by atomic replacement. A new feature that needs to change a record adds a registry method. | `tests/tier1_offline/test_p0000_item3b_registry.py`, `test_p0000_item3c2_registry.py` |
| **Never overwrite, never delete.** An existing point folder or output refuses the run. Redoing a point archives it first. | `tests/tier1_offline/test_p0000_item3c2_run.py`, `test_matrix_rerun.py` |
| **Read inputs through the tracked reader.** Every workspace file an adapter or the planner reads goes through `InputFiles` or `RowInputs`, so its digest is recorded and checked again later. | `tests/tier1_offline/test_p0000_item3b_registry.py`, `test_p0000_item3b_plan.py` |
| **Process success is not convergence.** A point is classified from numerical evidence. A coefficient is returned with its axis and its references. | `cfdbender.core.results`; `cfdbender.testing.assert_exit_evidence`, `assert_results` |
| **A refusal says what is wrong and what to do.** The message names the file, the row and the key or value at fault, and the accepted values or the next command. In the core raise `WorkspaceError` or one of its subclasses (`MatrixError`, `PlanError`, `RegistryError`); in an adapter raise `ValueError`. The commands print the message to standard error and exit with code 2. | convention; each refusal has its own test |
| **Refuse with care.** Refuse what would write a wrong result or lose evidence. An entry that simply does not apply to a row is reported as a warning and skipped. | convention |
| **Write the same bytes on every platform.** Open a text file for writing with `newline="\n"`, or write bytes. Never rely on the platform's line ending for a file whose digest is recorded. | convention |
| **One home per constant.** A limit, a file name or a pattern used in two modules is defined once, in the lower layer, and imported. | convention |
| **A shared behaviour is tested for every solver.** A behaviour every adapter must have is asserted in `cfdbender.testing` and run for every registered adapter, never proved on the mock alone. | the conformance suite in `tests/tier1_offline/` |
| **Launch safely.** A child process is started without a shell and without a console window, under the host guards of `cfdbender.solvers._host`. | `tests/tier1_offline/test_p0000_item3f_run.py`, `test_p0000_solver_host_helpers.py` |
| **Documentation moves with the code.** A change a user can see comes with its page under `docs/` and a fragment `changelog.d/<package>.md` in the same commit. A change to what an adapter must do changes the numbered clause in `docs/adapter-contract.md` and its test together. | `properdocs build --strict`; otherwise convention |
| **Public text stays public.** No private name, path or process note in tracked text. | `python tools/check_public_text.py` |

## Adding a solver

A solver is an adapter package that satisfies the
[adapter contract](docs/adapter-contract.md). The contract page ends with the
steps: the package and its five functions, the registration, the input
layout, the shared host helpers, the stand-in executable and the conformance
tests, and the documentation page.

Before opening the change, check that:

- every clause of the contract is either proved by a test of the new adapter
  or named in the change description as not yet proved;
- the adapter's module docstring states what `converged` means for the
  solver, where its reference values come from and how its axes are oriented;
- no test needs the real solver: a test that does is marked and skipped where
  the solver is absent.

## Testing without a solver

The tests run a Python stand-in in place of each solver. A setup whose
`[solver]` table names a `.py` file as `executable`, with no launcher, runs
that file with the Python interpreter; the PyFR tests use this to write a
synthetic `forces.csv`. A stand-in proves the workflow, never the physics: a
real run is the only evidence of accuracy for a given mesh.
