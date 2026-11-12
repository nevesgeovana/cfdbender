# Package boundaries

`cfdbender.core` owns solver-independent workspace declarations and layout. It
receives available solver names as data and never imports `cfdbender.solvers`.
`cfdbender.solvers` lists registered adapters and loads one by name. This
skeleton registers only `cfdbender.solvers.mock`, a placeholder for later
execution and conformance work. `cfdbender.workspace.cli` composes core with
solver discovery.

Run the source guard with:

```console
python tools/check_architecture.py
```

It parses source without executing adapters. Absolute and relative imports,
imports inside functions, and literal dynamic imports from core into solvers
are violations, reported with file and line and exit code 1. Upstream library
imports anywhere except a future `cfdbender._pyfs` facade are also violations.
Missing core sources and malformed Python are refusals with exit code 2.
Computed dynamic imports are outside this static check's coverage.

The offline tests run the guard against the live package and planted mutants,
including a core module importing `cfdbender.solvers.mock`. Legal imports,
comments and docstrings pass, so a mention alone is not treated as an import.
