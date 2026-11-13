## Added

- Every completed run record carries `solver_version`, the version the solver reported; PyFR now asks its executable (`--version`) instead of always reporting `unknown`, and a rebuilt record takes it from the collected result.
- A local run whose row states `SOLVER_BUILD` is recorded failed, with a clear message, when the version the solver printed differs from it (`8.3.0-mpi` is an alias of `8.3.0`); an unreadable version cannot be checked and is not refused.
