# The workspace declaration

A workspace belongs to exactly one solver. Its root `init.toml` declares that
solver and the file schema:

```toml
schema_version = 1
solver = "mock"
```

Only `mock` is registered in this skeleton. It is an offline registration
placeholder; it does not execute simulations. The declaration is read through
`cfdbender.core.read_init(root, available_solvers=available_solvers())`, where
`available_solvers` comes from `cfdbender.solvers`. Core receives names as data
and never loads an adapter. Adapter discovery and loading belong to
`cfdbender.solvers.available_solvers()` and `load_solver(name)`.

The reader refuses missing or unreadable files, invalid TOML or UTF-8, unknown
keys, schema versions other than the integer `1`, a missing or non-string solver,
and unknown solver names. Its error names `init.toml`, the problem and the
available solvers. Reading never changes the file.

All simulation inputs, including solver-specific inputs, will live in the s, r,
p files. There is no SOLVER-DNA column or DNA directory. Matrix expansion and
the contents of those input files belong to later work.
