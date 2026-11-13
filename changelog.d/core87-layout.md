## Changed

- `cfd-workspace init` creates `inputs/mesh/` for every library solver (su2, openfoam, pyfr, xfoil, qprop), in place of `inputs/geometries/`. One mesh resolver reads `inputs/mesh/` first, then the older `inputs/geometry/`, `inputs/geometries/` and `inputs/meshes/`, for the GEOMETRY column and for every mesh path a setup names.

## Migration

- A workspace that keeps its meshes in `inputs/geometry/`, `inputs/geometries/` or `inputs/meshes/` still plans and runs; the first use of an older name prints one warning. Move the files to `inputs/mesh/` to silence it. When a file exists in both `inputs/mesh/` and an older folder, `inputs/mesh/` wins.
