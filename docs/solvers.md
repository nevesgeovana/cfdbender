# Solvers, side by side

You choose a solver once, when you run `cfd-workspace init --solver <name>`, and every solver reads the same matrix. This table shows what changes from one solver to the next; each solver page gives the details.

| | [CFD++](cfdpp-workspaces.md) (incomplete) | [SU2](su2-workspaces.md) | [OpenFOAM](openfoam-workspaces.md) | [PyFR](pyfr-workspaces.md) | [XFOIL](xfoil-workspaces.md) | [QPROP](qprop-workspaces.md) | FlightStream |
|---|---|---|---|---|---|---|---|
| Computes | steady compressible RANS flow, input written by cfdbender; not yet verified against the solver | steady or time-accurate flow, any SU2 solver, with rotor run types | steady `simpleFoam` flow on a 2D extruded mesh | unsteady 2D compressible flow, time-averaged forces | 2D airfoil polar, one point per angle | isolated propeller performance | see [FlightStream](flightstream-workspaces.md) |
| Mesh | a native grid folder (`nodesin.bin`, `cellsin.bin`, `exbcsin.bin`, `mcfd.bc`) | `.su2`, `.cgns` | `.msh` (gmsh 2.2 ASCII); `.cgns`, `.ugrid` with `mesh_converter` | `.msh` (gmsh) | none: `NACA` designation or `.dat` coordinates | blade `.obj` or a polars-2D set | see [FlightStream](flightstream-workspaces.md) |
| MESH becomes | the grid folder in `inputs/mesh/` | `MESH_FILENAME` | `mesh` | `mesh` | `airfoil_file` | `blade` (an OBJ) | see [FlightStream](flightstream-workspaces.md) |
| Setup keys | `iterations`, `convergence_orders`, `cfl`, `turbulence`, `[physics]`, `[fluid]`, `[numerics]`, `[boundaries.<family>]` | `iterations`, SU2 keys in capitals | `application`, `speed`, `nu`, `turbulence`, `iterations` | `mach`, `reynolds`, `order`, `tend`, `window_start` | `airfoil`, `reynolds`, `mach`, `ncrit` | `blade`, `n_blades`, `n_sections` | see [FlightStream](flightstream-workspaces.md) |
| `[solver]` keys | none; builds are declared in `inputs/executables.toml` | `route`, `executable`, `pysu2_path` | `foam_bashrc`, `launcher` | `executable`, `launcher` | `executable`, `launcher` | `executable`, `xfoil_executable`, `launcher` | see [FlightStream](flightstream-workspaces.md) |
| Environment variable | none | `CFDBENDER_SU2_EXE`, `CFDBENDER_PYSU2_PATH` | `CFDBENDER_OPENFOAM_BASHRC` | none | `CFDBENDER_XFOIL_EXE` | `CFDBENDER_QPROP_EXE`, `CFDBENDER_XFOIL_EXE` | see [FlightStream](flightstream-workspaces.md) |
| Convergence evidence | residual files and log; layouts not yet verified | convergence line in the SU2 log, `history.csv` | `forceCoeffs` file and SIMPLE log | stationarity of window means in `forces.csv` | one polar row at the requested angle | one operating row in the QPROP table | see [FlightStream](flightstream-workspaces.md) |
| Coefficients | `CFx`..`CFz`, `CMx`..`CMz` and their inviscid and viscous parts | `CL`, `CD`, `CSF`, `CFx`..`CFz`, `CMx`..`CMz` | `Cd`, `Cl`, `CmPitch` | `Cl`, `Cd` | `CL`, `CD`, `CDp`, `CM` | `J`, `CT`, `CP`, `CT_QPROP`, `CP_QPROP`, `EFFICIENCY`, `EFFICIENCY_OVERALL`, `T`, `Q`, `P` | see [FlightStream](flightstream-workspaces.md) |
| Neutral `CL`, `CD`, `CM` in the tables | not yet verified against the solver | `CL`, `CD`, `CMy` (`CMz` in a two-dimensional case) | `Cl`, `Cd`, `CmPitch` | `Cl`, `Cd`, none | `CL`, `CD`, `CM` | none | see [Results and tables](results.md) |
| Parallel option | `mpi_launcher`, `partitioner` in the build entry | `mpi_launcher` (executable route) | `processes`, `mpi_launcher` | none | none | none | see [FlightStream](flightstream-workspaces.md) |

## What each solver supports

These are cfdbender adapter capabilities, rather than everything a solver can
do natively. A missing declaration means no optional behavior. The table is
checked against the declarations by
`test_p051_f3c_documented_capabilities_match_declarations`.

| Capability | cfdpp | mock | openfoam | pyfr | qprop | su2 | xfoil |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `restart` | Yes | No | Yes | Yes | No | Yes | No |
| `unsteady` | No | No | No | Yes | No | No | No |
| `boundary_groups` | Yes | No | Yes | No | No | Yes | No |
| `cluster_staging` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| `numerics_file` | No | No | Yes | Yes | No | Yes | No |

CFD++ is incomplete and not yet verified against the solver; see [CFD++](cfdpp-workspaces.md).

`numerics_file` means the adapter reads the matrix column `NUMERICS`, a separate
file of numerical keys; see [matrix columns](matrix-columns.md#setup-and-numerics).
`restart` means resuming a solver from saved solution state. The run command's
`--resume` selects unfinished points and is not a solver restart.
`boundary_groups` means separate per-boundary force results. SU2 and OpenFOAM
accept a setup `[groups]` table and return `force_groups` with each set plus TOTAL.
`unsteady` means the adapter reads the time window keys of the setup file: PyFR
marches in time and reduces its `window_start` to `tend` window to time-averaged
forces, and its matrix `WORKFLOW` remains `steady`. SU2 time-accurate rows are
not this capability: the `WORKFLOW` column selects them, with the clock and
windows of the row (see [SU2 run types](su2-workspaces.md#run-types)). Every adapter can stage a cluster point. QPROP requires a
supplied polar for every section; staging never runs XFOIL to make missing
polars. The mock stages deterministic offline inputs for contract tests.
FlightStream forwards commands to pyflightstream unchanged and is not an
adapter under this capability contract.

A cluster request (`run --no-local` with a selected profile, or the cluster
preview in `plan`) checks `cluster_staging`. Without it, the command warns
once and skips staging and submission, with no local fallback. Without a
profile, the existing local default is unchanged. `RESTART` and `run --restart`
request a saved solution, as described in [records](records.md). Only PyFR accepts the time
window setup keys, and only SU2 accepts a `WORKFLOW` other than `steady`; unknown setup
keys and unsupported `WORKFLOW` values keep their existing validation errors.

## Convergence control of the RANS solvers

The two solvers that solve the Reynolds-averaged equations, SU2 and OpenFOAM, start from a robust convergence control when the setup states none, so a poor initial field does not diverge. Each uses the control of its own solver, so the controls differ: SU2 accepts the neutral keys `cfl`, `cfl_max` and `cfl_growth` as well as native keys. A CFL number and OpenFOAM under-relaxation factors or local Courant limit do not map onto each other.

| Solver | Default | Other route | Keys |
|---|---|---|---|
| SU2 | CFL ramp from 5 to 25 | any CFL key of yours replaces it | see [CFL ramp](su2-workspaces.md#cfl-ramp) |
| OpenFOAM | steady `simpleFoam` with under-relaxation p 0.3, U 0.7, nuTilda 0.7 | pseudo-transient local time stepping with `time_integration = "lts"` | see [Convergence control](openfoam-workspaces.md#convergence-control) |

In both, a value you state natively wins entirely: cfdbender writes what you stated and merges nothing from its default. PyFR (not RANS), XFOIL and QPROP have no such control.
