# Solvers, side by side

You choose a solver once, when you run `cfd-workspace init --solver <name>`, and every solver reads the same matrix. This table shows what changes from one solver to the next; each solver page gives the details.

| | [SU2](su2-workspaces.md) | [OpenFOAM](openfoam-workspaces.md) | [PyFR](pyfr-workspaces.md) | [XFOIL](xfoil-workspaces.md) | [QPROP](qprop-workspaces.md) | FlightStream |
|---|---|---|---|---|---|---|
| Computes | steady flow, any SU2 solver | steady `simpleFoam` flow on a 2D extruded mesh | unsteady 2D compressible flow, time-averaged forces | 2D airfoil polar, one point per angle | isolated propeller performance | see [FlightStream](flightstream-workspaces.md) |
| Mesh | `.su2`, `.cgns` | `.msh` (gmsh 2.2 ASCII); `.cgns`, `.ugrid` with `mesh_converter` | `.msh` (gmsh) | none: `NACA` designation or `.dat` coordinates | blade `.obj` or a polars-2D set | see [FlightStream](flightstream-workspaces.md) |
| GEOMETRY becomes | `MESH_FILENAME` | `mesh` | `mesh` | `airfoil_file` | `blade` (an OBJ) | see [FlightStream](flightstream-workspaces.md) |
| Setup keys | `iterations`, SU2 keys in capitals | `application`, `speed`, `nu`, `turbulence`, `iterations` | `mach`, `reynolds`, `order`, `tend`, `window_start` | `airfoil`, `reynolds`, `mach`, `ncrit` | `blade`, `n_blades`, `n_sections` | see [FlightStream](flightstream-workspaces.md) |
| `[solver]` keys | `route`, `executable`, `pysu2_path` | `foam_bashrc`, `launcher` | `executable`, `launcher` | `executable`, `launcher` | `executable`, `xfoil_executable`, `launcher` | see [FlightStream](flightstream-workspaces.md) |
| Environment variable | `CFDBENDER_SU2_EXE`, `CFDBENDER_PYSU2_PATH` | `CFDBENDER_OPENFOAM_BASHRC` | none | `CFDBENDER_XFOIL_EXE` | `CFDBENDER_QPROP_EXE`, `CFDBENDER_XFOIL_EXE` | see [FlightStream](flightstream-workspaces.md) |
| Convergence evidence | convergence line in the SU2 log, `history.csv` | `forceCoeffs` file and SIMPLE log | stationarity of window means in `forces.csv` | one polar row at the requested angle | one operating row in the QPROP table | see [FlightStream](flightstream-workspaces.md) |
| Coefficients | `CL`, `CD`, `CSF`, `CFx`..`CFz`, `CMx`..`CMz` | `Cd`, `Cl`, `CmPitch` | `Cl`, `Cd` | `CL`, `CD`, `CDp`, `CM` | `J`, `CT`, `CP`, `CT_QPROP`, `CP_QPROP`, `EFFICIENCY`, `EFFICIENCY_OVERALL`, `T`, `Q`, `P` | see [FlightStream](flightstream-workspaces.md) |
| Neutral `CL`, `CD`, `CM` in the tables | `CL`, `CD`, `CMy` (`CMz` in a two-dimensional case) | `Cl`, `Cd`, `CmPitch` | `Cl`, `Cd`, none | `CL`, `CD`, `CM` | none | see [Results and tables](results.md) |
| Parallel option | `mpi_launcher` (executable route) | `processes`, `mpi_launcher` | none | none | none | see [FlightStream](flightstream-workspaces.md) |

## Convergence control of the RANS solvers

The two solvers that solve the Reynolds-averaged equations, SU2 and OpenFOAM, start from a robust convergence control when the setup states none, so a poor initial field does not diverge. Each uses the control of its own solver, so the controls differ: SU2 accepts the neutral keys `cfl`, `cfl_max` and `cfl_growth` as well as native keys. A CFL number and OpenFOAM under-relaxation factors or local Courant limit do not map onto each other.

| Solver | Default | Other route | Keys |
|---|---|---|---|
| SU2 | CFL ramp from 5 to 25 | any CFL key of yours replaces it | see [CFL ramp](su2-workspaces.md#cfl-ramp) |
| OpenFOAM | steady `simpleFoam` with under-relaxation p 0.3, U 0.7, nuTilda 0.7 | pseudo-transient local time stepping with `time_integration = "lts"` | see [Convergence control](openfoam-workspaces.md#convergence-control) |

In both, a value you state natively wins entirely: cfdbender writes what you stated and merges nothing from its default. PyFR (not RANS), XFOIL and QPROP have no such control.
