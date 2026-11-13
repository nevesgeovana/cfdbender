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
| Coefficients | `CL`, `CD`, `CSF`, `CFx`..`CFz`, `CMx`..`CMz` | `Cd`, `Cl`, `CmPitch` | `Cl`, `Cd` | `CL`, `CD`, `CDp`, `CM` | `J`, `CT`, `CP`, `CT_QPROP`, `CP_QPROP`, `EFFICIENCY`, `T`, `Q`, `P` | see [FlightStream](flightstream-workspaces.md) |
| Parallel option | `mpi_launcher` (executable route) | `processes`, `mpi_launcher` | none | none | none | see [FlightStream](flightstream-workspaces.md) |
