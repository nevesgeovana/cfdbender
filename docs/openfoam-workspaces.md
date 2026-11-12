# OpenFOAM workspaces

cfdbender runs ESI OpenFOAM `simpleFoam` cases on a tracked 2D extruded
gmsh mesh, then collects `forceCoeffs` and SIMPLE evidence. Install the solver
separately in WSL and install cfdbender and gmsh in the Windows Python environment.

## Install in WSL

Install micromamba using its official instructions, then create a user-space
environment with `micromamba create -n foam -c conda-forge openfoam`. Find the
installed ESI `etc/bashrc` and use its absolute Linux path below. The launcher
uses the Windows point directory mapped by WSL. Use an ESI distribution; the
adapter does not target the Foundation lineage.

## NACA 0012 at alpha 4

From a cfdbender source checkout, initialize a workspace and generate the kit
mesh. The destination parent must exist and the Python environment needs gmsh.

```console
cfd-workspace init --solver openfoam --workspace campaign
mkdir campaign/inputs/meshes -Force
python tools/kit/openfoam_naca0012_mesh.py campaign/inputs/meshes/naca0012.msh
```

`campaign/inputs/setups/s001.toml`:

```toml
mesh = "inputs/meshes/naca0012.msh"
application = "simpleFoam"
speed = 10.0
nu = 1.5e-5
turbulence = "SpalartAllmaras"
iterations = 500
residual_target = 1e-5

[solver]
launcher = ["wsl", "-d", "Ubuntu", "--"]
foam_bashrc = "/home/your-user/path/to/OpenFOAM/etc/bashrc"
```

`campaign/inputs/references/r001.toml`:

```toml
area = 0.1
length = 1.0
moment_origin = [0.25, 0.0, 0.0]
```

`campaign/inputs/pproc/p001.toml`:

```toml
history = true
```

`campaign/inputs/matrices/study.fs`:

```text
POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES
---|---|---|---|-----|----------------|------------
1001|1|001|001|001|alpha:4|
```

```console
cfd-matrix plan study --workspace campaign --local
cfd-matrix run study --workspace campaign --local
cfd-matrix collect study --workspace campaign
```

Collection writes `campaign/post/study/campaign_summary.json`. Planning reads
and claims outputs; execution writes the translated case and stages the mesh.

## Inputs and interpretation

The s file requires `mesh`, `application`, `speed`, `nu`, `turbulence` and
`iterations`; `residual_target` defaults to 1e-5. Turbulence is `laminar` or
`SpalartAllmaras`. The `[solver]` table accepts `foam_bashrc`, `launcher`,
`timeout_s`, `threads` (1), and `executable` for an offline stand-in.
`processes` defaults to 1; `mpi_launcher` defaults to `["mpirun"]`.
`CFDBENDER_OPENFOAM_BASHRC` supplies the bashrc when omitted. The r file
requires `area`, `length`, and three-coordinate `moment_origin`; the p file
requires `history = true`. Alpha sweeps use `alpha:sweep` and comma-separated
values. Native `.msh` import requires gmsh 2.2 ASCII with the expected named patches.

Convergence requires a successful process, valid force output matching the
final SIMPLE iteration, and either the matching SIMPLE convergence message or
all required initial residuals below the target. A valid nonconverged run at
the iteration limit is `iteration_limit`; missing, inconsistent, fatal or failed
evidence is `failed`. A successful exit alone is insufficient.

Cd follows the XY wind drag direction `(cos(alpha), sin(alpha), 0)` and Cl
the lift direction `(-sin(alpha), cos(alpha), 0)`. ESI derives pitch as
lift cross drag, body minus z. CmPitch retains the solver's value and sign;
do not interpret it as a body plus z moment without changing sign. The kit
mesh is a repeatable example, not a mesh-independence or accuracy claim.
Only steady `simpleFoam` is supported; assess physical convergence and
mesh adequacy for each application.

## Parallel execution on Linux

Use multiple processes for a decomposed case. For example, add these options
to the setup's `[solver]` table on a native Linux installation:

```toml
[solver]
foam_bashrc = "/opt/OpenFOAM/etc/bashrc"
processes = 12
mpi_launcher = ["mpirun"]
```

Omit `launcher` on native Linux. `launcher` wraps `bash Allrun`, while
`mpi_launcher` is a nonempty list of MPI command tokens inside Allrun. Tokens
are shell quoted individually. With `processes > 1`, the case gains a scotch
`system/decomposeParDict` with the requested `numberOfSubdomains`. After mesh
import and patch correction, Allrun executes:

```console
decomposePar
mpirun -np 12 simpleFoam -parallel
reconstructPar -latestTime
```

Each step has its own `log.*` file and a failed step stops Allrun. ESI writes
the reduced force coefficient history on the master rank in the undecomposed
case's `postProcessing/forceCoeffs/0/coefficient.dat`; reconstruction restores
the final fields and leaves that history in place. Collection reads the same
path as serial runs. This follows ESI's
[writeFile case-root and master-rank handling](https://api.openfoam.com/2506/writeFile_8C_source.html).
The default `processes = 1` keeps the serial script without decomposition or
reconstruction. Install a compatible MPI runtime and scotch support separately.

For SU2 mesh and launch options, see [SU2 workspaces](su2-workspaces.md).

## User-supplied mesh conversion

ESI has no native CGNS or UGRID import route in this adapter. For `.cgns` and
`.ugrid` meshes, supply `mesh_converter` in the s file, before `[solver]`:

```toml
mesh = "inputs/meshes/airfoil.cgns"
mesh_converter = "convert-to-fluent {mesh} {case}/converted.msh && fluentMeshToFoam {case}/converted.msh -case {case}"
application = "simpleFoam"
speed = 10.0
nu = 1.5e-5
turbulence = "SpalartAllmaras"
iterations = 500
```

`convert-to-fluent` is an illustrative user-supplied command, not an installed
cfdbender tool. Install and validate a converter for your grid and ESI version.
The command must create `constant/polyMesh` in the point case, with `airfoil`,
`farfield`, `front`, and `back` patches matching the adapter's extruded XY case.
This hook extends import formats, not the adapter's geometry or boundary model.

The source mesh remains tracked by `inputs_sha256` and is staged as `mesh.cgns`
or `mesh.ugrid`. Allrun executes the trusted shell template inside the point
directory before patch correction. `{mesh}` expands to the shell-quoted staged
filename and `{case}` to the shell-quoted case directory `.`. Do not add your
own quotes around placeholders. Only those two placeholders are accepted; use
`{{` and `}}` for literal braces. Output goes to `log.meshConverter`, and a
nonzero converter exit stops execution. No CGNS reader is included. Omitting
the converter for either format refuses during planning. Native `.msh` import
continues to use `gmshToFoam` by default.
