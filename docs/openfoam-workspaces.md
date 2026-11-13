# OpenFOAM

You run steady ESI OpenFOAM `simpleFoam` cases on a two-dimensional extruded mesh through cfdbender, and collect `forceCoeffs` and SIMPLE convergence evidence. This page lists only what is specific to OpenFOAM.

## Install

Use an ESI distribution of OpenFOAM; the Foundation lineage is not supported. In WSL, install micromamba with its official instructions, then create a user-space environment:

```bash
micromamba create -n foam -c conda-forge openfoam
```

Find the installed ESI `etc/bashrc` and use its absolute Linux path as `foam_bashrc` below. The launcher runs in the Windows point folder, which WSL maps. On native Linux, omit `launcher`.

| What | Setup key (`[solver]`) | Environment variable |
|---|---|---|
| ESI `etc/bashrc` | `foam_bashrc` | `CFDBENDER_OPENFOAM_BASHRC` |
| Command that starts the shell | `launcher` | none |

The key wins over the variable. cfdbender does not search the PATH for OpenFOAM: the generated `Allrun` script sources the bashrc, and the ESI commands run from there. If neither is set, `Allrun` sources no file.

A matrix row can also pick a build by name; see [Executables](executables.md).

## Mesh

OpenFOAM reads a gmsh `.msh` mesh natively, in format 2.2 ASCII. Any other gmsh format is refused: `mesh must be gmsh 2.2 ASCII`. Put the file in `inputs/mesh/` and name it in the matrix `GEOMETRY` column or in the setup key `mesh`. See [Workspace layout](workspace-layout.md) for how a mesh is linked. The point folder holds the link as `mesh.msh`, `mesh.cgns` or `mesh.ugrid`.

The mesh is one extruded layer in the XY plane with four named patches: `airfoil`, `farfield`, `front` and `back`. After import, cfdbender sets `front` and `back` to type `empty` and `airfoil` to type `wall`.

**Other formats.** OpenFOAM has no CGNS or UGRID import in cfdbender. For a `.cgns` or `.ugrid` mesh, give `mesh_converter` in the setup file, above the `[solver]` table. Without it, planning refuses: `mesh_converter must be a nonempty ASCII command template`.

```toml
mesh = "inputs/mesh/airfoil.cgns"
mesh_converter = "convert-to-fluent {mesh} {case}/converted.msh && fluentMeshToFoam {case}/converted.msh -case {case}"
application = "simpleFoam"
speed = 10.0
nu = 1.5e-5
turbulence = "SpalartAllmaras"
iterations = 500
```

`convert-to-fluent` stands for a converter you supply; cfdbender installs none. Install a converter that suits your grid and your ESI version. The command must create `constant/polyMesh` in the point case, with the four patches above.

`Allrun` runs the template inside the point folder, before the patch fixes. `{mesh}` becomes the quoted staged file name and `{case}` becomes the quoted case folder `.`. Do not add your own quotes around them. No other placeholder is accepted; write `{{` and `}}` for literal braces, or planning refuses with `mesh_converter supports only {mesh} and {case} placeholders`. The output goes to `log.meshConverter`, and a nonzero exit stops the run.

## Setup file

`inputs/setups/s001.toml`:

| Key | Meaning | Unit | Default |
|---|---|---|---|
| `mesh` | mesh file under `inputs/mesh/`, `.msh`, `.cgns` or `.ugrid` | - | required (or `GEOMETRY`) |
| `mesh_converter` | shell template for `.cgns` and `.ugrid`, see Mesh | - | none |
| `application` | solver; only `simpleFoam` is accepted | - | required |
| `speed` | free stream speed | m/s | required (or from the row) |
| `nu` | kinematic viscosity | m^2/s | required (or from the row) |
| `turbulence` | `"laminar"` or `"SpalartAllmaras"` | - | required |
| `iterations` | SIMPLE iterations (`endTime`), integer above 0 | - | required |
| `residual_target` | initial residual that counts as converged | - | 1e-5 |

Any other key is refused as `<file>: unknown key '<key>' (closest known key: <name>); known keys: ...`, naming the file.

**`[solver]` table**

| Key | Meaning | Unit | Default |
|---|---|---|---|
| `foam_bashrc` | Linux path of the ESI `etc/bashrc` | - | `CFDBENDER_OPENFOAM_BASHRC` |
| `launcher` | tokens placed before `bash Allrun`, for example `["wsl", "-d", "Ubuntu", "--"]` | - | `[]` |
| `timeout_s` | time after which cfdbender stops the run | s | 600 |
| `threads` | must be 1 | - | 1 |
| `processes` | MPI processes; above 1 decomposes the case | - | 1 |
| `mpi_launcher` | nonempty list of MPI command tokens | - | `["mpirun"]` |

The same `[solver]` options can be set for the whole workspace in `init.toml`; the setup file overrides them.

**Parallel run on Linux.** Add the parallel keys to `[solver]`:

```toml
[solver]
foam_bashrc = "/opt/OpenFOAM/etc/bashrc"
processes = 12
mpi_launcher = ["mpirun"]
```

`launcher` wraps `bash Allrun`. `mpi_launcher` is the list of MPI tokens used inside `Allrun`, each quoted on its own. With `processes` above 1, the case gains a scotch `system/decomposeParDict` with `numberOfSubdomains` set to `processes`, and `Allrun` runs:

```console
decomposePar
mpirun -np 12 simpleFoam -parallel
reconstructPar -latestTime
```

Each step writes its own `log.*` file, and a failed step stops `Allrun`. ESI writes the reduced coefficient history on the master rank in `postProcessing/forceCoeffs/0/coefficient.dat` of the undecomposed case, so cfdbender reads the same path as for a serial run, as described in ESI's [writeFile handling](https://api.openfoam.com/2506/writeFile_8C_source.html). With `processes = 1` there is no decomposition or reconstruction. Install a compatible MPI runtime and scotch support yourself. For a scheduler job, see [Cluster jobs](cluster-jobs.md).

**Reference file** `inputs/references/r001.toml` must hold `area` (m^2), `length` (m) and `moment_origin` (three coordinates, m). They become `Aref`, `lRef` and `CofR` of `forceCoeffs`. Other keys are refused.

**Pproc file** `inputs/pproc/p001.toml` must hold `history = true`; otherwise `post history must be true`.

**Flight condition.** The row carries `ALPHA` in degrees, or an alpha sweep. Any other condition key is refused as `FLIGHT_CONDITION: unknown key '<key>' (closest known key: <name>); known keys: ...`. The case flies at `BETA` 0: `the OpenFOAM case flies at BETA 0; remove BETA from the row`. A row that states the air (`TASmps` or `MACH`, with the other keys of [Matrix columns](matrix-columns.md)) replaces `speed` and `nu`:

| Matrix | OpenFOAM key |
|---|---|
| resolved velocity | `speed` |
| resolved viscosity over density | `nu` |
| `REmi` x 1e6, solved over the reference file's `length` | density, hence `nu` |
| `GEOMETRY` | `mesh` |

A bare `MACH` resolves at sea level. A setup value that disagrees with the row is refused: `conflict: <key> is <value> in the inputs and <value> from the matrix row; state it in one place`.

## What converged means

cfdbender reads `postProcessing/forceCoeffs/0/coefficient.dat` and the logs (`log.simpleFoam` and the other `log.*` files, `stdout.log`, `stderr.log`).

| Result | Rule |
|---|---|
| `converged` | The exit code is 0, the force file is valid and ends at the same iteration as `log.simpleFoam`, and either the log says `SIMPLE solution converged in <N> iterations` with N equal to that last iteration, or the initial residuals of that iteration are all below `residual_target`. For the residual route, `Ux`, `Uy` and `p` must be present, and `nuTilda` with `SpalartAllmaras`. |
| `iteration_limit` | The evidence is valid, the run is not converged, and the last iteration reached `iterations`. |
| `failed` | A nonzero exit code, a `FOAM FATAL ERROR` in a log, a missing, malformed or non-finite force file, force file and log that end at different iterations, or a run that stops before `iterations` without converging. |

A clean exit alone is not enough. The residual of an iteration is the largest initial residual of any solve in it.

| Name | Meaning | Axis |
|---|---|---|
| `Cd` | drag coefficient, along `(cos(alpha), sin(alpha), 0)` | x |
| `Cl` | lift coefficient, along `(-sin(alpha), cos(alpha), 0)` | y |
| `CmPitch` | pitching moment coefficient | z |

cfdbender reports the last row of the force file. `forceCoeffs` normalises with the free stream speed, `area` and `length`, and a density of 1. ESI derives the pitch axis as lift cross drag, which is body minus z. `CmPitch` keeps the solver's value and sign, so do not read it as a body plus z moment without changing its sign.

## Worked example

A NACA 0012 at 4 degrees, 10 m/s, with Spalart-Allmaras turbulence. In a cfdbender source checkout, create the workspace and generate the mesh. The generator needs gmsh in your Python environment.

```console
cfd-workspace init campaign --solver openfoam
python tools/kit/openfoam_naca0012_mesh.py campaign/inputs/mesh/naca0012.msh
cd campaign
```

`inputs/setups/s001.toml`:

```toml
mesh = "inputs/mesh/naca0012.msh"
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

`inputs/references/r001.toml`:

```toml
area = 0.1
length = 1.0
moment_origin = [0.25, 0.0, 0.0]
```

`inputs/pproc/p001.toml`:

```toml
history = true
```

`inputs/matrices/study.fs`:

```text
POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES
---|---|---|---|-----|----------------|------------
1001|1|001|001|001|ALPHA:4|
```

```console
cfd-matrix plan study
cfd-matrix run study
cfd-matrix collect study
```
