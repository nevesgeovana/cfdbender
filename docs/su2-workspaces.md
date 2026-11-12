# SU2 workspaces

SU2 accepts tracked workspace meshes in native `.su2` or CGNS `.cgns` format.
Each point stages `mesh.su2` with `MESH_FORMAT= SU2`, or `mesh.cgns` with
`MESH_FORMAT= CGNS`. Both formats are included in `inputs_sha256` and checked
before execution. Install SU2 with CGNS support to run CGNS inputs.

The mesh is never copied into a point folder. Staging creates a relative
symbolic link, so the link still resolves after the workspace moves to a
cluster; on a Windows host without the symbolic-link privilege it creates a hard
link instead (the same file on disk). It refuses if neither link works. Hard
links require the source and point directory to share a volume. Keep source
meshes immutable while planned points or linked results use them. The setup key
`mesh_link = true` is still accepted and changes nothing; `mesh_link = false` is
refused. Input validation and planning still read the mesh for provenance, so
allow memory for input hashing.

For example, `inputs/setups/s001.toml` can contain:

```toml
mesh = "inputs/meshes/transport.cgns"
iterations = 1000
mach = 0.85
alpha = 2.0
SOLVER = "EULER"
MARKER_EULER = ["wing", "fuselage", "windscreen"]
MARKER_SYM = ["symmetry"]
MARKER_FAR = ["farfield-xmax", "farfield-xmin", "farfield-ymax", "farfield-zmax", "farfield-zmin"]

[solver]
route = "executable"
executable = "/opt/su2/bin/SU2_CFD"
```

Marker names are supplied by the s file, not inferred from CGNS. Match the
names and boundary conditions to the grid. Supply reference `area`, `length`,
and `moment_origin` in the r file and `history = true` in the p file, then use
`cfd-matrix plan study`, `cfd-matrix run study`, and `cfd-matrix collect study`.

## A matrix that states the free stream and the mesh

The matrix can carry the flight condition, the mesh, the processor count and the
SU2 build, so the setup holds only the numerics. See
[matrix columns](matrix-columns.md) for every column.

`inputs/matrices/dpw.fs`:

```text
POL  | HIDDEN | RUN | AIRCRAFT | CONFIGURATION | DESCRIPTION       | FLIGHT_CONDITION                                   | SWEEP_VALUES | GEOMETRY | REF | SET | PPROC | NCPUS | WALLTIME | SOLVER_BUILD | VAR_NAMES_VALUES
-----|--------|-----|----------|---------------|-------------------|----------------------------------------------------|--------------|----------|-----|-----|-------|-------|----------|--------------|-----------------
5001 |   0    |  1  | WB       | cruise        | Polar near design | MACH:0.85, REmi:5, TK:310.928, ALPHA:sweep, BETA:0 | 2.5,2.75     | wb.su2   | 001 | 001 | 001   | 192   | 4h       | 8.3.0        | -
5002 |   0    |  1  | WB       | cruise        | Longer run        | MACH:0.85, REmi:5, TK:310.928, ALPHA:sweep, BETA:0 | 3.0          | wb.su2   | 001 | 001 | 001   | 192   | 6h       | 8.3.0        | iterations: 4000
```

`inputs/setups/s001.toml` then names no mesh and no free stream:

```toml
iterations = 2000
SOLVER = "RANS"
KIND_TURB_MODEL = "SA"
MARKER_HEATFLUX = ["wing", "fuselage"]
MARKER_FAR = ["farfield"]
```

and `inputs/references/r001.toml` gives `area`, `length` and `moment_origin`;
`length` is the length REmi is measured against. The mesh is
`inputs/geometries/wb.su2`. For each point the deck receives:

| Matrix | SU2 key |
|---|---|
| resolved Mach number | `MACH_NUMBER` |
| `ALPHA`, `BETA` | `AOA`, `SIDESLIP_ANGLE` |
| resolved temperature, pressure, density | `FREESTREAM_TEMPERATURE`, `FREESTREAM_PRESSURE`, `FREESTREAM_DENSITY` |
| `REmi` x 1e6 and the r file's `length` | `REYNOLDS_NUMBER`, `REYNOLDS_LENGTH` |
| `MUPas` | `VISCOSITY_MODEL= CONSTANT_VISCOSITY`, `MU_CONSTANT` |
| `GEOMETRY` | `MESH_FILENAME` (staged as `mesh.su2` or `mesh.cgns`) |

When the air is stated without `REmi`, the Reynolds number written is the
outcome of the resolved density, velocity and viscosity over the r file's
`length`. `ASMPS` is refused unless it equals the speed of sound SU2 derives from
the temperature. If the setup also states one of these keys, the two values must
agree. A row stating only `MACH` and angles keeps the setup's flow state.

With an HPC profile in `inputs/hpc/`, `NCPUS` fills `{ncpus}`, `WALLTIME` fills
`{walltime}`, `SOLVER_BUILD` selects the `[builds]` entry for
`{solver_version_alias}`, and `{solver_type}` is `SU2`:

```toml
application_id = "SU2"
max_walltime = "96:00:00"

[descriptor]
format = "yaml"
name = "submit.yaml"

[descriptor.fields]
job_name = "{job_name}"
walltime = "{walltime}"
ncpus = "{ncpus}"
version = "{solver_version_alias}"
working_directory = "{workdir}"
config_file = "{config_path}"
solver_type = "{solver_type}"

[submit]
command = ["submit-tool", "{descriptor_path}"]

[builds]
"8.3.0" = "8.3.0"
```

`cfd-matrix plan dpw` prints each point, its descriptor and the submit command
without writing anything; `cfd-matrix run dpw` stages `su2.cfg` and the linked
mesh in each point folder and submits. `{config_path}` is that `su2.cfg`. Send
the job's standard output to `{stdout_log}`, since convergence is read from the
SU2 log; see [point folders of cluster jobs](workspace-files.md#point-folders-of-cluster-jobs).

## MPI executable launch

For a native MPI-enabled SU2 executable, add `mpi_launcher` in the s file's
`[solver]` table:

```toml
[solver]
route = "executable"
executable = "/opt/su2/bin/SU2_CFD"
mpi_launcher = ["mpirun", "-np", "192"]
threads = 1
```

This launches `mpirun -np 192 /opt/su2/bin/SU2_CFD su2.cfg` in the point
directory. Under an existing scheduler allocation, `mpi_launcher = ["srun"]`
uses the allocation's task count. Tokens are passed as arguments without shell
expansion. The default empty list retains serial execution. This option applies
only to the executable route; the Python wrapper route remains serial. Configure
the allocation and MPI environment separately before starting the worker.
