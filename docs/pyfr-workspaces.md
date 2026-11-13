# PyFR

You run two-dimensional compressible Navier-Stokes cases with PyFR 3.1 through cfdbender, and collect time-averaged lift and drag from the PyFR fluid-force plugin. This page lists only what is specific to PyFR.

## Install

PyFR is a high-order flux reconstruction solver for unstructured meshes, on GPUs or CPUs. In WSL, install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), then build a user-space environment:

```bash
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
micromamba create -n cfdb-pyfr -c conda-forge python=3.12 mpi4py openmpi h5py cuda-nvrtc-dev libcublas-dev
micromamba run -n cfdb-pyfr pip install pyfr
```

The CUDA backend needs the Windows GPU driver, which WSL exposes in `/usr/lib/wsl/lib`. Run PyFR with `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$MAMBA_ROOT_PREFIX/envs/cfdb-pyfr/lib`. Put a small wrapper that sets this variable and calls `micromamba run -n cfdb-pyfr pyfr "$@"` on the WSL path, and name it as `executable` in the setup file. The openmp backend needs no GPU.

| What | Setup key (`[solver]`) | Default |
|---|---|---|
| PyFR program or wrapper | `executable` | `pyfr` |
| Command that starts it | `launcher` | `[]` |

PyFR has no environment variable. With no `launcher`, the executable name is looked up on the PATH by the operating system. With a `launcher`, the executable path is passed unchanged to the guest system, and `~` is not expanded.

A matrix row can also pick a build by name; see [Executables](executables.md).

## Mesh

PyFR reads a gmsh `.msh` file. Put it in `inputs/mesh/` and name it in the matrix `GEOMETRY` column or in the setup key `mesh`. See [Workspace layout](workspace-layout.md) for how a mesh is linked. The point folder holds the link as `mesh.msh`; PyFR imports it to `mesh.pyfrm`. A file that does not start with `$MeshFormat` is refused: `mesh must be gmsh format`.

The mesh is two-dimensional, with the physical groups `fluid`, `wall` and `far`. A mesh that mixes quadrilaterals and triangles is supported: both element types are written.

## Setup file

`inputs/setups/s001.toml` names the case. PyFR runs in a fixed nondimensional state: density 1, free stream speed 1, chord 1 and dynamic pressure 0.5. Lengths and times are in units of the chord and of chord over free stream speed.

| Key | Meaning | Unit | Default |
|---|---|---|---|
| `mesh` | gmsh file under `inputs/mesh/` | - | required (or `GEOMETRY`) |
| `mach` | free stream Mach number, above 0 | - | required (or from the row) |
| `reynolds` | Reynolds number over the chord, above 0 | - | required (or from the row) |
| `alpha` | angle of attack | degrees | 0 |
| `order` | polynomial order, integer from 1 | - | required |
| `tend` | end of the run | chord / speed | required |
| `window_start` | start of the averaging window, from 0 and below `tend` | chord / speed | required |
| `backend` | `"cuda"` or `"openmp"` | - | `"cuda"` |
| `tolerance` | stationarity tolerance, see below | fraction | 0.01 |

`mach`, `reynolds`, `alpha`, `order`, `tend`, `window_start` and `tolerance` may also be matrix conditions; `alpha` is normally the matrix condition. A key not in this table is refused: `unsupported PyFR setup: [...]`.

**`[solver]` table**

| Key | Meaning | Unit | Default |
|---|---|---|---|
| `executable` | PyFR program or wrapper | - | `pyfr` |
| `launcher` | tokens placed before it, for example `["wsl", "-d", "Ubuntu", "--"]` | - | `[]` |
| `timeout_s` | time after which cfdbender stops the run, import and solve together | s | 600 |

**Reference file** `inputs/references/r001.toml` stays empty. A reference file with keys is refused: `PyFR uses unit references and requires history=true`.

**Pproc file** `inputs/pproc/p001.toml` holds `history = true` and nothing else.

**Flight condition.** cfdbender derives the free stream from the row: the viscosity is `1/Reynolds`, the pressure is `1/(1.4 Mach^2)`, and the velocity components are the cosine and sine of the angle of attack. Translating the same row twice gives byte-identical `case.ini` text.

| Matrix | PyFR key |
|---|---|
| `MACH`, `REmi` | `mach`, `reynolds` (`REmi` x 1e6) |
| `ALPHA` | `alpha` |
| `GEOMETRY` | `mesh` |

`REmi` gives the Reynolds number, since the reference file is empty. A row that states only `MACH` and angles replaces the setup's `mach`. A row can also carry `ORDER`, `TEND`, `WINDOW_START` and `TOLERANCE`; any other key is refused as `unsupported PyFR condition: <key>`. `BETA` must be 0: `a two-dimensional case flies at BETA 0; remove BETA from the row`. A setup value that disagrees with the row is refused: `conflict: <key> is <value> in the inputs and <value> from the matrix row; state it in one place`. See [Matrix columns](matrix-columns.md).

cfdbender runs `pyfr import mesh.msh mesh.pyfrm`, then `pyfr -p run -b <backend> mesh.pyfrm case.ini`, in the point folder. The folder keeps `stdout.log`, `stderr.log`, `execution.json` and `forces.csv`. For a scheduler job, see [Cluster jobs](cluster-jobs.md).

## What converged means

Unsteady flow has no residual to drive to zero, so convergence is stationarity of the means. cfdbender reads the samples of `forces.csv` (columns `t`, `px`, `py`, `vx`, `vy`: pressure and viscous parts along x and y) between `window_start` and `tend`, splits them into two halves, and compares the two means.

| Result | Rule |
|---|---|
| `converged` | The exit code is 0 and the means of the two halves differ by less than `tolerance` times the absolute window mean, for both `Cl` and `Cd`. |
| `iteration_limit` | The window holds fewer than two samples, or the halves disagree. Extend `tend` or move `window_start` later. |
| `failed` | A nonzero exit code, a timeout, a missing or malformed `forces.csv`, a non-finite value, or time stamps that do not increase. |

The averages are the plain means over the window. With the angle of attack alpha and `F` the pressure plus viscous force:

- `Cl = 2 (Fy cos(alpha) - Fx sin(alpha))`, axis y.
- `Cd = 2 (Fx cos(alpha) + Fy sin(alpha))`, axis x.

The factor 2 is the inverse of the dynamic pressure 0.5, with chord 1. The plugin gives no moment, so the pitching moment is reported as `not available`. The result records the averaging window, the dynamic pressure 0.5 and the PyFR version.

## Worked example

A NACA 0012 section at Mach 0.2 and Reynolds 10000 per chord, at 4 degrees. Generate a gmsh mesh with a far field of about 30 chords and the three physical groups above, and save it as `naca0012.msh`.

```console
cfd-workspace init campaign --solver pyfr
cd campaign
```

Copy the mesh to `inputs/mesh/naca0012.msh`. `inputs/setups/s001.toml`:

```toml
mesh = "inputs/mesh/naca0012.msh"
mach = 0.2
reynolds = 10000
order = 3
tend = 30
window_start = 20
backend = "cuda"

[solver]
launcher = ["wsl", "-d", "Ubuntu", "--"]
executable = "/home/your-user/bin/pyfr-run"
timeout_s = 7200
```

`inputs/references/r001.toml` is an empty file. `inputs/pproc/p001.toml`:

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

`collect` reports `Cl` and `Cd` averaged over the window from `window_start` to `tend`.
