# PyFR workspaces

PyFR 3.1 is a high-order flux reconstruction solver for unstructured meshes on
GPUs and CPUs. cfdbender plans one two-dimensional compressible
Navier-Stokes case per matrix row, runs `pyfr import` then `pyfr run`, and
collects time-averaged lift and drag from the fluid-force plugin.

Initialize with `cfd-workspace init --solver pyfr`. Then plan, run and collect
with `cfd-matrix plan study`, `cfd-matrix run study` and
`cfd-matrix collect study`.

## Install in WSL

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
in WSL, then build a user-space environment:

```bash
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
micromamba create -n cfdb-pyfr -c conda-forge python=3.12 mpi4py openmpi h5py cuda-nvrtc-dev libcublas-dev
micromamba run -n cfdb-pyfr pip install pyfr
```

The CUDA backend needs the Windows GPU driver, which WSL exposes in
`/usr/lib/wsl/lib`. Run PyFR with
`LD_LIBRARY_PATH=/usr/lib/wsl/lib:$MAMBA_ROOT_PREFIX/envs/cfdb-pyfr/lib`. Put a
small wrapper that sets this variable and calls `micromamba run -n cfdb-pyfr
pyfr "$@"` on the WSL path, and name it in the setup file as the executable.
The openmp backend needs no GPU.

## Setup file

A setup file under `inputs/setups` names the case:

```toml
mesh = "inputs/meshes/naca0012.msh"
mach = 0.2
reynolds = 10000
order = 3
tend = 30
window_start = 20
backend = "cuda"          # or "openmp"
tolerance = 0.01          # optional, default 0.01

[solver]
launcher = ["wsl", "-d", "Ubuntu", "--"]
executable = "/home/user/bin/pyfr-run"
timeout_s = 7200
```

`mesh` is a gmsh `.msh` file in `inputs/meshes` with the physical groups
`fluid`, `wall` and `far`. A mesh that mixes quadrilaterals and triangles is
supported: both element sections are written. `mach`, `reynolds`, `alpha`,
`order`, `tend`, `window_start` and `tolerance` may also be matrix
conditions; `alpha` is normally the matrix condition. Reference files stay
empty and the post file reads `history = true`, because the nondimensional
state is fixed: density 1, freestream speed 1, chord 1 and dynamic pressure
0.5.

The planner derives the freestream from the row: viscosity is `1/Reynolds`,
pressure is `1/(1.4 Mach^2)`, and the velocity components are the cosine and
sine of the angle of attack. Planning is read-only. Translating the same row
twice returns byte-identical `case.ini` text. Execution stages the mesh as
`mesh.msh`, writes `case.ini`, runs import and then the solver in the point
directory, and keeps `stdout.log`, `stderr.log`, `execution.json` and
`forces.csv`.

## Worked example: NACA 0012

For a NACA 0012 section at Mach 0.2 and Reynolds 10000 per chord, generate a
gmsh mesh with a far field of about 30 chords and the three physical groups
above. The matrix row

```text
POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION
---|---|---|---|-----|----------------
1001|1|001|001|001|ALPHA:4
```

plans one case at four degrees. After the run, `collect` reports `Cl` and
`Cd` averaged over the window from `window_start` to `tend`, with chord 1,
dynamic pressure 0.5 and the window recorded in the result.

Forces come from the plugin columns `px`, `py`, `vx` and `vy` (pressure and
viscous parts along the x and y axes). With the angle of attack alpha,

- `Cl = 2 (Fy cos(alpha) - Fx sin(alpha))`,
- `Cd = 2 (Fx cos(alpha) + Fy sin(alpha))`,

where `F` is the pressure plus viscous force. The plugin gives no moment, so
the pitching moment is reported as not available.

## What converged means

Unsteady flow has no residual to drive to zero, so convergence is stationarity
of the means. The samples inside the window are split into two halves.
The run is `converged` when the means of the halves differ by less than
`tolerance` times the absolute value of the window mean, for both `Cl` and
`Cd` (default one percent).

- `iteration_limit`: the window holds too few samples to form two halves, or
  the halves disagree. Extend `tend` or move `window_start` later.
- `failed`: a nonzero exit code, a timeout, a missing or malformed force
  file, or a non-finite value.

## Offline testing

`executable = "stand_in.py"` with no launcher runs a Python stand-in in place
of PyFR, which the test suite uses to write a synthetic `forces.csv`. A real
run on a GPU is the only evidence of physical accuracy for a given mesh.
