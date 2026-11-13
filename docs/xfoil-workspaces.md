# XFOIL

You compute two-dimensional airfoil polars with XFOIL through cfdbender. It plans one independent headless XFOIL run per angle of attack and collects the printed coefficients. This page lists only what is specific to XFOIL.

## Install

Install cfdbender on Windows and build XFOIL in WSL. Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) in WSL, download and unpack the [MIT XFOIL source](https://web.mit.edu/drela/Public/web/xfoil/) to `~/src/Xfoil`, then build it in a user-space environment:

```bash
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
micromamba create -n cfdb-xfoil -c conda-forge gfortran_linux-64 gcc_linux-64 make xorg-libx11 xorg-xorgproto
E="$MAMBA_ROOT_PREFIX/envs/cfdb-xfoil"
export PATH="$E/bin:$PATH"
FC=$(ls "$E"/bin/*-gfortran | head -1)
CC=$(ls "$E"/bin/*-gcc | head -1)
cd "$HOME/src/Xfoil/plotlib"
cp config.make.gfortranDP config.make
sed -i "s#^FC = .*#FC = $FC#; s#^CC *= .*#CC = $CC#" config.make
make libPlt_gDP.a CFLAGS="-O2 -I$E/include"
cd "$HOME/src/Xfoil/bin"
cp Makefile_gfortran Makefile
sed -i "s#^FC = .*#FC = $FC#; s#^CC = .*#CC = $CC#" Makefile
sed -i "s#^PLTOBJ = .*#PLTOBJ = ../plotlib/libPlt_gDP.a#" Makefile
sed -i "s#^PLTLIB = .*#PLTLIB = -L$E/lib -lX11 -Wl,-rpath,$E/lib#" Makefile
sed -i 's/^CHK = $/CHK = -fallow-argument-mismatch -std=legacy/' Makefile
make xfoil
```

Change `MAMBA_ROOT_PREFIX` if your micromamba root is elsewhere. The executable is `~/src/Xfoil/bin/xfoil`.

cfdbender finds XFOIL in this order:

| Order | Source |
|---|---|
| 1 | `executable` in the `[solver]` table of the setup file |
| 2 | the environment variable `CFDBENDER_XFOIL_EXE` |
| 3 | `xfoil` on the PATH |

With a `launcher` (for example `["wsl", "-d", "Ubuntu", "--"]`), cfdbender passes the executable path unchanged to the guest system and does not search the host. Give an absolute Linux path; `~` is not expanded. Without a launcher, a missing executable refuses: `XFOIL executable not found; set solver.executable or CFDBENDER_XFOIL_EXE`.

A matrix row can also pick a build by name; see [Executables](executables.md).

## Mesh

XFOIL needs no mesh. You give the airfoil in one of two ways, and exactly one:

| Setup key | Airfoil |
|---|---|
| `airfoil` | a NACA four- or five-digit designation, for example `"NACA 0012"` |
| `airfoil_file` | a coordinate file `<name>.dat` in `inputs/mesh/` (or `inputs/airfoils/`) |

The matrix `GEOMETRY` column names the coordinate file and becomes `airfoil_file`; the setup then states no `airfoil`. Stating both, or neither, is refused: `specify exactly one of airfoil and airfoil_file`. A designation that is not `NACA` plus four or five digits is refused: `airfoil must be a NACA 4 or 5 digit designation, e.g. NACA 0012`. A coordinate file elsewhere is refused: `airfoil_file must name inputs/mesh/<name>.dat or inputs/airfoils/<name>.dat`.

cfdbender writes the coordinate file into each point folder as `airfoil.dat`, and `xfoil.in` loads it.

## Setup file

`inputs/setups/s001.toml`:

| Key | Meaning | Unit | Default |
|---|---|---|---|
| `airfoil` or `airfoil_file` | the airfoil, see Mesh | - | one is required |
| `reynolds` | Reynolds number over the chord; 0 is inviscid | - | 0 |
| `mach` | free stream Mach number | - | 0 |
| `ncrit` | transition criterion `N`, above 0 | - | 9 |
| `iterations` | viscous iteration limit, integer from 1 | - | 200 |
| `panels` | number of panels | - | XFOIL's own |

Any other key is refused as `unsupported XFOIL setup: <keys>`.

**`[solver]` table**

| Key | Meaning | Unit | Default |
|---|---|---|---|
| `executable` | path to XFOIL | - | see Install |
| `launcher` | tokens placed before the executable | - | `[]` |
| `timeout_s` | time after which cfdbender stops the run | s | 600 |
| `ram_estimate_gib` | memory the run is expected to need | GiB | 0.25 |

**Reference file** `inputs/references/r001.toml` may be empty. XFOIL reads only `length` from it, to turn a Reynolds number into a flow state when the row states the air with keys other than `REmi`. Any other key is skipped with a warning.

**Pproc file** `inputs/pproc/p001.toml` may be empty.

**Flight condition.** The row carries `ALPHA` in degrees, either one value or `ALPHA:sweep` with `SWEEP_VALUES`. Each angle is one point. cfdbender refuses any other condition key: `XFOIL matrix conditions require only scalar alpha or an alpha sweep`. The air keys of [Matrix columns](matrix-columns.md) are an exception, and set `mach` and `reynolds`:

| Matrix | XFOIL key |
|---|---|
| `MACH`, `REmi` | `mach`, `reynolds` (`REmi` x 1e6) |
| `GEOMETRY` | `airfoil_file` |

A row that states only `MACH` and angles sets the setup's `mach`. `BETA` must be 0: `a two-dimensional case flies at BETA 0; remove BETA from the row`. A setup value that disagrees with the row is refused: `conflict: <key> is <value> in the inputs and <value> from the matrix row; state it in one place`.

With a launcher, the WSL process inherits the mapped Windows point folder, and `xfoil.in`, `airfoil.dat` and `polar.txt` are relative to it. For a scheduler job, see [Cluster jobs](cluster-jobs.md).

## What converged means

cfdbender reads `polar.txt`, the polar that XFOIL accumulates. A point is `converged` when the exit code is 0 and the polar holds exactly one valid row whose alpha matches the requested angle (XFOIL prints alpha to three decimals; a match is within 0.0005 degrees). Otherwise the point is `failed`. XFOIL never gives `iteration_limit` here; a `VISCAL: Convergence failed` without a polar row is `failed`, and so is a clean exit without the row.

The polar holds no iterations or residuals. The history of a converged point has one entry, iteration 1 and residual 0, which marks that the row was accepted. They are not measured values.

| Name | Meaning | Axis |
|---|---|---|
| `CL` | lift coefficient | z, wind |
| `CD` | drag coefficient | x, wind |
| `CDp` | pressure drag coefficient | x, wind |
| `CM` | nose-up pitching moment coefficient | y |

Values are kept as XFOIL printed them. The references are unit chord, unit span and a moment origin at a quarter of the chord. XFOIL handles steady alpha points only. A polar row is numerical evidence, not a validation of the airfoil model.

## Worked example

A NACA 0012 polar at Reynolds 1 million, at 0, 2 and 4 degrees.

```console
cfd-workspace init campaign --solver xfoil
cd campaign
```

`inputs/setups/s001.toml`:

```toml
airfoil = "NACA 0012"
reynolds = 1000000

[solver]
launcher = ["wsl", "-d", "Ubuntu", "--"]
executable = "/home/your-user/bin/xfoil"
```

`inputs/references/r001.toml` and `inputs/pproc/p001.toml` are empty files. `inputs/matrices/study.fs`:

```text
POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES
---|---|---|---|-----|----------------|------------
1001|1|001|001|001|ALPHA:sweep|0,2,4
```

```console
cfd-matrix plan study
cfd-matrix run study
cfd-matrix collect study
```
