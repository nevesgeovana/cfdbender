# XFOIL workspaces

XFOIL computes two-dimensional airfoil polars. cfdbender plans one independent
headless XFOIL run per angle and collects the printed coefficients. Install
cfdbender on Windows and supply XFOIL separately in WSL.

## Install in WSL

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
in WSL. Download and unpack the [MIT XFOIL source](https://web.mit.edu/drela/Public/web/xfoil/)
to `~/src/Xfoil`, then build it in a user-space environment:

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

Adjust `MAMBA_ROOT_PREFIX` to your micromamba root if different. Use the
resulting executable's absolute Linux path below. The launcher passes that
path verbatim; `~` is not expanded.

## NACA 0012 alpha sweep

From PowerShell, initialize a workspace and edit the indicated files:

```console
cfd-workspace init --solver xfoil --workspace campaign
```

`campaign/inputs/setups/s001.toml`:

```toml
airfoil = "NACA 0012"
reynolds = 1000000

[solver]
launcher = ["wsl", "-d", "Ubuntu", "--"]
executable = "/home/your-user/bin/xfoil"
```

Use empty `campaign/inputs/references/r001.toml` and
`campaign/inputs/pproc/p001.toml`. Write `campaign/inputs/matrices/study.fs`:

```text
POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES
---|---|---|---|-----|----------------|------------
1001|1|001|001|001|alpha:sweep|0,2,4
```

```console
cfd-matrix plan study --workspace campaign --local
cfd-matrix run study --workspace campaign --local
cfd-matrix collect study --workspace campaign
```

The summary is `campaign/post/study/campaign_summary.json`. Planning reads
inputs without creating point folders. The WSL launcher inherits the mapped
Windows point directory, and input and output names are relative to it.

## Inputs and interpretation

The s file accepts `airfoil` as a NACA four- or five-digit designation, or
`airfoil_file` as a tracked coordinate file under the workspace. `reynolds`
defaults to zero (inviscid), `mach` to 0, `ncrit` to 9 and `iterations` to 200;
`panels` optionally sets the panel count. Its `[solver]` accepts `executable`,
`launcher`, `timeout_s` and `ram_estimate_gib`. Without an explicit executable,
`CFDBENDER_XFOIL_EXE` and then `xfoil` on PATH are used. The r and p files may
be empty. Scalar conditions use `alpha:4` with an empty sweep field.

A zero exit and exactly one valid polar row matching the requested alpha are
required for `converged`; otherwise the point is `failed`. XFOIL does not
produce `iteration_limit` here. The neutral history's iteration 1 and residual
0 are acceptance markers, not measured solver iterations or residuals.
CL is wind lift (z), CD and CDp are wind drag (x), and CM is nose-up pitch (y).
Values are retained as printed with unit chord and span and a quarter-chord
moment origin.

The adapter handles steady alpha points only. A polar row is numerical evidence,
not a validation of the airfoil model or a guarantee of physical accuracy.
