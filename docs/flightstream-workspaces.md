# FlightStream workspaces

FlightStream support uses pyflightstream to manage a campaign in its own
workspace format. Install the optional support first:

```console
pip install cfdbender[flightstream]
cfd-workspace init --solver flightstream campaign
```

The resulting workspace contains the pyflightstream files and folders plus a
root `init.toml` identifying its solver. In a FlightStream workspace, use
every `pyfs-matrix` command and flag through `cfd-matrix`, and every
`pyfs-workspace` command and flag through `cfd-workspace`. Arguments and help
are provided by pyflightstream. See the
[pyflightstream documentation](https://nevesgeovana.github.io/pyflightstream/)
for commands, inputs, and workflows.

Declare the FlightStream build in the matrix `FS_BUILD` column and configure
its executable in `inputs/executables.toml` as described by pyflightstream.

A cfdbender neutral reader for FlightStream results is pending implementation.
