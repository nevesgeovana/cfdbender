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
`pyfs-workspace` command and flag through `cfd-workspace`. The thirteen verbs
that act on what the workspace holds (`space-in-use`, `free-space`,
`delete-sims`, `sync`, `restore`, `rebuild`, `mark-failed`, `mark-converged`,
`show`, `log`, `trace`, `history` and `diff`) are typed on `cfd-workspace` for
every solver; `cfd-workspace` hands them to `pyfs-matrix`, which owns them. In a
FlightStream workspace `cfd-matrix` still hands the old form to `pyfs-matrix` as it
hands every other command. Arguments and help
are provided by pyflightstream. See the
[pyflightstream documentation](https://nevesgeovana.github.io/pyflightstream/)
for commands, inputs, and workflows.

Declare the FlightStream build in the matrix `FS_BUILD` column and configure
its executable in `inputs/executables.toml` as described by pyflightstream.

A cfdbender neutral reader for FlightStream results is pending implementation.
