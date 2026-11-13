# Cluster jobs

This page shows how to send the points of a matrix to a cluster as jobs and
how to pick their results up. You describe the scheduler of your cluster once,
in a profile; the matrix and the commands stay the same.

## The profile

A profile is a file `inputs/hpc/<name>.toml`. It says what the job description
file of your scheduler looks like and which command submits it. cfdbender
knows no scheduler: every field name and the submit command are yours.

`inputs/hpc/h001.toml`:

```toml
application_id = "CFD"
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
output_file = "{stdout_log}"

[submit]
command = ["submit-tool", "{descriptor_path}"]

[builds]
"1.0" = "cfd-1.0"
```

| Key | Meaning |
|---|---|
| `application_id` | The name your scheduler gives the application. Required. |
| `max_walltime` | The longest wall clock the cluster grants a job, written `HH:MM:SS`. Optional. |
| `walltime_arithmetic` | `wall` writes the `WALLTIME` cell as the row states it (`4h`); `seconds` writes whole seconds. Optional; `wall` when absent. |
| `[descriptor]` `format` | The format of the job description file: `yaml`, `json`, `toml` or `text`. |
| `[descriptor]` `name` | The name of that file. It is written in the folder of each point. |
| `[descriptor.fields]` | The lines of that file. The names on the left are those your scheduler expects. |
| `[submit]` `command` | The submit command, one word per list item. |
| `[builds]` | For each `SOLVER_BUILD` a row may name, the name your cluster gives that build. It fills `{solver_version_alias}`. |
| `[application_ids]` | For a `SOLVER_BUILD`, an `application_id` that replaces the general one. Optional. |

A key the profile reader does not know is refused, with the list of the keys
it reads, so a misspelt key cannot be ignored in silence.

A value in braces is filled for each point:

| Placeholder | Filled with |
|---|---|
| `{job_name}` | The `POL` and the point name, for example `1001_DP-M200AL+000`. |
| `{ncpus}`, `{walltime}` | The `NCPUS` and `WALLTIME` cells of the row. |
| `{solver_version_alias}` | The `[builds]` entry of the row's `SOLVER_BUILD`. |
| `{workdir}` | The folder of the point. |
| `{config_path}`, `{script_path}` | The file the solver reads, in the folder of the point. |
| `{stdout_log}`, `{stderr_log}` | `stdout.log` and `stderr.log` in the folder of the point. |
| `{descriptor_path}` | The job description file, for the submit command. |

## See what would be submitted

With a profile in `inputs/hpc/`, `plan` adds the job description of each point
and the command that would submit it. It submits nothing. For a matrix row
with `NCPUS` 16, `WALLTIME` `2h` and `SOLVER_BUILD` `1.0`, in a workspace at
`/home/you/cluster`:

```console
cfd-matrix plan polar
```

```text
descriptor: /home/you/cluster/sims/sim_1001/datapoints/DP-M200AL+000/submit.yaml
job_name: "1001_DP-M200AL+000"
walltime: "2h"
ncpus: 16
version: "cfd-1.0"
working_directory: "/home/you/cluster/sims/sim_1001/datapoints/DP-M200AL+000"
config_file: "/home/you/cluster/sims/sim_1001/datapoints/DP-M200AL+000/mock.in"
output_file: "/home/you/cluster/sims/sim_1001/datapoints/DP-M200AL+000/stdout.log"
would submit: submit-tool /home/you/cluster/sims/sim_1001/datapoints/DP-M200AL+000/submit.yaml
```

These lines follow the list of points that `plan` always prints.

## Submit

```console
cfd-matrix run polar
```

With a profile, `run` prepares the folder of each point, writes the job
description there, runs the submit command, and records the point as
`SUBMITTED`. It does not wait for the jobs.

The folder of a point holds every file the solver reads, so the job needs
nothing else from the workspace:

| Solver | Prepared in the folder of the point | `{config_path}` | The job must write |
|---|---|---|---|
| su2 | `su2.cfg`, the mesh (`mesh.su2` or `mesh.cgns`) | `su2.cfg` | `history.csv`, `stdout.log` |
| openfoam | the case folders `system`, `constant`, `0`, the script `Allrun`, the mesh | `Allrun` | `log.simpleFoam`, `postProcessing/forceCoeffs/0/coefficient.dat` |
| pyfr | `case.ini`, `mesh.msh` | `case.ini` | `forces.csv` |
| xfoil | `xfoil.in`, and `airfoil.dat` when the row names a coordinate file | `xfoil.in` | `polar.txt`, `stdout.log` |
| qprop | `propeller.prop`, `qcon.def`, `sections.json` | `propeller.prop` | `stdout.log` |

Point the output file of your scheduler at `{stdout_log}`. Several solvers
print their results there, and `collect` reads that file.

The mesh is a link to the file in `inputs/mesh/`, written relative to the
workspace, so it still works after the whole workspace is copied to the
cluster. A point whose files cannot be prepared is recorded as failed with
`FAILED_STAGING` and is not submitted.

## Choose where a matrix runs

| Situation | What `run` does |
|---|---|
| No file in `inputs/hpc/` | Runs on this machine. |
| One profile | Submits with it. `--local` runs on this machine instead. |
| Several profiles | Refuses until you name one with `--hpc NAME`, or choose `--local`. |

## Pick the results up

```console
cfd-matrix collect polar
```

`collect` looks at each submitted point. When the files its job must write are
there, it reads them, records the point as `COMPLETED` with its verdict, and
adds it to the tables. A point whose files are not there yet stays
`SUBMITTED` and is counted as outstanding; the exit code is then 3.

```console
cfd-matrix collect polar --watch
```

`--watch` repeats the collect until no point is outstanding. `--watch-interval`
sets the seconds between two rounds (60 when absent) and `--rounds` stops after
a number of rounds. Under `--watch` a file counts as finished when its size
and its time of change are the same on two looks 2 s apart; `--interval` sets
that time.

A cluster job leaves no record of how its process ended, so the verdict of a
submitted point rests on the numbers in its output alone.

## Builds on a cluster

A cluster job does not read `inputs/executables.toml`. The `SOLVER_BUILD` of
the row is translated by the `[builds]` table of the profile, and the job
description carries the result. [Executables](executables.md) covers runs on
your machine.

## Redoing a submitted point

`cfd-matrix run polar --force-rerun POINT` archives the point and submits it
again. It does not cancel the earlier job: cancel it in your scheduler first.
See [Run again, resume, records](records.md).
