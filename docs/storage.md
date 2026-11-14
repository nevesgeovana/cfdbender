# Storage and sync

This page shows how to see the disk space a campaign uses, how to free it, how
to delete a sim, and how to bring results from another copy of the workspace.

The three commands that delete or copy show what they would do and change
nothing until you add `--apply`. Every call is recorded in
`storage_management.json`, and `cfd-matrix log --storage` lists the calls.

## See the space in use

```console
cfd-matrix space-in-use
```

```text
by folder
     12.5 KB  post
      6.4 KB  archive
      1.7 KB  sims
       445 B  inputs
by sim
      1.7 KB  sim_1001
by extension
     24.9 KB  .json
      2.3 KB  .log
      2.1 KB  .csv
      1.6 KB  .jsonl
      1.6 KB  .in
       250 B  .fs
       207 B  .toml
        23 B  (none)
```

`--top N` sets how many lines each group shows (15 when absent).

## Free space with a recipe

A recipe is a file `inputs/management/<name>.toml` that lists the kinds of file
to delete under `sims/`. Create the folder `inputs/management/` if the
workspace does not have it.

`inputs/management/clean.toml`:

```toml
[[delete_extensions]]
extensions = [".csv", ".tmp"]
```

```console
cfd-matrix free-space clean --list
```

```text
  keep sims/sim_1001/datapoints/DP-M200AL+000/history.csv (123 B)
  keep sims/sim_1001/datapoints/DP-M200AL+020/history.csv (123 B)
  keep sims/sim_1001/datapoints/DP-M200AL+040/history.csv (123 B)
would free 0 B
preview only; add --apply to delete
```

A file that a record names as an output of its run is kept, whatever the
recipe says, because `collect` reads it. In this example every `.csv` file is
such an output, so nothing would be freed. `--list` prints each file with what
happens to it; without it only the total is printed.

```console
cfd-matrix free-space clean --apply
```

## Delete a sim

```console
cfd-matrix delete-sims --sims 1001 --matrix-products regenerate
```

```text
would delete sims/sim_1001 (1.7 KB)
  record 1001/DP-M200AL+000 (COMPLETED)
  record 1001/DP-M200AL+040 (COMPLETED)
  record 1001/DP-M200AL+020 (COMPLETED)
preview only; add --apply to delete
```

`delete-sims` deletes the folder of each named sim and takes its records out
of `runs.json`. A note of each deleted record stays in the history, so
`cfd-matrix history 1001` still shows that the sim existed.

When the collected tables of a matrix hold the points to delete, the command
asks what to do with them:

```text
the products of polar hold the deleted points; choose --matrix-products points-only or regenerate
```

| `--matrix-products` | Effect |
|---|---|
| `points-only` | Deletes the sim and leaves the tables as they are. |
| `regenerate` | Deletes the sim and rebuilds the tables without it. `--archive` keeps the replaced tables under `archive/<stamp>/`. |

A sim with a point that is still `SUBMITTED` or `RUNNING` is refused; `--force`
deletes it anyway. A sim that a manifest other than the one named by `--runs` still
records is refused, because every manifest shares the folder `sims/sim_<id>`: delete
its points from that manifest first.

To clear a sim and keep its files, use `cfd-workspace archive` instead; see
[Run again, resume, records](records.md).

## Bring results from another workspace

`sync` copies records and files into this workspace from other copies of it,
for example the copy on a cluster. Name the other workspaces in
`inputs/sync-workspaces.toml`:

```toml
[workspaces]
cluster = "/mnt/cluster/campaign"
```

```console
cfd-matrix sync runs
cfd-matrix sync runs --apply
```

The word after `sync` says how much to bring. Each level adds to the one
before:

| Level | Brings |
|---|---|
| `runs` | The records. |
| `post` | Also the collected files under `post/`. |
| `all` | Also the files of the sims. |

| Option | Effect |
|---|---|
| `--from NAME` | Reads one of the named workspaces instead of all of them. |
| `--prefer-other` | When both workspaces have a record of the same point, takes the other one. Without it this workspace keeps its own. At level `post` or `all` it needs `--overwrite` too, and is refused without it: records taken without their files would not match. |
| `--overwrite` | When both have a file of the same name, archives the file of this workspace and takes the other one. The files of a point whose record this workspace keeps are never replaced: a record and the files it describes stay together. |
| `--include-archives` | Also brings the files under folders named `archive`. |

Without the file, `sync` refuses:

```text
sync needs inputs/sync-workspaces.toml: [workspaces] name = "path"
```
