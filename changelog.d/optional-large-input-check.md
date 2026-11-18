## Added

- A planned input larger than 8 MiB, such as a mesh, can be left unread: `--no-read-large-inputs`
  on `cfd-matrix plan`, `run` and `collect`, the environment variable
  `CFDBENDER_NO_READ_LARGE_INPUTS=1`, or `large_input_check = "identity"` in `init.toml`. The file
  is then never opened; its size and modification time are recorded as
  `identity:size=<bytes>;mtime_ns=<ns>` and checked by every later command, and the checks that
  need its contents are skipped with one warning each. The default, `"digest"`, is unchanged.

## Changed

- A scheduler submission no longer refuses when the submitting host has less than 0.5 GiB of free
  memory or cannot measure it: it warns once and submits, because the job runs on the cluster. A
  local run keeps the refusal.
