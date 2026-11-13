## Added

- `plan`, `run`, `collect` and `post` write a live log `logs/<command>-<stamp>.log` and start and end events to `logs/activity.log` and `logs/activity.log.jsonl`; the `log` verb lists the events (stage `command`).

## Changed

- Every warning is now written to `logs/warnings.log`; `--warning-off` only silences the console.
