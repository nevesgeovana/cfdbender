## Fixed

- A run stopped by the RAM floor, or by any exception after a command finished, now records a non-zero `exit_code` in `execution.json` in every adapter; the PyFR record also names the `failed_command` index.
- PyFR classifies a run whose last force sample falls one output interval short of `tend` as `iteration_limit`, not `failed`.
