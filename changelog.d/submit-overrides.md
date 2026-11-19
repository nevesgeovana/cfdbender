## Added

- `hpc.<name>: <value>` entries in `VAR_NAMES_VALUES` replace one value of the row's HPC profile (a descriptor field, `application_id` or `cluster`), so the matrix states what is run and where. The profile is the base and everything else of it is kept. A name the profile does not declare is refused with the names it accepts; the override is recorded with its origin in the plan record and the run record. Double quotes in `VAR_NAMES_VALUES` now protect a `/` inside a value, such as a script path.
