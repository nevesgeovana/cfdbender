## Added

- Adapter capability declarations for restart, unsteady time marching,
  per-boundary force groups and cluster staging, validated on load against
  one core vocabulary. Solver support is documented in a table checked by tests.

## Changed

- Cluster planning and runs warn once and skip staging and submission when the
  selected adapter does not declare cluster staging support.
