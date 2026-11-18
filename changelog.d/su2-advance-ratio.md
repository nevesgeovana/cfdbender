## Added

- The SU2 adapter reads `ADVANCE_RATIO` in `FLIGHT_CONDITION`, as a value or as the swept
  variable, for a propeller in a steady rotating frame. With the setup's
  `GRID_MOVEMENT = "ROTATING_FRAME"` and a rotor declared in the REF file as a FlightStream
  reference file declares it (a table with `kind = "rotor"`, `axis`, `rpm_sign`,
  `diameter_m`, the hub `x_m`, `y_m`, `z_m` and `families_blades`), it writes `ROTATION_RATE`
  from `J = V / (n D)`, `MOTION_ORIGIN` and `MACH_MOTION`, one point per advance ratio. A
  missing rotor, a missing rotating frame, a second `ROTATION_RATE`, a non-positive
  `ADVANCE_RATIO` and more than one rotor are refused.
- The matrix columns `SYMMETRY` (`NONE` or `PERIODIC` for SU2) and `SYMMETRY_LOADS` are read
  for SU2 with the row variable `PERIODIC_COPIES`, as on the FlightStream side. `DELTA_THETA`, `REVOLUTIONS`,
  `LAST_REVS_AVG`, `CLOCK_MOTION`, `MOTIONS` and every `EXPORT_*` variable warn and are skipped. None of them reaches `su2.cfg`.
- `collect` computes thrust, torque, `CT`, `CQ`, `CP` and the efficiency of the whole rotor
  for such a point, the periodic copies summed as rotated vectors, into the point's `rotor`
  result and the new `post/<matrix>/rotor.csv`. The polar files keep their columns.
- `docs/su2-workspaces.md` has a section, Propeller in a steady rotating frame.

## Migration

- A matrix without `ADVANCE_RATIO` plans byte-identically. SU2 now reads a `SYMMETRY` and a
  `SYMMETRY_LOADS` cell; the other solvers still refuse them.
