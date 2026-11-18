## Added

- The SU2 adapter reads `ADVANCE_RATIO` in `FLIGHT_CONDITION`, as a value or as the swept
  variable, for a propeller in a steady rotating frame. With the setup's
  `GRID_MOVEMENT = "ROTATING_FRAME"` and a rotor block in the REF file
  (`[rotors.<NAME>]` with `diameter_m`, `blades`, `origin`, `shaft`, `rpm_sign` and
  `monitored_blades`), it writes `ROTATION_RATE` from `J = V / (n D)`, `MOTION_ORIGIN` and
  `MACH_MOTION`, one point per advance ratio. A missing rotor block, a missing rotating frame,
  a second `ROTATION_RATE`, a non-positive `ADVANCE_RATIO` and more than one rotor are refused.
- `collect` computes thrust, torque, `CT`, `CQ`, `CP` and the efficiency of the whole rotor
  for such a point, scaled by `blades / monitored_blades`, into the point's `rotor` result and
  the new `post/<matrix>/rotor.csv`. The polar files keep their columns.
- `docs/su2-workspaces.md` has a section, Propeller in a steady rotating frame.

## Migration

- A matrix without `ADVANCE_RATIO` plans byte-identically. A REF file may now hold a `rotors`
  table; SU2 reads it and no longer warns that it is skipped.
