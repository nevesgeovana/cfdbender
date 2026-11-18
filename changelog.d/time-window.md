## Added

- `cfdbender.core.unsteady_window`: the averaging window of an unsteady point and the time
  average over it, as pure functions with the semantics of pyflightstream's. Exactly one of
  `LAST_REVS_AVG` or `LAST_ITERS_AVG` states the window, with no default; it ends at the last
  step, inclusive and counted from one, and a fractional length is rounded to the nearest whole
  step, halves to even. `window_average` takes the arithmetic mean of each column over the
  window, and `window_is_covered` gives the reason a history that does not hold every step of
  the window is left out. Nothing reads a solver file and nothing is wired into the matrix, the
  plan or an adapter yet.
- `cfdbender.core.unsteady_clock`: the clock of an unsteady row, `DELTA_THETA` with
  `REVOLUTIONS` or `DELTA_TIME` with `TIME_ITERATIONS`, resolved against the signed rotor speed
  (`DELTA_TIME = DELTA_THETA / (6 abs(rpm))`, a whole number of steps or a refusal), the steps
  per revolution, and the azimuth of blade one at each step from the `azimuth_deg` of the
  reference rotor's `blade1` table and the sign of the rpm. Pure functions only; no row key is
  read by any command yet.
- `cfdbender.core.revolutions`: one mean of each column per complete revolution
  (`per_revolution_table`), the drift of a mean from the previous revolution in per cent
  (`revolution_drift_pct`), and `revolution_drift_warnings`, which judges the last revolution's
  change of each load column at 1 percent by default against the scale of its load group, so an
  in-plane force is judged against the thrust. The warning is advice and never blocks. The
  classifier that names a column's load kind and group is a parameter.
