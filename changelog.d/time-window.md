## Added

- `cfdbender.core.unsteady_window`: the averaging window of an unsteady point, with the
  semantics of pyflightstream's. Exactly one of `LAST_REVS_AVG` or `LAST_ITERS_AVG` states the
  window, with no default; it ends at the last step, inclusive and counted from one, and a
  fractional length is rounded to the nearest whole step, halves to even.
- `cfdbender.core.unsteady_clock`: the clock of an unsteady row, `DELTA_THETA` with
  `REVOLUTIONS` or `DELTA_TIME` with `TIME_ITERATIONS`, resolved against the signed rotor speed
  (`DELTA_TIME = DELTA_THETA / (6 abs(rpm))`, a whole number of steps or a refusal), the steps
  per revolution, and the azimuth of blade one at each step from the `azimuth_deg` of the
  reference rotor's `blade1` table and the sign of the rpm.
- `cfdbender.post.time_average`: the arithmetic mean of each column of a history over the
  window, and the reason a history that does not hold every step of the window is left out.
- `cfdbender.post.revolutions`: one mean of each column per complete revolution, the drift of a
  mean from the previous revolution in per cent, and a warning that judges the last
  revolution's change of each load column at 1 percent by default against the scale of its load
  group, so an in-plane force is judged against the thrust. The warning never blocks.
- Nothing reads a solver file and nothing is wired into the matrix, the plan or an adapter yet.

## Changed

- The SU2 adapter warns and skips `LAST_ITERS_AVG`, `DELTA_TIME` and `TIME_ITERATIONS` beside
  the other time-accurate row controls, and never forwards them to `su2.cfg`.
