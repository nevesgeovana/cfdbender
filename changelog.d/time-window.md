## Added

- `cfdbender.core.unsteady_window`: the averaging window of an unsteady point and the time
  average over it, as pure functions with the semantics of pyflightstream's. Exactly one of
  `LAST_REVS_AVG` or `LAST_ITERS_AVG` states the window, with no default; it ends at the last
  step, inclusive and counted from one, and a fractional length is rounded to the nearest whole
  step, halves to even. `window_average` takes the arithmetic mean of each column over the
  window, and `window_is_covered` gives the reason a history that does not hold every step of
  the window is left out. Nothing reads a solver file and nothing is wired into the matrix, the
  plan or an adapter yet.
