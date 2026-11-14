## Changed

- SU2 setups that state `KIND_TURB_MODEL = "SA"` without `SA_OPTIONS` now get `SA_OPTIONS= ( NEGATIVE, WITHFT2 )`, the standard SA-neg, instead of SU2's default SA-noft2, whose nu-tilde update limiter stalled the NACA 0012 runs. A stated `SA_OPTIONS` replaces the default entirely.
  migration: to keep SA-noft2, state `SA_OPTIONS = ["NONE"]`.
