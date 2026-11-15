## Changed

- SU2 decks scaled with `REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE` (the default for compressible setups) that state no convergence level now get `CONV_RESIDUAL_MINVAL= -12`, the level of SU2's own NACA 0012 case, instead of SU2's built-in -8. A point is `converged` only when SU2's log reports the criteria satisfied, so this is also the classification threshold.
  migration: state `convergence_residual` or `CONV_RESIDUAL_MINVAL` to keep another level.
