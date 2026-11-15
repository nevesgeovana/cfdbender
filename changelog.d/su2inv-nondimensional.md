## Changed

- SU2 compressible setups (`SOLVER` `EULER`, `NAVIER_STOKES` or `RANS`) that state no `REF_DIMENSIONALIZATION` now get `REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE`, as SU2's own NACA 0012 case does. SU2's default, `DIMENSIONAL`, made the implicit system so badly scaled that SU2's official NACA 0012 deck stalled at CFL 1000. Coefficients are unchanged; residual levels in `history.csv` become relative to the free stream.
  migration: a `CONV_RESIDUAL_MINVAL` chosen for a dimensional run now measures a different level; state `REF_DIMENSIONALIZATION = "DIMENSIONAL"` to keep the old scaling.
