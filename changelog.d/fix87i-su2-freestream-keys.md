## Changed

- SU2 decks no longer state free-stream values SU2 recomputes and ignores: the resolved `FREESTREAM_DENSITY` is written only with `FREESTREAM_OPTION = "DENSITY_FS"`, and the resolved `FREESTREAM_PRESSURE` is left out of a compressible viscous run with a Reynolds number unless `INIT_OPTION = "TD_CONDITIONS"`. Values stated in the setup are always written. Compressible EULER, NAVIER_STOKES and RANS decks default to ``FREESTREAM_PRESS_EQ_ONE`` unless the setup states another scaling.
