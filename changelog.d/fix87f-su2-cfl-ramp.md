## Changed

- SU2 runs now default to a CFL ramp (`CFL_NUMBER= 5`, `CFL_ADAPT= YES`, `CFL_ADAPT_PARAM= ( 0.5, 1.2, 5, 25 )`) when the setup states no CFL key; stating `cfl`, `CFL_NUMBER`, `CFL_ADAPT` or `CFL_ADAPT_PARAM` replaces the default entirely. The new neutral setup key `cfl` maps to `CFL_NUMBER`.
