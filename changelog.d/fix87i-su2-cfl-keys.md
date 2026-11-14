## Added

- SU2 setups accept the neutral keys `cfl_max` and `cfl_growth`, which keep the default CFL ramp and replace only its maximum or its up factor; `cfl` sets the start and the minimum. `cfl_max = 1000` writes `CFL_ADAPT_PARAM= ( 0.5, 1.2, 5, 1000 )`. The default ramp is unchanged.
