## Changed

- SU2 `RANS` setups that state no CFL key now get `CFL_NUMBER= 1000`, `CFL_ADAPT= YES` and `CFL_ADAPT_PARAM= ( 1, 1.2, 1000, 100000, 1, 300 )`, with `LINEAR_SOLVER_ITER= 50` and `NUM_METHOD_GRAD_RECON= LEAST_SQUARES` when the setup does not state them: the recipe that converged the TMR NACA 0012 at alpha 0, where the old ramp to 25 stalled. `cfl_max` and `cfl_growth` keep the recipe's other slots. Other solvers keep the ramp from 5 to 25.
  migration: state `cfl`, `CFL_NUMBER`, `CFL_ADAPT` or `CFL_ADAPT_PARAM` to keep your own CFL scheme; a stated `LINEAR_SOLVER_ITER` or `NUM_METHOD_GRAD_RECON` wins.
