## Fixed

- SU2 now receives `GAS_CONSTANT= 287.05287` and `GAMMA_VALUE= 1.4` from cfdbender's atmosphere constants (unless the setup states them), so its free-stream speed matches the other solvers; the PyFR case reads its `gamma` from the same constant.
