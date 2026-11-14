## Changed

- OpenFOAM steady runs keep the under-relaxation default (`p 0.3`, `U 0.7`, `nuTilda 0.7`) and now let the setup replace it whole with `relaxationFactors`. The new `time_integration = "lts"` selects a pseudo-transient route (`pimpleFoam`, `ddtSchemes localEuler`) with `lts_max_co`, `lts_smoothing`, `lts_damping`, `lts_max_delta_t` and a native `PIMPLE` body that wins entirely. See the OpenFOAM page.
