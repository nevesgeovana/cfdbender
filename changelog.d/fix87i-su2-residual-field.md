## Changed

- The residual cfdbender records for an SU2 iteration, and the `residual` column of the `collect` tables, is now the residual of `CONV_FIELD` (`rms[Rho]` by default), the one SU2 converges on, instead of the largest `rms[...]` column. The SU2 result names it under `residual_field`. A `CONV_FIELD` without a residual column keeps the largest `rms[...]`, named `largest rms`.
  migration: a script that compared SU2 `residual` values across runs reads them again; they now follow `CONV_FIELD`.
