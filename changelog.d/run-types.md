## Added

- SU2 run types selected by the matrix column `WORKFLOW` with pyflightstream's names: `steady`, `qsteady_rotor` (a rotor solved steady in its rotating frame), `unsteady` and `unsteady_rotor` (the whole mesh in rigid motion, or a sliding-mesh multizone layout declared by a `[sliding]` table of the setup). Each run type writes the SU2 keys that follow from it; a key the setup states itself wins.
- Time-accurate rows read pyflightstream's clock (`DELTA_THETA` and `REVOLUTIONS`, or `DELTA_TIME` and `TIME_ITERATIONS`), averaging window (`LAST_REVS_AVG` or `LAST_ITERS_AVG`) and export window (`EXPORT_UNSTEADY_LAST_REV`, `_LAST_ITER`, `_AFTER_REV`, `_AFTER_ITER`). The reported result is the mean over the window, the rotor row is formed from the mean force and moment, and `polars/P<sim>_<matrix>_uns_avg.csv` holds one row per point with `FIRST_STEP`, `LAST_STEP` and `STEPS`. A point with an export window runs in two segments of one job (`su2.cfg`, `su2_segment2.cfg`, `run_segments.sh`).
- The row variable `TURBULENCE_MODEL` (`SA` or `SST`), always fully turbulent.
- The rotor declaration reads `blade1`, `alias` and the table form `axis = { shaft, vector_wrt_global }` with pyflightstream's refusals.

## Changed

- The rotor table is pyflightstream's `polars/P<sim>-<alias>_rotor.csv` with its columns, the in-plane `CN`, `CS`, `CMN` and `CMS` among them; columns nothing fills are absent and named in `products.json`. `post/<matrix>/rotor.csv` is no longer written.
- Thrust and torque are counted along the declared shaft, as pyflightstream counts them (`T = F . shaft`, `Q = M_hub . shaft`): with `axis = "X"` and the flow along +x a pulling propeller's `CT` is negative.
- A duplicated column of an SU2 history that the adapter does not read (SU2 8.6.0 writes `Time(sec)` twice) no longer fails the point, and a history that cannot be read is recorded with its cause.

## Migration

- A rotor block with `axis = "X"` and no `blade1` is refused, as in pyflightstream: add `blade1 = { azimuth_deg = 0.0, zero = "Z" }` (or `"Y"`).
- Read rotor coefficients from `polars/P<sim>-<alias>_rotor.csv`; the signs of `CT` and `CQ` follow the declared shaft.
