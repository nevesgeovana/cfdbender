## Changed

- The collected CSV tables (`campaign_sweep.csv`, `polars/P<POL>.csv`) carry the fixed flight-condition and reference columns of pyflightstream after `POL` and `point`, write `NA` for a missing value instead of an empty cell, and quote no cell, so `numpy.genfromtxt` reads them.
