OpenFOAM accepts `solver.processes` and `solver.mpi_launcher`, emitting scotch
decomposition, MPI simpleFoam execution, and final-field reconstruction.

migration: The default process count remains 1. Parallel force coefficients
retain the case-root postProcessing path used by serial collection.
