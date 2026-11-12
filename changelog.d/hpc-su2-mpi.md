SU2 executable runs accept a `solver.mpi_launcher` argument list before the
solver command, including explicit mpirun task counts or scheduler launchers.

migration: The default empty launcher retains serial execution; pysu2 remains serial.
