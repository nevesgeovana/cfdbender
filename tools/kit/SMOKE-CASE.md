# A check of the SU2 chain

The builder uses gmsh's Python API to generate a neutral coarse 2D inviscid
NACA0012 mesh with airfoil and farfield markers. No research geometry is used.
The one-row smoke matrix selects EULER, Mach 0.3, angle 2 degrees, reference
area and length 1, moment origin (0.25, 0, 0), and 40 iterations. One thread
and a 0.25 GiB prelaunch estimate leave the adapter's 0.5 GiB free RAM floor.

Measured on 2026-11-10 with SU2 8.6.0: 1,280 cells, 40 iterations, classified
`iteration_limit`. A successful process exit does not mean converged.
A cluster running SU2 8.3.0 may differ in the last digits; that version is not
measured by this kit.
The first run is a check of the chain, not of aerodynamics.

Expected files are a point's su2.cfg, mesh.su2, stdout.log, stderr.log,
execution.json and history.csv, the root runs.json, and the neutral summary
post/smoke/campaign_summary.json. All paths are relative to smoke/.
Keep logs if the result differs. Do not rerun a finished point in place.

The kit explicitly selects the SU2 executable route. The pysu2 route can run
when available and can fall back to the executable when pysu2 is absent.
Grouped SU2 HPC CLI jobs (--batch) are not implemented. FlightStream needs
the optional extra and a licensed solver; neither is included in this kit.
OpenFOAM and XFOIL are supported separately; Fluent is not implemented.
The HPC template is only a field example; fill it yourself before any submission.
