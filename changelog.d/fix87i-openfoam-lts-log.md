## Fixed

- The OpenFOAM LTS route (`time_integration = "lts"`) now writes its solver log as `log.pimpleFoam`, named after the application that ran, instead of `log.simpleFoam`; collect and cluster jobs read and await that log. The steady route keeps `log.simpleFoam`.
  migration: a script that read `log.simpleFoam` of an LTS point reads `log.pimpleFoam`.
