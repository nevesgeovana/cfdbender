# Dependency licenses

cfdbender is distributed under the MIT license. Its direct runtime dependencies
have their own licenses. The offline kit retains their published wheels,
including their license files and bundled notices.

| Package | Project license |
| --- | --- |
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| trimesh | MIT |
| rtree | MIT |
| gmsh | GPL-2.0-or-later with the upstream linking exception |
| vtk | BSD-3-Clause |
| matplotlib | Matplotlib license, based on the Python Software Foundation license |
| pydantic | MIT |

gmsh and its OpenCASCADE kernel are installed from gmsh's own distribution as a
separate package. cfdbender imports gmsh from one module only,
`cfdbender/mesh/_gmsh.py`, when a command needs the kernel. The GNU General Public
License, version 2 or later, includes the upstream exception permitting linking
with other programs. Measured with gmsh 4.15.2.

The table describes the direct Python projects. Native libraries and other
dependencies bundled in a wheel can carry additional notices; retain the license
files of the exact wheels distributed in a kit.

See [Install](install.md) for Python installation, system libraries and solver
requirements, and [Geometry and mesh commands](cli-geometry-mesh.md) for the
commands that use the kernel.
