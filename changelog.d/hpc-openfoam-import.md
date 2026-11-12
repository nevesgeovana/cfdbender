OpenFOAM accepts tracked .cgns and .ugrid meshes with a user-supplied
`mesh_converter` shell template using `{mesh}` and `{case}` before patch fixes.

migration: Install and validate the converter separately. Native gmsh import
is unchanged; no CGNS or UGRID reader is included.
