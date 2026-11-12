SU2 accepts tracked CGNS meshes and optional `mesh_link = true` staging using
a relative symbolic link or hard link, with no copy fallback.

migration: Existing .su2 setups retain their format and copy staging by default.
