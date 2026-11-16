"""Surface mesh operations: OBJ input, welding and plane cuts.

The public names are :func:`read_mesh`, :func:`obj_group_surface`, :func:`weld` and
:func:`plane_cut_pieces`; they need numpy and trimesh only.
"""

from cfdbender.mesh.objio import obj_group_surface, plane_cut_pieces, read_mesh, weld

__all__ = ["obj_group_surface", "plane_cut_pieces", "read_mesh", "weld"]
