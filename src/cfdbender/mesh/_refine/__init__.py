"""Refining, coarsening and auditing a panel mesh from its OBJ.

A private package of the workspace layer. It imports nothing of ``run``; the
command layer reaches it only through ``refine_mesh``, ``audit_mesh`` and
``prepare_to_refine``, which read those names, and their result records, from this
module alone. Its modules, from
the floor up:

- ``_geometry``: the thresholds the requirements name, each defined once, and
  the shared geometric primitives;
- ``_obj``: the OBJ and trailing-edge points files;
- ``_audit``: the audit of a level against its source, which imports
  nothing of the refinement modules;
- ``_grid``: the structured grid of a family, recovered and resampled;
- ``_remesh_group`` and ``_remesh_pairs``: the group a remesh works on, and the
  pairing of its triangles into quadrilaterals;
- ``_remesh``: the remeshing of the other families, with conforming interfaces;
- ``_config``: the refinement file;
- ``_level``: the orchestration that writes a level;
- ``_lines``: the perimeter lines a grid is oriented by, and a family's
  trailing edge;
- ``_prepare``: the patches, refine file and node list of ``prepare-to-refine``.
"""

from cfdbender.mesh._refine._audit import MeshAudit as MeshAudit
from cfdbender.mesh._refine._audit import audit_mesh as audit_mesh
from cfdbender.mesh._refine._level import RefinedMesh as RefinedMesh
from cfdbender.mesh._refine._level import refine_mesh as refine_mesh
from cfdbender.mesh._refine._prepare import PreparedMesh as PreparedMesh
from cfdbender.mesh._refine._prepare import prepare_to_refine as prepare_to_refine
