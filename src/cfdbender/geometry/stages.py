"""The numerical building blocks of the geometry pipeline, for the runner and the kernel stages.

Pipeline role: the one home of the numerical building blocks that the stage runner
(:mod:`cfdbender.geometry.pipeline`) and the kernel stages call: the B-spline :class:`Patch`,
the generation of the components of a resolved configuration and their regions, the body and
lifting-surface skins and the flat tip cap, the placement of a wing in the frame of its body,
the mesh spacing rules and the mesh checks, the OBJ writer, the mathematics of reading a
lifting surface back from sections, the record boundary (schema export, pin verification,
configuration writing), and ``junctions``, the namespace of the measured checks and limits of
the S10 junction sub-steps.

The pin check of a component record is defined here; configuration writing delegates to the
file boundary, and the other names are defined in the private modules of this package and
re-exported, so a caller in another package imports this module and no private path. The functions
import no geometry kernel and no solver; they need numpy and scipy only. The names that matter
to a user are frozen in :mod:`cfdbender.geometry`; this module is the internal contract between
the numerical layer and the stages.

Examples
--------
>>> from cfdbender.geometry.stages import Mesh, Patch
>>> Patch.__name__, Mesh.__name__
('Patch', 'Mesh')
"""

from __future__ import annotations

from pathlib import Path

import cfdbender.geometry._control_surfaces as controls
import cfdbender.geometry._junctions as junctions
from cfdbender.geometry._blister import blister_model
from cfdbender.geometry._bodies import body
from cfdbender.geometry._boundary import export_schema, sha256_file, unpack
from cfdbender.geometry._boundary import write_config as _write_config
from cfdbender.geometry._caps import CAP_EDGE_GAP, cap_edge_gap, lifting_model, tip_cap
from cfdbender.geometry._fillet import fillet_model
from cfdbender.geometry._generation import (
    GeneratedPiece,
    assembly_regions,
    generate_components,
    generate_pieces,
    source_cells,
    stored_sections_config,
)
from cfdbender.geometry._identity import build_id, parameter_id, surface_id
from cfdbender.geometry._lifting import lifting_surface
from cfdbender.geometry._mesh_blocks import (  # noqa: F401  strategy 2 planning
    azimuth_split,
    block_from_walk,
    carry_pairs_growth,
    check_triangles,
    component_topology,
    irregular_nodes,
    ring_count,
    solve_counts,
)
from cfdbender.geometry._mesh_rules import (
    check_body,
    check_lifting,
    chord_fraction,
    chord_nodes,
    constant,
    count_classes,
    insert_seed,
    load_mesh_controls,
    tip_refinement,
)
from cfdbender.geometry._mesh_schema import (  # noqa: F401
    azimuth_count,
    mesh_controls,
    mesh_measurements,
)
from cfdbender.geometry._mesh_topology import (
    Mesh,
    check_mesh,
    orient_mesh,
    oriented_source_normals,
    repair_inverted,
    write_obj,
)
from cfdbender.geometry._patch import Patch, evaluate_patch
from cfdbender.geometry._placement import placement_transform, wing_in_body_frame
from cfdbender.geometry._propeller_checks import evaluate as evaluate_propeller_patch
from cfdbender.geometry._propeller_checks import measured as propeller_measurement
from cfdbender.geometry._propeller_input import adopt_propeller
from cfdbender.geometry._propeller_slice import (  # noqa: F401
    BladeFrame,
    BladeSurface,
    definition_text,
    profile_text,
    slice_blade,
)
from cfdbender.geometry._records import (
    CheckFailedError,
    check_entry,
    exit_code,
    geometry_report,
    stage_record,
    write_record,
)
from cfdbender.geometry._registry import manifest_checks, manifest_record, sidecar_identity
from cfdbender.geometry._reverse import (
    clean_band,
    closed_contour,
    contour_features,
    cosine,
    fit_section,
    leading_extension,
    reference_planform,
    refine_tip,
)
from cfdbender.geometry._toml_keys import adopt_surface_file
from cfdbender.geometry.component import ComponentRecord
from cfdbender.geometry.stage_table import STAGE_ROWS, StageRow, stage_index, stage_row

__all__ = [
    "adopt_propeller",
    "evaluate_propeller_patch",
    "propeller_measurement",
    "build_id",
    "manifest_record",
    "surface_id",
    "sidecar_identity",
    "stored_sections_config",
    "placement_transform",
    "CAP_EDGE_GAP",
    "cap_edge_gap",
    "lifting_model",
    "CheckFailedError",
    "GeneratedPiece",
    "Mesh",
    "Patch",
    "STAGE_ROWS",
    "StageRow",
    "adopt_surface_file",
    "assembly_regions",
    "blister_model",
    "azimuth_count",
    "body",
    "check_body",
    "check_entry",
    "check_lifting",
    "check_mesh",
    "chord_fraction",
    "chord_nodes",
    "clean_band",
    "closed_contour",
    "constant",
    "contour_features",
    "controls",
    "cosine",
    "count_classes",
    "evaluate_patch",
    "exit_code",
    "export_schema",
    "fillet_model",
    "fit_section",
    "generate_components",
    "generate_pieces",
    "geometry_report",
    "insert_seed",
    "junctions",
    "leading_extension",
    "lifting_surface",
    "load_mesh_controls",
    "mesh_controls",
    "azimuth_count",
    "azimuth_split",
    "block_from_walk",
    "carry_pairs_growth",
    "check_triangles",
    "component_topology",
    "irregular_nodes",
    "ring_count",
    "solve_counts",
    "manifest_checks",
    "orient_mesh",
    "oriented_source_normals",
    "parameter_id",
    "reference_planform",
    "refine_tip",
    "repair_inverted",
    "stage_index",
    "source_cells",
    "stage_record",
    "stage_row",
    "tip_cap",
    "tip_refinement",
    "unpack",
    "verify_pins",
    "wing_in_body_frame",
    "write_config",
    "write_obj",
    "write_record",
]


def write_config(config, path):
    """Write a resolved configuration as TOML that :func:`load_config` reads back unchanged.

    Parameters
    ----------
    config : dict
        Resolved configuration of plain values.
    path : str or Path
        File to write.
    """
    _write_config(config, path)


def verify_pins(record: ComponentRecord, asset_root: str | Path) -> list[str]:
    """Check every referenced file against its pin; return one message per failure.

    Paths resolve against ``asset_root``. An empty list means every referenced
    file exists with matching bytes. A path that escapes the root is a failure.

    Parameters
    ----------
    record : ComponentRecord
        The component record whose referenced files are checked.
    asset_root : str or Path
        Directory against which the referenced paths resolve.

    Returns
    -------
    list of str
        One message per failure; empty when every pin holds.
    """
    root = Path(asset_root).resolve()
    problems: list[str] = []
    for reference in record.referenced_files():
        target = (root / reference.path).resolve()
        if root != target and root not in target.parents:
            problems.append(f"{reference.path}: outside the asset root")
        elif not target.is_file():
            problems.append(f"{reference.path}: missing")
        elif sha256_file(target) != reference.sha256:
            problems.append(f"{reference.path}: SHA-256 differs from the pin")
    return problems
