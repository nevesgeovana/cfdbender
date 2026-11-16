"""The stage table of the geometry pipeline: ids, names, checks and where each limit lives.

Pipeline role: pure data of the geometry package. The runner
(:mod:`cfdbender.geometry.pipeline`) executes stages by id, a listing of the stages prints
this table, and a stage record names its checks from it. A stage that needs
the geometry kernel is marked, so a run that stops at the last pure stage imports none.
``limit`` is the sentence that says where the check's limit comes from, never a second copy of
the number: the number lives with the stage that measures it.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfdbender._errors import InputValueError

__all__ = ["STAGE_ROWS", "StageRow", "stage_index", "stage_row"]


@dataclass(frozen=True)
class StageRow:
    """One stage: its id, its name, whether it needs the kernel, and its checks.

    ``checks`` is a tuple of ``(check id, limit)`` pairs; ``limit`` states where the limit is
    taken from.
    """

    stage: str
    name: str
    kernel: bool
    checks: tuple[tuple[str, str], ...]


def _row(stage, name, kernel, *checks):
    return StageRow(stage, name, kernel, tuple(checks))


STAGE_ROWS: tuple[StageRow, ...] = (
    _row("S0", "Definition", False, ("input.valid", "no refused key")),
    _row(
        "S1",
        "Section law",
        False,
        ("section.residual", "acceptance.section_residual"),
        ("section.one_nose_radius", "acceptance"),
        ("section.te_thickness", "acceptance"),
        ("section.inflection", "acceptance"),
    ),
    _row(
        "S1.P",
        "Propeller section fit",
        False,
        ("propeller.section_fit", "definition.options.fit_tol_chord"),
    ),
    _row("S2", "Identity report", False, ("identity.listed", "every seed listed")),
    _row(
        "S3",
        "Compatibility",
        False,
        ("compat.common_vector", "identical knots and pins"),
        ("compat.shape_unchanged", "1e-9 of the chord"),
    ),
    _row(
        "S4",
        "Planform or body law",
        False,
        ("law.fewest_stations", "acceptance"),
        ("law.monotonic", "acceptance"),
        ("law.ruled_twist", "acceptance"),
        ("law.region_boundaries", "acceptance"),
    ),
    _row(
        "S4.P",
        "Propeller design measures",
        False,
        ("propeller.solidity", "positive disk area ratio"),
        ("propeller.activity_factor", "positive per-blade integral"),
        ("propeller.beta_reference", "1e-10 deg residual"),
        ("propeller.beta_0p7R", "absolute beta from plane at most 90 deg"),
    ),
    _row("S5", "Placement", False, ("place.frame", "acceptance")),
    _row(
        "S5.P",
        "Propeller planar placement",
        False,
        ("propeller.section_plane", "1e-8 mm"),
        ("propeller.handedness", "zero sign violations"),
    ),
    _row(
        "S6",
        "Skin",
        False,
        ("skin.section_exact", "1e-12 m"),
        ("skin.law_residual", "acceptance"),
        ("skin.distance", "acceptance, when an original exists"),
    ),
    _row(
        "S6.P",
        "Propeller cubic loft",
        False,
        ("propeller.loft_exact", "1e-8 mm at defining stations"),
        ("propeller.loft_c2", "zero radial knots below C2"),
    ),
    _row(
        "S7",
        "Trims and layout",
        False,
        ("trim.closed", "1e-12 in parameter units"),
        ("trim.in_domain", "domain of the skin"),
        ("trim.iso_exact", "axis-aligned"),
        ("trim.partition", "1e-9 relative"),
        ("trim.outer_only", "warning"),
    ),
    _row(
        "S8.P",
        "Propeller numerical closure",
        False,
        ("propeller.closure", "zero unexpected free or triple edges at 1e-6 mm"),
    ),
    _row(
        "S8",
        "Closures",
        False,
        ("closure.rim_shared", "1e-12 m"),
        ("closure.cap_edge_gap", "5e-6 m, 4001 boundary samples"),
        ("closure.rim_jump", "acceptance"),
        ("closure.base_present", "acceptance"),
        ("closure.no_ridge", "acceptance"),
        ("closure.retracted_identity", "acceptance"),
    ),
    # Sub-steps of S8 for the blend pieces, each with its own measured checks. A blister
    # hosts a fillet, so its sub-steps come first and its own check stops the run.
    _row(
        "S8.B1",
        "Blister build",
        False,
        ("section.residual", "body tolerance, 1e-4 of its length by default"),
        ("continuity.bluff_jump", "1 deg, geometry._blister.SEAM_LIMIT_DEG"),
    ),
    _row(
        "S8.B2",
        "Blister trim by host",
        False,
        ("closure.base_present", "every section enters the host"),
        ("blister.crease_on_host", "1e-9 m"),
        ("trim.closed", "1e-12 in parameter units"),
        ("trim.in_domain", "domain of the base"),
        ("trim.area_removed", "1e-2 relative, geometry._blister.AREA_REMOVED_LIMIT"),
    ),
    _row(
        "S8.B3",
        "Blister closure",
        False,
        ("closure.free_edges", "none: poles at the tips, symmetry plane at the keel"),
    ),
    _row(
        "S8.F1",
        "Fillet build",
        False,
        ("fillet.spine", "every section finds the hosts' intersection"),
        ("closure.tangent_host", "1 deg, geometry._fillet.TANGENT_LIMIT_DEG"),
        ("closure.tangent_wing", "1 deg, geometry._fillet.TANGENT_LIMIT_DEG"),
        ("closure.contact_gap_host", "body tolerance, 1e-4 of its length by default"),
        ("closure.contact_gap_wing", "body tolerance, 1e-4 of its length by default"),
    ),
    _row(
        "S8.F2",
        "Fillet trim by host",
        False,
        ("closure.base_present", "the base runs past both contact lines"),
        ("fillet.host_footprint", "2e-2 relative, geometry._fillet.FOOTPRINT_LIMIT"),
        ("fillet.contact_on_face", "no contact on the part of a blister its body trims away"),
    ),
    _row(
        "S8.F3",
        "Fillet closure",
        False,
        ("closure.free_edges", "none; the lower end when the lower face stops short of the TE"),
    ),
    _row(
        "S9",
        "Continuity",
        False,
        ("continuity.bluff_jump", "1 deg or a declared crease"),
        ("continuity.internal", "reported"),
    ),
    _row(
        "S10",
        "Junctions",
        False,
        ("dominance.plan", "no cycle; a row naming no member is warned and skipped"),
    ),
    # Sub-steps of S10, in the order they run; ids follow the junction stages: C body joins,
    # D assembly, E groups of three pieces or more. Every limit is a constant of
    # geometry._junctions, named here.
    _row(
        "S10.D1",
        "Pieces loaded",
        True,
        ("assembly.face_count", "the faces the generation built, per piece"),
        ("placement.contact", "the row's tolerance_m: the pieces of a row touch"),
    ),
    _row("S10.C5", "Piece regenerated", True, ("piece.regenerated", "REGENERATE_M, 1e-9 m")),
    _row("S10.C6", "Symmetry edges", True, ("piece.symmetry_edges", "SYMMETRY_M, 1e-6 m")),
    _row("S10.C8", "Rims of the dominated buried", True, ("join.rim_inside", "RIM_MARGIN_M")),
    _row("S10.C9", "Openings on their partner", True, ("join.opening_on_partner", "OPENING_M")),
    _row(
        "S10.D2",
        "Each piece alone, written, read back and sewn at 0.01 mm",
        True,
        ("reimport.faces", "faces read equal faces written"),
        ("piece.triple_edges", "none"),
        ("piece.free_edges", "none unexplained: symmetry, opening or buried"),
    ),
    _row(
        "S10.C7",
        "Each piece alone sewn at 0.001 mm",
        True,
        ("junction.free_edges_classified", "none unexplained at SEW_FINE_M"),
    ),
    _row(
        "S10.D3",
        "Body joins cut",
        True,
        ("join.crease_gap", "CREASE_GAP_M, 1e-6 m"),
        ("join.length", "MIN_JUNCTION_FRACTION, 0.04 of the smaller piece's extent"),
    ),
    _row(
        "S10.D4",
        "Cuts",
        True,
        ("junction.curve_gap", "JUNCTION_GAP_M, 5e-6 m"),
        ("junction.chain_ends", "CHAIN_END_GAP_M: no lonely end"),
        ("junction.loop_closure", "LOOP_CLOSURE_M, 5e-6 m"),
        ("junction.single_curve", "one curve per junction"),
        ("junction.shared_vertex", "one vertex where faces meet"),
        ("junction.pcurve_on_face", "JUNCTION_GAP_M, 5e-6 m"),
        ("junction.length", "MIN_JUNCTION_FRACTION, 0.04 of the smaller piece's extent"),
    ),
    _row(
        "S10.E1",
        "Groups of three pieces or more",
        True,
        ("group.junction_length", "MIN_JUNCTION_FRACTION of the smaller piece, per pair"),
        ("group.single_curve", "one curve per junction of the group"),
    ),
    _row(
        "S10.D5",
        "Projected junctions",
        True,
        ("junction.projected_length", "the dominant's contact edges, 1e-3 relative"),
        ("junction.offset", "the row's tolerance_m"),
        ("junction.ends_on_cut", "J_END_ON_CUT_M, 5e-5 m"),
    ),
    _row(
        "S10.D6",
        "Hosts split",
        True,
        ("junction.on_host_loops", "J_ON_LOOP_M, 5e-6 m"),
        ("junction.split_closure", "J_LOOP_CLOSURE_M, 2e-5 m"),
    ),
    _row(
        "S10.D8",
        "Inside removed",
        True,
        ("removal.seeds_reach", "every piece keeps a seed"),
        ("removal.kept_on_loops", "KEPT_ON_LOOP_M, 1e-5 m"),
        ("removal.area", "the area under the dominants, 1e-3 relative"),
        ("junction.no_vertex_inside", "none"),
        ("junction.no_triple_edge", "none"),
    ),
    _row(
        "S10.D8b",
        "Skin coverage",
        True,
        ("coverage.skin", "acceptance.coverage_fraction"),
        ("coverage.area_sum", "1e-3 relative"),
    ),
    _row("S10.C1", "Shared curves of the joins", True, ("join.shared_pairs", "at least one")),
    _row("S10.C2", "Joins written", True, ("join.written_pairs", "WRITTEN_PAIR_M, 1e-9 m")),
    _row("S10.C3", "Edges on their surface", True, ("join.edge_on_surface", "EDGE_ON_SURFACE_M")),
    _row(
        "S10.C4",
        "Joins sewn in one import",
        True,
        ("join.group_free_edges", "none unexplained at SEW_M and SEW_FINE_M"),
    ),
    _row(
        "S10.D9",
        "Assembly written and read back",
        True,
        ("write.read_back", "faces read equal faces written, per piece"),
        ("write.area", "1e-3 relative, per piece"),
    ),
    _row(
        "S10.D10",
        "Assembly sewn at 0.01 and 0.05 mm",
        True,
        ("sew.free_introduced", "none beyond the pieces' own"),
        ("sew.triple_edges", "none"),
        ("sew.slivers", "none beyond the pieces' own, SLIVER_WIDTH_M (theirs listed)"),
    ),
    _row(
        "S10.E2",
        "Separate pieces clear",
        True,
        # stricter than "> 0": a piece within the sewing tolerance of
        # another would be sewn to it
        ("assembly.clearance", "SEW_M, 1e-5 m, the sewing tolerance (min)"),
    ),
    _row(
        "S10.FW1",
        "Fowler cove bays",
        True,
        ("fowler.cove_bay_mismatch", "0; split each cove face at every skin span knot"),
    ),
    _row(
        "S10.FW2",
        "Fowler nesting",
        True,
        ("fowler.nested_clearance_pct", "control.clearance_pct, minimum percent chord"),
    ),
    _row(
        "S10.FW3",
        "Fowler motion laws",
        True,
        ("fowler.gap_error_pct", "below 0.01 percent chord, 49 stations"),
        ("fowler.overlap_error_pct", "below 0.01 percent chord, 49 stations"),
        ("fowler.law_threshold_hits", "0 errors at or above 0.01 percent chord"),
        ("fowler.inside_main", "0 profile points at 49 stations"),
    ),
    _row(
        "S10.FW4",
        "Fowler retracted exterior",
        True,
        ("closure.retracted_identity", "1e-12 m outside the nose and lip gap bands"),
    ),
    _row(
        "S10.F0",
        "Control pieces read",
        True,
        ("control.faces_read", "faces read equal faces written"),
    ),
    _row(
        "S10.F1",
        "Main with cove sewn",
        True,
        ("control.f1.free_edges", "0 unexplained at 0.01 mm"),
        ("control.f1.triple_edges", "0"),
    ),
    _row(
        "S10.F2",
        "Control alone sewn",
        True,
        ("control.f2.free_edges", "0 except named open ends at 0.01 mm"),
        ("control.f2.triple_edges", "0"),
    ),
    _row(
        "S10.F3",
        "Control host clearance",
        True,
        ("control.host_contact", "0 pairs with native distance <= 0 m"),
    ),
    _row(
        "S10.F4",
        "Main and control together",
        True,
        ("control.f4.free_delta", "0 relative to F1 plus F2"),
        ("control.f4.free_edges", "0 unnamed slot edges"),
        ("control.f4.triple_edges", "0"),
    ),
    _row(
        "S10.F5",
        "Controlled wing body cut",
        True,
        ("control.f5.free_delta", "0 relative to F4"),
        ("control.f5.junction_length", "equal to the clean assembly reference"),
        ("control.f5.triple_edges", "0"),
    ),
    _row(
        "S10.F6",
        "Controlled wing nacelle cuts",
        True,
        ("control.f6.free_delta", "0 relative to F5"),
        ("control.f6.junction_length", "equal to the clean assembly reference"),
        ("control.f6.triple_edges", "0"),
    ),
    _row(
        "S10.F7",
        "Capped tail group",
        True,
        ("control.f7.free_edges", "0 at 0.01 mm"),
        ("control.f7.triple_edges", "0"),
    ),
    _row(
        "S11",
        "Export",
        True,
        ("write.face_count", "faces written equal retained faces"),
        ("profile.surfaces_only", "no solid entity"),
    ),
    _row("S11.V", "Export registry", False, ("manifest.piece_version", "version of each piece")),
    _row(
        "S11.PU",
        "Propeller output units",
        False,
        ("propeller.output_units", "zero files outside schema 1 mm units"),
    ),
    _row(
        "S11.P",
        "Export profile",
        False,
        ("profile.reader", "reader profile"),
        ("profile.base_shared", "1e-12"),
    ),
    _row(
        "S11.PC",
        "Propeller written trim curves",
        False,
        ("propeller.trim_preference", "zero missing curves or non-PREF-1 boundaries"),
        ("propeller.pcurve_3d", "1e-4 mm between the pcurve image and spatial trim"),
    ),
    _row(
        "S11.R",
        "Export reimport",
        True,
        ("area.read_back", "1e-3 relative"),
        ("area.source_excess", "acceptance.area_excess"),
        ("reimport.each_component", "0 lost faces"),
        ("reimport.free_edges", "0 changed edges at 1e-5 m"),
        ("reimport.triple_edges", "0 edges"),
    ),
    _row(
        "S11.S",
        "Export sew",
        True,
        ("sew.free_introduced", "0 new edges at 1e-5 and 5e-5 m"),
        ("sew.triple_edges", "0 edges"),
        ("sew.slivers", "0 new faces below 5 mm mean width"),
    ),
    _row(
        "S11.PS",
        "Propeller sewn topology",
        True,
        ("propeller.sewn_free_edges", "zero beyond explicitly uncapped rims"),
        ("propeller.sewn_triple_edges", "zero"),
    ),
    _row("S12.C", "Reader conversion", True, ("gate.convert", "0 changed face counts")),
    _row("S12.G", "Reader gridding", True, ("gate.grid", "0 mismatched or triple edges")),
    _row(
        "S12",
        "Reader gate",
        False,
        ("gate.conformal", "0 mismatched or triple edges"),
        ("gate.te_found", "positive upper TE rim from sidecar roles"),
    ),
    _row(
        "S13.R", "Regenerate stored sections", False, ("roundtrip.same_surface", "equal surface_id")
    ),
    _row(
        "S13",
        "Round trip",
        True,
        ("roundtrip.identify", "acceptance"),
    ),
)

_BY_ID = {row.stage: row for row in STAGE_ROWS}


def stage_row(stage: str) -> StageRow:
    """Return the row of ``stage``, refusing an id the table does not hold.

    Parameters
    ----------
    stage : str
        Stage id from the geometry pipeline table, such as ``S10.D1``.

    Returns
    -------
    StageRow
        The stage's name, kernel requirement and measured checks.

    Raises
    ------
    InputValueError
        If ``stage`` is absent from the table; the message lists accepted ids.
    """
    try:
        return _BY_ID[stage]
    except KeyError:
        raise InputValueError(
            f"stage {stage!r}: unknown; the stages are {', '.join(_BY_ID)}"
        ) from None


def stage_index(stage: str) -> int:
    """Return the position of ``stage`` in the table, refusing an unknown id.

    Parameters
    ----------
    stage : str
        A stage id, ``"S0"`` to ``"S13"``.

    Returns
    -------
    int
        The index of its row in ``STAGE_ROWS``.

    Raises
    ------
    InputValueError
        If the table holds no such stage.

    Examples
    --------
    >>> from cfdbender.geometry.stages import stage_index
    >>> stage_index("S0")
    0
    """
    return STAGE_ROWS.index(stage_row(stage))
