"""The junction sub-steps of stage S10: dominance applied on the kernel, one check per step.

Pipeline role: kernel part of stage S10 of the geometry pipeline. An :class:`Assembly` holds the
pieces of one configuration as kernel faces and runs the sub-steps of the stage table in order:
pieces loaded, each piece alone, body joins and cuts by one fragment per group, projected
junctions imprinted on their hosts, the inside removed by a flood from each piece's own exposed
side, skin coverage in parameter space, the joins and the whole written, read back and sewn. Each
sub-step returns its named checks with the measured value and the limit, and a run stops at the
first failed check. The dominant imprints on the dominated, the dominated loses its inside,
and the dominant is split along the junction too; no
area constant decides a removal.

The kernel is reached only through :class:`~cfdbender.mesh._cad.CadKernel`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import label

from cfdbender.geometry._geom_parts import (
    BLUFF,
    SEAT_REACH_M,
    add_patch,
    bbox_hit,
    free_boundaries,
    hull_contact,
    sewn_free_edges,
    skip,
)
from cfdbender.geometry._geom_removal import RemovalSteps
from cfdbender.geometry.stages import Patch, source_cells
from cfdbender.geometry.stages import junctions as jn
from cfdbender.mesh._cad import CadKernel, Face, FileTopology

__all__ = ["BLUFF", "Assembly", "assemble", "sewn_free_edges"]


#: A piece of this many faces or fewer is fused in one call; a larger one face by face (see
#: ``Assembly._fragment_in_turn``). The faces of a lifting surface and its controls fuse in one
#: call in a few seconds; a body of sixty faces did not return.
_ONE_CALL_FACES = 24
#: Boxes closer than this, in meters, are taken to touch.
_BOX_GAP_M = 1e-6


def _boxes_meet(a: tuple, b: tuple) -> bool:
    """Whether two axis-aligned boxes, each ``(minimum, maximum)``, touch or overlap."""
    return all(
        a[0][k] <= b[1][k] + _BOX_GAP_M and b[0][k] <= a[1][k] + _BOX_GAP_M for k in range(3)
    )


#: Fuzzy value of a two-piece cut, as a fraction of the smaller piece's extent.
PAIR_FUZZY_FRACTION = 2e-5


class Assembly(RemovalSteps):
    """The pieces of one configuration on a kernel, and the sub-steps of S10 run on them.

    Parameters
    ----------
    kernel : CadKernel
        An open kernel; the caller closes it.
    pieces : Mapping
        Component name to its placed patches.
    rows : Sequence of Mapping
        The ``[[dominance]]`` rows as resolved.
    out : path
        The folder the written files go to.
    members : sequence of str, optional
        The members of the configuration; every piece by default.
    kinds : Mapping, optional
        Component name to its type; two pieces of a type in :data:`BLUFF` form a body join.
    plane_axis : int, optional
        The axis normal to the symmetry plane of a half model (1 for y); None for a full model.
    expected_faces : Mapping, optional
        Face counts the generation reported, per piece; the patch counts by default.
    regenerated : Mapping, optional
        The pieces built a second time from their own resolved inputs.
    coverage_fraction : float, optional
        The largest missing area of a piece's skin, as a fraction of its area.
    coverage_spacing : float, optional
        Sample spacing as a share of each piece's extent for source-base coverage.
    """

    STEPS = (
        ("S10.D1", "load"),
        ("S10.C5", "regenerate"),
        ("S10.C6", "symmetry"),
        ("S10.C8", "rims"),
        ("S10.C9", "openings"),
        ("S10.D2", "alone"),
        ("S10.C7", "alone_fine"),
        ("S10.D3", "join_cut"),
        ("S10.D4", "cut"),
        ("S10.E1", "group"),
        ("S10.D5", "project"),
        ("S10.D6", "split"),
        ("S10.D8", "remove"),
        ("S10.D8b", "coverage"),
        ("S10.C1", "shared"),
        ("S10.C2", "written"),
        ("S10.C3", "on_surface"),
        ("S10.C4", "join_sewn"),
        ("S10.D9", "write"),
        ("S10.D10", "sew"),
        ("S10.E2", "clearance"),
    )

    def _source_coverage(self, name, index, patch):
        model = self.models.get(name)
        trims = (
            None
            if model is None
            else [t for t in model.trims if t.base == list(model.bases)[index]]
        )
        cells = source_cells(
            patch, spacing=self.coverage_spacing, scale=self.extents[name], trims=trims
        )
        points, areas = cells["points"], cells["areas"]
        covered = np.zeros(len(points), bool)
        for face in self.faces.get(name, []):
            if self.origin.get(face) != (name, index):
                continue
            projected, uv = self.k.project(face, points)
            near = np.linalg.norm(projected - points, axis=1) <= jn.SEW_M
            near &= np.isfinite(uv).all(axis=1)
            if near.any():
                covered[near] |= self.k.inside(face, uv[near])
        covered &= cells["included"]
        removed = ~cells["included"]
        for i in np.flatnonzero(~covered & ~removed):
            removed[i] = self._legit(name, points[i], cells["normals"][i])
        missing = ~(covered | removed)
        groups, _ = label(missing.reshape(areas.shape))
        group_areas = np.bincount(groups.ravel(), weights=areas.ravel())
        group_areas[0] = 0.0
        chosen = int(group_areas.argmax())
        largest = float(group_areas[chosen])
        centroid = (
            np.average(
                points[groups.ravel() == chosen],
                axis=0,
                weights=areas.ravel()[groups.ravel() == chosen],
            ).tolist()
            if largest
            else []
        )
        return (
            {
                "area_m2": float(areas.sum()),
                "covered_m2": float(areas.ravel()[covered].sum()),
                "removed_m2": float(areas.ravel()[removed].sum()),
                "missing_m2": float(areas.ravel()[missing].sum()),
            },
            largest,
            centroid,
        )

    def _coverage(self):
        """Check generated bases in world space, retaining original trim exclusions.

        Patch-only callers cover the full base domain. Generated source models
        additionally retain their original trim exclusions.
        """
        worst, area_error, where, totals = 0.0, 0.0, {}, {}
        for name, patches in self.sources.items():
            ledger = {key: 0.0 for key in ("area_m2", "covered_m2", "removed_m2", "missing_m2")}
            expected = self.source_areas[name]
            for index, patch in enumerate(patches):
                measured, largest, centroid = self._source_coverage(name, index, patch)
                for key in ledger:
                    ledger[key] += measured[key]
                fraction = largest / max(expected, 1e-30)
                if fraction >= worst:
                    worst, where = (
                        fraction,
                        {
                            "component": name,
                            "patch": index,
                            "missing_m2": largest,
                            "centroid_m": centroid,
                        },
                    )
            balance = ledger["covered_m2"] + ledger["removed_m2"] + ledger["missing_m2"]
            area_error = max(area_error, abs(balance - expected) / max(expected, 1e-30))
            totals[name] = {**ledger, "source_area_m2": expected}
        return [
            jn.measured(
                "coverage.skin",
                worst,
                self.coverage_fraction,
                unit="fraction",
                where=where,
                cause="BUG: source skin is neither kept nor legitimately removed",
            ),
            jn.measured(
                "coverage.area_sum",
                area_error,
                1e-3,
                unit="relative",
                cause="BUG: covered, removed and missing areas do not sum to the source area",
            ),
        ], {"pieces": totals, "inputs": {"spacing": self.coverage_spacing, "distance_m": jn.SEW_M}}

    # ---- D1, C5, C6, C8, C9: the pieces before any operation ------------------------------

    def _load(self):
        worst, where = 0, {}
        for name, patches in self.sources.items():
            model = self.models.get(name)
            if model is None:
                built = [add_patch(self.k, p) for p in patches]
                self.source_areas[name] = sum(self.k.area(f) for f in built)
                for index, face in enumerate(built):
                    self.origin[face] = (name, index)
            else:
                bases = {key: add_patch(self.k, p) for key, p in model.bases.items()}
                self.source_areas[name] = sum(self.k.area(f) for f in bases.values())
                built = []
                for trim in model.trims:
                    face = self.k.add_trimmed_face(bases[trim.base], [trim.loop, *trim.inner])
                    self.origin[face] = (name, list(bases).index(trim.base))
                    built.append(face)
                self.k.remove(list(bases.values()))
            expected = len(model.trims) if model else len(patches)
            gap = abs(len(built) - int(self.expected_faces.get(name, expected)))
            if gap > worst or not where:
                worst, where = gap, {"component": name, "built": len(built)}
            self.faces[name] = self._self_fragment(name, built)
            self.input_free_edges += [
                self._samples(edge, 17)
                for edge, count in self.edge_counts(self.faces[name]).items()
                if count == 1
            ]
            if model is not None and not self.plan:
                self._record_input_slivers(name)
        self._resolve_auto()
        contact, pair, limit = 0.0, {}, jn.SEW_M
        for op in self.plan:
            d = hull_contact(self.hulls[op.dominant], self.hulls[op.dominated])
            if d - op.tolerance_m >= contact - limit or not pair:
                contact, limit, pair = d, op.tolerance_m, {"pair": self._op_name(op)}
        checks = [
            jn.measured(
                "assembly.face_count",
                worst,
                0,
                unit="faces",
                where=where,
                cause="BUG: a piece holds another count of faces than its generation",
            ),
            jn.measured(
                "placement.contact",
                contact,
                limit,
                unit="m",
                where=pair,
                cause="STRATEGY: the pieces of a row do not touch (placement)",
            ),
        ]
        return checks, {"inputs": {"patches": {n: len(p) for n, p in self.sources.items()}}}

    def _record_input_slivers(self, name):
        """Keep the thin source faces as the export sewing reference."""
        for index, face in enumerate(self.faces[name]):
            perimeter = sum(
                jn.polyline_length(self._samples(abs(e), 257)) for e in self.k.boundary(face)
            )
            area = self.k.area(face)
            width = jn.mean_width(area, perimeter)
            if width < jn.SLIVER_WIDTH_M:
                self.input_sliver_faces.append(
                    {"component": name, "face": index, "area_m2": area, "width_m": width}
                )

    def _self_fragment(self, name: str, built: list[Face]) -> list[Face]:
        """Fragment a piece's own faces together, so faces of one piece share their seams."""
        inert = [f for f in built if self._inert(f)]
        active = [f for f in built if not self._inert(f)]
        if len(active) < 2:
            return active + inert
        if len(active) > _ONE_CALL_FACES:
            return self._fragment_in_turn(active) + inert
        result = self.k.fragment(active[:1], active[1:])
        return self._inherit(active, [*result.from_a, *result.from_b]) + inert

    def _fragment_in_turn(self, faces: Sequence[Face]) -> list[Face]:
        """Fragment many faces by adding each to the set made so far, against its neighbors only.

        One call of the kernel's general fuse on every face of a body (about 60 B-spline faces
        that touch along their seams) did not return in twenty minutes, while the same faces fuse
        pairwise in seconds in all: the cost of the call grows much faster than the number of
        faces. A face that touches nothing already placed is set down as it is; one that touches
        some is fused with those only, and the pieces replace them. The pieces are the same as
        from the one call, since a face crosses only the faces its box meets.
        """
        done: list[Face] = []
        rank: dict[Face, int] = {}
        for position, face in enumerate(faces):
            box = self.k.bbox(face)
            near = [p for p in done if _boxes_meet(box, self.k.bbox(p))]
            if not near:
                done.append(face)
                rank[face] = position
                continue
            inputs = [face, *near]
            before = {f: rank[f] if f in rank else position for f in inputs}
            result = self.k.fragment([face], near)
            rows = [*result.from_a, *result.from_b]
            self._inherit(inputs, rows)
            for p in near:
                rank.pop(p, None)
            fresh: dict[Face, int] = {}
            for source, row in zip(inputs, rows, strict=True):
                for piece in row:
                    fresh[piece] = min(fresh.get(piece, before[source]), before[source])
            done = [p for p in done if p not in near] + list(fresh)
            rank.update(fresh)
        return sorted(done, key=lambda f: rank[f])

    def _inherit(
        self,
        inputs: Sequence[Face],
        rows: Sequence[Sequence[Face]],
        before: Mapping[Face, tuple[str, int]] | None = None,
    ) -> list[Face]:
        # the kernel numbers new faces again with tags of faces it removed, so the origin of a
        # piece is set from this call's inputs, never kept from an earlier face of the same tag
        out: list[Face] = []
        fresh: dict[Face, tuple[str, int]] = {}
        for face, row in zip(inputs, rows, strict=True):
            origin = (self.origin if before is None else before).get(face)
            for piece in row:
                if origin is not None and piece not in fresh:
                    fresh[piece] = origin
                if piece not in out:
                    out.append(piece)
        self.origin.update(fresh)
        return out

    def _resolve_auto(self) -> None:
        """Read an ``auto`` row as a projection when the dominant lies along the dominated."""
        resolved = []
        for op in self.plan:
            if op.operation == "auto":
                free = free_boundaries(self._source_faces(op.dominant), jn.SEW_M)
                touching = any(
                    jn.distance_to_triangles(c, self.hulls[op.dominated]).max() <= op.tolerance_m
                    for c in free
                )
                kind = "project" if touching else "cut"
                self.warnings.append(f"{self._op_name(op)}: auto read as {kind}")
                op = jn.Operation(
                    kind, op.dominant, op.dominated, op.row, op.tolerance_m, op.contact_deg
                )
            resolved.append(op)
        self.plan = [o for o in resolved if o.operation == "cut"] + [
            o for o in resolved if o.operation == "project"
        ]

    def _regenerate(self):
        if self.regenerated is None:
            return skip(["piece.regenerated"], "no second build of the pieces was given"), {}
        worst, where = 0.0, {}
        for name, patches in self.sources.items():
            try:
                change = jn.pole_change(patches, self.regenerated.get(name, []))
            except ValueError as error:
                change, where = 1.0, {"component": name, "error": str(error)}
            if change >= worst:
                worst, where = change, {**where, "component": name}
        return [
            jn.measured(
                "piece.regenerated",
                worst,
                jn.REGENERATE_M,
                unit="m",
                where=where,
                cause="BUG: the same inputs build other poles",
            )
        ], {}

    def _symmetry(self):
        if self.plane_axis is None:
            return skip(["piece.symmetry_edges"], "a full model has no symmetry plane"), {}
        worst, where = 0.0, {}
        for name in self.sources:
            near = [
                c
                for c in free_boundaries(self._source_faces(name), jn.SEW_M)
                if np.abs(c[:, self.plane_axis]).max() <= jn.SHARED_PAIR_NEAR_M
            ]
            offset = jn.plane_offset(near, self.plane_axis)
            if offset >= worst:
                worst, where = offset, {"component": name, "edges": len(near)}
        return [
            jn.measured(
                "piece.symmetry_edges",
                worst,
                jn.SYMMETRY_M,
                unit="m",
                where=where,
                cause="BUG: a half model's open edges are off the symmetry plane",
            )
        ], {}

    def _rims(self):
        cuts = self._ops("cut")
        if not cuts:
            return skip(["join.rim_inside"], "no cut row"), {}
        total, where = 0, {}
        for op in cuts:
            hull = self.hulls[op.dominant]
            for rim in self._rim_edges(op.dominated):
                if not bbox_hit(rim, hull, jn.RIM_MARGIN_M):
                    continue
                count = jn.points_outside(rim, hull, margin=jn.RIM_MARGIN_M)
                if count:
                    total += count
                    where = {"pair": self._op_name(op), "at": rim[0].tolist()}
        return [
            jn.measured(
                "join.rim_inside",
                total,
                0,
                unit="points",
                where=where,
                cause="STRATEGY: an open rim of the dominated is not buried in the "
                "dominant, so the cut leaves it free",
            )
        ], {}

    def _rim_edges(self, name: str) -> list[np.ndarray]:
        return [
            c
            for c in free_boundaries(self._source_faces(name), jn.SEW_M)
            if not (self.plane_axis is not None and self._on_plane_curve(c))
        ]

    def _openings(self):
        cuts = self._ops("cut")
        if not cuts:
            return skip(["join.opening_on_partner"], "no cut row"), {}
        worst, where = 0.0, {}
        for op in cuts:
            hull = self.hulls[op.dominated]
            for edge in self._rim_edges(op.dominant):
                if not bbox_hit(edge, hull, jn.SEW_M) or jn.parity_inside(edge, hull).all():
                    continue
                d = float(jn.distance_to_triangles(edge, hull).max())
                if d >= worst:
                    worst, where = d, {"pair": self._op_name(op), "at": edge[0].tolist()}
        return [
            jn.measured(
                "join.opening_on_partner",
                worst,
                jn.OPENING_M,
                unit="m",
                where=where,
                cause="STRATEGY: the dominant's opening does not follow the dominated",
            )
        ], {}

    # ---- D2, C7: each piece alone, written, read back and sewn ---------------------------

    def _partners(self, name: str):
        hulls = [h for n, h in self.hulls.items() if n != name and len(h)]

        def distance(points: np.ndarray) -> np.ndarray:
            if not hulls:
                return np.full(len(points), np.inf)
            return np.min([jn.distance_to_triangles(points, h) for h in hulls], axis=0)

        def inside(points: np.ndarray) -> np.ndarray:
            if not hulls:
                return np.zeros(len(points), bool)
            return np.any([jn.parity_inside(points, h) for h in hulls], axis=0)

        return distance, inside

    def _unexplained(self, name: str, topology: FileTopology) -> int:
        free = [s for s, faces in topology.edges if len(faces) == 1]
        distance, inside = self._partners(name)
        classes = jn.classify_free_edges(
            free, distance=distance, inside=inside, plane_axis=self.plane_axis
        )
        return classes.count("unexplained")

    def _alone(self):
        read_gap, triple, worst, where = 0, 0, 0, {}
        per_piece = {}
        for name, faces in self.faces.items():
            path = self.out / f"S10-alone-{name}.igs"
            self.k.export_iges({name: faces}, path)
            self.alone_files[name] = path
            topology = self.k.inspect_file(path, jn.SEW_M)
            read_gap = max(read_gap, abs(len(topology.areas) - len(faces)))
            triple += sum(1 for _, f in topology.edges if len(f) > 2)
            self.alone_free += [s for s, f in topology.edges if len(f) == 1]
            widths = [
                jn.mean_width(a, p)
                for a, p in zip(topology.areas, topology.perimeters, strict=True)
            ]
            self.alone_slivers += len(jn.slivers(widths))
            count = self._unexplained(name, topology)
            per_piece[name] = count
            if count > worst or not where:
                worst, where = count, {"component": name}
        return [
            jn.measured(
                "reimport.faces",
                read_gap,
                0,
                unit="faces",
                cause="BUG: the writer or the reader lost a face",
            ),
            jn.measured(
                "piece.triple_edges",
                triple,
                0,
                unit="edges",
                cause="BUG: a piece alone has an edge of three faces",
            ),
            jn.measured(
                "piece.free_edges",
                worst,
                0,
                unit="edges",
                where=where,
                cause="BUG: a piece alone has an open edge nothing closes",
            ),
        ], {"unexplained": per_piece, "files": {n: p.name for n, p in self.alone_files.items()}}

    def _alone_fine(self):
        worst, where = 0, {}
        for name, path in self.alone_files.items():
            count = self._unexplained(name, self.k.inspect_file(path, jn.SEW_FINE_M))
            if count > worst or not where:
                worst, where = count, {"component": name}
        return [
            jn.measured(
                "junction.free_edges_classified",
                worst,
                0,
                unit="edges",
                where=where,
                cause="BUG: a seam of the piece closes only at the coarser tolerance",
            )
        ], {}

    # ---- D3, D4, E1: the cuts, one fragment per group ------------------------------------

    def _cut_all(self) -> None:
        if self.cut_done:
            return
        self.cut_done = True
        self.groups = self._cut_groups()
        with self._active_faces():
            for group in self.groups:
                a = list(self.faces[group[0]])
                b = [f for name in group[1:] for f in self.faces[name]]
                result = self.k.fragment(a, b, fuzzy=self._group_fuzzy(group))
                self._regroup(group, a + b, [*result.from_a, *result.from_b])

    def _group_fuzzy(self, group: Sequence[str]) -> float:
        """Return the fuzzy value of one group's fragment.

        Only a body join of two pieces is cut exactly: its crease gap is what ``join.crease_gap``
        measures, and a fuzzy value would close it. Every other cut takes the group value, a
        pair too: a body seam lying in the plane of a blunt trailing-edge base is a curve on a
        face, which an exact fragment resolves by the last bits of the arithmetic, and on some
        processors it lost every junction curve but those across that base.
        """
        if len(group) == 2 and all(self.kinds.get(name) in BLUFF for name in group):
            return 0.0
        if len(group) == 2:
            # Relative to the smaller piece, so a model scaled by any factor keeps its
            # junctions: 2e-5 of a 2.5 m wing is the group value, a 0.2 mm junction of a model
            # at a hundredth of that size is not merged away.
            smaller = min(self.extents[name] for name in group)
            return min(jn.GROUP_FUZZY_M, PAIR_FUZZY_FRACTION * smaller)
        return jn.GROUP_FUZZY_M

    def _remove(self):
        with self._active_faces():
            return super()._remove()

    def _legit(self, name, position, normal):
        cut, under = self._partners_of(name)
        return any(self._inside_source(other, position) for other in cut) or any(
            jn.ray_hits(position, normal, self.hulls[other]) for other in under
        )

    def _inside_kept(self):
        return sum(
            self._inside_source(op.dominant, self._interior(face)[0])
            for op in self._ops("cut")
            for face in self.faces[op.dominated]
        )

    def _cut_groups(self) -> list[list[str]]:
        groups: list[set[str]] = []
        for op in self._ops("cut"):
            pair = {op.dominant, op.dominated}
            touching = [g for g in groups if g & pair]
            merged = set(pair).union(*touching)
            groups = [g for g in groups if not g & pair] + [merged]
        order = list(self.sources)
        return [sorted(g, key=order.index) for g in groups]

    def _regroup(self, group: Sequence[str], inputs: Sequence[Face], rows) -> None:
        holder = {f: name for name in group for f in self.faces[name]}
        before = {f: self.origin[f] for f in inputs if f in self.origin}
        for name in group:
            mine = [(f, r) for f, r in zip(inputs, rows, strict=True) if holder[f] == name]
            self.faces[name] = self._inherit([f for f, _ in mine], [r for _, r in mine], before)

    def cut_pairs_separately(self) -> None:
        """Cut each row on its own pair, one fragment after another.

        This alternative to the group fragment is retained for comparing junction topology.
        :meth:`step` ``S10.E1`` measures whether sequential cuts leave extra curves where
        three pieces meet.
        """
        self.cut_done = True
        self.groups = self._cut_groups()
        for op in self._ops("cut"):
            a, b = list(self.faces[op.dominant]), list(self.faces[op.dominated])
            result = self.k.fragment(a, b)
            self._regroup([op.dominant, op.dominated], a + b, [*result.from_a, *result.from_b])

    def _gap_to_faces(self, edges: Sequence[int]) -> float:
        edge_map, worst = self._edge_map(), 0.0
        for e in edges:
            points = self._samples(e, 11)
            for face in edge_map.get(e, []):
                worst = max(worst, float(self._to_faces(points, [face]).max()))
        return worst

    def _join_cut(self):
        self._cut_all()
        joins = [op for op in self._ops("cut") if self._is_join(op)]
        if not joins:
            return skip(["join.crease_gap", "join.length"], "no body join among the rows"), {}
        gap, ratio, shortest, limit, where = 0.0, np.inf, 0.0, 0.0, {}
        for op in joins:
            edges = self._junction(op.dominant, op.dominated)
            if not edges:
                d = hull_contact(self.hulls[op.dominant], self.hulls[op.dominated])
                gap = max(gap, d)
            else:
                gap = max(gap, self._gap_to_faces(edges))
            length = sum(self._length(e) for e in edges)
            least = self._min_junction(op)
            if length / max(least, 1e-300) < ratio:
                ratio, shortest, limit = length / max(least, 1e-300), length, least
                where = {"pair": self._op_name(op), "length_m": length}
        return [
            jn.measured(
                "join.crease_gap",
                gap,
                jn.CREASE_GAP_M,
                unit="m",
                where=where,
                cause="STRATEGY: the bodies only touch; a crease that is not crossed is never cut",
            ),
            jn.measured(
                "join.length",
                shortest,
                limit,
                unit="m",
                bound="min",
                where=where,
                cause="STRATEGY: the join found no crease",
            ),
        ], {}

    def _cut(self):
        self._cut_all()
        ops = [op for op in self._ops("cut") if not self._is_join(op)]
        ids = [
            "junction.curve_gap",
            "junction.chain_ends",
            "junction.loop_closure",
            "junction.single_curve",
            "junction.shared_vertex",
            "junction.pcurve_on_face",
            "junction.length",
        ]
        if not ops:
            return skip(ids, "no cut row other than body joins"), {}
        m = {"gap": 0.0, "lonely": 0, "closure": 0.0, "dup": 0, "vertex": 0, "pcurve": 0.0}
        m["chain"] = {}
        ratio, shortest, limit, where, lengths = np.inf, 0.0, 0.0, {}, {}
        for op in ops:
            edges = self._junction(op.dominant, op.dominated)
            self._measure_junction(op, edges, m)
            length = sum(self._length(e) for e in edges)
            lengths[self._op_name(op)] = length
            least = self._min_junction(op)
            if length / max(least, 1e-300) < ratio:
                ratio, shortest, limit = length / max(least, 1e-300), length, least
                where = {"pair": self._op_name(op)}
        return [
            jn.measured(
                "junction.curve_gap",
                m["gap"],
                jn.JUNCTION_GAP_M,
                unit="m",
                cause="BUG: a junction curve is off one of its surfaces",
            ),
            jn.measured(
                "junction.chain_ends",
                m["lonely"],
                0,
                unit="ends",
                where=m["chain"],
                cause="BUG: a junction chain stops with no partner",
            ),
            jn.measured(
                "junction.loop_closure",
                m["closure"],
                jn.LOOP_CLOSURE_M,
                unit="m",
                cause="BUG: a rebuilt loop does not close",
            ),
            jn.measured(
                "junction.single_curve",
                m["dup"],
                0,
                unit="curves",
                cause="BUG: a junction is two curves, one per side",
            ),
            jn.measured(
                "junction.shared_vertex",
                m["vertex"],
                0,
                unit="ends",
                cause="BUG: faces meet at a junction end without a common vertex",
            ),
            jn.measured(
                "junction.pcurve_on_face",
                m["pcurve"],
                jn.JUNCTION_GAP_M,
                unit="m",
                cause="BUG: a junction's parameter-space curve leaves its 3D curve",
            ),
            jn.measured(
                "junction.length",
                shortest,
                limit,
                unit="m",
                bound="min",
                where=where,
                cause="STRATEGY: the pieces of a cut do not intersect",
            ),
        ], {"lengths_m": lengths}

    def _measure_junction(self, op: jn.Operation, edges: Sequence[int], m: dict) -> None:
        edge_map = self._edge_map()
        m["gap"] = max(m["gap"], self._gap_to_faces(edges))
        ends = [(s[0], s[-1]) for s in (self._samples(e, 3) for e in edges)]
        _, lonely = jn.chain(ends, jn.CHAIN_END_GAP_M)
        kept = [(c, side) for c, side in lonely if not self._on_plane(ends[c][side])]
        m["lonely"] += len(kept)
        if kept and not m["chain"]:  # the first chain that stops, for a record read later
            m["chain"] = {"pair": self._op_name(op), **jn.lonely_ends(ends, kept, edges)}
        faces = {f for e in edges for f in edge_map.get(e, [])}
        m["closure"] = max([m["closure"], *(self._loop_gap(f) for f in faces)])
        m["dup"] += self._duplicates(op)
        m["vertex"] += self._loose_ends(ends, edge_map, faces)
        m["pcurve"] = max([m["pcurve"], *(self._pcurve_gap(e, edge_map[e]) for e in edges)])

    def _loop_gap(self, face: Face) -> float:
        """Return the largest closure gap of the loops of ``face``, in loop order."""
        ends = []
        for signed in self.k.boundary(face):
            s = self._samples(abs(signed), 3)
            ends.append((s[0], s[-1]) if signed > 0 else (s[-1], s[0]))
        loops, current = [], [ends[0]] if ends else []
        for prev, nxt in zip(ends, ends[1:], strict=False):
            if np.linalg.norm(prev[1] - nxt[0]) > jn.CHAIN_END_GAP_M:
                loops.append(current)
                current = []
            current.append(nxt)
        if current:
            loops.append(current)
        return max((jn.loop_closure(loop) for loop in loops), default=0.0)

    def _duplicates(self, op: jn.Operation) -> int:
        """Count free edges of the dominated that lie on the dominant: a junction left double."""
        edge_map, holder = self._edge_map(), self._owner()
        dominant = self.faces[op.dominant]
        count = 0
        for e, faces in edge_map.items():
            if len(faces) != 1 or holder[faces[0]] != op.dominated:
                continue
            points = self._samples(e, 9)
            if not bbox_hit(points, self.hulls[op.dominant], jn.SEW_M):
                continue
            if float(self._to_faces(points, dominant).max()) <= jn.JUNCTION_GAP_M:
                count += 1
        return count

    def _loose_ends(self, ends, edge_map, faces) -> int:
        """Count junction ends where a free edge of the same faces ends: no common vertex."""
        free_ends = [
            p
            for e, fs in edge_map.items()
            if len(fs) == 1 and fs[0] in faces
            for p in (lambda s: (s[0], s[-1]))(self._samples(e, 3))
        ]
        return sum(
            1
            for pair in ends
            for p in pair
            if not self._on_plane(p)
            and any(np.linalg.norm(p - q) <= jn.CHAIN_END_GAP_M for q in free_ends)
        )

    def _pcurve_gap(self, edge: int, faces: Sequence[Face]) -> float:
        low, high = self.k.edge_bounds(edge)
        ts = np.linspace(low, high, 9)
        points = self.k.edge_values(edge, ts.tolist())
        worst = 0.0
        for face in faces:
            for t, p in zip(ts, points, strict=True):
                uv = self.k.reparametrize(edge, face, float(t))
                worst = max(worst, float(np.linalg.norm(np.subtract(self.k.value(face, uv), p))))
        return worst

    def _group(self):
        self._cut_all()
        groups = [g for g in self.groups if len(g) >= 3]
        ids = ["group.junction_length", "group.single_curve"]
        if not groups:
            return skip(ids, "no group of three pieces or more"), {}
        lengths, dup = {}, 0
        ratio, shortest, limit = np.inf, 0.0, 0.0
        for op in self._ops("cut"):
            if not any({op.dominant, op.dominated} <= set(g) for g in groups):
                continue
            edges = self._junction(op.dominant, op.dominated)
            length = sum(self._length(e) for e in edges)
            lengths[f"{op.dominant} x {op.dominated}"] = length
            least = self._min_junction(op)
            if length / max(least, 1e-300) < ratio:
                ratio, shortest, limit = length / max(least, 1e-300), length, least
            dup += self._duplicates(op)
        return [
            jn.measured(
                "group.junction_length",
                shortest,
                limit,
                unit="m",
                bound="min",
                where={"lengths_m": lengths},
                cause="STRATEGY: a pair of the group does not intersect",
            ),
            jn.measured(
                "group.single_curve",
                dup,
                0,
                unit="curves",
                cause="STRATEGY: pairs cut one after another leave a junction double; "
                "cut the group in one fragment",
            ),
        ], {"groups": groups}

    # ---- D5, D6: projected junctions and the hosts they split ----------------------------

    def _project(self):
        ops = self._ops("project")
        ids = ["junction.projected_length", "junction.offset", "junction.ends_on_cut"]
        if not ops:
            return skip(ids, "no projection row"), {}
        self._cut_all()
        rel, offset, lonely, where = 0.0, 0.0, 0, {}
        lengths = {}
        for op in ops:
            edges, samples, distances = self._seat(op)
            self.j_edges[op], self.j_samples[op] = edges, samples
            found = sum(jn.polyline_length(s) for s in samples)
            probe = sum(jn.polyline_length(c) for c in self._probe(op))
            lengths[self._op_name(op)] = {"projected_m": found, "probe_m": probe}
            r = abs(found - probe) / probe if probe > 0 else 1.0
            if r >= rel:
                rel, where = r, {"pair": self._op_name(op)}
            offset = max(offset, max(distances, default=0.0))
            lonely += self._open_j_ends(op, samples)
        tolerance = min(op.tolerance_m for op in ops)
        return [
            jn.measured(
                "junction.projected_length",
                rel,
                1e-3,
                unit="relative",
                where=where,
                cause="BUG: the projected curves are not the dominant's contact edges",
            ),
            jn.measured(
                "junction.offset",
                offset,
                tolerance,
                unit="m",
                cause="BUG: the dominant's contact edges are off the host",
            ),
            jn.measured(
                "junction.ends_on_cut",
                lonely,
                0,
                unit="ends",
                cause="STRATEGY: the projected curves do not close a footprint on the "
                "host with the cut's junction",
            ),
        ], {"lengths": lengths}

    def _seat(self, op: jn.Operation):
        """Return the dominant's free edges within :data:`SEAT_REACH_M` of the host."""
        edge_map, holder = self._edge_map(), self._owner()
        hosts = self.faces[op.dominated]
        edges, samples, distances = [], [], []
        for e, faces in edge_map.items():
            if len(faces) != 1 or holder[faces[0]] != op.dominant:
                continue
            points = self._samples(e)
            if not bbox_hit(points, self.hulls[op.dominated], SEAT_REACH_M):
                continue
            d = float(self._to_faces(points, hosts).max())
            if d <= SEAT_REACH_M:
                edges.append(e)
                samples.append(points)
                distances.append(d)
        return edges, samples, distances

    def _probe(self, op: jn.Operation) -> list[np.ndarray]:
        """Return the dominant's contact curves from its own patches, apart from the kernel."""
        hull = self.hulls[op.dominated]
        return [
            c
            for c in free_boundaries(self._source_faces(op.dominant), jn.SEW_M)
            if float(jn.distance_to_triangles(c, hull).max()) <= SEAT_REACH_M
        ]

    def _open_j_ends(self, op: jn.Operation, samples: Sequence[np.ndarray]) -> int:
        ends = [(s[0], s[-1]) for s in samples]
        _, lonely = jn.chain(ends, jn.CHAIN_END_GAP_M)
        edge_map, holder = self._edge_map(), self._owner()
        stops = [
            self._samples(e, 41)
            for e, faces in edge_map.items()
            if any(holder[f] == op.dominated for f in faces)
            and (len(faces) == 1 or {holder[f] for f in faces} - {op.dominated, op.dominant})
        ]
        count = 0
        for c, side in lonely:
            p = ends[c][side]
            if self._on_plane(p):
                continue
            if not any(jn.distance_to_polyline(p[None], s)[0] <= jn.J_END_ON_CUT_M for s in stops):
                count += 1
        return count

    def _split(self):
        with self._active_faces():
            return self._split_active()

    def _split_active(self):
        ops = self._ops("project")
        ids = ["junction.on_host_loops", "junction.split_closure"]
        if not ops:
            return skip(ids, "no projection row"), {}
        on_loop, closure = 0.0, 0.0
        for op in ops:
            faces = list(self.faces[op.dominated]) + list(self.faces[op.dominant])
            # an earlier imprint numbers the edges again, so the curves are found anew
            edges = self._seat(op)[0]
            result = self.k.imprint(faces, edges, fuzzy=op.tolerance_m)
            n = len(self.faces[op.dominated])
            before = {f: self.origin[f] for f in faces if f in self.origin}
            self.faces[op.dominated] = self._inherit(faces[:n], result.from_a[:n], before)
            self.faces[op.dominant] = self._inherit(faces[n:], result.from_a[n:], before)
            loops = self._host_loops(op)
            for s in self.j_samples.get(op, []):
                on_loop = max(on_loop, self._to_curves(s, loops))
            touched = [f for f in self.faces[op.dominated] if self._near_j(f, op)]
            closure = max([closure, *(self._loop_gap(f) for f in touched)])
        return [
            jn.measured(
                "junction.on_host_loops",
                on_loop,
                jn.J_ON_LOOP_M,
                unit="m",
                cause="BUG: a projected junction is on the host but not on its loops",
            ),
            jn.measured(
                "junction.split_closure",
                closure,
                jn.J_LOOP_CLOSURE_M,
                unit="m",
                cause="BUG: a split host face's loop does not close",
            ),
        ], {}

    def _near_j(self, face: Face, op: jn.Operation) -> bool:
        low, high = self.k.bbox(face)
        for s in self.j_samples.get(op, []):
            if np.all(s.min(axis=0) <= np.asarray(high) + SEAT_REACH_M) and np.all(
                s.max(axis=0) >= np.asarray(low) - SEAT_REACH_M
            ):
                return True
        return False


def assemble(
    kernel: CadKernel,
    pieces: Mapping[str, Sequence[Patch]],
    rows: Sequence[Mapping[str, Any]],
    **options: Any,
) -> list[dict[str, Any]]:
    """Run every sub-step of S10 on the pieces and return their records, in order.

    ``options`` are those of :class:`Assembly` (``out`` is required). The run stops after the
    first sub-step with a failed check, so the last record names where it stopped.
    """
    return Assembly(kernel, pieces, rows, **options).run()
