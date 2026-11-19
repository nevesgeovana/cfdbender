"""The S10 sub-steps after the cuts: removal, coverage, the joins, writing and sewing.

Pipeline role: the later half of the sub-steps of
:class:`cfdbender.geometry._geom_assembly.Assembly` (D8, D8b, C1 to C4, D9, D10, E2):
the inside removed region by region from each region's seed, the coverage of every source
patch in parameter space, the joins and the whole written, read back and sewn, and the
clearance of the pieces in no row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cfdbender.geometry._geom_parts import (
    AssemblyBase,
    hull_contact,
    skip,
    unexplained_edges,
)
from cfdbender.geometry.stages import Patch, evaluate_patch
from cfdbender.geometry.stages import junctions as jn
from cfdbender.mesh._cad import Face

__all__ = ["RemovalSteps", "largest_group"]


class RemovalSteps(AssemblyBase):
    """The sub-steps of S10 from the removal of the inside to the clearance."""

    # ---- D8, D8b: the inside removed, and the skin it leaves --------------------------------

    def _interior(self, face: Face) -> tuple[np.ndarray, np.ndarray]:
        """Return a point inside the trimmed face, nearest its center, and the normal there."""
        (u0, v0), (u1, v1) = self.k.uv_bounds(face)
        grid = [
            (u0 + (u1 - u0) * a, v0 + (v1 - v0) * b)
            for a in np.linspace(0.05, 0.95, 7)
            for b in np.linspace(0.05, 0.95, 7)
        ]
        inside = self.k.inside(face, grid)
        centre = np.asarray(self.k.centroid(face))
        candidates = [uv for uv, ok in zip(grid, inside, strict=True) if ok] or grid
        points = self.k.values(face, candidates)
        k = int(np.argmin(np.linalg.norm(points - centre, axis=1)))
        return points[k], np.asarray(self.k.normal(face, candidates[k]))

    def _partners_of(self, name: str) -> tuple[list[str], list[str]]:
        """Return the pieces ``name`` is cut against, and the dominants projected on it."""
        cut = [
            op.dominant if op.dominated == name else op.dominated
            for op in self._ops("cut")
            if name in (op.dominant, op.dominated)
        ]
        under = [op.dominant for op in self._ops("project") if op.dominated == name]
        return cut, under

    def _legit(self, name: str, point: np.ndarray, normal: np.ndarray) -> bool:
        """Whether a point of ``name`` lies inside a cut partner or under a projected dominant."""
        cut, under = self._partners_of(name)
        for other in cut:
            hull = self.hulls[other]
            if jn.dominant_inside(point[None], hull, hull, near=jn.SEW_M)[0]:
                return True
        return any(jn.ray_hits(point, normal, self.hulls[other]) for other in under)

    def _remove(self):
        edge_map, holder = self._edge_map(), self._owner()
        barriers = {e for e, fs in edge_map.items() if len({holder[f] for f in fs}) > 1}
        expected = {}
        for name, faces in self.faces.items():
            expected[name] = sum(
                self.k.area(f) for f in faces if self._legit(name, *self._interior(f))
            )
        removed, dropped, emptied = {}, 0, 0
        for name in list(self.faces):
            kept, gone = self._flood(name, edge_map, holder, barriers)
            widths = [self._width(f) for f in kept]
            kept, thin = jn.drop_zero_width(kept, widths)
            dropped += len(thin)
            removed[name] = sum(self.k.area(f) for f in gone + thin)
            if gone or thin:
                self.k.remove(gone + thin)
            self.faces[name] = kept
            emptied += 0 if kept else 1
        rel, where = 0.0, {"removed_m2": removed, "independent_m2": expected, "dropped": dropped}
        for name, area in removed.items():
            scale = max(area, expected[name], 1e-300)
            rel = max(rel, abs(area - expected[name]) / scale if scale > 1e-300 else 0.0)
        if dropped:
            self.warnings.append(
                f"{dropped} region(s) under {jn.ZERO_WIDTH_M} m mean width dropped"
            )
        return [
            jn.measured(
                "removal.seeds_reach",
                emptied,
                0,
                unit="pieces",
                cause="BUG: a piece kept no face; its seed is not on its exposed side",
            ),
            *self._kept_on_loops(),
            jn.measured(
                "removal.area",
                rel,
                1e-3,
                unit="relative",
                where=where,
                cause="BUG: the flood removed another area than lies inside or under the dominants",
            ),
            jn.measured(
                "junction.no_vertex_inside",
                self._inside_kept(),
                0,
                unit="faces",
                cause="BUG: a kept face of a dominated piece lies inside its dominant",
            ),
            jn.measured(
                "junction.no_triple_edge",
                self._triple(),
                0,
                unit="edges",
                cause="BUG: an edge of three faces is left",
            ),
        ], {"removed_m2": removed, "dropped": dropped}

    def _width(self, face: Face) -> float:
        perimeter = sum(self._length(abs(e)) for e in self.k.boundary(face))
        return jn.mean_width(self.k.area(face), perimeter)

    def _flood(self, name, edge_map, holder, barriers):
        faces = self.faces[name]
        neighbours: dict[Face, list[tuple[int, Face]]] = {f: [] for f in faces}
        mine_barriers = set()
        for e, fs in edge_map.items():
            mine = [f for f in fs if holder[f] == name]
            if e in barriers and mine:
                mine_barriers.add(e)
            for f in mine:
                neighbours[f] += [(e, g) for g in mine if g != f]
        if not mine_barriers:
            return list(faces), []
        barrier_points = np.concatenate([self._samples(e, 9) for e in mine_barriers])
        far = {
            f: float(
                jn.distance_to_polyline(barrier_points, np.asarray(self.k.centroid(f))[None]).min()
            )
            for f in faces
        }
        # The junctions cut the piece into regions; each region is kept or lost whole, as its
        # face farthest from the junctions says (its seed). A dominant can cut a dominated piece
        # in two exposed halves (a fin through a tail plane), so every region gets a seed.
        kept: set[Face] = set()
        seen: set[Face] = set()
        for face in sorted(faces, key=lambda f: -far[f]):
            if face in seen:
                continue
            region = jn.flood(neighbours, seeds=[face], barriers=barriers)
            seen |= region
            if not self._legit(name, *self._interior(face)):
                kept |= region
        return [f for f in faces if f in kept], [f for f in faces if f not in kept]

    def _kept_on_loops(self) -> list[dict[str, Any]]:
        ops = self._ops("project")
        if not ops:
            return skip(["removal.kept_on_loops"], "no projection row")
        worst = 0.0
        for op in ops:
            loops = self._host_loops(op)
            for s in self.j_samples.get(op, []):
                worst = max(worst, self._to_curves(s, loops))
        return [
            jn.measured(
                "removal.kept_on_loops",
                worst,
                jn.KEPT_ON_LOOP_M,
                unit="m",
                cause="BUG: the host kept the side under the dominant",
            )
        ]

    def _inside_kept(self) -> int:
        count = 0
        for op in self._ops("cut"):
            hull = self.hulls[op.dominant]
            for face in self.faces[op.dominated]:
                point, _ = self._interior(face)
                if jn.dominant_inside(point[None], hull, hull, near=jn.SEW_M)[0]:
                    count += 1
        return count

    def _triple(self) -> int:
        return sum(1 for fs in self._edge_map().values() if len(fs) > 2)

    def _coverage(self):
        worst, where, totals = 0.0, {}, {}
        for name, patches in self.sources.items():
            area, missing, largest = 0.0, 0.0, (0.0, -1)
            for index, patch in enumerate(patches):
                cells = self._cells(name, index, patch)
                area += cells["total"]
                missing += cells["missing"]
                if cells["largest"] > largest[0]:
                    largest = (cells["largest"], index)
            totals[name] = {"area_m2": area, "missing_m2": missing}
            fraction = largest[0] / area if area > 0 else 0.0
            if fraction >= worst:
                worst = fraction
                where = {"component": name, "patch": largest[1], "missing_m2": missing}
        return [
            jn.measured(
                "coverage.skin",
                worst,
                self.coverage_fraction,
                unit="fraction",
                where=where,
                cause="BUG: a part of a source skin is neither kept, "
                "inside a dominant nor under one",
            )
        ], {"pieces": totals}

    def _cells(self, name: str, index: int, patch: Patch, count: int = 24) -> dict[str, float]:
        """Sample one patch's domain: cell areas covered, removed with a reason, or missing."""
        g = evaluate_patch(patch, count + 1)
        a, b, c, d = g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]
        areas = 0.5 * (
            np.linalg.norm(np.cross(b - a, c - a), axis=-1)
            + np.linalg.norm(np.cross(c - a, d - a), axis=-1)
        )
        centres = (a + b + c + d) / 4.0
        normals = np.cross(c - a, d - b)
        su = np.linspace(0.5 / count, 1 - 0.5 / count, count)
        ku, kv = patch.knots_u, patch.knots_v
        uvs = [(ku[0] + (ku[-1] - ku[0]) * s, kv[0] + (kv[-1] - kv[0]) * t) for s in su for t in su]
        covered = np.zeros(count * count, bool)
        for face in (f for f in self.faces.get(name, []) if self.origin.get(f) == (name, index)):
            covered |= np.asarray(self.k.inside(face, uvs), bool)
        missing = np.zeros(count * count, bool)
        flat_c, flat_n = centres.reshape(-1, 3), normals.reshape(-1, 3)
        for k in np.flatnonzero(~covered):
            missing[k] = not self._legit(name, flat_c[k], flat_n[k])
        return {
            "total": float(areas.sum()),
            "missing": float(areas.reshape(-1)[missing].sum()),
            "largest": largest_group(missing.reshape(count, count), areas),
        }

    # ---- C1 to C4: the joins, written and sewn -------------------------------------------

    def _joins(self) -> list[jn.Operation]:
        return [op for op in self._ops("cut") if self._is_join(op)]

    def _shared(self):
        joins = self._joins()
        if not joins:
            return skip(["join.shared_pairs"], "no body join among the rows"), {}
        counts = {self._op_name(op): len(self._junction(op.dominant, op.dominated)) for op in joins}
        return [
            jn.measured(
                "join.shared_pairs",
                min(counts.values()),
                1,
                unit="curves",
                bound="min",
                where={"pairs": counts},
                cause="STRATEGY: the join shares no curve; the crease was never cut",
            )
        ], {}

    def _written(self):
        joins = self._joins()
        if not joins:
            return skip(["join.written_pairs"], "no body join among the rows"), {}
        worst, where = 0.0, {}
        for op in joins:
            path = self.out / f"S10-join-{op.dominant}-{op.dominated}.igs"
            self.k.export_iges(
                {n: self.faces[n] for n in (op.dominant, op.dominated) if self.faces[n]}, path
            )
            self.join_files[(op.dominant, op.dominated)] = path
            topology = self.k.inspect_file(path, None)
            curves = [(topology.names[f[0]], s) for s, f in topology.edges if len(f) == 1 and f]
            pairs = jn.pair_curves(curves, near=jn.SHARED_PAIR_NEAR_M, end_gap=jn.SHARED_PAIR_END_M)
            gap = max((p[2] for p in pairs), default=jn.SHARED_PAIR_NEAR_M)
            if gap >= worst:
                worst, where = gap, {"pair": self._op_name(op), "pairs": len(pairs)}
        return [
            jn.measured(
                "join.written_pairs",
                worst,
                jn.WRITTEN_PAIR_M,
                unit="m",
                where=where,
                cause="BUG: the two sides of a shared curve were written apart",
            )
        ], {}

    def _on_surface(self):
        joins = self._joins()
        if not joins:
            return skip(["join.edge_on_surface"], "no body join among the rows"), {}
        worst = 0.0
        for name in {n for op in joins for n in (op.dominant, op.dominated)}:
            for face in self.faces[name]:
                for e in self.k.boundary(face):
                    points = self._samples(abs(e), 9)
                    worst = max(worst, float(self._to_faces(points, [face]).max()))
        return [
            jn.measured(
                "join.edge_on_surface",
                worst,
                jn.EDGE_ON_SURFACE_M,
                unit="m",
                cause="BUG: a face's edge is off the face's own surface",
            )
        ], {}

    def _join_sewn(self):
        if not self.join_files:
            return skip(["join.group_free_edges"], "no body join among the rows"), {}
        worst, where = 0, {}
        for (a, b), path in self.join_files.items():
            others = {n: h for n, h in self.hulls.items() if n not in (a, b)}
            for tolerance in (jn.SEW_M, jn.SEW_FINE_M):
                free = [s for s, f in self.k.inspect_file(path, tolerance).edges if len(f) == 1]
                count = unexplained_edges(free, others, self.plane_axis)
                if count >= worst:
                    worst, where = count, {"join": f"{a} + {b}", "tolerance_m": tolerance}
        return [
            jn.measured(
                "join.group_free_edges",
                worst,
                0,
                unit="edges",
                where=where,
                cause="BUG: the join written in one file does not sew closed",
            )
        ], {}

    # ---- D9, D10, E2: the whole written, read back and sewn ------------------------------

    def _write(self):
        path = self.out / "S10-assembly.igs"
        groups = {n: f for n, f in self.faces.items() if f}
        self.k.export_iges(groups, path)
        topology = self.k.inspect_file(path, None)
        gap, rel, where = 0, 0.0, {}
        for name, faces in groups.items():
            read = [i for i, n in enumerate(topology.names) if n == name]
            area = sum(self.k.area(f) for f in faces)
            back = sum(topology.areas[i] for i in read)
            if abs(len(read) - len(faces)) >= gap:
                gap, where = abs(len(read) - len(faces)), {"component": name}
            rel = max(rel, abs(back - area) / area if area > 0 else 0.0)
        return [
            jn.measured(
                "write.read_back",
                gap,
                0,
                unit="faces",
                where=where,
                cause="BUG: the written file reads back another count of faces",
            ),
            jn.measured(
                "write.area",
                rel,
                1e-3,
                unit="relative",
                cause="BUG: a written face reads back with another area",
            ),
        ], {"iges": path.name, "faces": {n: len(f) for n, f in groups.items()}}

    def _sew(self):
        path = self.out / "S10-assembly.igs"
        new_free, triple, new_slivers, where = 0, 0, 0, {}
        for tolerance in (jn.SEW_M, jn.SEW_WIDE_M):
            topology = self.k.inspect_file(path, tolerance)
            free = [s for s, f in topology.edges if len(f) == 1]
            fresh = [s for s in free if not self._brought(s)]
            if len(fresh) >= new_free:
                new_free = len(fresh)
                where = {"tolerance_m": tolerance, "at": [s[0].tolist() for s in fresh[:4]]}
            triple = max(triple, sum(1 for _, f in topology.edges if len(f) > 2))
            widths = [
                jn.mean_width(a, p)
                for a, p in zip(topology.areas, topology.perimeters, strict=True)
            ]
            new_slivers = max(
                new_slivers, jn.introduced(len(jn.slivers(widths)), self.alone_slivers)
            )
        return [
            jn.measured(
                "sew.free_introduced",
                new_free,
                0,
                unit="edges",
                where=where,
                cause="BUG: the assembly opened an edge its pieces did not have",
            ),
            jn.measured(
                "sew.triple_edges",
                triple,
                0,
                unit="edges",
                cause="BUG: the written assembly sews an edge of three faces",
            ),
            jn.measured(
                "sew.slivers",
                new_slivers,
                0,
                unit="faces",
                cause="BUG: the assembly made a face under 5 mm wide",
            ),
        ], {}

    def _brought(self, samples: np.ndarray) -> bool:
        """Whether a free edge of the assembly is on the plane or was a free edge of a piece."""
        if self.plane_axis is not None and self._on_plane_curve(samples):
            return True
        return any(
            float(jn.distance_to_polyline(samples, s).max()) <= jn.OPENING_M
            for s in self.alone_free
        )

    def _clearance(self):
        """E2: the least distance from a piece in no row to any other piece.

        The floor is the sewing tolerance ``SEW_M``: a piece closer than that to another
        could be sewn to it when the file is read. Positive distance alone is insufficient.
        """
        joined = {n for op in self.plan for n in (op.dominant, op.dominated)}
        separate = [n for n in self.sources if n not in joined]
        if not separate:
            return skip(["assembly.clearance"], "every piece takes part in a row"), {}
        least, where = np.inf, {}
        for name in separate:
            for other in self.sources:
                if other == name:
                    continue
                d = hull_contact(self.hulls[name], self.hulls[other])
                if d < least:
                    least, where = d, {"pieces": f"{name} and {other}"}
        return [
            jn.measured(
                "assembly.clearance",
                least,
                jn.SEW_M,
                unit="m",
                bound="min",
                where=where,
                cause="STRATEGY: a piece in no row meets another piece or lies within the "
                "sewing tolerance of it",
            )
        ], {}


def largest_group(missing: np.ndarray, areas: np.ndarray) -> float:
    """Return the area of the largest group of missing cells joined by a side."""
    seen = np.zeros_like(missing)
    largest = 0.0
    rows, cols = missing.shape
    for i in range(rows):
        for j in range(cols):
            if not missing[i, j] or seen[i, j]:
                continue
            total, stack = 0.0, [(i, j)]
            seen[i, j] = True
            while stack:
                a, b = stack.pop()
                total += float(areas[a, b])
                for x, y in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if 0 <= x < rows and 0 <= y < cols and missing[x, y] and not seen[x, y]:
                        seen[x, y] = True
                        stack.append((x, y))
            largest = max(largest, total)
    return largest
