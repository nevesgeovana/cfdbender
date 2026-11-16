"""State and shared queries of the S10 assembly, and the pure helpers its sub-steps use.

Pipeline role: the base of :class:`cfdbender.geometry._geom_assembly.Assembly`: the
pieces as kernel faces, their origin patches and hulls, the driver that runs one sub-step
and records it, and the queries every sub-step asks (edges and the faces that share them,
junction curves of a pair, distances to faces and curves). The kernel is reached only
through :class:`~cfdbender.mesh._cad.CadKernel`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import GeometryError
from cfdbender.geometry.stages import (
    GeneratedPiece,
    Patch,
    check_entry,
    evaluate_patch,
    stage_row,
)
from cfdbender.geometry.stages import junctions as jn
from cfdbender.mesh._cad import CadKernel, Face

__all__ = ["AssemblyBase", "sewn_free_edges"]

#: The pieces whose join with each other is a body join.
BLUFF = ("body", "blister")
#: A dominant's free edge within 0.5 mm of a host is a candidate projected junction.
SEAT_REACH_M = 5e-4
EDGE_SAMPLES = 17


def plain(value: Any) -> Any:
    """Return ``value`` with numpy scalars and arrays made JSON values."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def skip(ids: Sequence[str], note: str) -> list[dict[str, Any]]:
    return [check_entry(cid, "skipped", note=note) for cid in ids]


def point(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def as_points(values: np.ndarray) -> list[tuple[float, float, float]]:
    return [point(v) for v in np.asarray(values, float).reshape(-1, 3)]


def add_patch(kernel: CadKernel, patch: Patch) -> Face:
    controls = np.asarray(patch.controls, float)
    net = [
        [point(controls[i, j]) for j in range(controls.shape[1])] for i in range(controls.shape[0])
    ]
    return kernel.add_bspline_surface(
        net,
        knots_u=list(patch.knots_u),
        knots_v=list(patch.knots_v),
        mult_u=list(patch.multiplicities_u),
        mult_v=list(patch.multiplicities_v),
        deg_u=patch.degree_u,
        deg_v=patch.degree_v,
    )


def hull(patches: Sequence[Patch], count: int = 9) -> np.ndarray:
    """Return the triangles of a piece's untrimmed patches, its hull for inside tests."""
    triangles = []
    for patch in patches:
        g = evaluate_patch(patch, count)
        a, b, c, d = g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]
        triangles.append(np.stack([a, b, c], axis=2).reshape(-1, 3, 3))
        triangles.append(np.stack([a, c, d], axis=2).reshape(-1, 3, 3))
    return np.concatenate(triangles) if triangles else np.zeros((0, 3, 3))


def free_boundaries(patches: Sequence[Patch], tolerance: float) -> list[np.ndarray]:
    """Return the boundary curves of a piece's patches that no other patch of it shares."""
    curves = []
    for patch in patches:
        g = evaluate_patch(patch, EDGE_SAMPLES)
        curves += [g[0, :], g[-1, :], g[:, 0], g[:, -1]]
    curves = [c for c in curves if jn.polyline_length(c) > tolerance]
    free = []
    for i, c in enumerate(curves):
        others = [d for j, d in enumerate(curves) if j != i]
        shared = any(
            min(
                max(np.linalg.norm(c[0] - d[0]), np.linalg.norm(c[-1] - d[-1])),
                max(np.linalg.norm(c[0] - d[-1]), np.linalg.norm(c[-1] - d[0])),
            )
            <= tolerance
            and jn.curve_distance(c, d) <= tolerance
            for d in others
        )
        # an edge that meets several shorter ones end to end (a T junction) is shared when
        # every sample of it lies on one of them
        if not shared and others:
            near = np.min([jn.distance_to_polyline(c, d) for d in others], axis=0)
            shared = bool(near.max() <= tolerance)
        if not shared:
            free.append(c)
    return free


def gap(found: Any, points: np.ndarray) -> np.ndarray:
    """Return the distance of each point to the point the kernel gave for it.

    A point the kernel could not give (NaN) is infinitely far, never 0: the built-in ``max`` of
    the sub-steps would drop a NaN and read a lost projection as a pass, and an infinite value is
    refused by the check, so the sub-step fails where it stands.
    """
    d = np.linalg.norm(np.asarray(found, float) - points, axis=1)
    return np.where(np.isnan(d), np.inf, d)


def bbox_hit(samples: np.ndarray, triangles: np.ndarray, margin: float) -> bool:
    """Whether any sample lies in the bounding box of the triangles, grown by ``margin``."""
    if not len(triangles):
        return False
    points = triangles.reshape(-1, 3)
    low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
    return bool(np.any(np.all((samples >= low) & (samples <= high), axis=1)))


def hull_contact(a: np.ndarray, b: np.ndarray) -> float:
    """Return 0 when two hulls cross, otherwise the least distance between them (sampled)."""
    pa, pb = np.unique(a.reshape(-1, 3), axis=0), np.unique(b.reshape(-1, 3), axis=0)
    inside_a = jn.parity_inside(pb, a)
    inside_b = jn.parity_inside(pa, b)
    if (inside_a.any() and not inside_a.all()) or (inside_b.any() and not inside_b.all()):
        return 0.0
    return float(min(jn.distance_to_triangles(pb, a).min(), jn.distance_to_triangles(pa, b).min()))


def sewn_free_edges(kernel: CadKernel, paths: Sequence[str | Path], tolerance: float) -> list:
    """Return the free edges of the files read in ONE import each and sewn at ``tolerance``.

    A list of files is read file by file; faces of two imports are never sewn together, which
    is why a join is always written to one file before it is sewn.
    """
    out = []
    for path in paths:
        topology = kernel.inspect_file(path, tolerance)
        out += [samples for samples, faces in topology.edges if len(faces) == 1]
    return out


def unexplained_edges(free, hulls: Mapping[str, np.ndarray], plane_axis: int | None) -> int:
    partners = [h for h in hulls.values() if len(h)]

    def distance(points: np.ndarray) -> np.ndarray:
        if not partners:
            return np.full(len(points), np.inf)
        return np.min([jn.distance_to_triangles(points, h) for h in partners], axis=0)

    def inside(points: np.ndarray) -> np.ndarray:
        if not partners:
            return np.zeros(len(points), bool)
        return np.any([jn.parity_inside(points, h) for h in partners], axis=0)

    classes = jn.classify_free_edges(free, distance=distance, inside=inside, plane_axis=plane_axis)
    return classes.count("unexplained")


class AssemblyBase:
    """The state of an assembly and the queries its sub-steps share."""

    STEPS: tuple[tuple[str, str], ...] = ()

    def __init__(
        self,
        kernel: CadKernel,
        pieces: Mapping[str, Sequence[Patch] | GeneratedPiece],
        rows: Sequence[Mapping[str, Any]],
        *,
        out: str | Path,
        members: Sequence[str] | None = None,
        kinds: Mapping[str, str] | None = None,
        plane_axis: int | None = None,
        expected_faces: Mapping[str, int] | None = None,
        regenerated: Mapping[str, Sequence[Patch]] | None = None,
        coverage_fraction: float = 1e-4,
        coverage_spacing: float = 1 / 200,
    ) -> None:
        self.k = kernel
        self.models = {name: p for name, p in pieces.items() if isinstance(p, GeneratedPiece)}
        self.sources = {
            name: list(p.bases.values()) if isinstance(p, GeneratedPiece) else list(p)
            for name, p in pieces.items()
        }
        self.kinds = dict(kinds or {})
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.warnings: list[str] = []
        self.plan = jn.dominance_plan(rows, list(members or pieces), self.warnings)
        self.plane_axis = plane_axis
        self.expected_faces = dict(expected_faces or {})
        self.regenerated = regenerated
        self.coverage_fraction = float(coverage_fraction)
        self.coverage_spacing = float(coverage_spacing)
        self.hulls = {
            name: hull(self.models[name].hull_patches or p) if name in self.models else hull(p)
            for name, p in self.sources.items()
        }
        #: each piece's largest extent, the size a relative junction limit is taken of
        self.extents = {
            name: float(np.ptp(h.reshape(-1, 3), axis=0).max()) if len(h) else 0.0
            for name, h in self.hulls.items()
        }
        self.faces: dict[str, list[Face]] = {}
        self.origin: dict[Face, tuple[str, int]] = {}
        self.records: list[dict[str, Any]] = []
        self.cut_done = False
        self.alone_free: list[np.ndarray] = []
        self.alone_slivers = 0
        self.alone_files: dict[str, Path] = {}
        self.join_files: dict[tuple[str, str], Path] = {}
        self.j_edges: dict[jn.Operation, list[int]] = {}
        self.j_samples: dict[jn.Operation, list[np.ndarray]] = {}
        self.groups: list[list[str]] = []
        self.input_free_edges: list[np.ndarray] = []
        self.input_sliver_faces: list[dict[str, Any]] = []
        self.source_areas: dict[str, float] = {}
        self.classification_hulls: dict[str, np.ndarray] = {}

    def export_state(self) -> dict[str, Any]:
        """Snapshot retained ownership and measured areas while the kernel is still open.

        Source identity belongs to generation, not to the requested registry version.
        The original free edges remain available for the export sewing comparison.
        """
        components = {}
        for name, faces in self.faces.items():
            model = self.models.get(name)
            bases = list(model.bases) if model else [p.family for p in self.sources[name]]
            components[name] = {
                "version": model.version if model else None,
                "source_surface_id": model.source_surface_id if model else None,
                "source_area_m2": self.source_areas.get(name),
                "free_edges": sum(n == 1 for n in self.edge_counts(faces).values()),
                "faces": [
                    {
                        "name": f"{name}_{i}",
                        "tag": f.tag,
                        "base": bases[self.origin[f][1]],
                        "area_m2": self.k.area(f),
                    }
                    for i, f in enumerate(faces)
                ],
            }
        return plain(
            {
                "components": components,
                "input_free_edges": self.alone_free or self.input_free_edges,
                "input_slivers": self.alone_slivers
                if self.alone_files
                else len(self.input_sliver_faces),
                "input_sliver_faces": self.input_sliver_faces,
                "plane_axis": self.plane_axis,
            }
        )

    def edge_counts(self, faces: Sequence[Face]) -> dict[int, int]:
        """Count face incidences for each edge, without combining separate pieces."""
        counts: dict[int, int] = {}
        for face in faces:
            for edge in self.k.boundary(face):
                key = abs(edge)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _inert(self, face: Face) -> bool:
        name, index = self.origin[face]
        model = self.models.get(name)
        return model is not None and list(model.bases)[index] in model.inert_bases

    def _source_faces(self, name):
        model = self.models.get(name)
        return (
            model.hull_patches if model is not None and model.hull_patches else self.sources[name]
        )

    def _inside_source(self, name, position):
        """Use the near-face rule locally and refined source parity away from it.

        Placement's nine-point hull can put a curved boundary on the wrong
        side of a narrow cut region. The 33-point classification hull refines
        that sampling; the independent removal-area check retains its limit.
        """
        points = np.asarray(position)[None]
        triangles = self.hulls[name]
        near = float(jn.distance_to_triangles(points, triangles)[0]) <= jn.SEW_M
        if name not in self.models or near:
            return bool(jn.dominant_inside(points, triangles, triangles, near=jn.SEW_M)[0])
        if name not in self.classification_hulls:
            self.classification_hulls[name] = hull(self._source_faces(name), count=33)
        return bool(jn.parity_inside(points, self.classification_hulls[name])[0])

    @contextmanager
    def _active_faces(self):
        """Keep moving and cove faces out of host operations, retaining their handles."""
        inert = {name: [f for f in faces if self._inert(f)] for name, faces in self.faces.items()}
        for name, faces in self.faces.items():
            self.faces[name] = [f for f in faces if f not in inert[name]]
        try:
            yield
        finally:
            for name, faces in inert.items():
                self.faces[name].extend(faces)

    # ---- driving --------------------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """Run every sub-step in order; stop after the first one with a failed check."""
        for stage, _ in self.STEPS:
            record = self.step(stage)
            if any(c["status"] == "failed" for c in record["checks"]):
                break
        return self.records

    def step(self, stage: str) -> dict[str, Any]:
        """Run one sub-step and return its record (checks, inputs, outputs, wall time).

        A sub-step that raises is recorded as failed at that sub-step (check ``stage.raised``,
        the error in its note), as the runner records a raising stage, so the failure keeps its
        place instead of ending the assembly as a refusal of its first sub-step.
        """
        method = dict(self.STEPS)[stage]
        before = len(self.warnings)
        clock = time.perf_counter()
        try:
            checks, outputs = getattr(self, f"_{method}")()
        except (GeometryError, ValueError, OSError) as error:
            note = f"class BUG: the sub-step raised: {error}"
            checks, outputs = [check_entry("stage.raised", "failed", 1, 0, note=note)], {}
        inputs = {"rows": [self._op_name(op) for op in self.plan], **outputs.pop("inputs", {})}
        outputs = {"inputs": inputs, **outputs}
        record = {
            "stage": stage,
            "name": stage_row(stage).name,
            "checks": checks,
            "outputs": plain(outputs),
            "warnings": self.warnings[before:],
            "wall_s": round(time.perf_counter() - clock, 3),
        }
        self.records.append(record)
        return record

    @staticmethod
    def _op_name(op: jn.Operation) -> str:
        return f"{op.operation} {op.dominant} > {op.dominated}"

    def _min_junction(self, op: jn.Operation) -> float:
        """Return the shortest junction the pieces of ``op`` may have, relative to their size."""
        return jn.min_junction(self.extents[op.dominant], self.extents[op.dominated])

    def _ops(self, operation: str) -> list[jn.Operation]:
        return [op for op in self.plan if op.operation == operation]

    def _is_join(self, op: jn.Operation) -> bool:
        return self.kinds.get(op.dominant) in BLUFF and self.kinds.get(op.dominated) in BLUFF

    def _owner(self) -> dict[Face, str]:
        return {f: name for name, faces in self.faces.items() for f in faces}

    def _edge_map(self) -> dict[int, list[Face]]:
        edges: dict[int, list[Face]] = {}
        for faces in self.faces.values():
            for face in faces:
                for e in self.k.boundary(face):
                    edges.setdefault(abs(e), []).append(face)
        return edges

    def _samples(self, edge: int, count: int = EDGE_SAMPLES) -> np.ndarray:
        low, high = self.k.edge_bounds(edge)
        return np.asarray(self.k.edge_values(edge, np.linspace(low, high, count).tolist()), float)

    def _length(self, edge: int) -> float:
        return jn.polyline_length(self._samples(edge, 41))

    def _to_faces(self, points: np.ndarray, faces: Sequence[Face]) -> np.ndarray:
        best = np.full(len(points), np.inf)
        for face in faces:
            xyz, _ = self.k.project(face, as_points(points))
            best = np.minimum(best, gap(xyz, points))
        return best

    def _on_plane(self, point: np.ndarray) -> bool:
        return self.plane_axis is not None and abs(point[self.plane_axis]) <= jn.SYMMETRY_M

    def _junction(self, a: str, b: str) -> list[int]:
        holder = self._owner()
        return [e for e, faces in self._edge_map().items() if {holder[f] for f in faces} >= {a, b}]

    def _on_plane_curve(self, curve: np.ndarray) -> bool:
        return bool(np.abs(curve[:, self.plane_axis]).max() <= jn.SYMMETRY_M)

    def _host_loops(self, op: jn.Operation) -> list[int]:
        return sorted({abs(e) for f in self.faces[op.dominated] for e in self.k.boundary(f)})

    def _to_curves(self, points: np.ndarray, edges: Sequence[int]) -> float:
        """Return the largest distance of the points to the nearest of the curves (exact)."""
        best = np.full(len(points), np.inf)
        low, high = points.min(axis=0) - SEAT_REACH_M, points.max(axis=0) + SEAT_REACH_M
        for e in edges:
            box_low, box_high = self.k.edge_bbox(e)
            if np.any(np.asarray(box_high) < low) or np.any(np.asarray(box_low) > high):
                continue
            projected = self.k.edge_project(e, as_points(points))
            best = np.minimum(best, gap(projected, points))
        # the box only skips work: a point no curve's box reached is measured against them all,
        # so a junction far from every loop reads its distance, never an infinity
        far = ~np.isfinite(best)
        if far.any():
            for e in edges:
                projected = self.k.edge_project(e, as_points(points[far]))
                best[far] = np.minimum(best[far], gap(projected, points[far]))
        return float(best.max()) if len(best) else 0.0
