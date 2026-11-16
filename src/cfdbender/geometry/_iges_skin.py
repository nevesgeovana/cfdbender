"""The outer skin of a gridded IGES assembly: the faces that lie inside it are removed.

Untrimmed components cross: a wing runs through the body to the plane of symmetry, a fillet lies
over the wing and the body, a fin root sinks into the body, a tailplane runs through the fin.
After the join every crossing is a curve more than two faces meet along, and the faces split
into regions there (faces of one component joined across edges only two faces use). A selection
of regions is kept when every edge keeps no face or two, every component keeps a region, and
every region it leaves out lies inside what it keeps. Of those selections the one keeping the
largest area is the outer skin.

Inside is measured by the generalized winding number of the kept faces, oriented outward, at the
centres of the region's deepest faces (those farthest from its edge, away from a tangent line
where a fillet starts on the skin it covers): a region is inside when the median is above one
half. The winding number of a surface with an opening, such as a lifting surface without its tip
cap, falls gently near the opening instead of flipping as a ray count would. A half model is
mirrored across the plane of symmetry the caller states, an axis plane through the origin; with
none stated (the default) nothing is mirrored and no symmetry is inferred from the geometry.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

__all__ = ["outer_skin"]

Faces = list[list[int]]
#: The deepest faces of a region whose centres are measured, at most.
_SAMPLES = 9
#: The selections tried at most; past it the assembly is kept whole and the record says so.
_SEARCH_LIMIT = 200_000
#: The axis a stated plane of symmetry is normal to.
_PLANE_AXIS = {"x": 0, "y": 1, "z": 2}


def _key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _edges(face: list[int]) -> Iterator[tuple[int, int]]:
    for i, a in enumerate(face):
        yield _key(a, face[(i + 1) % len(face)])


def _area_vector(verts: NDArray[np.float64], face: list[int]) -> NDArray[np.float64]:
    p = verts[face]
    return np.asarray(0.5 * np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0))


def _winding(
    points: NDArray[np.float64], tris: NDArray[np.float64], holder: NDArray[np.int64], regions: int
) -> NDArray[np.float64]:
    """Return the winding number of each region's triangles at each point (points x regions)."""
    out = np.empty((len(points), regions))
    for k, p in enumerate(points):
        a, b, c = tris[:, 0] - p, tris[:, 1] - p, tris[:, 2] - p
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        num = np.einsum("ij,ij->i", a, np.cross(b, c))
        den = (
            la * lb * lc
            + np.einsum("ij,ij->i", a, b) * lc
            + np.einsum("ij,ij->i", a, c) * lb
            + np.einsum("ij,ij->i", b, c) * la
        )
        angle = 2.0 * np.arctan2(num, den) / (4.0 * math.pi)
        out[k] = np.bincount(holder, weights=angle, minlength=regions)
    return out


class _Assembly:
    """The faces of every component, their edges and their regions."""

    def __init__(self, verts: NDArray[np.float64], families: dict[str, Faces]) -> None:
        self.verts = verts
        self.faces = [(name, list(f)) for name, fs in families.items() for f in fs]
        self.users: dict[tuple[int, int], list[int]] = {}
        for i, (_, face) in enumerate(self.faces):
            for e in _edges(face):
                self.users.setdefault(e, []).append(i)
        parent = list(range(len(self.faces)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for used in self.users.values():
            if len(used) == 2 and self.faces[used[0]][0] == self.faces[used[1]][0]:
                parent[find(used[0])] = find(used[1])
        roots: dict[int, int] = {}
        self.region = [roots.setdefault(find(i), len(roots)) for i in range(len(self.faces))]
        self.members: list[list[int]] = [[] for _ in roots]
        for i, r in enumerate(self.region):
            self.members[r].append(i)
        self.holder = [self.faces[m[0]][0] for m in self.members]
        self.area = [
            sum(float(np.linalg.norm(_area_vector(verts, self.faces[i][1]))) for i in m)
            for m in self.members
        ]

    def deepest(self, r: int) -> list[int]:
        """Return the faces of region ``r`` farthest, in steps across edges, from its edge."""
        members = set(self.members[r])
        depth: dict[int, int] = {}
        front = []
        for i in members:
            for e in _edges(self.faces[i][1]):
                inside = [j for j in self.users[e] if j in members]
                if len(inside) != 2 or len(self.users[e]) != 2:
                    depth[i] = 0
                    front.append(i)
                    break
        while front:
            nxt = []
            for i in front:
                for e in _edges(self.faces[i][1]):
                    for j in self.users[e]:
                        if j in members and j not in depth:
                            depth[j] = depth[i] + 1
                            nxt.append(j)
            front = nxt
        return sorted(members, key=lambda i: (-depth.get(i, 0), i))[:_SAMPLES]

    def _partner(self, i: int, edge: tuple[int, int]) -> int | None:
        """Return the face wound against ``i`` across ``edge``, or None.

        It is the other user of an edge two faces use or, where components cross, the other
        face of ``i``'s own component there.
        """
        used = self.users[edge]
        if len(used) != 2:
            used = [j for j in used if self.faces[j][0] == self.faces[i][0]]
        if len(used) != 2:
            return None
        return used[0] if used[1] == i else used[1]

    def oriented(self, mirror: int | None) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Every face as triangles wound alike across its partners and outward, with its region.

        Outward: the volume a piece encloses (by the divergence theorem about the origin, which
        lies on the plane of symmetry when there is one, so the missing cap adds nothing) is
        positive. A mirrored copy follows when there is a plane of symmetry.
        """
        loops = [list(f) for _, f in self.faces]
        seen = [False] * len(loops)
        tris: list[list[int]] = []
        holder: list[int] = []
        for start in range(len(loops)):
            if seen[start]:
                continue
            seen[start] = True
            piece, stack = [start], [start]
            while stack:
                i = stack.pop()
                for a, b in zip(loops[i], loops[i][1:] + loops[i][:1], strict=True):
                    j = self._partner(i, _key(a, b))
                    if j is None or seen[j]:
                        continue
                    if (a, b) in set(zip(loops[j], loops[j][1:] + loops[j][:1], strict=True)):
                        loops[j].reverse()
                    seen[j] = True
                    piece.append(j)
                    stack.append(j)
            volume = sum(
                float(self.verts[loops[i]].mean(axis=0) @ _area_vector(self.verts, loops[i]))
                for i in piece
            )
            for i in piece:
                loop = loops[i] if volume >= 0 else loops[i][::-1]
                tris += [[loop[0], loop[k], loop[k + 1]] for k in range(1, len(loop) - 1)]
                holder += [self.region[i]] * (len(loop) - 2)
        out = self.verts[np.array(tris, dtype=np.int64).reshape(-1, 3)]
        regions = np.array(holder, dtype=np.int64)
        if mirror is not None:
            flip = out.copy()
            flip[:, :, mirror] *= -1.0
            out = np.concatenate([out, flip[:, ::-1]])
            regions = np.concatenate([regions, regions])
        return out, regions


def _selections(
    variable: list[int], checks: list[tuple[int, ...]], fixed: set[int]
) -> Iterator[dict[int, int]]:
    """Yield the keep (1) or drop (0) choices of ``variable`` that leave every edge 0 or 2 faces."""
    position = {r: i for i, r in enumerate(variable)}
    # A check is the regions of the faces along one edge; it is decided with its last variable.
    due: dict[int, list[tuple[int, ...]]] = {}
    for check in checks:
        due.setdefault(max(position[r] for r in check if r in position), []).append(check)
    choice: dict[int, int] = {}
    tried = 0

    def kept(check: tuple[int, ...]) -> int:
        return sum(choice.get(r, 1) if r in position else int(r in fixed) for r in check)

    def walk(i: int) -> Iterator[dict[int, int]]:
        nonlocal tried
        tried += 1
        if tried > _SEARCH_LIMIT:
            raise OverflowError
        if i == len(variable):
            yield dict(choice)
            return
        for value in (1, 0):
            choice[variable[i]] = value
            if all(kept(c) in (0, 2) for c in due.get(i, [])):
                yield from walk(i + 1)
        del choice[variable[i]]

    yield from walk(0)


def outer_skin(
    verts: NDArray[np.float64], families: dict[str, Faces], *, symmetry: str = "none"
) -> tuple[dict[str, Faces], dict[str, dict[str, float]], str]:
    """Remove the faces that lie inside the assembly; return the families, what went, and why.

    The second value maps each component that lost faces to its ``faces`` and ``area`` removed;
    the third says in a few words what was done, for the record. ``symmetry`` is ``"none"`` or
    the axis (``"x"``, ``"y"``, ``"z"``) normal to the origin plane the model is mirrored across.
    """
    every = _Assembly(verts, families)
    crossing = [e for e, used in every.users.items() if len(used) > 2]
    if not crossing:
        return families, {}, "no crossing: nothing inside"
    variable = sorted({every.region[i] for e in crossing for i in every.users[e]})
    fixed = set(range(len(every.members))) - set(variable)
    checks = sorted(
        {
            tuple(every.region[i] for i in used)
            for used in every.users.values()
            if len(used) >= 2 and any(every.region[i] in variable for i in used)
        }
    )
    tris, holder = every.oriented(_PLANE_AXIS.get(symmetry))
    regions = len(every.members)
    # The winding number is a sum over faces: each region's share at each sample, once.
    share = {
        r: _winding(
            np.array([verts[every.faces[i][1]].mean(axis=0) for i in every.deepest(r)]),
            tris,
            holder,
            regions,
        )
        for r in variable
    }
    best: tuple[float, dict[int, int]] | None = None
    try:
        for choice in _selections(variable, checks, fixed):
            keep = fixed | {r for r, v in choice.items() if v}
            if {every.holder[r] for r in keep} != set(every.holder):
                continue
            area = sum(every.area[r] for r in keep)
            if best is not None and area <= best[0]:
                continue
            mask = np.zeros(regions)
            mask[sorted(keep)] = 1.0
            dropped = [r for r, v in choice.items() if not v]
            if all(float(np.median(share[r] @ mask)) > 0.5 for r in dropped):
                best = (area, choice)
    except OverflowError:
        return families, {}, f"kept whole: more than {_SEARCH_LIMIT} selections to try"
    if best is None:
        return families, {}, "kept whole: no selection leaves every edge to two faces"
    keep = fixed | {r for r, v in best[1].items() if v}
    out: dict[str, Faces] = {name: [] for name in families}
    removed: dict[str, dict[str, float]] = {}
    for i, (name, face) in enumerate(every.faces):
        if every.region[i] in keep:
            out[name].append(face)
        else:
            entry = removed.setdefault(name, {"faces": 0, "area": 0.0})
            entry["faces"] += 1
            entry["area"] += float(np.linalg.norm(_area_vector(verts, face)))
    return out, removed, "removed" if removed else "nothing inside"
