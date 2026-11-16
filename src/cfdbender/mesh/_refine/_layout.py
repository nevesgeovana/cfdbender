"""Plan and resample connected grid families as one layout."""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict, deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy

from cfdbender._errors import GeometryWarning, InputArtifactError, warn
from cfdbender.mesh._refine._blocks import (
    Blocks,
    Edge,
    count,
    kept_sides,
    recover_joined,
    report,
    resample,
    sheet_blocks,
    sheet_directions,
    te_directions,
)
from cfdbender.mesh._refine._config import FamilySpec
from cfdbender.mesh._refine._geometry import Faces, Points, edge_faces, order_like
from cfdbender.mesh._refine._grid import Grid, GridLevel, least_factors

Recovered = Grid | Blocks


@dataclass(frozen=True)
class ArcDirection:
    """The orientation of an arc and the family and reference that supplied it.

    Attributes
    ----------
    direction : str
        ``parallel`` or ``normal`` to the reference.
    source_family : str
        The family supplying the orientation, including a propagated seed.
    provenance : str
        ``listed_te``, ``longest_side``, ``sheet_index``,
        ``propagated_from:<family>``, ``own_orientation_conflict`` or
        ``own_orientation_unlinked``.
    """

    direction: str
    source_family: str
    provenance: str


@dataclass(frozen=True, eq=False)
class Layout:
    """Grid families resampled as one multiblock layout.

    Attributes
    ----------
    names : tuple of str
        Its families, in the source's family order.
    kinds : tuple of str
        Each family's own layout: ``"sheet"`` or ``"multiblock"``.
    blocks : Blocks
        The layout of their faces taken together, in the order of ``names``.
    holder : tuple of int
        Per patch, the index in ``names`` of the family holding it.
    counts : tuple of int
        Per arc, its new interval count.
    kept : dict
        Per family, the patch sides its factor cannot coarsen.
    directions : mapping of int to mapping of int to ArcDirection
        Per patch and arc, the orientation and its source, including conflicting patch references.
    band_leads : mapping of int to (str, str)
        Per band, the family whose count was selected and its local direction.
    labels : mapping of int to str
        Per arc, its band's stable parallel (P) or normal (N) label.
    leads : mapping of str to (str, int)
        Per label, the lead family and the arc that supplied its selected count.
    governing : mapping of str to str
        The selected factor source in each direction.
    continuity : mapping of str to dict
        Per keep family changed by a band, its directions, labels and causing families.
    unchanged : tuple of str
        Keep families whose every arc retains its source count; copied without resampling.
    """

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    blocks: Blocks
    holder: tuple[int, ...]
    counts: tuple[int, ...]
    kept: Mapping[str, list[dict[str, object]]]
    directions: Mapping[int, Mapping[int, ArcDirection]] = field(default_factory=dict)
    band_leads: dict[int, tuple[str, str]] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)
    leads: dict[str, tuple[str, int]] = field(default_factory=dict)
    governing: Mapping[str, str] = field(default_factory=dict)
    continuity: Mapping[str, dict[str, object]] = field(default_factory=dict)
    unchanged: tuple[str, ...] = ()
    #: Component factor sources for each member, where component governing is active.
    governing_by_family: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def patches_of(self, name: str) -> list[int]:
        """Return the patches of one family, in layout order."""
        i = self.names.index(name)
        return [p for p, o in enumerate(self.holder) if o == i]


@dataclass
class _Planner:
    """Shared inputs for grouping, orientation and counting in one planning call."""

    verts: Points
    families: Mapping[str, Faces]
    grids: dict[str, Recovered]
    specs: dict[str, FamilySpec]
    te_edges: Mapping[str, Collection[Edge]]
    where: str
    followers: dict[str, str]
    assumed: Collection[str]
    prepare: bool = False
    failures: dict[str, str] = field(default_factory=dict)
    component_leads: dict[str, dict[str, str]] = field(default_factory=dict)

    def govern_components(self, components, result):
        """Select component governors from geometric side totals in source order."""
        if not components:
            return
        totals = self.component_totals()
        original = dict(self.specs)
        for component, members in components.items():
            # A kept member (no table, or method keep) is copied unchanged: it neither
            # governs nor follows, so it never replaces the factors the others declare.
            names = [
                n
                for n in self.families
                if n in members and n in self.grids and original[n].method != "keep"
            ]
            if not names:
                continue
            leads = {d: max(names, key=lambda n: totals[n, d]) for d in ("parallel", "normal")}
            result[component] = leads
            for name in names:
                self.component_leads[name] = leads
                self.warn_overridden(name, component, leads, original)
                normal, parallel = self.component_factors(name, leads, original)
                self.specs[name] = replace(
                    original[name],
                    chordwise=normal,
                    spanwise=parallel,
                    governing=False,
                    governing_normal=name == leads["normal"],
                    governing_parallel=name == leads["parallel"],
                    method="grid" if name in leads.values() else "grid_auto",
                )
                if name not in leads.values():
                    self.followers[name] = leads["parallel"]

    def component_totals(self):
        """Measure every directional side, including the closing edge of a tube ring."""
        totals = Counter()
        for name, grid in self.grids.items():
            if isinstance(grid, Grid):
                pts = self.verts[grid.ids]
                totals[name, "parallel"] = sum(
                    float(numpy.linalg.norm(numpy.diff(pts[:, i], axis=0), axis=1).sum())
                    for i in (0, -1)
                )
                if grid.wrap:
                    pts = numpy.concatenate((pts, pts[:, :1]), axis=1)
                totals[name, "normal"] = sum(
                    float(numpy.linalg.norm(numpy.diff(pts[i], axis=0), axis=1).sum())
                    for i in (0, -1)
                )
            else:
                told = self.arc_directions(
                    grid, range(len(grid.patches)), name, runs=None, source=None
                )
                for directions in told.values():
                    for arc, direction in directions.items():
                        totals[name, direction.direction] += float(
                            numpy.linalg.norm(
                                numpy.diff(self.verts[list(grid.arcs[arc])], axis=0), axis=1
                            ).sum()
                        )
        return totals

    def warn_overridden(self, name, component, leads, original):
        """Warn when a member's own stated factors give way to its component governors'."""
        own = original[name]
        if own.unchanged:
            return
        given = (
            original[leads["normal"]].directions[0],
            original[leads["parallel"]].directions[1],
        )
        if own.directions == given:
            return
        warn(
            f"{self.where} family {name}: its stated normal and parallel factors "
            f"{own.directions[0]:g} and {own.directions[1]:g} are replaced by "
            f"{given[0]:g} and {given[1]:g} from the governors of component {component} "
            f"({leads['normal']} normal, {leads['parallel']} parallel); state the factors "
            "on the governors, or refine without --keep-within-component to keep them",
            GeometryWarning,
            stacklevel=5,
        )

    def component_factors(self, name, leads, original):
        """Keep inherited tube factors above the same minimum as directly stated factors."""
        factors = (
            original[leads["normal"]].directions[0],
            original[leads["parallel"]].directions[1],
        )
        grid = self.grids[name]
        if not isinstance(grid, Grid) or not grid.wrap:
            return factors
        least = least_factors(grid, *factors)
        if least != factors:
            warn(
                f"{self.where} family {name}: inherited normal and parallel factors "
                f"{factors[0]:g} and {factors[1]:g} are below the least the tube allows "
                f"and are raised to {least[0]:g} and {least[1]:g}",
                GeometryWarning,
                stacklevel=5,
            )
        return least

    def counted(self, names, found, *, prepare=False) -> Layout:
        """Count one recovered layout and retain its directions and governing decisions."""
        verts, grids, specs = self.verts, self.grids, self.specs
        where, followers = self.where, self.followers
        blocks, holder = found
        lengths = {
            a: float(numpy.linalg.norm(numpy.diff(verts[list(arc)], axis=0), axis=1).sum())
            for a, arc in enumerate(blocks.arcs)
        }
        perimeter = _perimeters(lengths, names, blocks, holder)
        governing = [n for n in names if any(specs[n].leads)]
        declared = [n for n in names if specs[n].method not in ("grid_auto", "keep")] or list(names)
        leaders = {
            d: _band_lead(declared, names, specs, perimeter, direction=d)
            for d in ("normal", "parallel")
        }
        for name in names:
            if specs[name].method == "grid_auto" and governing and name not in self.component_leads:
                lead = _band_lead(governing, names, specs, perimeter)
                specs[name] = FamilySpec(
                    chordwise=specs[leaders["normal"]].directions[0],
                    spanwise=specs[leaders["parallel"]].directions[1],
                    method="grid_auto",
                )
                followers[name] = lead
        proposed: dict[int, list[tuple[str, int, int]]] = {}
        kept: dict[str, list[dict[str, object]]] = {}
        seeds = (
            leaders
            if any(specs[n].method == "keep" for n in names)
            else {d: n for d, n in leaders.items() if specs[n].leads[d == "parallel"]}
        )
        runs, provenance = self.follower_directions(names, found, seeds)
        directions = {}
        for i, name in enumerate(names):
            mine = [p for p, o in enumerate(holder) if o == i]
            directions.update(
                self.arc_directions(
                    blocks,
                    mine,
                    name,
                    runs=runs.get(name),
                    source=provenance.get(name, self.failures.get(name)),
                )
            )
            kept[name] = []
            for number, p in enumerate(mine, 1):
                factors = {
                    a: specs[name].directions[d.direction == "parallel"]
                    for a, d in directions[p].items()
                }
                for arc, f in factors.items():
                    proposed.setdefault(arc, []).append(
                        (name, p, count(len(blocks.arcs[arc]) - 1, f))
                    )
                kept[name].extend(
                    dict(side, patch=number)
                    for side in kept_sides(blocks, verts, factors, patches=[p])
                )
        kinds = tuple(grids[n].layout for n in names)
        component_sources = [self.component_leads[n] for n in names if n in self.component_leads]
        if component_sources and all(s == component_sources[0] for s in component_sources):
            leaders = component_sources[0]
        layout = Layout(
            names,
            kinds,
            blocks,
            holder,
            (),
            kept,
            directions,
            governing=leaders,
            governing_by_family={
                n: self.component_leads[n] for n in names if n in self.component_leads
            },
        )
        if prepare:
            self.prepare_leads(layout, lengths)
            return self.counted(names, found)
        failed = (
            {
                n: s
                for n, s in provenance.items()
                if isinstance(s, str) and n not in self.component_leads
            }
            if self.prepare
            else {}
        )
        if failed:
            self.failures.update(failed)
            for n in failed:
                specs[n] = replace(specs[n], method="grid")
                followers.pop(n, None)
            return self.counted(names, found)
        counts = _band_counts(layout, proposed, specs, perimeter, where=where)
        return _continuity(
            replace(layout, counts=tuple(counts[a] for a in range(len(blocks.arcs)))), specs
        )

    def prepare_leads(self, layout, lengths):
        """Mark the longest total parallel and normal sides, with source-order ties."""
        names = tuple(n for n in layout.names if n not in self.component_leads)
        if not names:
            return
        totals = Counter()
        for p, told in layout.directions.items():
            for a, d in told.items():
                name = layout.names[layout.holder[p]]
                if name in names:
                    totals[name, d.direction] += lengths[a]
        leads = {d: max(names, key=lambda n: totals[n, d]) for d in ("parallel", "normal")}
        for n in names:
            self.specs[n] = replace(
                self.specs[n],
                governing_parallel=n == leads["parallel"],
                governing_normal=n == leads["normal"],
                method="grid" if n in leads.values() else "grid_auto",
            )

    def arc_directions(self, blocks, patches, name, *, runs, source):
        """Orient this family's patches using the planner's geometry and reference."""
        verts, grid, spec = self.verts, self.grids[name], self.specs[name]
        edges, where, assumed = self.te_edges[name], self.where, name in self.assumed
        local = {}
        if runs is None:
            # Equal factors did not request geometric warnings before directions were
            # retained. Recording their orientation must not add those diagnostics.
            with warnings.catch_warnings():
                if spec.directions[0] == spec.directions[1] and not any(spec.leads):
                    warnings.simplefilter("ignore", GeometryWarning)
                runs = _reference_directions(
                    verts,
                    blocks,
                    patches,
                    grid,
                    edges,
                    family=f"{where} family {name}",
                    patch_runs=local,
                )
        provenance = (
            "listed_te" if edges else "sheet_index" if isinstance(grid, Grid) else "longest_side"
        )
        if assumed:
            provenance = "longest_side"
        if isinstance(source, str):
            provenance = "own_orientation_" + source.removeprefix("own reference: ")
        if not local:
            local = {p: {a: runs[a] for a, _ in blocks.sides[p]} for p in patches}
        return {
            p: {
                a: ArcDirection(
                    "parallel" if d == "spanwise" else "normal",
                    source[a] if isinstance(source, dict) else name,
                    f"propagated_from:{source[a]}" if isinstance(source, dict) else provenance,
                )
                for a, d in told.items()
            }
            for p, told in local.items()
        }

    def follower_directions(self, names, found, leaders):
        """Propagate each directional leader through the selected followers."""
        if any(n in self.component_leads for n in names):
            runs, provenance = {}, {}
            groups = {}
            for name in names:
                sources = self.component_leads.get(name, leaders)
                groups.setdefault(tuple(sources.items()), []).append(name)
            for entries, selected in groups.items():
                sources = dict(entries)
                targets = [n for n in selected if n not in sources.values()]
                local = {d: n for d, n in sources.items() if n in names}
                reached, reasons = self._propagated(names, found, local or sources, targets)
                runs.update(reached)
                provenance.update(reasons)
            return runs, provenance
        selected = [n for n in names if n in self.followers or self.specs[n].method == "keep"]
        return self._propagated(names, found, leaders, selected)

    def _propagated(self, names, found, leaders, selected):
        """Propagate the given seeds to the selected component or layout followers."""
        verts, grids, followers = self.verts, self.grids, self.followers
        te_edges, where = self.te_edges, self.where
        blocks, indices = found
        holder = tuple(names[i] for i in indices)
        runs, provenance = {}, {}
        if not selected:
            return runs, provenance
        evidence = {}
        for lead in dict.fromkeys(leaders.values() if leaders else followers.values()):
            seed = {}
            if lead in names:
                patches = [p for p, n in enumerate(holder) if n == lead]
                grid, edges, label = grids[lead], te_edges[lead], f"{where} family {lead}"
                seed = _reference_directions(verts, blocks, patches, grid, edges, family=label)
            evidence[lead] = propagate_directions(blocks, holder, seed, selected)
        for name in selected:
            arcs = {a for p, n in enumerate(holder) if n == name for a, _ in blocks.sides[p]}
            failures = [failed[name] for _, failed in evidence.values() if name in failed]
            values = {a: {told[a] for told, _ in evidence.values() if a in told} for a in arcs}
            if failures or any(len(ds) != 1 for ds in values.values()):
                conflict = "conflict" in failures or any(len(ds) > 1 for ds in values.values())
                reason = "conflict" if conflict else "unlinked"
                warn_direction_fallback(name, ", ".join(evidence), reason, where=where)
                provenance[name] = f"own reference: {reason}"
            else:
                runs[name] = {a: next(iter(ds)) for a, ds in values.items()}
                provenance[name] = {
                    a: leaders.get(
                        "parallel" if d == "spanwise" else "normal",
                        next(iter(leaders.values()), followers.get(name, name)),
                    )
                    for a, d in runs[name].items()
                }
        return runs, provenance

    def grouped(self, groups, *, accept):
        """Plan each connected group with this planner's recovery and counting context."""
        families, grids, specs = self.families, self.grids, self.specs
        te_edges, where = self.te_edges, self.where
        layouts: list[Layout] = []
        why: dict[str, str] = {}
        for group in groups:
            if all(specs[n].method == "keep" for n in group):
                continue
            if not any(any(specs[n].leads) or n in self.component_leads for n in group):
                for name in group:
                    if specs[name].method == "grid_auto":
                        _skip_unled(grids, specs, name, where)
                group = [n for n in group if n in grids]
            if len(group) < 2 and not any(n in self.component_leads for n in group):
                continue
            te = {v for n in group for e in te_edges[n] for v in e}
            found = self.joined(group, te, {e for n in group for e in te_edges[n]})
            if isinstance(found, str):
                why.update(
                    dict.fromkeys(group, f"; the grids {', '.join(group)} form no layout: {found}")
                )
                continue
            try:
                layouts.append(self.counted(tuple(group), found, prepare=self.prepare))
            except InputArtifactError as error:
                error.where = {
                    "families": list(group),
                    "nodes": sorted({v for n in group for f in families[n] for v in f}),
                    **error.where,
                }
                if not accept and not any(specs[n].method == "keep" for n in group):
                    raise
                why.update(dict.fromkeys(group, _skipped_layout(error, group, where)))
        return layouts, why

    def joined(self, group, te, lines=None):
        """Recover a connected layout, including a component's isolated sheet.

        ``lines`` are the group's reference edges, the only edges traced as
        trailing edges across the joined families.
        """
        grid = self.grids[group[0]]
        if len(group) == 1 and isinstance(grid, Grid) and grid.layout == "sheet":
            return sheet_blocks(grid, self.families[group[0]], te), (0,)
        return _join(self.verts, self.families, group, te, lines)


def _joinable(grid: Recovered, faces: Faces) -> bool:
    """Return whether a grid family can join a layout: quadrilaterals as a sheet or multiblock."""
    sheet = isinstance(grid, Grid) and grid.layout == "sheet"
    return (sheet or isinstance(grid, Blocks)) and all(len(f) == 4 for f in faces)


def _touching(names: Sequence[str], nodes: Mapping[str, set[int]]) -> list[list[str]]:
    """Return the groups of families joined by shared nodes, each in the order of ``names``."""
    groups: list[list[str]] = []
    for name in names:
        hit = [g for g in groups if any(nodes[name] & nodes[m] for m in g)]
        merged = sorted([name, *(m for g in hit for m in g)], key=names.index)
        groups = [g for g in groups if g not in hit] + [merged]
    return sorted(groups, key=lambda g: names.index(g[0]))


def _join(
    verts: Points,
    families: Mapping[str, Faces],
    group: Sequence[str],
    te: Collection[int],
    lines: Collection[Edge] | None = None,
) -> tuple[Blocks, tuple[int, ...]] | str:
    """Return the layout of a group of families taken as one and each patch's family, or why not."""
    faces = [f for n in group for f in families[n]]
    family = numpy.repeat(numpy.arange(len(group)), [len(families[n]) for n in group])
    joins = {
        e for e, fs in edge_faces(faces).items() if len(fs) == 2 and family[fs[0]] != family[fs[1]]
    }
    found = recover_joined(verts, faces, te, joins, lines)
    if isinstance(found, str):
        return found
    return found, tuple(int(family[int(cells[0, 0])]) for cells in found.cells)


def _direction_links(blocks: Blocks, holder: Sequence[str], followers: Collection[str]):
    """Join follower sides with equal or opposite direction constraints."""
    links: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for p, name in enumerate(holder):
        if name not in followers:
            continue
        arcs = [a for a, _ in blocks.sides[p]]
        for i, a in enumerate(arcs):
            for j, b in enumerate(arcs):
                links[a].append((b, (i < 2) != (j < 2)))
    return links


def propagate_directions(
    blocks: Blocks, holder: Sequence[str], seed: Mapping[int, str], followers: Collection[str]
) -> tuple[dict[int, str], dict[str, str]]:
    """Propagate seeded arc directions through follower patches, reporting conflict or unlinked.

    Opposite sides retain a direction and adjacent sides switch it. Only follower
    patches transmit directions; a corner or a remeshed family cannot transmit one.
    Both directions are retained during traversal so a conflict cannot be hidden
    by patch order. Failed families must use their own geometric reference.
    """
    links = _direction_links(blocks, holder, followers)
    reached: dict[int, set[bool]] = defaultdict(set)
    queue = deque((a, d == "spanwise") for a, d in seed.items())
    while queue:
        arc, direction = queue.popleft()
        if direction in reached[arc]:
            continue
        reached[arc].add(direction)
        queue.extend((other, direction != flip) for other, flip in links.get(arc, ()))
    failures = {}
    for name in followers:
        arcs = {a for p, n in enumerate(holder) if n == name for a, _ in blocks.sides[p]}
        if any(len(reached[a]) == 2 for a in arcs):
            failures[name] = "conflict"
        elif not arcs or any(not reached[a] for a in arcs):
            failures[name] = "unlinked"
    return {
        a: "spanwise" if True in values else "chordwise"
        for a, values in reached.items()
        if len(values) == 1
    }, failures


def _reference_directions(verts, blocks, patches, grid, te_edges, *, family, patch_runs=None):
    """Read a family's directions using the existing geometric reference path."""
    arcs = {a for p in patches for a, _ in blocks.sides[p]}
    if isinstance(grid, Grid):
        return sheet_directions(grid, blocks, arcs)
    te = numpy.asarray(verts, dtype=float)[sorted(te_edges)].reshape(-1, 2, 3)
    local = {} if patch_runs is None else patch_runs
    runs = te_directions(
        verts, blocks, patches, te, family=family, what="its patches", patch_runs=local
    )
    return {
        a: d for a, d in runs.items() if len({told[a] for told in local.values() if a in told}) <= 1
    }


def warn_direction_fallback(name: str, lead: str, reason: str, *, where: str) -> None:
    """Warn that a follower's topology cannot inherit its governing directions."""
    warn(
        f"{where} family {name}: not a follower of {lead} in directions: {reason}; "
        "using its own reference. Declare its own grid table to set independent factors.",
        GeometryWarning,
        stacklevel=4,
    )


def _perimeters(
    lengths: Mapping[int, float], names: Sequence[str], blocks: Blocks, holder: Sequence[int]
) -> dict[str, float]:
    """Return each family's perimeter in the layout: the arcs one of its patches holds."""
    out = {}
    for i, name in enumerate(names):
        held = Counter(a for p, o in enumerate(holder) if o == i for a, _ in blocks.sides[p])
        out[name] = sum(lengths[a] for a, n in held.items() if n == 1)
    return out


def arc_bands(blocks: Blocks, arcs: Iterable[int]) -> dict[int, int]:
    """Return each arc's band: arcs joined as opposite sides of a patch hold one count."""
    parent = {a: a for a in arcs}

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for sides in blocks.sides:
        ring = [arc for arc, _ in sides]
        for x, y in ((ring[0], ring[1]), (ring[2], ring[3])):
            if x in parent and y in parent:
                parent[root(x)] = root(y)
    return {a: root(a) for a in parent}


def _lead_rank(name, names, specs, perimeter, direction):
    """Rank a band's eligible directional governor, then declared perimeter and source order."""
    governing = (
        any(specs[name].leads) if direction is None else specs[name].leads[direction == "parallel"]
    )
    return (
        not governing,
        specs[name].method == "keep",
        specs[name].method == "grid_auto",
        -perimeter[name],
        names.index(name),
    )


def _band_lead(candidates, names, specs, perimeter, *, direction=None):
    """Select a governing or declared family by perimeter, then source order."""
    return min(candidates, key=lambda n: _lead_rank(n, names, specs, perimeter, direction))


def _band_counts(
    layout: Layout,
    proposed: Mapping[int, list[tuple[str, int, int]]],
    specs: Mapping[str, FamilySpec],
    perimeter: Mapping[str, float],
    *,
    where: str,
) -> dict[int, int]:
    """Return one count per arc: each band's count from the family that leads it, warned."""
    names = layout.names
    band = arc_bands(layout.blocks, proposed)
    members: dict[int, list[tuple[str, int, str, int]]] = {}
    for arc, offers in proposed.items():
        members.setdefault(band[arc], []).extend(
            (n, c, layout.directions[p][arc].direction, arc) for n, p, c in offers
        )
    chosen: dict[int, int] = {}
    dominant: dict[int, int] = {}
    followed: dict[tuple[str, str], int] = {}
    for key, candidates in members.items():
        lead, value, direction, dominant[key] = min(
            candidates, key=lambda row: _lead_rank(row[0], names, specs, perimeter, row[2])
        )
        chosen[key] = value
        layout.band_leads[key] = (lead, direction)
        for n in {n for n, c, _, _ in candidates if c != value}:
            followed[(n, lead)] = followed.get((n, lead), 0) + 1
    _label_bands(layout, band, dominant)
    for (n, lead), bands in sorted(followed.items()):
        how = "governing" if any(specs[lead].leads) else "declared, of the largest perimeter"
        advice = (
            f'To give this {specs[n].method} family its own counts, declare method = "grid", '
            "its factors and governing = true in its [families] table."
            if specs[n].method in ("grid_auto", "keep")
            else "Mark the family that should lead with governing = true in its [families] table."
        )
        warn(
            f"{where} family {n} follows {lead} ({how}) on {bands} band(s) of the curves they "
            f"share: those curves take {lead}'s counts. {advice}",
            GeometryWarning,
            stacklevel=5,
        )
    return {arc: chosen[band[arc]] for arc in proposed}


def _label_bands(layout, bands, dominant):
    """Assign labels by the selected lead's source order, then its count-setting arc."""
    serial = {"parallel": 0, "normal": 0}
    ordered = sorted(
        layout.band_leads, key=lambda b: (layout.names.index(layout.band_leads[b][0]), dominant[b])
    )
    for band in ordered:
        lead, direction = layout.band_leads[band]
        serial[direction] += 1
        label = ("P" if direction == "parallel" else "N") + str(serial[direction])
        layout.leads[label] = (lead, dominant[band])
        layout.labels.update({a: label for a, b in bands.items() if b == band})


def _continuity(layout, specs):
    """Retain the minimum changes and untouched keep families from the resolved counts."""
    changed, untouched = {}, []
    for name in layout.names:
        if specs[name].method != "keep":
            continue
        arcs = {
            a: d.direction
            for p in layout.patches_of(name)
            for a, d in layout.directions[p].items()
            if layout.counts[a] != len(layout.blocks.arcs[a]) - 1
        }
        if not arcs:
            untouched.append(name)
            continue
        labels = [label for label in layout.leads if label in {layout.labels[a] for a in arcs}]
        causes = {layout.leads[label][0] for label in labels}
        changed[name] = {
            "directions": [d for d in ("parallel", "normal") if d in arcs.values()],
            "bands": labels,
            "because": [n for n in layout.names if n in causes and n != name],
        }
    return replace(layout, continuity=changed, unchanged=tuple(untouched))


def _warn_kept(layout: Layout, specs: Mapping[str, FamilySpec], where: str) -> None:
    """Warn once per family naming each patch side its factor cannot coarsen."""
    for name in layout.names:
        kept = layout.kept[name]
        if not kept:
            continue
        chordwise, spanwise = specs[name].directions
        label = (
            f"factor {chordwise:g}"
            if chordwise == spanwise
            else f"factors normal {chordwise:g} and parallel {spanwise:g}"
        )
        lines = "; ".join(
            f"patch {k['patch']} {k['side']}: {k['source_intervals']} source interval(s), "
            f"{k['intervals']} written, centroid {k['centroid']}, "
            f"box {k['bbox_min']} to {k['bbox_max']}"
            for k in kept
        )
        what = f"{len(layout.patches_of(name))} patches of {', '.join(layout.names)}"
        if len(layout.names) == 1:
            what = layout.blocks.describe()
        warn(
            f"{where} family {name}: the {label} cannot coarsen {len(kept)} patch side(s) of "
            f"{what}; they are kept and the rest of the family is resampled: {lines}",
            GeometryWarning,
            stacklevel=6,  # past _warn_kept, plan, _grid_after_grid, _decide and refine_mesh
        )


def plan(
    verts: Points,
    families: Mapping[str, Faces],
    grids: dict[str, Recovered],
    specs: dict[str, FamilySpec],
    te_edges: Mapping[str, Collection[Edge]],
    *,
    where: str,
    accept: bool = False,
    followers: dict[str, str] | None = None,
    assumed: Collection[str] = (),
    prepare: bool = False,
    components: Mapping[str, Sequence[str]] | None = None,
    component_governors: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, Layout], list[tuple[str, str, str]]]:
    """Group the grid families that share nodes into layouts, before any work.

    Under ``accept`` (``accept_remesh``) a layout that is refused (two
    families counting a shared curve apart, a patch whose directions cannot
    be told apart) is warned and left to the rule of the first grid: the
    first of its families in the source's order is refined as a grid alone
    and the others are remeshed; a lone multiblock family refused alone is
    remeshed, listed with its own name as the grid it follows.

    Parameters
    ----------
    verts : numpy.ndarray
        All the mesh's vertices, shape (V, 3).
    families : mapping of str to faces
        Every family of the mesh, in the source's order.
    grids : mapping of str to Grid or Blocks
        The families recovered as grids.
    specs : mapping of str to FamilySpec
        Working specs, updated with each layout's governing factors for its followers.
    te_edges : mapping of str to edges
        The trailing-edge edges of each family.
    where : str
        The source, for a refusal or a warning.
    accept : bool
        Leave a refused layout to the rule of the first grid, with a
        warning, instead of refusing it (``accept_remesh``).
    followers : mapping of str to str, optional
        Working mapping populated with each grid_auto family's governing family.
    assumed : collection of str
        Families whose reference edges were assumed from their longest boundary side.
    prepare : bool
        Select each group's directional leads by total side length, then retain
        only followers whose every arc can propagate with the final declarations.
    components : mapping, optional
        Components whose grids share directional governors; absent keeps layout governing.
    component_governors : dict, optional
        Output mapping populated with the selected component governors.

    Returns
    -------
    dict, list
        The layout of each family resampled in one (every member of a group,
        and each lone multiblock family), and, per grid left to the rule of
        the first grid, its name, the grid refined first that it shares
        nodes with, and why it joined no layout ("" when it could not).

    Raises
    ------
    InputArtifactError
        Two families giving a shared curve different counts; nothing was written.
    """
    names = tuple(n for n in families if n in grids)
    nodes = {n: {v for f in families[n] for v in f} for n in names}
    joinable = [n for n in names if _joinable(grids[n], families[n])]
    planner = _Planner(
        verts,
        families,
        grids,
        specs,
        te_edges,
        where,
        {} if followers is None else followers,
        assumed,
        prepare,
    )
    planner.govern_components(
        components or {}, {} if component_governors is None else component_governors
    )
    layouts, why = planner.grouped(_touching(joinable, nodes), accept=accept)
    done = [n for lay in layouts for n in lay.names]
    late: list[tuple[str, str, str]] = []
    for name in sorted(names, key=lambda n: specs.get(n, FamilySpec()).method == "keep"):
        if name not in grids or name in done:
            continue
        if specs[name].method == "grid_auto" and name not in planner.component_leads:
            _skip_unled(grids, specs, name, where)
            continue
        first = [m for m in names if m in done and nodes[name] & nodes[m]]
        if first:
            late.append((name, first[0], why.get(name, "")))
            continue
        if specs[name].method == "keep":
            continue
        done.append(name)
        grid = grids[name]
        if isinstance(grid, Blocks):
            lone = (grid, (0,) * len(grid.patches))
            try:
                layouts.append(planner.counted((name,), lone))
            except InputArtifactError as error:
                error.where = {"families": [name], "nodes": sorted(nodes[name]), **error.where}
                if not accept:
                    raise
                done.remove(name)
                late.append((name, name, _skipped_layout(error, [name], where)))
    for layout in layouts:
        _warn_kept(layout, specs, where)
    return {n: lay for lay in layouts for n in lay.names}, late


def _skip_unled(grids, specs, name, where):
    """Keep a grid_auto family unchanged when its layout has no governing family."""
    warn(
        f"{where} family {name}: grid_auto follows a governing family and the "
        "refinement file marks none in its grid-connected layout (governing = true); "
        "it is kept unchanged",
        GeometryWarning,
        stacklevel=4,
    )
    del grids[name], specs[name]


def _skipped_layout(error: InputArtifactError, group: Sequence[str], where: str) -> str:
    """Warn that a refused layout is left to the first-grid rule and return the reason."""
    text = str(error).removesuffix(". Nothing was written.").removeprefix(f"{where} ")
    names = ", ".join(group)
    warn(
        f"{where}: {text}; the first of {names} in the source's order is refined as a grid "
        "alone; other families follow their remesh or continuity-only rule",
        GeometryWarning,
        stacklevel=5,  # past _skipped_layout, plan, _grid_after_grid and _decide
    )
    return f"; {text}"


def refine_layout(
    verts: Points, families: Mapping[str, Faces], layout: Layout
) -> dict[str, GridLevel]:
    """Resample a layout and write each new cell back to its family.

    Every family's level holds the layout's points; its faces are its
    patches' new faces in the order of its source faces. The new trailing-edge
    mid-points are the first family's.
    """
    verts = numpy.asarray(verts, dtype=float)
    faces = [f for n in layout.names for f in families[n]]
    copied = {p for name in layout.unchanged for p in layout.patches_of(name)}
    done = resample(verts, faces, layout.blocks, layout.counts, keep=copied)
    out: dict[str, GridLevel] = {}
    for i, name in enumerate(layout.names):
        mine = layout.patches_of(name)
        built = [f for p in mine for f in done.cells[p]]
        ordered = (
            [
                f
                for _, f in sorted(
                    (int(source), f)
                    for p in mine
                    for source, f in zip(layout.blocks.cells[p].flat, done.cells[p], strict=True)
                )
            ]
            if name in layout.unchanged
            else order_like(done.points, built, verts, families[name])
        )
        entry = report(layout.blocks, done, mine, (len(families[name]), len(ordered)))
        if len(layout.names) > 1:
            entry["layout"] = layout.kinds[i]
            entry["grouped_with"] = [n for n in layout.names if n != name]
        te = done.te_midpoints if i == 0 else numpy.zeros((0, 3))
        out[name] = GridLevel(done.points, ordered, te, entry)
    return out
