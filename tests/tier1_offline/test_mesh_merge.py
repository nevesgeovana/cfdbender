"""``cfd-mesh merge``, written from.

Two parts of one synthetic closed body, a capped tube cut at x = 0, are joined
along the cut. Every check reads the written files, by the OBJ's own text, so a
node is the kept one when the output's face names the kept line. The check of
the interface is scored against an unmerged control, and the check of the band
against a node moved off the surface.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy
import pytest

from cfdbender._errors import (
    GeometryWarning,
    InputArtifactError,
    InputValueError,
    NotApplicableWarning,
)
from cfdbender.mesh import cli
from cfdbender.mesh._merge import merge_objs

TAU = 2.0 * math.pi
KEPT_STATIONS = tuple(numpy.linspace(-1.0, 0.0, 5).tolist())
ADAPTED_STATIONS = tuple(numpy.linspace(0.0, 1.0, 9).tolist())


def _tube(
    count: int,
    stations: tuple[float, ...],
    *,
    cap: int | None = None,
    sweep: float = TAU,
    shift=None,
    family: str = "Tube",
) -> tuple[list[list[float]], list[list[int]], str]:
    """Return a tube's nodes and faces: quads between stations, triangles closing a cap."""
    closed = math.isclose(sweep, TAU)
    around = count if closed else count + 1
    angles = [sweep * j / count for j in range(around)]
    nodes = [
        [x + (shift(i, t) if shift else 0.0), math.cos(t), math.sin(t)]
        for i, x in enumerate(stations)
        for t in angles
    ]

    def at(i: int, j: int) -> int:
        return i * around + j % around

    faces = [
        [at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)]
        for i in range(len(stations) - 1)
        for j in range(count)
    ]
    if cap is not None:
        centre = len(nodes)
        nodes.append([stations[cap], 0.0, 0.0])
        for j in range(count):
            fan = [at(cap, j), at(cap, j + 1), centre]
            faces.append(fan if cap else fan[::-1])
    return nodes, faces, family


def _kept(count: int, **options) -> tuple[list[list[float]], list[list[int]], str]:
    """Return the kept part: x from -1 to the cut, its far end capped unless an arc is asked."""
    options.setdefault("family", "Kept")
    cap = None if "sweep" in options else 0
    return _tube(count, KEPT_STATIONS, cap=cap, **options)


def _adapted(count: int, **options) -> tuple[list[list[float]], list[list[int]], str]:
    """Return the adapted part: x from the cut to 1, its far end capped unless an arc is asked."""
    options.setdefault("family", "Adapted")
    stations = options.pop("stations", ADAPTED_STATIONS)
    cap = None if "sweep" in options else len(stations) - 1
    return _tube(count, stations, cap=cap, **options)


def _write(path: Path, part, *, digits: int = 9) -> Path:
    """Write a part as an OBJ with a fixed number of digits, so its text is not the writer's."""
    nodes, faces, family = part
    lines = [f"v {x:.{digits}f} {y:.{digits}f} {z:.{digits}f}" for x, y, z in nodes]
    lines.append(f"g {family}")
    lines += ["f " + " ".join(str(v + 1) for v in f) for f in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _Obj:
    """An OBJ as text: its vertex lines and, per group, its faces as vertex numbers."""

    def __init__(self, path: Path) -> None:
        self.nodes: list[str] = []
        self.groups: dict[str, list[list[int]]] = {}
        group = "default"
        for line in path.read_text(encoding="utf-8").splitlines():
            word, _, rest = line.partition(" ")
            if word == "v":
                self.nodes.append(rest)
            elif word == "g":
                group = rest
                self.groups.setdefault(group, [])
            elif word == "f":
                self.groups.setdefault(group, []).append([int(v) - 1 for v in rest.split()])

    @property
    def points(self) -> numpy.ndarray:
        return numpy.array([[float(w) for w in n.split()] for n in self.nodes])

    @property
    def faces(self) -> list[list[int]]:
        return [f for faces in self.groups.values() for f in faces]

    def edges(self) -> Counter[tuple[int, int]]:
        """Return how many faces use each undirected edge."""
        seen: Counter[tuple[int, int]] = Counter()
        for f in self.faces:
            for a, b in zip(f, f[1:] + f[:1], strict=True):
                seen[(min(a, b), max(a, b))] += 1
        return seen

    def face_text(self, faces: list[list[int]]) -> list[tuple[str, ...]]:
        """Return faces as the text of their nodes, comparable across two files."""
        return [tuple(self.nodes[v] for v in f) for f in faces]


def _open_edges(obj: _Obj) -> list[tuple[int, int]]:
    return [edge for edge, used in obj.edges().items() if used == 1]


def _non_manifold(obj: _Obj) -> list[tuple[int, int]]:
    return [edge for edge, used in obj.edges().items() if used > 2]


def _seam_is_closed(obj: _Obj) -> bool:
    """Return whether no open edge lies on the plane of the cut and no node is doubled on it."""
    points = obj.points
    on_cut = abs(points[:, 0]) < 1e-9
    open_on_cut = [e for e in _open_edges(obj) if on_cut[e[0]] and on_cut[e[1]]]
    return not open_on_cut


def _merge(tmp_path: Path, kept, adapted, threshold: float, **options):
    """Merge two parts written beside each other; return the record, the OBJ and the parts."""
    first = _write(tmp_path / "kept.obj", kept)
    second = _write(tmp_path / "adapted.obj", adapted)
    out = tmp_path / "out.obj"
    merged = merge_objs(first, second, threshold=threshold, adapt=second, out=out, **options)
    return merged, _Obj(out), _Obj(first), _Obj(second)


def _band_of(adapted: _Obj, chain: set[int]) -> list[int]:
    """Return the faces within two face layers of the chain's nodes, independently of the code."""
    faces = adapted.faces
    first = {i for i, f in enumerate(faces) if chain & set(f)}
    reached = chain | {v for i in first for v in faces[i]}
    return sorted(first | {i for i, f in enumerate(faces) if reached & set(f)})


def _cut_nodes(obj: _Obj) -> set[int]:
    return {i for i, x in enumerate(obj.points[:, 0]) if abs(x) < 1e-9}


def _reach(a: numpy.ndarray, ends: numpy.ndarray, others: numpy.ndarray) -> numpy.ndarray:
    """Return each point's distance to the nearest segment, from the requirement's words."""
    best = numpy.full(len(a), numpy.inf)
    for p, q in zip(ends, others, strict=True):
        d = q - p
        t = numpy.clip((a - p) @ d / max(float(d @ d), 1e-300), 0.0, 1.0)
        best = numpy.minimum(best, numpy.linalg.norm(p + t[:, None] * d - a, axis=1))
    return best


def _distance_to_surface(points: numpy.ndarray, obj: _Obj) -> numpy.ndarray:
    """Return each point's distance to the surface of an OBJ, its polygons fanned."""
    import trimesh

    triangles = [[f[0], f[i], f[i + 1]] for f in obj.faces for i in range(1, len(f) - 1)]
    mesh = trimesh.Trimesh(obj.points, numpy.array(triangles), process=False)
    return trimesh.proximity.closest_point(mesh, points)[1]


def _quiet_merge(*args, **options):
    """Merge and return the merge with its warnings' messages."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _merge(*args, **options)
    return result, [str(w.message) for w in caught]


def test_mesh_merge_merge_control_the_seam_check_says_no_on_an_unmerged_mesh(tmp_path):
    """The derangement: the two parts in one file, not merged, must fail the checks used below."""
    kept = _write(tmp_path / "kept.obj", _kept(16))
    adapted = _write(tmp_path / "adapted.obj", _adapted(29))
    one, two = _Obj(kept), _Obj(adapted)
    count = len(one.nodes)
    lines = [f"v {n}" for n in one.nodes + two.nodes]
    lines += ["g K"] + ["f " + " ".join(str(v + 1) for v in f) for f in one.faces]
    lines += ["g A"] + ["f " + " ".join(str(v + count + 1) for v in f) for f in two.faces]
    unmerged = tmp_path / "unmerged.obj"
    unmerged.write_text("\n".join(lines) + "\n", encoding="utf-8")
    obj = _Obj(unmerged)
    assert len(_open_edges(obj)) == 16 + 29
    assert not _seam_is_closed(obj)


def test_mesh_merge_merge_one_node_per_node_moves_nodes_and_keeps_every_face(tmp_path):
    """Equal counts within the threshold: only node moves, quads stay quads, no band."""
    shift = lambda i, t: 0.001 * math.cos(3.0 * t) if i == 0 else 0.0  # noqa: E731
    (merged, out, kept, adapted), messages = _quiet_merge(
        tmp_path, _kept(24), _adapted(24, shift=shift), 0.005
    )
    assert merged.moved_nodes == 24
    assert merged.band_faces == 0
    assert merged.triangles_created == 0
    assert not _open_edges(out)
    assert not _non_manifold(out)
    assert Counter(len(f) for f in out.faces) == Counter(len(f) for f in kept.faces + adapted.faces)
    assert len(out.nodes) == len(kept.nodes) + len(adapted.nodes) - 24
    # The same faces in the same order: a face on the cut names the kept node at that place.
    count = len(kept.nodes)
    faces = out.groups["Adapted"]
    assert len(faces) == len(adapted.faces)
    for source, written in zip(adapted.faces, faces, strict=True):
        for v, w in zip(source, written, strict=True):
            if v < 24:
                assert w < count
                assert numpy.linalg.norm(out.points[w] - adapted.points[v]) <= 0.005
            else:
                assert out.nodes[w] == adapted.nodes[v]
    assert not [m for m in messages if "open edge" in m or "degenerate" in m]


@pytest.mark.parametrize(
    ("kept_count", "adapted_count"), [(16, 32), (32, 16), (24, 29), (29, 24)], ids=str
)
def test_mesh_merge_merge_a_closed_body_cut_with_different_counts_closes_cleanly(
    tmp_path, kept_count, adapted_count
):
    """The adapted side finer or coarser, a closed loop: a closed body with nothing left open."""
    (merged, out, kept, _), messages = _quiet_merge(
        tmp_path, _kept(kept_count), _adapted(adapted_count), 0.05
    )
    assert merged.interface_edges == kept_count
    assert merged.open_edges_left == 0
    assert merged.facing_edges_left == 0
    assert not _open_edges(out)
    assert not _non_manifold(out)
    assert merged.audit is not None
    assert not merged.audit.degenerate_faces
    assert merged.audit.passed
    assert not messages or all("rebuilt" in m for m in messages)
    # Every node of the output on the plane of the cut is a line of the kept file.
    on_cut = _cut_nodes(out)
    assert on_cut
    assert all(v < len(kept.nodes) for v in on_cut)


@pytest.mark.parametrize(
    ("kept_count", "adapted_count"), [(16, 23), (23, 16)], ids=["finer", "coarser"]
)
def test_mesh_merge_merge_an_open_arc_ends_at_each_parts_other_boundary(
    tmp_path, kept_count, adapted_count
):
    """A half tube: the arc closes, the boundary it ends at stays open and is not warned."""
    (merged, out, kept, adapted), messages = _quiet_merge(
        tmp_path,
        _kept(kept_count, sweep=math.pi),
        _adapted(adapted_count, sweep=math.pi),
        0.05,
    )
    assert merged.interface_edges == kept_count
    assert _seam_is_closed(out)
    assert not _non_manifold(out)
    # What stays open is each part's own boundary away from the cut; the kept one is exact.
    count = len(kept.nodes)
    own = len(_open_edges(kept)) - kept_count
    assert len([e for e in _open_edges(out) if e[1] < count]) == own
    assert len(_open_edges(out)) == merged.open_edges_left
    assert merged.open_edges_left >= own + len(_open_edges(adapted)) - adapted_count - 4
    assert merged.facing_edges_left == 0
    assert not [m for m in messages if "open edge" in m]


def test_mesh_merge_merge_only_boundaries_within_the_threshold_become_the_interface(tmp_path):
    """A far open boundary of the adapted part stays as it was; so does the kept one's."""
    kept = _tube(24, KEPT_STATIONS, family="Kept")
    adapted = _tube(29, ADAPTED_STATIONS, family="Adapted")
    (merged, out, _, source), _ = _quiet_merge(tmp_path, kept, adapted, 0.05)
    assert merged.interface_edges == 24
    assert merged.open_edges_left == 24 + 29
    far = [e for e in _open_edges(out) if abs(out.points[e[0], 0]) > 0.5]
    assert len(far) == 24 + 29
    assert _seam_is_closed(out)
    written = set(out.nodes)
    end = [v for v, x in enumerate(source.points[:, 0]) if x > 0.99]
    assert all(source.nodes[v] in written for v in end)


def test_mesh_merge_merge_a_gap_partly_beyond_the_threshold_stays_open_and_warns_the_count(
    tmp_path,
):
    """Half the cut within the threshold, half beyond: that half stays open, counted."""
    threshold = 0.02
    shift = lambda i, t: 0.06 if i == 0 and math.cos(t) > 0.1 else 0.0  # noqa: E731
    kept_part, adapted_part = _kept(24), _adapted(29, shift=shift)
    (merged, out, kept, adapted), messages = _quiet_merge(
        tmp_path, kept_part, adapted_part, threshold
    )
    # The expected counts from the requirement's words, over the two sources.
    k_edges, a_edges = _open_edges(kept), _open_edges(adapted)

    def within(edges, at, other_edges, other):
        ends = other[[e[0] for e in other_edges]]
        others = other[[e[1] for e in other_edges]]
        near = _reach(at, ends, others) <= threshold
        return [e for e in edges if near[e[0]] and near[e[1]]]

    k_near = within(k_edges, kept.points, a_edges, adapted.points)
    assert 0 < len(k_near) < len(k_edges)
    assert merged.interface_edges == len(k_near)
    count = len(kept.nodes)
    k_left = [e for e in _open_edges(out) if e[1] < count]
    assert len(k_left) == len(k_edges) - len(k_near)
    assert not _non_manifold(out)
    # The adapted side of what stays open is still on the adapted part's open boundary.
    ends = adapted.points[[e[0] for e in a_edges]]
    others = adapted.points[[e[1] for e in a_edges]]
    own = sorted({v for e in _open_edges(out) for v in e if v >= count})
    assert own
    assert float(_reach(out.points[own], ends, others).max()) < 1e-6
    left = merged.open_edges_left
    assert left == len(_open_edges(out)) > len(k_left)
    assert merged.facing_edges_left == left
    assert [m for m in messages if f"{left} open edge(s)" in m and "threshold" in m]
    record = json.loads((tmp_path / "out.merge.json").read_text(encoding="utf-8"))
    assert record["open_edges_left"] == left


def test_mesh_merge_merge_no_interface_warns_writes_nothing_and_exits_non_zero(tmp_path, capsys):
    """A threshold finding no interface: a warning, no file, a non-zero exit; a control merges."""
    first = _write(tmp_path / "kept.obj", _kept(24))
    second = _write(tmp_path / "adapted.obj", _adapted(29, stations=(0.5, 0.75, 1.0)))
    out = tmp_path / "out.obj"
    with pytest.warns(GeometryWarning, match="nothing was written"):
        code = cli.main(
            [
                "merge",
                str(first),
                str(second),
                "--threshold",
                "0.05",
                "--adapt",
                str(second),
                "--out",
                str(out),
            ]
        )
    assert code != 0
    assert not out.exists()
    assert not list(tmp_path.glob("out.*"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        code = cli.main(
            [
                "merge",
                str(first),
                str(second),
                "--threshold",
                "0.6",
                "--adapt",
                str(second),
                "--out",
                str(out),
            ]
        )
    assert code == 0, capsys.readouterr()
    assert out.exists()


def test_mesh_merge_merge_the_command_writes_the_obj_its_record_inventory_and_audit(
    tmp_path, capsys
):
    """The verb's form: kept first, adapted after, the record's fields, the console's statement."""
    first = _write(tmp_path / "wing.obj", _kept(16))
    second = _write(tmp_path / "nacelle.obj", _adapted(23))
    out = tmp_path / "joined.obj"
    code = cli.main(
        [
            "merge",
            str(first),
            str(second),
            "--threshold",
            "0.05",
            "--adapt",
            str(second),
            "--out",
            str(out),
        ]
    )
    seen = capsys.readouterr()
    printed = seen.out
    assert code == 0
    assert "not merged" in seen.err
    obj = _Obj(out)
    assert list(obj.groups) == ["Kept", "Adapted"]
    record = json.loads((tmp_path / "joined.merge.json").read_text(encoding="utf-8"))
    assert record["threshold"] == 0.05
    assert record["kept"] == str(first)
    assert record["adapted"] == str(second)
    assert record["interface_edges"] == 16
    assert record["declarations_merged"] is False
    triangles = sum(len(f) == 3 for f in obj.faces)
    source = sum(len(f) == 3 for f in _Obj(first).faces + _Obj(second).faces)
    assert record["triangles_created"] == triangles - source > 0
    for key in ("band_faces", "open_edges_left"):
        assert key in record
    side = tmp_path / "joined.obj.boundaries.toml"
    assert side.is_file()
    assert "Kept" in side.read_text(encoding="utf-8")
    assert (tmp_path / "joined.audit.json").is_file()
    assert json.loads(printed)["merged"] == str(out)
    # An existing output stays unless asked; the second run replaces it and its sidecar.
    with pytest.raises(InputArtifactError, match="--overwrite"):
        merge_objs(first, second, threshold=0.05, adapt=second, out=out)
    assert merge_objs(first, second, threshold=0.05, adapt=second, out=out, overwrite=True).obj


def test_mesh_merge_merge_adapt_is_required_and_decides_which_part_changes(tmp_path, capsys):
    """--adapt is required, must name one of the two, and swaps which part stays unchanged."""
    first = _write(tmp_path / "one.obj", _kept(16))
    second = _write(tmp_path / "two.obj", _adapted(23))
    out = tmp_path / "out.obj"
    with pytest.raises(SystemExit):
        cli.main(["merge", str(first), str(second), "--threshold", "0.05", "--out", str(out)])
    capsys.readouterr()
    with pytest.raises(InputArtifactError, match="names neither"):
        merge_objs(first, second, threshold=0.05, adapt=tmp_path / "three.obj", out=out)
    assert not out.exists()
    merge_objs(first, second, threshold=0.05, adapt=first, out=out)
    source = _Obj(second)
    written = _Obj(out)
    assert written.nodes[: len(source.nodes)] == source.nodes
    assert list(written.groups) == ["Adapted", "Kept"]
    assert written.groups["Adapted"] == source.faces


@pytest.mark.parametrize(
    "threshold", [-1.0, float("nan"), float("inf")], ids=["negative", "nan", "inf"]
)
def test_mesh_merge_merge_refuses_what_no_merge_can_answer(tmp_path, threshold):
    """A threshold that is no length, one part named twice, a source as the output."""
    first = _write(tmp_path / "one.obj", _kept(16))
    second = _write(tmp_path / "two.obj", _adapted(23))
    with pytest.raises(InputValueError, match="threshold"):
        merge_objs(first, second, threshold=threshold, adapt=second, out=tmp_path / "out.obj")
    with pytest.raises(InputArtifactError, match="both parts"):
        merge_objs(first, first, threshold=0.05, adapt=first, out=tmp_path / "out.obj")
    with pytest.raises(InputArtifactError, match="one of the two parts"):
        merge_objs(first, second, threshold=0.05, adapt=second, out=first, overwrite=True)


@pytest.mark.parametrize(
    ("kept_count", "adapted_count"), [(16, 32), (32, 16)], ids=["finer", "coarser"]
)
def test_mesh_merge_merge_the_kept_part_is_unchanged_to_the_byte(
    tmp_path, kept_count, adapted_count
):
    """The kept file's vertex lines lead in their order and its faces stay as they were."""
    (_, out, kept, _), _ = _quiet_merge(tmp_path, _kept(kept_count), _adapted(adapted_count), 0.05)
    assert out.nodes[: len(kept.nodes)] == kept.nodes
    assert out.groups["Kept"] == kept.faces
    assert next(iter(out.groups)) == "Kept"


@pytest.mark.parametrize(
    ("kept_count", "adapted_count"), [(16, 32), (32, 16)], ids=["finer", "coarser"]
)
def test_mesh_merge_merge_the_adapted_part_outside_the_band_is_unchanged_and_in_order(
    tmp_path, kept_count, adapted_count
):
    """Nodes and faces past two face layers keep their text and their relative order."""
    (merged, out, _, adapted), _ = _quiet_merge(
        tmp_path, _kept(kept_count), _adapted(adapted_count), 0.05
    )
    band = set(_band_of(adapted, _cut_nodes(adapted)))
    assert merged.band_faces == len(band)
    rest = [f for i, f in enumerate(adapted.faces) if i not in band]
    expected = adapted.face_text(rest)
    assert expected
    written = out.face_text(out.groups["Adapted"])
    remaining = set(expected)
    assert [f for f in written if f in remaining] == expected
    # Their nodes follow the kept part in the source's order.
    untouched = sorted({v for f in rest for v in f})
    lines = [adapted.nodes[v] for v in untouched]
    positions = [out.nodes.index(line) for line in lines]
    assert positions == sorted(positions)


def test_mesh_merge_merge_the_bands_faces_sit_where_the_band_sits_in_the_family(tmp_path):
    """A band listed mid-family stays mid-family: far faces before it, it, far faces after it."""
    nodes, faces, family = _adapted(29)
    source = _write(tmp_path / "probe.obj", (nodes, faces, family))
    band = set(_band_of(_Obj(source), _cut_nodes(_Obj(source))))
    far = [f for i, f in enumerate(faces) if i not in band]
    near = [f for i, f in enumerate(faces) if i in band]
    middle = len(far) // 2
    ordered = far[:middle] + near + far[middle:]
    (_, out, _, adapted), _ = _quiet_merge(tmp_path, _kept(24), (nodes, ordered, family), 0.05)
    written = out.face_text(out.groups["Adapted"])
    before = adapted.face_text(adapted.faces[:middle])
    after = adapted.face_text(adapted.faces[middle + len(near) :])
    assert written[:middle] == before
    assert written[-len(after) :] == after
    assert not set(written[middle : -len(after)]) & set(before + after)


@pytest.mark.parametrize(("kept_count", "adapted_count"), [(16, 32), (32, 16), (24, 29)], ids=str)
def test_mesh_merge_merge_the_band_stays_on_the_adapted_surface(
    tmp_path, kept_count, adapted_count
):
    """Every adapted node off the interface lies on the source surface; a moved node would not."""
    (_, out, kept, adapted), _ = _quiet_merge(
        tmp_path, _kept(kept_count), _adapted(adapted_count), 0.05
    )
    nodes = out.points[len(kept.nodes) :]
    assert float(_distance_to_surface(nodes, adapted).max()) < 1e-6
    off = nodes.copy()
    off[0] *= 1.01
    assert float(_distance_to_surface(off, adapted).max()) > 1e-4  # the control


def test_mesh_merge_merge_a_gap_across_the_cut_closes_on_the_kept_nodes(tmp_path):
    """The adapted part starts short of the cut: its boundary crosses the gap to the kept nodes."""
    stations = tuple(x + 0.02 for x in ADAPTED_STATIONS)
    (merged, out, kept, adapted), _ = _quiet_merge(
        tmp_path, _kept(24), _adapted(29, stations=stations), 0.05
    )
    assert merged.interface_edges == 24
    assert merged.open_edges_left == 0
    assert not _non_manifold(out)
    assert merged.audit is not None
    assert not merged.audit.degenerate_faces
    nodes = out.points[len(kept.nodes) :]
    assert float(nodes[:, 0].min()) >= 0.02 - 1e-9
    assert float(_distance_to_surface(nodes, adapted).max()) < 1e-6


def _mixed(part, every: int = 3):
    """Return a part with every third quad cut into two triangles."""
    nodes, faces, family = part
    out: list[list[int]] = []
    for i, f in enumerate(faces):
        if len(f) == 4 and i % every == 0:
            out += [f[:3], [f[0], f[2], f[3]]]
        else:
            out.append(f)
    return nodes, out, family


def test_mesh_merge_merge_quads_and_triangles_mixed_in_the_input(tmp_path):
    """Both parts carry quads and triangles along the cut; the merge closes all the same."""
    (merged, out, kept, _), _ = _quiet_merge(
        tmp_path, _mixed(_kept(20)), _mixed(_adapted(27)), 0.05
    )
    assert merged.interface_edges == 20
    assert merged.open_edges_left == 0
    assert not _non_manifold(out)
    assert out.groups["Kept"] == kept.faces
    assert {len(f) for f in out.groups["Adapted"]} == {3, 4}


def test_mesh_merge_merge_a_name_in_both_parts_is_warned_and_takes_the_suffix(tmp_path):
    """A family of one name in both parts: warned, the adapted one is written with _2."""
    (merged, out, _, _), messages = _quiet_merge(
        tmp_path, _kept(24, family="Body"), _adapted(24, family="Body"), 0.01
    )
    assert list(out.groups) == ["Body", "Body_2"]
    assert merged.renamed == {"Body": "Body_2"}
    assert [m for m in messages if "Body_2" in m]
    side = (tmp_path / "out.obj.boundaries.toml").read_text(encoding="utf-8")
    assert "Body_2" in side


def test_mesh_merge_merge_a_band_that_covers_a_family_rebuilds_it_warned(tmp_path):
    """Two face layers hold the whole adapted family: it is rebuilt whole, with a warning."""
    (merged, out, _, _), messages = _quiet_merge(
        tmp_path, _kept(8), _tube(12, (0.0, 0.5), cap=1, family="Adapted"), 0.2
    )
    assert [m for m in messages if "rebuilt whole" in m]
    assert merged.open_edges_left == 0
    assert not _non_manifold(out)
    record = json.loads((tmp_path / "out.merge.json").read_text(encoding="utf-8"))
    assert record["rebuilt_families"] == ["Adapted"]


def test_mesh_merge_merge_a_band_crossing_two_families_keeps_each_familys_faces(tmp_path):
    """The cut runs through two adapted families: each keeps its name and its far faces."""
    nodes, faces, _ = _adapted(29)
    half = len(faces) // 2
    lines = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in nodes]
    lines += ["g A1"] + ["f " + " ".join(str(v + 1) for v in f) for f in faces[:half]]
    lines += ["g A2"] + ["f " + " ".join(str(v + 1) for v in f) for f in faces[half:]]
    second = tmp_path / "two.obj"
    second.write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = _write(tmp_path / "one.obj", _kept(24))
    merged = merge_objs(first, second, threshold=0.05, adapt=second, out=tmp_path / "o.obj")
    out = _Obj(tmp_path / "o.obj")
    assert list(out.groups) == ["Kept", "A1", "A2"]
    assert merged.open_edges_left == 0
    assert not _non_manifold(out)
    source = _Obj(second)
    far = [f for f in source.groups["A2"] if min(source.points[f, 0]) > 0.4]
    assert set(source.face_text(far)) <= set(out.face_text(out.groups["A2"]))


def test_mesh_merge_merge_an_overlap_past_a_face_layer_splits_the_faces_and_warns(tmp_path):
    """The remesh cannot hold the interface (a chain past the next ring): a local split does."""
    stations = tuple(x - 0.2 for x in ADAPTED_STATIONS)
    (merged, out, kept, _), messages = _quiet_merge(
        tmp_path, _kept(24), _adapted(29, stations=stations), 0.3
    )
    record = json.loads((tmp_path / "out.merge.json").read_text(encoding="utf-8"))
    assert record["faces_split"] is True
    assert [m for m in messages if "split" in m]
    assert [m for m in messages if "turned against" in m]
    assert merged.turned_faces > 0
    assert merged.interface_edges == 24
    assert merged.open_edges_left == 0
    assert not _non_manifold(out)
    assert out.nodes[: len(kept.nodes)] == kept.nodes
    assert out.groups["Kept"] == kept.faces


def _spun(part, angle: float):
    """Return a part turned about x, so its nodes fall elsewhere around the cut."""
    nodes, faces, family = part
    c, s = math.cos(angle), math.sin(angle)
    return [[x, c * y - s * z, s * y + c * z] for x, y, z in nodes], faces, family


def _one_node_off(i: int, t: float) -> float:
    """Move the cut's node at angle zero out of reach, breaking that chain into an arc."""
    return 0.1 if i == 0 and t == 0.0 else 0.0


def test_mesh_merge_merge_a_kept_loop_against_an_adapted_arc_closes_what_the_arc_reaches(tmp_path):
    """Every kept node is near the adapted arc; the kept edge over its opening stays open."""
    kept = _spun(_kept(8), math.pi / 8.0)
    (merged, out, source, _), messages = _quiet_merge(
        tmp_path, kept, _adapted(29, shift=_one_node_off), 0.08
    )
    assert merged.interface_edges == 7
    assert not _non_manifold(out)
    assert [e for e in _open_edges(out) if e[1] < len(source.nodes)] != []
    assert out.groups["Kept"] == source.faces
    assert [m for m in messages if "split" in m]


def test_mesh_merge_merge_an_adapted_loop_against_a_kept_arc(tmp_path):
    """The kept chain is the arc: all its edges close and the adapted loop does not fold."""
    adapted = _spun(_adapted(8), math.pi / 8.0)
    last = len(KEPT_STATIONS) - 1
    kept = _kept(29, shift=lambda i, t: -_one_node_off(last - i, t))
    (merged, out, source, _), _ = _quiet_merge(tmp_path, kept, adapted, 0.08)
    assert merged.interface_edges == 27
    assert not _non_manifold(out)
    assert out.groups["Kept"] == source.faces


def _two_cuts(stations: tuple[float, ...]):
    """Return a kept part of two capped tubes, and the tube between them, 24 nodes on its left."""
    left = _tube(24, KEPT_STATIONS, cap=0, family="Left")
    right_stations = tuple(stations[-1] + x + 1.0 for x in KEPT_STATIONS)
    right = _tube(16, right_stations, cap=len(right_stations) - 1, family="Right")
    count = len(left[0])
    nodes = left[0] + right[0]
    groups = {"Left": left[1], "Right": [[v + count for v in f] for f in right[1]]}
    middle = _tube(24, stations, family="Middle")
    return nodes, groups, middle


@pytest.mark.parametrize(
    ("stations", "moved"),
    [(ADAPTED_STATIONS, 24), ((0.0, 0.2, 0.4), 0)],
    ids=["band apart", "band reaches the other chain"],
)
def test_mesh_merge_merge_each_chain_takes_its_own_way_unless_a_band_reaches_the_other(
    tmp_path, stations, moved
):
    """Two chains: one node for node, one by band; a band reaching the first remeshes it too."""
    nodes, groups, middle = _two_cuts(stations)
    lines = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in nodes]
    for name, faces in groups.items():
        lines += [f"g {name}"] + ["f " + " ".join(str(v + 1) for v in f) for f in faces]
    first = tmp_path / "kept.obj"
    first.write_text("\n".join(lines) + "\n", encoding="utf-8")
    second = _write(tmp_path / "middle.obj", middle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        merged = merge_objs(first, second, threshold=0.05, adapt=second, out=tmp_path / "o.obj")
    out = _Obj(tmp_path / "o.obj")
    assert merged.moved_nodes == moved
    assert merged.band_faces > 0
    assert merged.interface_edges == 24 + 16
    assert merged.open_edges_left == 0
    assert not _non_manifold(out)
    assert out.nodes[: len(nodes)] == _Obj(first).nodes


def test_mesh_merge_merge_the_same_input_writes_the_same_bytes(tmp_path):
    """Two runs of one call write one OBJ: nothing in the band depends on an accident of order."""
    first = _write(tmp_path / "one.obj", _kept(24))
    second = _write(tmp_path / "two.obj", _adapted(29))
    merge_objs(first, second, threshold=0.05, adapt=second, out=tmp_path / "a.obj")
    merge_objs(first, second, threshold=0.05, adapt=second, out=tmp_path / "b.obj")
    assert _Obj(tmp_path / "a.obj").nodes == _Obj(tmp_path / "b.obj").nodes
    assert _Obj(tmp_path / "a.obj").groups == _Obj(tmp_path / "b.obj").groups


def test_mesh_merge_merge_normals_are_not_reoriented_and_the_audit_reports_them(tmp_path):
    """The docs' limit: parts whose faces run opposite ways stay so, and the audit says so."""
    nodes, faces, family = _adapted(24)
    (merged, _, _, _), messages = _quiet_merge(
        tmp_path, _kept(24), (nodes, [f[::-1] for f in faces], family), 0.01
    )
    assert merged.audit is not None
    assert not merged.audit.passed
    assert [m for m in messages if "neighbours_of_opposite_orientation" in m]


def _strip(boundary, *, up: bool, family: str, rows: int = 3):
    """Return a flat strip of quads: the boundary at the cut, rows of it offset away from it."""
    count = len(boundary)
    sign = 1.0 if up else -1.0
    nodes = [[x, y + sign * r / rows, 0.0] for r in range(rows + 1) for x, y in boundary]
    faces = []
    for r in range(rows):
        for i in range(count - 1):
            a, b = r * count + i, r * count + i + 1
            faces.append([a, b, b + count, a + count] if up else [a, a + count, b + count, b])
    return nodes, faces, family


def _bend(x: float) -> float:
    """Return a cut straight up to x = 4 that bends away after it."""
    return 0.15 * (x - 4.0) ** 2 if x > 4.0 else 0.0


def test_mesh_merge_merge_an_adapted_node_past_the_kept_chain_is_not_dragged_onto_it(tmp_path):
    """A coarse boundary touching the dense one past the part that merges stays where it is.

    On the bend every adapted node lies on the kept boundary but no kept node lies on an
    adapted chord, so the kept chain stops at x = 4. Those adapted nodes are no part of it.
    """
    kept = _strip([(0.1 * i, _bend(0.1 * i)) for i in range(81)], up=False, family="Kept")
    adapted = _strip([(0.8 * i, _bend(0.8 * i)) for i in range(11)], up=True, family="Adapted")
    (merged, out, source_kept, source), _ = _quiet_merge(tmp_path, kept, adapted, 0.001)
    assert merged.interface_edges == 40
    written = set(out.nodes[len(source_kept.nodes) :])
    past = [v for v in range(11) if source.points[v, 0] > 5.0]
    assert [source.nodes[v] for v in past if source.nodes[v] not in written] == []
    assert merged.turned_faces == 0
    assert out.nodes[: len(source_kept.nodes)] == source_kept.nodes


def test_mesh_merge_merge_a_boundary_running_on_from_the_cut_is_no_part_of_the_interface(tmp_path):
    """A threshold past the cells beside a half body's plane closes the cut node for node.

    The plane's boundaries of the two parts meet at the cut's two ends, within the
    threshold of the other part. They run on from the cut, and joining them folds the faces.
    """
    kept = _tube(24, tuple(numpy.linspace(-1.0, 0.0, 41).tolist()), sweep=math.pi, family="Kept")
    adapted = _tube(
        24, tuple(numpy.linspace(0.0, 1.0, 41).tolist()), sweep=math.pi, family="Adapted"
    )
    (merged, out, source_kept, _), _ = _quiet_merge(tmp_path, kept, adapted, 0.2)
    assert merged.interface_edges == 24
    assert merged.moved_nodes == 25
    assert len(merged.audit.degenerate_faces) == 0
    plane = [
        e
        for e in _open_edges(out)
        if abs(out.points[e[0], 2]) < 1e-9 and abs(out.points[e[1], 2]) < 1e-9
    ]
    assert len(plane) == 4 * 40
    assert out.groups["Kept"] == source_kept.faces


@pytest.mark.parametrize("scale", [1.0, 1000.0, 0.001], ids=["unit", "thousand", "thousandth"])
def test_mesh_merge_merge_a_short_parts_far_boundary_faces_nothing(tmp_path, scale):
    """Two short collars closed along the cut: their far rings are near and no gap, at any unit.

    The warning tells the caller to raise the threshold, which would join those two rings.
    A real gap on the same parts is counted, at any unit.
    """

    def part(stations, family, **options):
        nodes, faces, name = _tube(24, stations, family=family, **options)
        return [[scale * c for c in n] for n in nodes], faces, name

    (merged, _, _, _), messages = _quiet_merge(
        tmp_path,
        part((-0.2, -0.1, 0.0), "Kept"),
        part((0.0, 0.1, 0.2), "Adapted"),
        0.01 * scale,
    )
    assert merged.interface_edges == 24
    assert merged.open_edges_left == 48
    assert merged.facing_edges_left == 0
    assert not [m for m in messages if "still face" in m]

    def lifted(i: int, t: float) -> float:
        return 0.06 if i == 0 and math.cos(t) > 0.1 else 0.0

    gap = tmp_path / "gap"
    gap.mkdir()
    (merged, _, _, _), _ = _quiet_merge(
        gap,
        part((-0.2, -0.1, 0.0), "Kept"),
        part((0.0, 0.1, 0.2), "Adapted", shift=lifted),
        0.01 * scale,
    )
    assert merged.facing_edges_left > 0

    def pushed(i: int, t: float) -> float:
        return -0.03 if i == 0 and math.cos(t) > 0.1 else 0.0

    overlap = tmp_path / "overlap"
    overlap.mkdir()
    (merged, _, _, _), _ = _quiet_merge(
        overlap,
        part((-0.2, -0.1, 0.0), "Kept"),
        part((0.0, 0.1, 0.2), "Adapted", shift=pushed),
        0.01 * scale,
    )
    # A cut overlapped by less than a face's depth is a gap like another, past the threshold.
    assert merged.facing_edges_left > 0


def test_mesh_merge_merge_a_family_opened_twice_is_warned_since_its_order_changes(tmp_path):
    """A group that comes back later in the file: the output's face order is not the file's."""
    nodes, faces, _ = _kept(24)
    third = len(faces) // 3
    lines = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in nodes]
    for name, part in (("A", faces[:third]), ("B", faces[third:-third]), ("A", faces[-third:])):
        lines.append(f"g {name}")
        lines += ["f " + " ".join(str(v + 1) for v in f) for f in part]
    first = tmp_path / "kept.obj"
    first.write_text("\n".join(lines) + "\n", encoding="utf-8")
    second = _write(tmp_path / "adapted.obj", _adapted(24))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merge_objs(first, second, threshold=0.01, adapt=second, out=tmp_path / "out.obj")
    said = [str(w.message) for w in caught if "more than one place" in str(w.message)]
    assert len(said) == 1
    assert "family A" in said[0]
    written = [line for line in (tmp_path / "out.obj").read_text().splitlines() if line[:2] == "f "]
    assert written[: len(faces)] != [line for line in lines if line[:2] == "f "]


@pytest.mark.parametrize("name", ["out.merge.json", "out.audit.json", "out.obj.boundaries.toml"])
def test_mesh_merge_merge_a_file_beside_the_output_is_replaced_only_with_overwrite(tmp_path, name):
    """A boundaries file declared by hand before the merge is not lost without the word."""
    first = _write(tmp_path / "one.obj", _kept(24))
    second = _write(tmp_path / "two.obj", _adapted(24))
    beside = tmp_path / name
    beside.write_text("# declared by hand\n", encoding="utf-8")
    with pytest.raises(InputArtifactError, match="--overwrite"):
        merge_objs(first, second, threshold=0.01, adapt=second, out=tmp_path / "out.obj")
    assert beside.read_text(encoding="utf-8") == "# declared by hand\n"
    assert not (tmp_path / "out.obj").exists()
    merged = merge_objs(
        first, second, threshold=0.01, adapt=second, out=tmp_path / "out.obj", overwrite=True
    )
    assert merged.interface_edges == 24
    assert beside.read_text(encoding="utf-8") != "# declared by hand\n"


def test_mesh_merge_merge_an_output_in_a_new_folder_is_written_as_on_the_other_verbs(tmp_path):
    """The wheel and the remesh verbs make the folder; this one reported a temporary file."""
    first = _write(tmp_path / "one.obj", _kept(24))
    second = _write(tmp_path / "two.obj", _adapted(24))
    out = tmp_path / "new" / "folder" / "out.obj"
    merged = merge_objs(first, second, threshold=0.01, adapt=second, out=out)
    assert merged.interface_edges == 24
    assert sorted(p.name for p in out.parent.iterdir()) == [
        "out.audit.json",
        "out.merge.json",
        "out.obj",
        "out.obj.boundaries.toml",
    ]


def test_mesh_merge_merge_two_parts_stating_different_units_are_warned(tmp_path):
    """Nothing is scaled, and the output's inventory states no unit: the words must say so."""
    first = _write(tmp_path / "one.obj", _kept(24))
    second = _write(tmp_path / "two.obj", _adapted(24))
    (tmp_path / "one.obj.boundaries.toml").write_text('[import]\nunits = "m"\n', encoding="utf-8")
    (tmp_path / "two.obj.boundaries.toml").write_text('[import]\nunits = "mm"\n', encoding="utf-8")

    def said(out: str) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            merge_objs(first, second, threshold=0.01, adapt=second, out=tmp_path / out)
        return [str(w.message) for w in caught if "states units" in str(w.message)]

    assert len(said("mixed.obj")) == 1
    (tmp_path / "two.obj.boundaries.toml").write_text('[import]\nunits = "M"\n', encoding="utf-8")
    assert said("same.obj") == []
    (tmp_path / "two.obj.boundaries.toml").write_text("not toml [", encoding="utf-8")
    assert said("broken.obj") == []


@pytest.mark.parametrize("name", ["merged", "merged.txt"])
def test_mesh_merge_merge_an_output_not_named_obj_is_warned_and_takes_the_obj_form(tmp_path, name):
    """The output takes the parts' kind with its record beside it; the name asked is not made."""
    first = _write(tmp_path / "one.obj", _kept(24))
    second = _write(tmp_path / "two.obj", _adapted(24))
    with pytest.warns(NotApplicableWarning, match=r"not the parts' kind \(\.obj\)"):
        merged = merge_objs(first, second, threshold=0.01, adapt=second, out=tmp_path / name)
    assert merged.obj == tmp_path / "merged.obj"
    assert (tmp_path / "merged.obj").is_file() and (tmp_path / "merged.merge.json").is_file()
    assert not (tmp_path / "merged.txt").exists()


def test_mesh_merge_merge_a_split_says_how_far_past_the_threshold_it_moved_a_node(tmp_path):
    """The threshold measures a node against a kept EDGE; a sparse kept chain moves it further."""
    stations = tuple(x - 0.2 for x in ADAPTED_STATIONS)
    (merged, out, kept, _), messages = _quiet_merge(
        tmp_path, _kept(8), _adapted(29, stations=stations), 0.3
    )
    record = json.loads((tmp_path / "out.merge.json").read_text(encoding="utf-8"))
    assert record["faces_split"] is True
    said = [m for m in messages if "further than the threshold" in m]
    assert len(said) == 1
    assert "of 29 interface node(s)" in said[0]
    assert merged.interface_edges == 8
    assert out.nodes[: len(kept.nodes)] == kept.nodes


def test_mesh_merge_merge_a_family_is_said_rebuilt_only_when_it_was(tmp_path):
    """A band covering the family that then fell back on the split rebuilt nothing."""
    triangle = [
        [0.0, 1.0, 0.0],
        [0.0, math.cos(TAU / 24.0), math.sin(TAU / 24.0)],
        [0.5, 1.0, 0.0],
    ]
    _, messages = _quiet_merge(tmp_path, _kept(24), (triangle, [[0, 1, 2]], "Lone"), 1.0)
    record = json.loads((tmp_path / "out.merge.json").read_text(encoding="utf-8"))
    assert record["faces_split"] is True
    assert record["rebuilt_families"] == []
    assert not [m for m in messages if "rebuilt whole" in m]


def test_mesh_merge_merge_the_split_warning_names_both_causes(tmp_path):
    """A chain ending away from a kept node also splits, with no overlap anywhere in the pair."""
    kept = _strip([(0.1 * i, _bend(0.1 * i)) for i in range(81)], up=False, family="Kept")
    adapted = _strip(
        [(0.8 * i + 0.05, _bend(0.8 * i + 0.05)) for i in range(10)], up=True, family="A"
    )
    _, messages = _quiet_merge(tmp_path, kept, adapted, 0.02)
    record = json.loads((tmp_path / "out.merge.json").read_text(encoding="utf-8"))
    assert record["faces_split"] is True
    said = [m for m in messages if "were split onto them" in m]
    assert len(said) == 1
    assert "a chain's end has no kept node beside it" in said[0]
