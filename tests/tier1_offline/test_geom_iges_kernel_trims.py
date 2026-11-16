"""Trimmed faces of one base, their splits, and the preference the writer states (, G2R)."""

from __future__ import annotations

import math

import pytest
from kernel_support import kernel_or_skip

from cfdbender._errors import InputValueError
from cfdbender.geometry._iges_write import pack_records, read_iges


@pytest.fixture
def kernel():
    kernel_or_skip()
    from cfdbender.mesh._gmsh import open_kernel

    k = open_kernel("test")
    try:
        yield k
    finally:
        k.close()


def _skin(kernel):
    """A degree 5 by 1 skin over a curved net, 2 by 3 m, its domain the unit square."""
    net = [[(2.0 * i / 5.0, 3.0 * j, 0.1 * math.sin(i)) for j in range(2)] for i in range(6)]
    return kernel.add_bspline_surface(
        net,
        knots_u=[0.0, 1.0],
        knots_v=[0.0, 1.0],
        mult_u=[6, 6],
        mult_v=[2, 2],
        deg_u=5,
        deg_v=1,
    )


def _box(u0, u1, v0=0.0, v1=1.0):
    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    return [[[corners[k], corners[(k + 1) % 4]] for k in range(4)]]


def _two_faces(kernel):
    base = _skin(kernel)
    lower = kernel.add_trimmed_face(base, _box(0.0, 0.5))
    upper = kernel.add_trimmed_face(base, _box(0.5, 1.0))
    kernel.remove([base])
    return lower, upper


def test_geom_iges_kernel_trims_a_skin_with_two_iso_trims_writes_and_reimports_as_two_faces(
    kernel, tmp_path
):
    lower, upper = _two_faces(kernel)
    areas = [kernel.area(lower), kernel.area(upper)]
    assert [f.tag for f in kernel.faces()] == [lower.tag, upper.tag]
    path = tmp_path / "skin.igs"
    kernel.export_iges({"WING": [lower, upper]}, path)
    named = kernel.import_iges(path)
    assert len(named.faces) == 2
    back = sorted(kernel.area(f) for f in named.faces)
    assert back == pytest.approx(sorted(areas), rel=1e-6)
    assert sum(areas) > 0.0 and areas[0] != pytest.approx(areas[1], rel=1e-3)


def test_geom_iges_kernel_trims_the_file_states_parameter_space_as_the_preference(kernel, tmp_path):
    lower, upper = _two_faces(kernel)
    path = tmp_path / "pref.igs"
    kernel.export_iges({"WING": [lower, upper]}, path)
    trims = [e for e in read_iges(path)["entities"] if e["type"] == 142]
    assert len(trims) == 2 and {e["parameters"][5] for e in trims} == {"1"}
    with pytest.warns(UserWarning, match="IGES.PREF: 3 .*PREF 1 is written instead"):
        kernel.export_iges({"WING": [lower, upper]}, path, preference=3)
    trims = [e for e in read_iges(path)["entities"] if e["type"] == 142]
    assert {e["parameters"][5] for e in trims} == {"1"}
    with pytest.raises(InputValueError, match="IGES.PREF"):
        kernel.export_iges({"WING": [lower]}, path, preference=4)


def test_geom_iges_kernel_trims_a_model_space_preference_is_warned_and_written_as_parameter_space(
    kernel, tmp_path
):
    """The solver's CAD import reads no face from a 142 of PREF 2 (record g13d-reads.json, P2)."""
    from cfdbender._errors import GeometryWarning

    lower, upper = _two_faces(kernel)
    path = tmp_path / "pref2.igs"
    with pytest.warns(GeometryWarning, match="IGES.PREF: 2 .*PREF 1 is written instead"):
        kernel.export_iges({"WING": [lower, upper]}, path, preference=2)
    trims = [e for e in read_iges(path)["entities"] if e["type"] == 142]
    assert len(trims) == 2 and {e["parameters"][5] for e in trims} == {"1"}


def test_geom_iges_kernel_trims_parameter_space_is_preferred_only_where_the_file_has_that_curve():
    """A 142 with no parameter-space curve (BPTR 0, the kernel's planar faces) states model space.

    PREF 1 on such a boundary names a curve the file does not carry.
    """
    from cfdbender.geometry._iges_write import set_preference

    def boundary(bptr):
        return {"type": 142, "parameters": ["142", "0", "5", str(bptr), "19", "3"]}

    document = {"entities": [boundary(9), boundary(0), {"type": 126, "parameters": ["126"]}]}
    set_preference(document, 1)
    assert [e["parameters"][5] for e in document["entities"][:2]] == ["1", "2"]
    set_preference(document, 2)
    assert [e["parameters"][5] for e in document["entities"][:2]] == ["2", "2"]


@pytest.mark.parametrize(
    ("loop", "word"),
    [
        (_box(0.5, 1.5), "domain"),  # the reviewed code extrapolated the base: area above half
        ([[[(0.2, 0.2), (0.8, 0.8)], [(0.8, 0.8), (0.2, 0.2)]]], "no area"),
        (
            [[[(0.1, 0.5), (0.5, 0.5)], [(0.5, 0.5), (0.9, 0.5)], [(0.9, 0.5), (0.1, 0.5)]]],
            "no area",
        ),
        ([[[(0.0, 0.0), (0.0, 0.0)], *_box(0.0, 1.0)[0]]], "zero length"),
    ],
)
def test_geom_iges_kernel_trims_a_loop_off_the_domain_or_of_no_area_is_refused(kernel, loop, word):
    base = _skin(kernel)
    before = len(kernel.faces())
    with pytest.raises(InputValueError, match=word):
        kernel.add_trimmed_face(base, loop)
    assert len(kernel.faces()) == before
    edge = kernel.add_trimmed_face(base, _box(0.0, 1.0))  # exactly on the domain's edge: kept
    assert kernel.area(edge) == pytest.approx(kernel.area(base), rel=1e-9)


def test_geom_iges_kernel_trims_an_open_loop_is_refused_and_a_hole_is_a_second_loop(kernel):
    base = _skin(kernel)
    open_loop = [[[(0.1, 0.1), (0.9, 0.1)], [(0.9, 0.1), (0.9, 0.9)], [(0.9, 0.9), (0.1, 0.5)]]]
    with pytest.raises(InputValueError, match="open"):
        kernel.add_trimmed_face(base, open_loop)
    with pytest.raises(InputValueError, match="outer loop"):
        kernel.add_trimmed_face(base, [])
    whole = kernel.area(base)
    hole = kernel.area(kernel.add_trimmed_face(base, _box(0.4, 0.6, 0.4, 0.6)))
    ring = kernel.add_trimmed_face(base, [*_box(0.0, 1.0), *_box(0.4, 0.6, 0.4, 0.6)])
    assert kernel.area(ring) == pytest.approx(whole - hole, rel=1e-6)
    reversed_hole = [[[b, a] for a, b in reversed(_box(0.4, 0.6, 0.4, 0.6)[0])]]
    ring = kernel.add_trimmed_face(base, [*_box(0.0, 1.0), *reversed_hole])
    assert kernel.area(ring) == pytest.approx(whole - hole, rel=1e-6)


def test_geom_iges_kernel_trims_split_face_uv_cuts_exactly_on_an_iso_line_in_either_direction(
    kernel,
):
    base = _skin(kernel)
    face = kernel.add_trimmed_face(base, _box(0.1, 0.9, 0.05, 0.95))
    whole = kernel.area(face)
    low, high = kernel.split_face_uv(face, "u", 0.5)
    assert kernel.area(low) + kernel.area(high) == pytest.approx(whole, rel=1e-9)
    assert kernel.bbox(low)[1][0] < kernel.bbox(high)[1][0] + 1e-9
    assert kernel.uv_bounds(low)[1][0] == pytest.approx(0.5)
    assert kernel.uv_bounds(high)[0][0] == pytest.approx(0.5)
    before = kernel.area(high)
    first, second = kernel.split_face_uv(high, "v", 0.5)
    assert kernel.area(first) + kernel.area(second) == pytest.approx(before, rel=1e-9)
    assert kernel.uv_bounds(second)[0][1] == pytest.approx(0.5)
    (same,) = kernel.split_face_uv(second, "u", 0.99)
    assert kernel.area(same) == pytest.approx(kernel.area(second), rel=1e-9)


def test_geom_iges_kernel_trims_split_face_uv_refuses_what_it_cannot_cut_exactly(kernel):
    base = _skin(kernel)
    face = kernel.add_trimmed_face(base, _box(0.0, 1.0))
    with pytest.raises(InputValueError, match="domain"):
        kernel.split_face_uv(face, "u", 1.5)
    with pytest.raises(InputValueError, match="'u' or 'v'"):
        kernel.split_face_uv(face, "w", 0.5)
    plane = kernel.add_planar_face([(0, 0, 5), (1, 0, 5), (1, 1, 5), (0, 1, 5)])
    with pytest.raises(InputValueError, match="B-spline base"):
        kernel.split_face_uv(plane, "u", 0.5)


def test_geom_iges_kernel_trims_reparametrize_and_shared_curve_read_the_edge_between_two_faces(
    kernel,
):
    first, second = _two_faces(kernel)
    made = kernel.fragment([first], [second])
    (lower,), (upper,) = made.from_a[0], made.from_b[0]
    edge = kernel.shared_curve(lower, upper)
    assert edge is not None
    low, high = kernel.edge_bounds(edge)
    for t in (low, 0.5 * (low + high), high):
        for face in (lower, upper):
            uv = kernel.reparametrize(edge, face, t)
            assert uv[0] == pytest.approx(0.5)
            on_face = kernel.value(face, uv)
            on_edge = kernel.edge_values(edge, [t])[0]
            assert tuple(on_face) == pytest.approx(tuple(on_edge), abs=1e-9)
    far = kernel.add_planar_face([(9, 9, 9), (10, 9, 9), (10, 10, 9), (9, 10, 9)])
    assert kernel.shared_curve(lower, far) is None
    edges = {abs(e) for e in kernel.boundary(lower)} & {abs(e) for e in kernel.boundary(upper)}
    assert edges == {edge}


def test_geom_iges_kernel_trims_shared_curve_refuses_two_faces_that_share_two_edges(
    kernel, monkeypatch
):
    from cfdbender.mesh._cad import Face

    edges = {1: (1, 2, -3), 2: (-2, 3, 4)}
    monkeypatch.setattr(kernel, "boundary", lambda face: edges[face.tag])
    with pytest.raises(InputValueError, match="share 2 edges"):
        kernel.shared_curve(Face(1), Face(2))


def test_geom_iges_kernel_trims_parameter_lines_break_between_values_never_inside_a_number():
    values = [f"{1.0 / 3.0 + k:.17E}" for k in range(40)]
    records = pack_records(values, 64)
    assert all(len(r) <= 64 for r in records)
    joined = "".join(records)
    assert joined == ",".join(values) + ";"
    pieces = [piece for r in records for piece in r.replace(";", ",").split(",") if piece]
    assert sorted(pieces) == sorted(values)
    assert all(r.endswith(",") or r.endswith(";") for r in records)


# --- the reader profile gate on trimmed skins (entity rules) -----------------------------------


@pytest.fixture
def written(kernel, tmp_path):
    """A two-face skin written to IGES: its path and the entities read back."""
    lower, upper = _two_faces(kernel)
    path = tmp_path / "skin.igs"
    kernel.export_iges({"WING": [lower, upper]}, path)
    return path, read_iges(path)


def _checks(report):
    return sorted({f.check for f in report.findings})


_SKIN_PROFILE = {"degrees_u": (5,), "degrees_v": (1,)}


def _rewrite(tmp_path, document, name="planted.igs"):
    from cfdbender._textio import write_text
    from cfdbender.geometry._iges_write import serialize_iges

    path = tmp_path / name
    write_text(path, serialize_iges(document), encoding="ascii")
    return path


def _curves(document, kind):
    """The 126 curves a first face's outer boundary uses: ``kind`` is 'uv' or 'model'."""
    by_id = {e["id"]: e for e in document["entities"]}
    face = next(e for e in document["entities"] if e["type"] == 144)
    boundary = by_id[int(face["parameters"][4])]
    composite = by_id[int(boundary["parameters"][3 if kind == "uv" else 4])]
    n = int(composite["parameters"][1])
    return [by_id[int(p)] for p in composite["parameters"][2 : 2 + n]]


def _move(curve, axis, amount):
    """Add ``amount`` to coordinate ``axis`` of every control point of a 126 or 110."""
    values = curve["parameters"]
    if curve["type"] == 110:
        places = [1 + axis, 4 + axis]
    else:
        k, m = int(values[1]), int(values[2])
        places = [7 + 2 * k + m + 3 + 3 * i + axis for i in range(k + 1)]
    for at in places:
        values[at] = f"{float(values[at].replace('D', 'E')) + amount:.17E}"


def _gate(path, **fields):
    from cfdbender.geometry._geom_gate import check_profile
    from cfdbender.geometry.iges_profile import ReaderProfile

    return check_profile(path, ReaderProfile(**{**_SKIN_PROFILE, **fields}))


def test_geom_iges_kernel_trims_a_written_two_face_skin_passes_the_gate_with_no_finding(written):
    path, _ = written
    report = _gate(path)
    assert report.findings == () and report.surfaces == 2


def test_geom_iges_kernel_trims_each_planted_loop_defect_is_caught_and_named(written, tmp_path):
    path, document = written
    open_loop = read_iges(path)
    interior = next(  # the straight piece at u = 0.5: moving it down stays inside the domain
        c for c in _curves(open_loop, "uv") if 0.1 < float(c["parameters"][1]) < 0.9
    )
    _move(interior, 0, -0.05)  # its two ends leave the neighbours, so the loop opens
    assert _checks(_gate(_rewrite(tmp_path, open_loop, "open.igs"))) == ["profile.loop_open"]
    out_of_domain = read_iges(path)
    for curve in _curves(out_of_domain, "uv"):
        _move(curve, 0, 0.7)  # the whole loop moves out of the unit domain and stays closed
    report = _gate(_rewrite(tmp_path, out_of_domain, "domain.igs"))
    assert _checks(report) == ["profile.loop_domain"] and not report.passed
    off_surface = read_iges(path)
    for curve in _curves(off_surface, "model"):
        _move(curve, 2, 0.01)  # the model-space boundary leaves the surface by 1 cm
    report = _gate(_rewrite(tmp_path, off_surface, "off.igs"))
    assert _checks(report) == ["profile.loop_surface"] and not report.passed
    assert _checks(_gate(path)) == []  # the control: the unplanted file has none of the three


def test_geom_iges_kernel_trims_a_parameter_gap_inside_the_file_resolution_is_not_an_open_loop(
    written, tmp_path
):
    """The kernel's own fragment writes pcurves that meet to 6e-7 in (u, v) (probe P7).

    A gap counts as open only when the two ends are also apart in model space by more than the
    resolution the file states; a gap a user would see still fails.
    """
    path, _ = written
    document = read_iges(path)
    document["global"][18] = "1E-05"
    interior = next(c for c in _curves(document, "uv") if 0.1 < float(c["parameters"][1]) < 0.9)
    _move(interior, 0, -5e-7)  # 5e-7 in u is about 1e-6 m on this 2 m skin
    assert _checks(_gate(_rewrite(tmp_path, document, "near.igs"))) == []
    _move(interior, 0, -1e-4)  # about 2e-4 m: open
    assert _checks(_gate(_rewrite(tmp_path, document, "far.igs"))) == ["profile.loop_open"]


def test_geom_iges_kernel_trims_a_126_is_read_over_its_own_parameter_range_not_its_knots(
    written, tmp_path
):
    """IGES 126 states V(0) and V(1); another writer may use only part of the knot range.

    The interior straight pcurve is rewritten as a 126 whose knots run past both ends and whose
    V(0), V(1) select exactly the old segment: the same loop, which must stay clean.
    """
    path, document = written
    interior = next(c for c in _curves(document, "uv") if 0.1 < float(c["parameters"][1]) < 0.9)
    assert interior["type"] == 110
    p0 = [float(v) for v in interior["parameters"][1:4]]
    p1 = [float(v) for v in interior["parameters"][4:7]]
    far0 = [a - (b - a) for a, b in zip(p0, p1, strict=True)]
    far1 = [b + (b - a) for a, b in zip(p0, p1, strict=True)]
    numbers = [0, 0, 1, 1, 1, 1, *far0, *far1, 1 / 3, 2 / 3, 0, 0, 1]
    interior["type"] = 126
    for line in (interior["d1"], interior["d2"]):
        line[0] = f"{126:8d}"
    interior["parameters"] = ["126", "1", "1", "0", "0", "1", "0", *(f"{x!r}" for x in numbers)]
    report = _gate(_rewrite(tmp_path, document, "subrange.igs"))
    assert _checks(report) == []


def test_geom_iges_kernel_trims_the_gate_reads_the_property_pointers_of_the_144_not_a_fixed_layout(
    written, tmp_path
):
    path, document = written
    faces = [e for e in document["entities"] if e["type"] == 144]
    for face in faces:
        name = face["parameters"][-1]
        face["parameters"][5:] = ["1", "0", "1", name]  # one association (pointer 0) first
    report = _gate(_rewrite(tmp_path, document, "assoc.igs"))
    assert report.findings == ()
    for face in faces:
        face["parameters"][5:] = ["0", "0"]  # no property at all
    assert _checks(_gate(_rewrite(tmp_path, document, "bare.igs"))) == ["profile.name"]


def test_geom_iges_kernel_trims_an_inner_boundary_and_a_120_warn_and_do_not_fail(written, tmp_path):
    path, document = written
    face = next(e for e in document["entities"] if e["type"] == 144)
    face["parameters"][3:5] = ["1", face["parameters"][4]]
    face["parameters"][5:5] = [face["parameters"][4]]  # the outer curve doubles as an inner one
    top = max(e["id"] for e in document["entities"]) + 2
    d1 = [f"{120:8d}"] + [f"{0:8d}"] * 7 + ["00000000"]
    d2 = [f"{120:8d}"] + [f"{0:8d}"] * 3 + [f"{0:8d}"] + [" " * 8] * 4
    document["entities"].append(
        {"id": top, "type": 120, "d1": d1, "d2": d2, "parameters": ["120", "0", "0", "0.", "6.28"]}
    )
    report = _gate(_rewrite(tmp_path, document, "warn.igs"))
    assert _checks(report) == ["profile.entity", "profile.inner_loop"] and report.passed
    assert {f.severity for f in report.findings} == {"warn"}
    # R21-014: a profile that records inner loops as read does not warn about them
    observed = _gate(_rewrite(tmp_path, document, "observed.igs"), inner_loops=True)
    assert _checks(observed) == ["profile.entity"]


def test_geom_iges_kernel_trims_degrees_are_observed_per_direction_and_a_net_limit_fails(written):
    path, _ = written
    swapped = _gate(path, degrees_u=(1,), degrees_v=(5,))
    assert _checks(swapped) == ["profile.degree"] and swapped.passed
    assert len(swapped.findings) == 4  # two skins, each 5 by 1 read as 1 by 5: wrong twice
    assert _checks(_gate(path, max_net=(6, 2))) == []
    big = _gate(path, max_net=(5, 2))
    assert _checks(big) == ["profile.net_size"] and not big.passed
    assert _checks(_gate(path, max_net=(6, 1))) == ["profile.net_size"]
