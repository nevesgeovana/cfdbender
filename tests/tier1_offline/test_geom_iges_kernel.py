"""The named IGES writer is deterministic and carries no account name."""

from __future__ import annotations

import getpass
import hashlib
import os
import subprocess
import sys
import time

import pytest
from kernel_support import kernel_or_skip

from cfdbender.geometry._iges_write import (
    FIXED_STAMP,
    normalize_global,
    read_iges,
    serialize_iges,
)

_WRITE = (
    "import sys\n"
    "from cfdbender.mesh._gmsh import session\n"
    "from cfdbender.geometry._iges_write import write_iges\n"
    "with session('test') as g:\n"
    "    t = g.model.occ.addRectangle(0, 0, 0, 1, 2)\n"
    "    g.model.occ.synchronize()\n"
    "    write_iges(g, {'PANEL': [t]}, sys.argv[1])\n"
)


def _write_in_fresh_process(path):
    done = subprocess.run(
        [sys.executable, "-c", _WRITE, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ),
    )
    assert done.returncode == 0, done.stderr[-400:]
    return path.read_bytes()


def test_geom_iges_kernel_normalize_global_fixes_stamps_and_empties_the_author():
    values = [""] * 26
    values[17], values[20], values[24] = "15H20261109.114612", "5Hgeova", "15H20261109.114612"
    out = normalize_global(values)
    assert out[17] == out[24] == FIXED_STAMP
    assert out[20] == ""
    assert values[20] == "5Hgeova"  # the input is not changed


def test_geom_iges_kernel_serialize_iges_without_stamp_positions_is_left_alone():
    document = {"start": ["x"], "global": ["", "", "1HA"], "entities": []}
    assert "1HA" in serialize_iges(document)


def _need_gmsh():
    kernel_or_skip()


def test_geom_iges_kernel_two_fresh_processes_write_byte_identical_iges_with_no_login(tmp_path):
    _need_gmsh()
    first = _write_in_fresh_process(tmp_path / "a.igs")
    time.sleep(1.1)  # the kernel's own stamp has a resolution of one second
    second = _write_in_fresh_process(tmp_path / "b.igs")
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    text = first.decode("ascii")
    assert getpass.getuser() not in text
    document = read_iges(tmp_path / "a.igs")
    assert document["global"][17] == document["global"][24] == FIXED_STAMP
    assert document["global"][20] == ""
    assert any(e["type"] == 406 for e in document["entities"])


# --- the kernel adapter ----------------------------------------------------------------------


def _flat_net(size=3.0, z=0.0, x0=0.0):
    return [[(x0 + size * i / 3, size * j / 3, z) for j in range(4)] for i in range(4)]


def _bicubic(kernel, net):
    return kernel.add_bspline_surface(
        net,
        knots_u=[0.0, 1.0],
        knots_v=[0.0, 1.0],
        mult_u=[4, 4],
        mult_v=[4, 4],
        deg_u=3,
        deg_v=3,
    )


@pytest.fixture
def kernel():
    _need_gmsh()
    from cfdbender.mesh._gmsh import open_kernel

    k = open_kernel("test")
    try:
        yield k
    finally:
        k.close()


def test_geom_iges_kernel_the_kernel_satisfies_the_protocol(kernel):
    from cfdbender.mesh._cad import CadKernel

    assert isinstance(kernel, CadKernel)
    assert kernel.name == "gmsh"
    assert kernel.version.count(".") >= 1


def test_geom_iges_kernel_a_bicubic_face_answers_the_geometric_queries(kernel):
    face = _bicubic(kernel, _flat_net(3.0))
    assert kernel.area(face) == pytest.approx(9.0, rel=1e-9)
    assert kernel.centroid(face) == pytest.approx((1.5, 1.5, 0.0), abs=1e-9)
    low, high = kernel.bbox(face)
    assert low == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert high == pytest.approx((3.0, 3.0, 0.0), abs=1e-6)
    (u0, v0), (u1, v1) = kernel.uv_bounds(face)
    assert (u1 - u0) > 0 and (v1 - v0) > 0
    mid = ((u0 + u1) / 2, (v0 + v1) / 2)
    assert kernel.value(face, mid) == pytest.approx((1.5, 1.5, 0.0), abs=1e-9)
    assert abs(kernel.normal(face, mid)[2]) == pytest.approx(1.0, abs=1e-9)
    du, dv = kernel.derivative(face, mid)
    assert du[2] == pytest.approx(0.0, abs=1e-9) and dv[2] == pytest.approx(0.0, abs=1e-9)
    ((xyz, _),) = kernel.closest(face, [(1.0, 2.0, 5.0)])
    assert xyz == pytest.approx((1.0, 2.0, 0.0), abs=1e-6)
    assert kernel.is_inside(face, mid)
    assert len(kernel.boundary(face)) == 4
    assert kernel.faces() == (face,)


def test_geom_iges_kernel_the_batched_queries_agree_with_the_single_ones_on_a_curved_face(kernel):
    net = [[(3.0 * i / 3, 3.0 * j / 3, 0.4 * (i - 1.5) ** 2) for j in range(4)] for i in range(4)]
    face = _bicubic(kernel, net)
    (u0, v0), (u1, v1) = kernel.uv_bounds(face)
    uvs = [
        (u0 + (u1 - u0) * a, v0 + (v1 - v0) * b) for a, b in ((0.2, 0.3), (0.5, 0.5), (0.9, 0.1))
    ]
    points = kernel.values(face, uvs)
    slopes = kernel.derivatives(face, uvs)
    for uv, point, slope in zip(uvs, points, slopes, strict=True):
        assert tuple(point) == pytest.approx(kernel.value(face, uv), abs=1e-12)
        single = [c for row in kernel.derivative(face, uv) for c in row]
        assert [c for row in slope for c in row] == pytest.approx(single, abs=1e-12)
    assert all(kernel.inside(face, uvs))
    assert not kernel.inside(face, [(u1 + 1.0, v1 + 1.0)])[0]
    off = [(0.7, 1.1, 3.0), (2.2, 0.4, -2.0)]
    xyz, uv = kernel.project(face, off)
    for point, target, parameters in zip(xyz, off, uv, strict=True):
        assert tuple(point) == pytest.approx(kernel.value(face, tuple(parameters)), abs=1e-9)
        assert point[2] != pytest.approx(target[2], abs=0.5)  # moved onto the surface
    back = kernel.parametrize(face, list(points))
    assert [c for row in back for c in row] == pytest.approx(
        [c for uv in uvs for c in uv], abs=1e-6
    )


def test_geom_iges_kernel_the_curve_queries_of_an_edge_agree_with_its_geometry(kernel):
    face = _bicubic(kernel, _flat_net(3.0))
    lengths = []
    for edge in kernel.boundary(face):
        low, high = kernel.edge_bounds(edge)
        ends = kernel.edge_ends(edge)
        a, b = kernel.edge_values(edge, [low, high])
        assert {tuple(round(c, 9) for c in p) for p in (a, b)} == {
            tuple(round(c, 9) for c in p) for p in ends
        }
        lengths.append(float(((a - b) ** 2).sum() ** 0.5))
        box_low, box_high = kernel.edge_bbox(edge)
        assert all(
            lo <= c + 1e-9 and c <= hi + 1e-9
            for lo, c, hi in zip(box_low, a, box_high, strict=True)
        )
        mid = kernel.edge_values(edge, [(low + high) / 2])[0]
        (projected,) = kernel.edge_project(edge, [tuple(mid + (0.0, 0.0, 0.5))])
        assert tuple(projected) == pytest.approx(tuple(mid), abs=1e-6)
    assert lengths == pytest.approx([3.0] * 4, abs=1e-9)


def test_geom_iges_kernel_fragment_imprints_a_crossing_pair_and_reports_what_each_became(kernel):
    a = _bicubic(kernel, _flat_net(3.0))
    corners = [(1.5, -1.0, -1.0), (1.5, 4.0, -1.0), (1.5, 4.0, 1.0), (1.5, -1.0, 1.0)]
    b = kernel.add_planar_face(corners)
    result = kernel.fragment([a], [b])
    assert len(result.from_a) == 1 and len(result.from_b) == 1
    assert len(result.from_a[0]) == 2  # the plate is cut in two along the crossing
    assert sum(kernel.area(f) for f in result.from_a[0]) == pytest.approx(9.0, rel=1e-6)
    halves = set(result.from_a[0])
    joined = [e for e in kernel.boundary(result.from_a[0][0]) if set(kernel.adjacent(e)) == halves]
    assert len(joined) == 1, "the two halves meet along exactly one imprinted edge"


def test_geom_iges_kernel_copy_transform_and_remove_edit_the_model(kernel):
    face = _bicubic(kernel, _flat_net(3.0))
    (twin,) = kernel.copy([face])
    kernel.transform([twin], [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert kernel.centroid(twin) == pytest.approx((11.5, 1.5, 0.0), abs=1e-9)
    kernel.remove([twin])
    assert kernel.faces() == (face,)


def test_geom_iges_kernel_ribbon_and_triangulation(kernel):
    pts = [(float(i), 0.0, 0.0) for i in range(5)]
    ribbon = kernel.add_ribbon(pts, [(0.0, 1.0, 0.0)] * 5)
    assert kernel.area(ribbon) == pytest.approx(4.0, rel=1e-6)
    nodes, triangles = kernel.triangulate([ribbon], 1.0)
    assert nodes and triangles
    assert all(0 <= i < len(nodes) for t in triangles for i in t)


def test_geom_iges_kernel_export_then_import_keeps_the_group_names_and_the_area(kernel, tmp_path):
    a = _bicubic(kernel, _flat_net(3.0))
    b = _bicubic(kernel, _flat_net(2.0, z=5.0, x0=10.0))
    path = tmp_path / "two.igs"
    kernel.export_iges({"ALPHA": [a], "BETA": [b]}, path)
    named = kernel.import_iges(path)
    assert sorted(named.names) == ["ALPHA", "BETA"]
    areas = {n: kernel.area(f) for n, f in zip(named.names, named.faces, strict=True)}
    assert areas["ALPHA"] == pytest.approx(9.0, rel=1e-6)
    assert areas["BETA"] == pytest.approx(4.0, rel=1e-6)


def test_geom_iges_kernel_the_kernel_initializes_off_the_main_thread():
    _need_gmsh()
    import threading

    from cfdbender.mesh._gmsh import open_kernel

    box = {}

    def work():
        try:
            k = open_kernel("thread")
            box["area"] = k.area(_bicubic(k, _flat_net(3.0)))
            k.close()
        except Exception as error:  # noqa: BLE001
            box["error"] = error

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert "error" not in box, box.get("error")
    assert box["area"] == pytest.approx(9.0, rel=1e-9)


def _block_gmsh(monkeypatch, error):
    import importlib.abc

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "gmsh":
                raise error
            return None

    monkeypatch.delitem(sys.modules, "gmsh", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Finder(), *sys.meta_path])


@pytest.mark.parametrize(
    ("error", "word"),
    [
        (ImportError("No module named 'gmsh'"), "No module named"),
        (OSError("libGLU.so.1: cannot open shared object file"), "libGLU.so.1"),
    ],
)
def test_geom_iges_kernel_an_unusable_kernel_library_gives_one_named_error(
    monkeypatch, error, word
):
    from cfdbender.mesh._extras import KernelUnavailableError, MissingExtraError
    from cfdbender.mesh._gmsh import gmsh_module

    _block_gmsh(monkeypatch, error)
    with pytest.raises(KernelUnavailableError) as caught:
        gmsh_module("the test")
    assert isinstance(caught.value, MissingExtraError)
    message = str(caught.value)
    assert word in message and "libglu1-mesa" in message and "still run" in message
    assert caught.value.__cause__ is error


def test_geom_iges_kernel_the_unavailable_error_pickles_and_is_catalogued():
    import pickle

    from cfdbender.mesh._extras import KernelUnavailableError

    error = KernelUnavailableError(purpose="p", reason="r")
    clone = pickle.loads(pickle.dumps(error))
    assert str(clone) == str(error) and clone.reason == "r"


# --- the reader profile gate -----------------------------------------------------------------


def _entity(ident, kind, parameters, form=0):
    d1 = [f"{kind:8d}"] + [f"{0:8d}"] * 7 + ["00000000"]
    d2 = [f"{kind:8d}", f"{0:8d}", f"{0:8d}", f"{0:8d}", f"{form:8d}"] + [" " * 8] * 4
    return {"id": ident, "type": kind, "d1": d1, "d2": d2, "parameters": parameters}


@pytest.fixture
def control(tmp_path):
    """A written file with one named, trimmed, clamped, non-rational bicubic face of four sides."""
    _need_gmsh()
    from cfdbender.geometry._iges_write import read_iges
    from cfdbender.mesh._gmsh import open_kernel

    k = open_kernel("gate")
    try:
        face = _bicubic(k, _flat_net(3.0))
        k.export_iges({"PANEL": [face]}, tmp_path / "c.igs")
    finally:
        k.close()
    document = read_iges(tmp_path / "c.igs")
    return tmp_path, _with_boundary(document, 4)


def _with_boundary(document, sides):
    face = next(e for e in document["entities"] if e["type"] == 144)
    surface = next(e for e in document["entities"] if e["type"] == 128)
    written = next(e for e in document["entities"] if e["type"] == 142)
    top = max(e["id"] for e in document["entities"]) + 2
    curve, trim = top, top + 2
    # the written parameter-space curve stays the BPTR (R21-004: a 142 with BPTR 0 is read by
    # the solver as no face, and the gate warns on it); the planted model-space composite
    # carries the side count
    bptr = written["parameters"][3]
    document["entities"] += [
        _entity(curve, 102, ["102", str(sides), *["0"] * sides]),
        _entity(trim, 142, ["142", "1", str(surface["id"]), bptr, str(curve), "1"]),
    ]
    face["parameters"][4] = str(trim)
    return document


def _write(directory, document, name):
    from cfdbender._textio import write_text

    path = directory / name
    write_text(path, serialize_iges(document), encoding="ascii")
    return path


def _checks(report):
    return sorted({f.check for f in report.findings})


def test_geom_iges_kernel_the_control_file_passes_the_gate_with_no_finding(control):
    from cfdbender.geometry._geom_gate import check_profile

    directory, document = control
    report = check_profile(_write(directory, document, "ok.igs"))
    assert report.findings == () and report.passed and report.surfaces == 1


def _rational(document):
    surface = next(e for e in document["entities"] if e["type"] == 128)
    weights_at = 10 + 8 + 8
    surface["parameters"][weights_at + 5] = "2."


def _unclamped(document):
    surface = next(e for e in document["entities"] if e["type"] == 128)
    surface["parameters"][10 + 1] = "0.1"


def _unclamped_at(position):
    def plant(document):
        surface = next(e for e in document["entities"] if e["type"] == 128)
        surface["parameters"][position] = "0.1" if position in (11, 19) else "0.9"

    return plant


def _no_name(document):
    face = next(e for e in document["entities"] if e["type"] == 144)
    del face["parameters"][5:]


@pytest.mark.parametrize(
    ("plant", "check"),
    [
        (_rational, "profile.rational"),
        (_unclamped, "profile.unclamped"),
        (_unclamped_at(16), "profile.unclamped"),
        (_unclamped_at(19), "profile.unclamped"),
        (_unclamped_at(24), "profile.unclamped"),
        (_no_name, "profile.name"),
    ],
)
def test_geom_iges_kernel_each_planted_property_is_caught_and_named(control, plant, check):
    from cfdbender.geometry._geom_gate import check_profile

    directory, document = control
    plant(document)
    report = check_profile(_write(directory, document, "bad.igs"))
    assert _checks(report) == [check] and not report.passed


def test_geom_iges_kernel_a_three_sided_face_fails(control):
    from cfdbender.geometry._geom_gate import check_profile
    from cfdbender.geometry._iges_write import read_iges

    directory, document = control
    path = _write(directory, _with_boundary(read_iges(directory / "c.igs"), 3), "tri.igs")
    assert _checks(check_profile(path)) == ["profile.sides"]


def test_geom_iges_kernel_an_unobserved_degree_warns_and_does_not_fail(control):
    from cfdbender.geometry._geom_gate import check_profile
    from cfdbender.geometry.iges_profile import ReaderProfile

    directory, document = control
    path = _write(directory, document, "deg.igs")
    report = check_profile(path, ReaderProfile(degrees=(5,)))
    assert _checks(report) == ["profile.degree"] and report.passed
    assert {f.severity for f in report.findings} == {"warn"}
    limited = check_profile(path, ReaderProfile(degrees=(5,), max_degree=2))
    assert not limited.passed


def test_geom_iges_kernel_solid_entity_refuses(control):
    from cfdbender._errors import InputArtifactError
    from cfdbender.geometry._geom_gate import check_profile

    directory, document = control
    document["entities"].append(_entity(99, 186, ["186", "0"]))
    path = _write(directory, document, "solid.igs")
    with pytest.raises(InputArtifactError, match="surfaces only"):
        check_profile(path)
