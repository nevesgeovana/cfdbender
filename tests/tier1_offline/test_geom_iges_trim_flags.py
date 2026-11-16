"""A written face has a parameter-space outer boundary flagged 2D parametric.

The curves a 142 names by BPTR carry entity use flag 05, and a face with no BPTR curve (a
natural boundary, N1 = 0, or a plane) is written as a B-spline copy that carries one. The
structural tests read the written file only; no test runs a solver.
"""

from __future__ import annotations

import inspect
import math

import pytest
from kernel_support import kernel_or_skip

from cfdbender.geometry._iges_write import read_iges


@pytest.fixture
def kernel():
    kernel_or_skip()
    from cfdbender.mesh._gmsh import open_kernel

    k = open_kernel("test")
    try:
        yield k
    finally:
        k.close()


def _use(entity) -> str:
    """The entity use flag: digits 5 and 6 of the directory entry status number (field 9)."""
    return entity["d1"][8][4:6]


def _outer_boundaries(document):
    """Per 144: its outer 142 (None when N1 = 0) and the curves its BPTR reaches."""
    by_id = {e["id"]: e for e in document["entities"]}
    out = []
    for face in (e for e in document["entities"] if e["type"] == 144):
        values = face["parameters"]
        outer = by_id.get(int(values[4] or 0)) if int(values[2] or 0) else None
        curves = []
        if outer is not None:
            stack = [int(outer["parameters"][3] or 0)]
            while stack:
                member = by_id.get(stack.pop())
                if member is None:
                    continue
                curves.append(member)
                if member["type"] == 102:
                    p = member["parameters"]
                    stack += [int(v) for v in p[2 : 2 + int(p[1])]]
        out.append((outer, curves))
    return out


def _assert_readable(path):
    boundaries = _outer_boundaries(read_iges(path))
    assert boundaries
    for outer, curves in boundaries:
        assert outer is not None, "a 144 with N1 = 0 (no outer 142) is read as no face"
        assert curves, "an outer 142 with no BPTR curve is read as no face"
        assert [_use(c) for c in curves] == ["05"] * len(curves)
        assert _use(outer) == "00"


def test_geom_iges_trim_flags_trimmed_faces_flag_their_parameter_space_curves_05(kernel, tmp_path):
    net = [[(2.0 * i / 5.0, 3.0 * j, 0.1 * math.sin(i)) for j in range(2)] for i in range(6)]
    base = kernel.add_bspline_surface(
        net, knots_u=[0.0, 1.0], knots_v=[0.0, 1.0], mult_u=[6, 6], mult_v=[2, 2], deg_u=5, deg_v=1
    )  # fmt: skip
    box = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
    face = kernel.add_trimmed_face(base, [[[box[k], box[(k + 1) % 4]] for k in range(4)]])
    kernel.remove([base])
    path = tmp_path / "trimmed.igs"
    kernel.export_iges({"WING": [face]}, path)
    _assert_readable(path)


def test_geom_iges_trim_flags_natural_and_planar_faces_are_written_with_a_parameter_space_boundary(
    kernel, tmp_path
):
    net = [[(float(i), float(j), 0.1 * i * j) for j in range(2)] for i in range(4)]
    skin = kernel.add_bspline_surface(
        net, knots_u=[0.0, 1.0], knots_v=[0.0, 1.0], mult_u=[4, 4], mult_v=[2, 2], deg_u=3, deg_v=1
    )  # fmt: skip
    plate = kernel.add_planar_face(
        [(0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (1.0, 1.0, 2.0), (0.0, 1.0, 2.0)]
    )
    before = kernel.faces()
    areas = sorted([kernel.area(skin), kernel.area(plate)])
    path = tmp_path / "natural.igs"
    kernel.export_iges({"SKIN": [skin], "PLATE": [plate]}, path)
    _assert_readable(path)
    assert kernel.faces() == before  # the B-spline copies left the model
    named = kernel.import_iges(path)
    assert sorted(named.names) == ["PLATE", "SKIN"]
    assert sorted(kernel.area(f) for f in named.faces) == pytest.approx(areas, rel=1e-6)


@pytest.mark.parametrize("planted", ["none", "use_00", "pref_2", "bptr_0", "n1_0"])
def test_geom_iges_trim_flags_the_gate_warns_on_a_boundary_the_direct_import_reads_as_no_face(
    kernel, tmp_path, planted
):
    """Each condition of the record, planted alone in a written file, is warned as profile.trim."""
    from cfdbender._textio import write_text
    from cfdbender.geometry._geom_gate import check_profile
    from cfdbender.geometry._iges_write import serialize_iges

    net = [[(float(i), float(j), 0.1 * i * j) for j in range(4)] for i in range(4)]
    skin = kernel.add_bspline_surface(
        net, knots_u=[0.0, 1.0], knots_v=[0.0, 1.0], mult_u=[4, 4], mult_v=[4, 4], deg_u=3, deg_v=3
    )  # fmt: skip
    path = tmp_path / "skin.igs"
    kernel.export_iges({"SKIN": [skin]}, path)
    document = read_iges(path)
    ((outer, curves),) = _outer_boundaries(document)
    face = next(e for e in document["entities"] if e["type"] == 144)
    if planted == "use_00":
        curves[0]["d1"][8] = curves[0]["d1"][8][:4] + "00" + curves[0]["d1"][8][6:]
    elif planted == "pref_2":
        outer["parameters"][5] = "2"
    elif planted == "bptr_0":
        outer["parameters"][3] = "0"
    elif planted == "n1_0":
        face["parameters"][2] = "0"
        face["parameters"][4] = "0"
    planted_path = tmp_path / f"{planted}.igs"
    write_text(planted_path, serialize_iges(document), encoding="ascii")
    trims = [f for f in check_profile(planted_path).findings if f.check == "profile.trim"]
    if planted == "none":
        assert trims == []
    else:
        assert [f.severity for f in trims] == ["warn"] and "iges-convert" in trims[0].detail


# -- R21-013: one rule for the parameter-space flags, used by the writer and by iges-convert ------


def _entity(number, kind, parameters):
    d1 = [f"{kind:8d}", f"{0:8d}", *[f"{0:8d}"] * 6, "00000000"]
    d2 = [f"{kind:8d}", f"{0:8d}", f"{0:8d}", f"{0:8d}", f"{0:8d}", *[" " * 8] * 4]
    return {"id": number, "type": kind, "d1": d1, "d2": d2, "parameters": parameters}


def _trim_document():
    """A 142 whose BPTR is a composite holding a line and a nested composite, and a 142 without."""
    line = ["110", "0.0", "0.0", "0.0", "1.0", "0.0", "0.0"]
    entities = [
        _entity(1, 142, ["142", "0", "5", "7", "9", "1"]),
        _entity(3, 142, ["142", "0", "5", "0", "9", "2"]),
        _entity(5, 110, line),
        _entity(7, 102, ["102", "2", "11", "13"]),
        _entity(9, 110, line),
        _entity(11, 110, line),
        _entity(13, 102, ["102", "2", "15", "7"]),
        _entity(15, 110, line),
        _entity(17, 110, line),
    ]
    return {"start": ["r21-013"], "global": ["1H,", "1H;"], "entities": entities}


def test_geom_iges_trim_flags_the_parameter_space_rule_reaches_nested_members_once():
    from cfdbender.geometry._iges_names import parameter_space_curves

    composites = {7: [11, 13], 13: [15, 7]}
    known = {5, 7, 9, 11, 13, 15, 17}

    def members(number):
        return composites.get(number, []) if number in known else None

    assert parameter_space_curves([7, 0, 99], members) == {7, 11, 13, 15}
    assert parameter_space_curves([], members) == set()


def test_geom_iges_trim_flags_the_writer_and_iges_convert_flag_the_same_curves_by_the_one_rule(
    tmp_path,
):
    from cfdbender._textio import write_text
    from cfdbender.geometry import _iges_names, _iges_write

    source = tmp_path / "trims.igs"
    write_text(source, _iges_write.serialize_iges(_trim_document()), encoding="ascii")
    written = _iges_write.flag_parameter_space(read_iges(source))
    converted = read_iges(_iges_names.flag_curves_on_surfaces(source, tmp_path / "flagged.igs"))
    by_writer = {e["id"]: _use(e) for e in written["entities"]}
    by_convert = {e["id"]: _use(e) for e in converted["entities"]}
    assert by_writer == by_convert
    assert {n for n, use in by_writer.items() if use == "05"} == {7, 11, 13, 15}

    # both callers delegate to the one rule, which lives once
    assert _iges_write.parameter_space_curves is _iges_names.parameter_space_curves
    for caller in (_iges_write.flag_parameter_space, _iges_names.flag_curves_on_surfaces):
        source_text = inspect.getsource(caller)
        assert "parameter_space_curves(" in source_text and "stack" not in source_text
