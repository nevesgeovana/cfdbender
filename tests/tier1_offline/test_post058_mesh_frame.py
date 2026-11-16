"""P058-2: a mesh is expected in x aft, y right, z up; another frame is declared and converted.

The expected values follow from the declaration's meaning alone: a mesh whose +x points
forward and +y left has every x and y of the expected frame negated.
"""

from __future__ import annotations

import hashlib

import pytest

from cfdbender.core.mesh_frame import (
    EXPECTED_FRAME,
    convert_mesh_text,
    convert_point,
    declared_frame,
    mesh_frame_matrix,
    mesh_frame_record,
)

OBJ = "# mesh\nv 1.0 2.0 3.0\nv 4 5 6\ng WING\nf 1 2 1\ng TAIL\nvn 0 1 0\nf 2 1 2\n"


def test_absent_is_the_expected_frame_and_converts_nothing():
    assert declared_frame(None) == EXPECTED_FRAME == ("aft", "right", "up")
    assert convert_mesh_text("m.obj", OBJ, None) is OBJ
    assert convert_mesh_text("m.fsm", "anything", ["AFT", "Right", "up"]) == "anything"


def test_forward_left_up_is_converted_with_faces_and_groups_kept():
    out = convert_mesh_text("m.obj", OBJ, ["forward", "left", "up"])
    lines = out.splitlines()
    assert lines[1] == "v -1.0 -2.0 3.0" and lines[2] == "v -4.0 -5.0 6.0"
    assert lines[6] == "vn 0.0 -1.0 0.0"
    kept = [line for line in lines if not line.startswith(("v ", "vn "))]
    assert kept == [line for line in OBJ.splitlines() if not line.startswith(("v ", "vn "))]


def test_a_rotated_frame_maps_each_axis_where_it_points():
    # +x up, +y right, +z forward: right-handed (up x right = forward).
    assert convert_point((1.0, 0.0, 0.0), ["up", "right", "forward"]) == (0.0, 0.0, 1.0)
    assert convert_point((0.0, 0.0, 1.0), ["up", "right", "forward"]) == (-1.0, 0.0, 0.0)
    assert mesh_frame_matrix(None)[0] == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("declared", "match"),
    [
        (["aft", "left", "up"], "left-handed"),
        (["aft", "forward", "up"], "one axis twice"),
        (["aft", "right", "sideways"], "sideways"),
        (["aft", "right"], "three words"),
    ],
)
def test_a_bad_declaration_is_refused_by_name(declared, match):
    with pytest.raises(ValueError, match=match):
        declared_frame(declared)


def test_another_format_declared_in_another_frame_is_refused():
    with pytest.raises(ValueError, match="only an OBJ mesh"):
        convert_mesh_text("wing.fsm", "x", ["forward", "left", "up"])


def test_the_record_states_declaration_matrix_and_both_digests():
    out = convert_mesh_text("m.obj", OBJ, ["forward", "left", "up"])
    record = mesh_frame_record(["forward", "left", "up"], OBJ.encode(), out.encode())
    assert record["declared"] == ["forward", "left", "up"]
    assert record["matrix"] == [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    assert record["source_sha256"] == hashlib.sha256(OBJ.encode()).hexdigest()
    assert record["converted_sha256"] == hashlib.sha256(out.encode()).hexdigest()
