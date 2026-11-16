"""geometry: check_against_original, the deviation of a rebuilt surface from an original.

Ported from the research check tests. The tests that need the geometry kernel skip where the
optional extra is absent.
"""

import numpy as np
import pytest
from kernel_support import kernel_or_skip

from cfdbender._errors import InputValueError

LIMITS = {"chord_m": 0.2, "forward_count": 13, "reverse_count": 13, "edge_count": 19}


@pytest.fixture
def occ():
    kernel_or_skip()
    from cfdbender.mesh._gmsh import session

    with session("a test") as gmsh:
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.add("test")
        yield gmsh.model.occ, gmsh


def check(*args, **kwargs):
    from cfdbender.geometry._geometry_check import check_against_original

    return check_against_original(*args, **kwargs)


def test_geom_kernel_check_self_and_one_mm_offset(occ):
    model, _ = occ
    face = model.addRectangle(0, 0, 0, 0.2, 1)
    offset = model.addRectangle(0, 0, 0.001, 0.2, 1)
    model.synchronize()
    identity = check([face], [face], LIMITS)
    assert identity["accepted"]
    assert identity["forward"]["max_m"] == pytest.approx(0, abs=1e-12)
    assert identity["restricted_reverse"]["max_m"] == pytest.approx(0, abs=1e-12)
    result = check([offset], [face], LIMITS)
    assert not result["accepted"]
    assert result["forward"]["max_mm"] == pytest.approx(1, abs=1e-9)
    assert result["restricted_reverse"]["max_mm"] == pytest.approx(1, abs=1e-9)
    assert result["forward"]["exceedance_count"] == result["forward"]["samples"]


def test_geom_kernel_check_reverse_excludes_extension_beyond_trim(occ):
    model, _ = occ
    original = model.addRectangle(0, 0, 0, 1, 1)
    rebuilt = model.addRectangle(-1, -1, 0, 3, 3)
    model.synchronize()
    result = check([rebuilt], [original], LIMITS)
    assert result["accepted"]
    assert result["unrestricted_reverse"]["max_m"] > 1
    assert result["restricted_reverse"]["max_m"] < 1e-12
    assert result["reverse_excluded_boundary_samples"] > 0


def test_geom_kernel_check_forward_includes_trim_hole_edges(occ):
    model, _ = occ
    outer = model.addRectangle(0, 0, 0, 1, 1)
    hole = model.addDisk(0.5, 0.5, 0, 0.15, 0.15)
    original, _ = model.cut([(2, outer)], [(2, hole)])
    rebuilt = model.addRectangle(0, 0, 0.001, 1, 1)
    model.synchronize()
    result = check([rebuilt], original, LIMITS)
    assert not result["accepted"]
    assert result["forward"]["max_mm"] == pytest.approx(1, abs=1e-8)
    assert result["reverse_excluded_boundary_samples"] > 0


def test_geom_kernel_check_check_names_its_profile_and_refuses_an_unknown_one(occ):
    model, _ = occ
    face = model.addRectangle(0, 0, 0, 0.2, 1)
    model.synchronize()
    result = check([face], [face], LIMITS)
    assert result["profile"] == "study_default"
    assert result["tolerance_rule"] == "max(0.5 mm, 0.001 times local chord)"
    with pytest.raises(InputValueError, match="unknown; available"):
        check([face], [face], LIMITS, profile="nope")


def test_geom_kernel_check_check_of_a_generated_iges_against_itself_is_accepted(tmp_path):
    kernel_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges

    path = tmp_path / "fus.igs"
    generate_iges({"components": {"FUS": {"type": "body", "length_mm": 2000.0}}}, path)
    result = check(
        f"{path}#FUS",
        f"{path}#FUS",
        {"chord_m": 2.0, "forward_count": 9, "reverse_count": 9, "edge_count": 9},
    )
    assert result["accepted"]
    assert result["forward"]["max_m"] < 1e-9
    assert result["restricted_reverse"]["samples"] > 0


def test_geom_kernel_check_check_refuses_an_unknown_component_name(tmp_path):
    kernel_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges

    path = tmp_path / "fus.igs"
    generate_iges({"components": {"FUS": {"type": "body", "length_mm": 2000.0}}}, path)
    with pytest.raises(InputValueError, match="no faces named NOPE"):
        check(f"{path}#NOPE", f"{path}#FUS", LIMITS)


def test_geom_kernel_check_local_limit_floor_and_chord():
    from cfdbender.geometry._geometry_check import local_limits

    points = np.array([[0, 0, 0], [0, 1, 0]])
    limits = local_limits(points, {"stations_m": [0, 1], "chords_m": [0.2, 2]})
    np.testing.assert_allclose(limits, [0.0005, 0.002])


def test_geom_kernel_check_local_limit_matches_the_named_profile_function():
    from cfdbender.geometry._boundary import deviation_limit_mm
    from cfdbender.geometry._geometry_check import local_limits

    chords = np.array([0.05, 0.4, 3.0])
    points = np.zeros((3, 3))
    got = local_limits(points, {"local_chord": lambda p: chords})
    expected = [deviation_limit_mm(c * 1000.0) / 1000.0 for c in chords]
    np.testing.assert_allclose(got, expected, rtol=1e-15, atol=0)


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"stations_m": [0, 0], "chords_m": [1, 1]}, "increasing stations"),
        ({"chord_m": -1.0}, "positive finite local chord"),
        ({"chord_m": float("nan")}, "positive finite local chord"),
    ],
)
def test_geom_kernel_check_local_limits_refuses_a_bad_chord(limits, message):
    from cfdbender.geometry._geometry_check import local_limits

    with pytest.raises(InputValueError, match=message):
        local_limits(np.zeros((2, 3)), limits)


def test_geom_kernel_check_local_limits_refuses_limits_that_name_no_chord():
    from cfdbender.geometry._geometry_check import local_limits

    with pytest.raises(InputValueError, match="chord_m"):
        local_limits(np.zeros((2, 3)), {"forward_count": 13})
