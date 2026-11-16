"""P058-5: the families, the default groups, their ids and names, and the column registry."""

from __future__ import annotations

import pytest

from cfdbender.matrix._family_columns import (
    SOLVER_ROTORS,
    Family,
    Rotor,
    acoef_columns,
    check_group_name,
    column_names,
    dim_columns,
    family_file_name,
    group_file_name,
    group_id,
    polar_groups,
    rcoef_columns,
    solver_families,
)


def test_group_ids_are_total_zero_airframe_one_and_ten_plus_the_rotor_id():
    assert (group_id("TOTAL"), group_id("AIRFRAME"), group_id("rotor", rotor_id=2)) == (0, 1, 12)
    with pytest.raises(ValueError):
        group_id("rotor")


def test_an_airframe_solver_writes_total_and_airframe_and_names_the_rotor_skip():
    groups, skipped = polar_groups(solver_families("su2"), SOLVER_ROTORS.get("su2", ()))
    assert [(g.name, g.group_id, g.members) for g in groups] == [
        ("TOTAL", 0, (1,)),
        ("AIRFRAME", 1, (1,)),
    ]
    assert [name for name, _ in skipped] == ["ROTOR"]
    assert "no rotor group" in skipped[0][1]


def test_qprop_writes_total_and_its_rotor_and_skips_the_empty_airframe():
    families = solver_families("qprop")
    assert families == (Family("BLADES", 1, 0),)
    groups, skipped = polar_groups(families, SOLVER_ROTORS["qprop"])
    assert [(g.name, g.group_id, g.rotor_id) for g in groups] == [
        ("TOTAL", 0, 0),
        ("PROPELLER", 10, 0),
    ]
    assert skipped == (
        ("AIRFRAME", "group AIRFRAME has no member family, so no AIRFRAME file is written"),
    )


def test_two_rotors_and_an_airframe_family():
    families = (Family("WING", 1), Family("B1", 2, 0), Family("B2", 3, 1))
    rotors = (Rotor("LEFT", 1, (-1, 0, 0)), Rotor("RIGHT", 0, (-1, 0, 0)))
    groups, skipped = polar_groups(families, rotors)
    assert [(g.name, g.group_id, g.members, g.rotor_id) for g in groups] == [
        ("TOTAL", 0, (1, 2, 3), None),
        ("AIRFRAME", 1, (1,), None),
        ("RIGHT", 10, (2,), 0),
        ("LEFT", 11, (3,), 1),
    ]
    assert skipped == ()


@pytest.mark.parametrize("name", ["TOTAL", "airframe", "FAM", "GRP", "P01x", "A-B", ""])
def test_a_reserved_or_unreadable_group_name_is_refused(name):
    with pytest.raises(ValueError, match=repr(name)):
        check_group_name(name)


def test_file_names_and_headers():
    assert family_file_name("1001", "ACOEF") == "FAM-ACOEF-P1001.csv"
    assert group_file_name("1001", "TOTAL", "DIM") == "P1001-TOTAL-DIM.csv"
    with pytest.raises(ValueError):
        family_file_name("1001", "SUPER")
    assert column_names(dim_columns()) == (
        "FAMILY", "FAMILY_ID", "FX_MRP", "FY_MRP", "FZ_MRP", "MX_MRP", "MY_MRP", "MZ_MRP",
        "MESH_FACES",
    )  # fmt: skip
    acoef = column_names(acoef_columns((0,), own=("CDW", "CT"), entity="group"))
    assert acoef[:8] == ("GROUP", "GROUP_ID", "CDW", "CYW", "CLW", "ClW", "CmW", "CnW")
    assert acoef[-8:] == ("CDR0", "CYR0", "CLR0", "ClR0", "CmR0", "CnR0", "CT", "MESH_FACES")
    rcoef = column_names(rcoef_columns((0,)))
    assert rcoef[:6] == ("FAMILY", "FAMILY_ID", "ROTOR_ID", "T", "Q", "P")
    assert rcoef[-4:] == ("CTR0", "CQR0", "ETAR0", "MESH_FACES")
    with pytest.raises(ValueError):
        rcoef_columns(())
