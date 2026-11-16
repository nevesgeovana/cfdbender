"""Geometry: the S2 identity report lists a planted seed mismatch and never fails.

The seed numbers are closed forms: a NACA 4-digit section of thickness t peaks at 0.30 chord
and its maximum thickness is t (finite trailing edge: within the fit tolerance).
"""

from __future__ import annotations

import pytest

from cfdbender.geometry._airfoils import naca_to_cst
from cfdbender.geometry._sections import (
    CstSection,
    identity_report,
    section_from_stored,
    thickness_profile,
)


def naca(digits):
    return section_from_stored(naca_to_cst(f"NACA{digits}"))


def test_identity_seed_numbers_follow_the_family():
    seed = thickness_profile(naca("0012"))
    assert seed["thickness"] == pytest.approx(0.12, abs=2e-3)
    assert seed["position"] == pytest.approx(0.30, abs=2e-2)
    assert thickness_profile(naca("0015"))["thickness"] == pytest.approx(0.15, abs=2e-3)


def test_identity_lists_a_planted_mismatch_and_a_match():
    report = identity_report("0015", {"cut_a": naca("0012"), "cut_b": naca("0015")})
    by_label = {r["section"]: r for r in report}
    assert by_label["cut_a"]["status"] == "mismatch"
    assert by_label["cut_a"]["difference"]["thickness"] == pytest.approx(-0.03, abs=3e-3)
    assert by_label["cut_b"]["status"] == "match"
    assert all(r["id"] == "identity.listed" for r in report)


def test_identity_reports_a_moved_maximum_thickness_position():
    # A thinner section whose maximum lies aft of the seed's, as the cuts of an original may.
    aft = CstSection((0.2, 0.1, 0.45, 0.15, 0.1), (-0.2, -0.1, -0.45, -0.15, -0.1), 0.0)
    record = identity_report("0012", {"aft": aft})[0]
    assert record["status"] == "mismatch"
    assert record["measured"]["position"] > record["seed"]["position"] + 0.05


def test_identity_without_measurement_is_seed_only_and_accepts_both_spellings():
    only = identity_report("NACA0012")
    assert only[0]["status"] == "seed only"
    assert "measured" not in only[0]
    assert identity_report("0012")[0]["seed"] == only[0]["seed"]
    assert identity_report("0012", {})[0]["status"] == "seed only"


def test_identity_never_raises_on_a_mismatch_and_refuses_a_bad_designation():
    from cfdbender._errors import InputValueError

    assert identity_report("0012", {"x": naca("0030")})[0]["status"] == "mismatch"
    with pytest.raises(InputValueError):
        identity_report("00x2")
