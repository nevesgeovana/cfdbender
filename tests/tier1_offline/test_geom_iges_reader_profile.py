"""The IGES profile gate: the observed sets and the findings it names on a written file."""

from __future__ import annotations

from cfdbender.geometry.iges_profile import OBSERVED, ReaderProfile, surface_findings


def test_geom_iges_reader_profile_a_degree_the_probe_did_not_read_still_warns():
    entity = _entity(k1=5, k2=5, m1=4, m2=4)
    found = surface_findings(entity, OBSERVED)
    assert [f.check for f in found] == ["profile.degree", "profile.degree"]
    assert all(f.severity == "warn" for f in found)


def test_geom_iges_reader_profile_observed_degrees_raise_no_finding():
    for du, dv in ((5, 1), (3, 3), (1, 1), (5, 5)):
        assert surface_findings(_entity(k1=du + 2, k2=dv + 2, m1=du, m2=dv), OBSERVED) == []


def test_geom_iges_reader_profile_a_profile_without_sets_keeps_the_single_degree_rule():
    assert ReaderProfile(degrees_u=None, degrees_v=None).observed("u") == (3,)


def _entity(k1: int, k2: int, m1: int, m2: int) -> dict:
    """A clamped, non-rational 128 parameter list for a net of (k1 + 1) by (k2 + 1) points."""
    n1, n2 = k1 + m1 + 2, k2 + m2 + 2
    inner1 = [i / (n1 - 2 * m1 - 1) for i in range(1, n1 - 2 * m1 - 1)]
    t1 = [0.0] * (m1 + 1) + inner1 + [1.0] * (m1 + 1)
    inner2 = [i / (n2 - 2 * m2 - 1) for i in range(1, n2 - 2 * m2 - 1)]
    t2 = [0.0] * (m2 + 1) + inner2 + [1.0] * (m2 + 1)
    weights = [1.0] * ((k1 + 1) * (k2 + 1))
    values = ["128", str(k1), str(k2), str(m1), str(m2), "0", "0", "0", "0", "0"]
    values += [str(v) for v in t1 + t2 + weights]
    return {"id": 1, "parameters": values}
