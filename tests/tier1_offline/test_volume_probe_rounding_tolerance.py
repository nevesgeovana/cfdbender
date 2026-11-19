import pytest
from tests.tier1_offline import test_post_volume_layouts_reference as reference


def test_probe_comparison_rejects_a_wrong_center(monkeypatch):
    original = reference.circle_points

    def shifted(*args):
        result = original(*args)
        result[0, 0] += 0.00001
        return result

    monkeypatch.setattr(reference, "circle_points", shifted)
    with pytest.raises(AssertionError):
        reference.test_a_probe_circle_lays_the_recorded_points(reference.REFERENCE["circles"][0])
