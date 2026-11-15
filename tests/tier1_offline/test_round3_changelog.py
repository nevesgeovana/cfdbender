from pathlib import Path


def test_freestream_changelog_does_not_claim_dimensional_default():
    root = Path(__file__).resolve().parents[2]
    text = (root / 'changelog.d/fix87i-su2-freestream-keys.md').read_text(encoding='utf-8')
    assert "The default nondimensionalization is unchanged (SU2's `DIMENSIONAL`)" not in text
    assert 'FREESTREAM_PRESS_EQ_ONE' in text
