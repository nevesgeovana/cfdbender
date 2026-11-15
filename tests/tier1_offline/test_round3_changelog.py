from pathlib import Path


def test_freestream_changelog_does_not_claim_dimensional_default():
    root = Path(__file__).resolve().parents[2]
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "The default nondimensionalization is unchanged (SU2's `DIMENSIONAL`)" not in text
    assert "FREESTREAM_PRESS_EQ_ONE" in text


def test_solver_comparison_acknowledges_neutral_cfl_controls():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/solvers.md").read_text(encoding="utf-8")
    assert "there is no neutral key" not in text
    assert "`cfl`, `cfl_max` and `cfl_growth`" in text


def test_openfoam_docs_distinguish_relaxation_from_residual_control():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/openfoam-workspaces.md").read_text(encoding="utf-8")
    assert "`nuTilda` is written only for the turbulent case" not in text
    assert "default relaxation factors include `nuTilda` even for laminar cases" in text
