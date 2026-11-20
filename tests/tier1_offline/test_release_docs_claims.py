"""Release guides must retain the implemented limits and upgrade exceptions."""

import runpy
from pathlib import Path

import pytest

import cfdbender
from cfdbender.core.run_types import RESTART_KEYS

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "path,claim",
    [
        ("docs/upgrading.md", "Nothing breaks."),
        (
            "guide/latex-sources/solvers/text/02-su2-openfoam-pyfr.tex",
            "a key the setup states wins",
        ),
        (
            "guide/latex-sources/solvers/text/03-xfoil-qprop-choosing.tex",
            "The package makes none",
        ),
    ],
)
def test_guides_do_not_make_superseded_universal_claims(path, claim):
    text = " ".join((ROOT / path).read_text(encoding="utf-8").split())
    assert claim not in text


def test_guide_checks_find_checkout_when_package_is_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cfdbender, "__file__", str(tmp_path / "site-packages/cfdbender/__init__.py")
    )
    loaded = runpy.run_path(__file__)
    assert loaded["ROOT"] == Path(__file__).resolve().parents[2]


def test_docs_matrix_columns_include_restart_keys():
    text = (ROOT / "docs/matrix-columns.md").read_text(encoding="utf-8")
    assert all(key in text for key in RESTART_KEYS)
