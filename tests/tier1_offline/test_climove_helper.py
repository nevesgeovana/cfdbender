"""CLIM-1: warn-and-skip goes through one helper; init.toml may carry warning_refusals."""

from __future__ import annotations

import warnings

import pytest

from cfdbender.core import WorkspaceError, read_init
from cfdbender.core.refusals import refusing, warn_or_refuse, warning_refusals
from cfdbender.core.settings import read_warning_refusals


def test_clim1_a_skip_warns_by_default():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        warn_or_refuse("entry x is skipped")
    assert [str(w.message) for w in seen] == ["entry x is skipped"]
    assert not refusing()


def test_clim1_a_skip_is_a_refusal_inside_the_block_and_warns_again_after_it():
    with warning_refusals(True):
        assert refusing()
        with pytest.raises(WorkspaceError, match="entry x is skipped.*--warning-refusals"):
            warn_or_refuse("entry x is skipped")
    assert not refusing()


def test_clim1_the_warning_points_at_the_caller_not_the_helper():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        warn_or_refuse("entry x is skipped")
    assert seen[0].filename == __file__


def test_clim1_init_toml_carries_the_option(tmp_path):
    (tmp_path / "init.toml").write_text('schema_version = 1\nsolver = "mock"\n', encoding="utf-8")
    assert read_warning_refusals(tmp_path) is False
    (tmp_path / "init.toml").write_text(
        'schema_version = 1\nsolver = "mock"\nwarning_refusals = true\n', encoding="utf-8"
    )
    assert read_warning_refusals(tmp_path) is True
    assert read_init(tmp_path, available_solvers=["mock"]).warning_refusals is True
    (tmp_path / "init.toml").write_text(
        'schema_version = 1\nsolver = "mock"\nwarning_refusals = "yes"\n', encoding="utf-8"
    )
    with pytest.raises(WorkspaceError, match="warning_refusals must be true or false"):
        read_warning_refusals(tmp_path)
