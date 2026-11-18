"""Mode precedence, weaker verification, context lifetime and command transitions."""

import itertools
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_optcheck_identity import cli, small_threshold, workspace  # noqa: F401

from cfdbender.core import plan
from cfdbender.core.registry import RunRegistry
from cfdbender.matrix import cli as cli_module
from cfdbender.matrix import receipt


@pytest.mark.parametrize("setting", ["digest", "identity"])
@pytest.mark.parametrize("env", ["", "0", "false", " 1 "])
def test_switch_precedence(tmp_path, monkeypatch, setting, env):
    (tmp_path / "init.toml").write_text(f'large_input_check="{setting}"\n')
    monkeypatch.setenv(plan.NO_READ_ENV, env)
    expected = env in ("", "0") and setting == "digest"
    assert plan.reads_large_inputs(tmp_path) == expected
    with plan.large_input_check("identity"):
        assert not plan.reads_large_inputs(tmp_path)
        with plan.rehash_inputs():
            assert plan.reads_large_inputs(tmp_path)
    with plan.rehash_inputs():
        assert plan.reads_large_inputs(tmp_path)
    assert plan.reads_large_inputs(tmp_path) == expected


def test_context_resets_after_exception_and_threads_are_independent(tmp_path):
    with pytest.raises(RuntimeError):
        with plan.large_input_check("identity"), plan.rehash_inputs():
            assert plan.reads_large_inputs(tmp_path)
            raise RuntimeError("abort")
    assert plan.reads_large_inputs(tmp_path)
    with plan.large_input_check("identity"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(plan.reads_large_inputs, tmp_path).result()
        assert not plan.reads_large_inputs(tmp_path)
    assert plan.reads_large_inputs(tmp_path)


@pytest.mark.parametrize("replace", [False, True])
def test_preserved_size_and_mtime_blind_spot(tmp_path, replace):
    path = tmp_path / "mesh"
    path.write_bytes(b"original" * 256)
    os.utime(path, ns=(10**18, 10**18))
    expected = plan.identity_value(path.stat())
    if replace:
        path.unlink()
    path.write_bytes(b"different"[:8] * 256)
    os.utime(path, ns=(10**18, 10**18))
    with plan.large_input_check("identity"):
        assert plan.InputFiles(tmp_path).agrees("mesh", expected)


def test_timestamp_rounding_can_refuse_unchanged_bytes(tmp_path):
    path = tmp_path / "mesh"
    path.write_bytes(b"x" * 2048)
    os.utime(path, ns=(10**18, 10**18 + 123456700))
    expected = plan.identity_value(path.stat())
    os.utime(path, ns=(10**18, 10**18))
    assert not plan.InputFiles(tmp_path).agrees("mesh", expected)


@pytest.mark.parametrize("modes", list(itertools.product(["digest", "identity"], repeat=3)))
def test_all_plan_run_collect_modes(tmp_path, monkeypatch, modes):
    monkeypatch.setattr(cli_module, "require_receipt", receipt.require_receipt)
    root = workspace(tmp_path / "ws", "su2")
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text().replace('[groups]\nwing = ["wall"]\n', ""))
    for command, mode in zip(("plan", "run", "collect"), modes, strict=True):
        (root / plan.DigestCache.NAME).unlink(missing_ok=True)
        words = ["--local"] if command == "run" else []
        if mode == "identity":
            words.append("--no-read-large-inputs")
        assert cli(root, command, *words) == 0
        if command == "run":
            value = RunRegistry(root).read()[0].point.inputs_sha256["inputs/geometries/large.su2"]
            assert plan.is_identity_value(value) == (mode == "identity")


def test_warning_refusals_does_not_escalate_optional_check(tmp_path):
    root = workspace(tmp_path / "ws", "su2")
    assert cli(root, "plan", "--no-read-large-inputs", "--warning-refusals") == 0


def test_warning_scope_resets_per_command():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            with plan.large_input_check("identity"):
                plan.warn_once("file", "not read")
                plan.warn_once("file", "not read")
    assert len(caught) == 2
