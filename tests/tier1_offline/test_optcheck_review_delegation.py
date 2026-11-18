"""The optional input check is a facade option, not an upstream CLI argument."""

import argparse

import pytest

from cfdbender import _pyfs
from cfdbender.matrix.cli import main


@pytest.mark.parametrize("verb", ["plan", "run", "collect"])
def test_flightstream_warns_and_strips_local_option(tmp_path, monkeypatch, capsys, verb):
    (tmp_path / "init.toml").write_text('schema_version=1\nsolver="flightstream"\n')
    calls = []

    def upstream(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("verb")
        parser.add_argument("matrix")
        parser.add_argument("--workspace")
        try:
            parser.parse_args(argv)
        except SystemExit as error:
            return error.code
        calls.append(argv)
        return 0

    monkeypatch.setattr(_pyfs, "matrix_main", upstream)
    argv = [verb, "study", "--workspace", str(tmp_path)]
    assert main([*argv, "--no-read-large-inputs"]) == 0
    assert calls == [argv]
    assert "FlightStream" in capsys.readouterr().err
