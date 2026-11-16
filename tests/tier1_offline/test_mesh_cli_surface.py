"""cfd-mesh mesh and mesh-generated: generated geometry meshed into inputs/mesh/ with a sidecar."""

from __future__ import annotations

import json

import pytest
from kernel_support import kernel_or_skip

from cfdbender.geometry.stages import write_config
from cfdbender.mesh._cli_support import provenance_path
from cfdbender.mesh.cli import build_parser, main

BODY = {"components": {"FUS": {"type": "body"}}}


def _run(args, capsys):
    capsys.readouterr()
    try:
        code = main([str(a) for a in args])
    except SystemExit as error:
        code = error.code
    return code, capsys.readouterr()


def test_mesh_cli_surface_mesh_generated_writes_the_obj_and_its_sidecar(tmp_path, capsys):
    kernel_or_skip()
    config = tmp_path / "body.toml"
    write_config(BODY, config)
    workspace = tmp_path / "ws"
    code, captured = _run(
        [
            "--workspace",
            workspace,
            "mesh-generated",
            config,
            "--obj",
            "fus.obj",
            "--component",
            "FUS",
        ],
        capsys,
    )
    assert code == 0, captured.err
    obj = workspace / "inputs" / "mesh" / "fus.obj"
    assert obj.is_file() and obj.read_text(encoding="utf-8").count("\nf ") > 10
    record = json.loads(provenance_path(obj).read_text(encoding="utf-8"))
    assert record["command"] == "cfd-mesh mesh-generated"
    assert record["mesh"] == "inputs/mesh/fus.obj"
    assert [s["path"] for s in record["sources"]] == [str(config.resolve())]


def test_mesh_cli_surface_mesh_refuses_a_descriptor_on_a_dry_run(tmp_path, capsys):
    code, captured = _run(
        ["mesh", tmp_path / "c.toml", "--controls", tmp_path / "m.toml", "--dry-run",
         "--descriptor", "x"],
        capsys,
    )  # fmt: skip
    assert code == 2 and "--descriptor" in captured.err


def test_mesh_cli_surface_mesh_refuses_a_missing_configuration(tmp_path, capsys):
    code, _ = _run(["mesh", tmp_path / "none.toml", "--controls", tmp_path / "m.toml"], capsys)
    assert code == 2


@pytest.mark.parametrize(
    "args", [["mesh-gen", "c.toml", "--obj", "o.obj"], ["--works", ".", "mesh"]]
)
def test_mesh_cli_surface_abbreviations_are_refused(args, capsys):
    code, _ = _run(args, capsys)
    assert code == 2


def test_mesh_cli_surface_every_verb_refuses_abbreviated_options():
    parser = build_parser()
    for action in parser._actions:
        for sub in getattr(action, "choices", None) or {}:
            assert action.choices[sub].allow_abbrev is False, sub
    assert parser.allow_abbrev is False
