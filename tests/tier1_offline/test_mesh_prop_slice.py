"""The prop-slice verb: a propeller OBJ is cut into sections and a propeller definition."""

from __future__ import annotations

import csv
import json

import pytest
from qprop_support import R_ROOT, R_TIP, beta_law, blade_obj

from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender.mesh.cli import main
from cfdbender.mesh.prop_slice import slice_propeller

ARGS = ["--blades", "1", "--axis", "1", "0", "0", "--blade-dir", "0", "1", "0", "--unit", "m"]
STATIONS = ["--stations", "0.3", "0.45", "0.6", "0.75", "0.9"]


@pytest.fixture
def blade(tmp_path):
    path = tmp_path / "blade.obj"
    path.write_text(blade_obj(rings=41), encoding="utf-8")
    return path


def test_the_cli_writes_the_definition_the_table_and_the_sections(blade, tmp_path, capsys):
    out = tmp_path / "sliced"
    code = main(["prop-slice", "--geometry", str(blade), "--out", str(out), *ARGS, *STATIONS])
    assert code == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["files"]["definition"] == "propeller.toml"
    for name in (
        "propeller.toml",
        "table.csv",
        "prop-slice.json",
        "propeller.toml.provenance.json",
    ):
        assert (out / name).is_file()
    rows = list(csv.DictReader((out / "table.csv").open(encoding="utf-8")))
    assert [round(float(row["r_R"]), 3) for row in rows] == [0.3, 0.45, 0.6, 0.75, 0.9]
    for row in rows:
        r = float(row["r_R"]) * R_TIP
        assert R_ROOT * 0.99 <= r <= R_TIP * 1.01
        assert abs(abs(float(row["beta_deg"])) - beta_law(r)) < 0.5
    assert len(list((out / "sections").glob("s*.dat"))) == len(rows)


def test_an_existing_output_is_refused_unless_overwrite(blade, tmp_path, capsys):
    out = tmp_path / "sliced"
    base = ["prop-slice", "--geometry", str(blade), "--out", str(out), *ARGS, *STATIONS]
    assert main(base) == 0
    capsys.readouterr()
    assert main(base) == 2
    assert "already holds" in capsys.readouterr().err
    assert main([*base, "--overwrite"]) == 0


def test_a_bare_relative_output_lands_under_the_workspace(blade, tmp_path, capsys):
    ws = tmp_path / "ws"
    code = main(
        [
            "--workspace",
            str(ws),
            "prop-slice",
            "--geometry",
            str(blade),
            "--out",
            "p",
            *ARGS,
            *STATIONS,
        ]
    )
    assert code == 0
    assert (ws / "p" / "propeller.toml").is_file()
    capsys.readouterr()


def test_refusals_name_the_fault_and_write_nothing(blade, tmp_path):
    out = tmp_path / "none"
    with pytest.raises(InputArtifactError, match="no such geometry file"):
        slice_propeller(tmp_path / "x.obj", blades=1, out=out)
    igs = tmp_path / "p.igs"
    igs.write_text("x", encoding="utf-8")
    with pytest.raises(InputArtifactError, match="give an .obj"):
        slice_propeller(igs, blades=1, out=out)
    with pytest.raises(InputValueError, match="positive integer"):
        slice_propeller(blade, blades=0, out=out)
    with pytest.raises(InputValueError, match="--unit"):
        slice_propeller(blade, blades=1, out=out, unit="yd")
    with pytest.raises(InputValueError, match="--axis"):
        slice_propeller(blade, blades=1, out=out, axis=[1, 0])
    assert not out.exists()
