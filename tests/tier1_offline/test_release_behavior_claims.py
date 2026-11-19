"""Release notes must describe the implemented clock, exports and missing values."""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from farfield_support import build_lattice
from test_post_volume_files import make_grid
from test_runtypes_b1_workflows import deck, workspace

from cfdbender.post.farfield.balances import shaft_torque
from cfdbender.post.farfield.fields import lattice_fields
from cfdbender.post.farfield.quadrature import grid_integral, sample_coverage
from cfdbender.post.volume_files import ProbeEntry, VolumeStep, write_probe_table
from cfdbender.post.volume_frames import REFERENCE

ROOT = Path(__file__).resolve().parents[2]


def release():
    return (
        (ROOT / "CHANGELOG.md")
        .read_text(encoding="utf-8")
        .split("## [0.7.0]", 1)[1]
        .split("## [0.6.0]", 1)[0]
    )


def test_release_clock_precedence_and_early_export_window(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady",
        rotor="",
        condition="MACH:0.05",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 10 / LAST_ITERS_AVG: 5 / "
        "EXPORT_UNSTEADY_AFTER_ITER: 1",
        setup="TIME_ITER = 999\nTIME_STEP = 9.0\n",
    )
    _, values, extras = deck(tmp_path)
    assert values["TIME_ITER"] == "10" and float(values["TIME_STEP"]) == 0.001
    assert extras == {}
    notes = release()
    assert "a key the setup states itself wins" not in notes
    assert "clock and restart accounting determine its time keys" in notes
    assert "early windows use one segment" in notes


def test_release_missing_probe_token(tmp_path):
    report = write_probe_table(
        tmp_path,
        "point",
        [ProbeEntry(np.array([[2.0, 2.0, 2.0]]), REFERENCE)],
        [VolumeStep(None, None, make_grid())],
        {"Pressure": "p"},
    )
    with report.file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["Pressure"] == "NA"
    assert "a probe table leaves their values empty" not in release()
    assert "a probe table writes `NA`" in release()


def test_release_torque_includes_dimensional_lever_arm():
    lattice = build_lattice(tip_radius=1.0, stations=(0.0,), n_r=2, n_psi=4)
    shape = (1, 2, 4)
    fields = lattice_fields(lattice, fields={"u": np.ones(shape), "v_theta": np.ones(shape)})
    assert shaft_torque(replace(fields, tip_radius=2.0), 1.0) == pytest.approx(
        8.0 * shaft_torque(fields, 1.0)
    )
    assert "every ledger is in the field's units times the square" not in release()
    assert "torque and moment also include a dimensional lever arm" in release()


def test_release_qualifies_compatibility_and_cold_start_outputs():
    notes = release()
    assert "A 0.6.0 workspace needs nothing to keep working" not in notes
    assert "on a cold start also at step 1" in notes
    upgrade = " ".join((ROOT / "docs/upgrading.md").read_text(encoding="utf-8").split())
    assert "the file holds the pressure loads, as before" not in upgrade


def test_release_distinguishes_missing_sample_integrals_from_coverage():
    samples = np.array([[1.0, np.nan], [1.0, 1.0]])
    axis = np.array([0.0, 1.0])
    assert sample_coverage(samples) == 0.75
    assert np.isnan(grid_integral(axis, z=axis, integrand=samples))
    assert "coverage diagnostics report the sampled fraction" in release()
