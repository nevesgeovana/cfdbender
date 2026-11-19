"""The files of a volume cut and of probes: names, columns, absent columns, removed points."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from post_volume_support import box_grid, write_vtu_ascii

from cfdbender.post.volume import VolumeError
from cfdbender.post.volume_files import (
    ProbeEntry,
    VolumeStep,
    write_probe_table,
    write_volume_section,
)
from cfdbender.post.volume_frames import REFERENCE, Frame, RotorSpin
from cfdbender.post.volume_layouts import volume_section_points
from cfdbender.post.vtu import UnstructuredGrid


def velocity(p: np.ndarray) -> np.ndarray:
    return np.column_stack([1.0 + p[:, 0], 2.0 * p[:, 1], -p[:, 2] + 0.25 * p[:, 0]])


def pressure(p: np.ndarray) -> np.ndarray:
    return 100.0 + 3.0 * p[:, 0] - p[:, 1]


def make_grid(scale: float = 1.0, with_pressure: bool = True) -> UnstructuredGrid:
    base = box_grid("hex", (4, 4, 4))
    data = {"vel": scale * velocity(base.points)}
    if with_pressure:
        data["p"] = scale * pressure(base.points)
    return UnstructuredGrid(base.points, base.connectivity, base.offsets, base.types, data)


COLUMNS = {"Velocity": "vel", "Pressure": "p"}


def vtk_blocks(text: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lines = text.splitlines()
    n = int(lines[4].split()[1])
    points = np.array([[float(x) for x in line.split()] for line in lines[5 : 5 + n]])
    fields: dict[str, np.ndarray] = {}
    cursor = 5 + n + 1 + n + 1  # POINTS rows, VERTICES line and rows, POINT_DATA line
    while cursor < len(lines):
        head = lines[cursor].split()
        if head[0] == "SCALARS":
            fields[head[1]] = np.array([float(x) for x in lines[cursor + 2 : cursor + 2 + n]])
            cursor += 2 + n
        else:
            fields[head[1]] = np.array(
                [[float(x) for x in line.split()] for line in lines[cursor + 1 : cursor + 1 + n]]
            )
            cursor += 1 + n
    return points, fields


def test_a_cut_is_written_with_the_removed_points_left_out(tmp_path: Path) -> None:
    # the plane x = 0.5 reaches beyond the grid in y and z: part of the rectangle has no cell
    points = volume_section_points(
        "rectangle", "YZ", corners_m=(-0.5, 0.0, 1.5, 1.0), points=(9, 5), offset_m=0.5
    )
    report = write_volume_section(
        tmp_path, "pt", points, REFERENCE, [VolumeStep(None, None, make_grid())], COLUMNS
    )
    (step,) = report.steps
    inside = (
        (points[:, 1] >= 0.0)
        & (points[:, 1] <= 1.0)
        & (points[:, 2] >= 0.0)
        & (points[:, 2] <= 1.0)
    )
    assert (
        step.n_points == 45
        and step.n_found == int(inside.sum())
        and step.n_missing == 45 - step.n_found
    )
    assert report.files == (tmp_path / "fields" / "pt_vsec.vtk",)
    placed, fields = vtk_blocks(report.files[0].read_text(encoding="utf-8"))
    np.testing.assert_allclose(placed, points[inside], atol=1e-8)
    np.testing.assert_allclose(fields["Velocity"], velocity(points[inside]), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(fields["Pressure"], pressure(points[inside]), rtol=1e-8)
    record = json.loads((tmp_path / "fields" / "pt_vsec.vtk.provenance.json").read_text())
    assert record["declared_points"] == 45 and record["removed_points"] == 45 - step.n_found
    assert record["topology"] == "vertex-cloud" and record["coordinate_units"] == "m"


def test_a_series_writes_one_file_per_step_named_by_the_step(tmp_path: Path) -> None:
    points = volume_section_points(
        "rectangle", "XY", corners_m=(0.1, 0.1, 0.9, 0.9), points=(3, 3), offset_m=0.5
    )
    volumes = [VolumeStep(4, 0.04, make_grid(1.0)), VolumeStep(8, 0.08, make_grid(2.0))]
    report = write_volume_section(tmp_path, "pt", points, REFERENCE, volumes, COLUMNS)
    assert [f.name for f in report.files] == ["pt_vsec_step_4.vtk", "pt_vsec_step_8.vtk"]
    _, first = vtk_blocks(report.files[0].read_text(encoding="utf-8"))
    _, second = vtk_blocks(report.files[1].read_text(encoding="utf-8"))
    np.testing.assert_allclose(second["Pressure"], 2.0 * first["Pressure"], rtol=1e-8)
    record = json.loads((tmp_path / "fields" / "pt_vsec_step_8.vtk.provenance.json").read_text())
    assert record["step"] == 8 and record["time"] == 0.08


def test_a_cut_in_a_rotating_frame_is_placed_at_each_step(tmp_path: Path) -> None:
    frame = Frame("rotor", (0.5, 0.5, 0.5), spin=RotorSpin((0, 0, 1), 0.0, 1.0, 8.0))
    local = np.array([[0.3, 0.0, 0.0], [0.2, 0.1, 0.1]])
    volumes = [VolumeStep(0, 0.0, make_grid()), VolumeStep(2, 0.2, make_grid())]
    report = write_volume_section(tmp_path, "r", local, frame, volumes, {"Velocity": "vel"})
    placed0, _ = vtk_blocks(report.files[0].read_text(encoding="utf-8"))
    placed2, _ = vtk_blocks(report.files[1].read_text(encoding="utf-8"))
    np.testing.assert_allclose(placed0, [[0.8, 0.5, 0.5], [0.7, 0.6, 0.6]], atol=1e-8)
    # two steps of eight is 90 degrees: (x, y) -> (-y, x) about the axis
    np.testing.assert_allclose(placed2, [[0.5, 0.8, 0.5], [0.4, 0.7, 0.6]], atol=1e-8)


def test_a_tecplot_cut_has_one_zone_per_sample_and_expanded_vectors(tmp_path: Path) -> None:
    points = np.array([[0.25, 0.25, 0.25], [9.0, 9.0, 9.0], [0.5, 0.5, 0.5]])
    report = write_volume_section(
        tmp_path,
        "t",
        points,
        REFERENCE,
        [VolumeStep(None, None, make_grid())],
        COLUMNS,
        format="tecplot",
    )
    text = report.files[0].read_text(encoding="utf-8")
    assert report.files[0].name == "t_vsec.dat"
    lines = text.splitlines()
    assert lines[1] == 'VARIABLES = "X" "Y" "Z" "Velocity_x" "Velocity_y" "Velocity_z" "Pressure"'
    assert text.count("ZONE T=") == 2
    row = [float(x) for x in lines[3].split()]
    np.testing.assert_allclose(row[:3], [0.25, 0.25, 0.25])
    np.testing.assert_allclose(row[3:6], velocity(points[:1])[0], rtol=1e-12)
    assert row[6] == pytest.approx(pressure(points[:1])[0], rel=1e-12)


def test_columns_the_volume_cannot_fill_are_absent_and_named(tmp_path: Path) -> None:
    points = np.array([[0.3, 0.3, 0.3], [0.6, 0.4, 0.2]])
    columns = {
        "Velocity": "vel",
        "Pressure": "p",
        "Density": "rho",
        "Ratio": lambda f: f["p"] / f["rho"],
        "Speed": lambda f: np.linalg.norm(f["vel"], axis=1),
    }
    grid = make_grid()
    report = write_volume_section(
        tmp_path, "a", points, REFERENCE, [VolumeStep(None, None, grid)], columns
    )
    (step,) = report.steps
    assert step.absent_columns == ("Density", "Ratio")
    _, fields = vtk_blocks(report.files[0].read_text(encoding="utf-8"))
    assert list(fields) == ["Velocity", "Pressure", "Speed"]
    np.testing.assert_allclose(fields["Speed"], np.linalg.norm(velocity(points), axis=1), rtol=1e-8)


def test_a_cut_that_finds_no_point_writes_no_file(tmp_path: Path) -> None:
    report = write_volume_section(
        tmp_path,
        "z",
        np.array([[5.0, 5.0, 5.0]]),
        REFERENCE,
        [VolumeStep(1, None, make_grid())],
        COLUMNS,
    )
    assert report.files == () and report.steps[0].file is None and report.steps[0].n_found == 0
    assert not (tmp_path / "fields" / "z_vsec_step_1.vtk").exists()


def test_the_files_are_byte_stable_and_a_file_source_is_hashed(tmp_path: Path) -> None:
    grid = make_grid()
    source = tmp_path / "flow.vtu"
    write_vtu_ascii(source, grid)
    points = volume_section_points("circle", "YZ", radii_m=(0.0, 0.4), points=(3, 6), offset_m=0.5)
    points = points + np.array([0.0, 0.5, 0.5])
    out = []
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        write_volume_section(
            directory,
            "c",
            points,
            REFERENCE,
            [VolumeStep(3, 0.3, source)],
            COLUMNS,
            description={"shape": "circle", "plane": "YZ"},
        )
        out.append(
            [
                (directory / "fields" / "c_vsec_step_3.vtk").read_bytes(),
                (directory / "fields" / "c_vsec_step_3.vtk.provenance.json").read_bytes(),
            ]
        )
    assert out[0] == out[1]
    record = json.loads(out[0][1])
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["source_file"] == "flow.vtu" and record["shape"] == "circle"
    assert b"\r" not in out[0][0] and out[0][0].endswith(b"\n")
    # the file route and the grid route give the same values to the printed precision
    placed, fields = vtk_blocks(out[0][0].decode())
    np.testing.assert_allclose(fields["Pressure"], pressure(placed), rtol=1e-8)


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_probe_rows_values_and_columns(tmp_path: Path) -> None:
    entry = ProbeEntry(np.array([[0.25, 0.5, 0.5], [3.0, 3.0, 3.0], [0.75, 0.5, 0.5]]), REFERENCE)
    columns = {"P": "p", "VX": lambda f: f["vel"][:, 0], "VY": lambda f: f["vel"][:, 1]}
    report = write_probe_table(
        tmp_path,
        "pt",
        [entry],
        [VolumeStep(None, None, make_grid())],
        columns,
        pol="POL-1",
        condition={"RHO": 1.225, "TEMP": "288.15"},
    )
    text = report.file.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "POL,PROBE,X,Y,Z,FRAME,STEP,RHO,TEMP,P,VX,VY"
    rows = read_table(report.file)
    assert [r["PROBE"] for r in rows] == ["1", "2", "3"]
    assert {r["FRAME"] for r in rows} == {"REFERENCE"} and {r["STEP"] for r in rows} == {"NA"}
    assert (
        {r["POL"] for r in rows} == {"POL-1"}
        and rows[0]["RHO"] == "1.225"
        and rows[0]["TEMP"] == "288.15"
    )
    assert float(rows[0]["P"]) == pytest.approx(
        pressure(np.array([[0.25, 0.5, 0.5]]))[0], rel=1e-12
    )
    assert float(rows[2]["VX"]) == pytest.approx(1.75, rel=1e-12)
    assert rows[1]["P"] == rows[1]["VX"] == rows[1]["VY"] == ""
    assert float(rows[1]["X"]) == 3.0
    assert report.n_rows == 3 and report.steps[0].n_missing == 1


def test_a_series_keeps_one_row_per_point_and_step_with_the_rotor_placement(tmp_path: Path) -> None:
    frame = Frame("rotor", (0.5, 0.5, 0.5), spin=RotorSpin((0, 0, 1), 0.0, 1.0, 8.0))
    entries = [
        ProbeEntry(np.array([[0.3, 0.0, 0.0]]), frame),
        ProbeEntry(np.array([[0.1, 0.1, 0.1], [0.0, 0.0, 0.2]]), REFERENCE),
    ]
    volumes = [VolumeStep(0, 0.0, make_grid()), VolumeStep(2, 0.2, make_grid(2.0))]
    report = write_probe_table(tmp_path, "s", entries, volumes, {"P": "p"}, history=True)
    assert report.file.name == "s_probes_history.csv"
    rows = read_table(report.file)
    assert [(r["STEP"], r["PROBE"]) for r in rows] == [
        ("0", "1"),
        ("0", "2"),
        ("0", "3"),
        ("2", "1"),
        ("2", "2"),
        ("2", "3"),
    ]
    assert (float(rows[0]["X"]), float(rows[0]["Y"])) == pytest.approx((0.8, 0.5))
    assert (float(rows[3]["X"]), float(rows[3]["Y"])) == pytest.approx((0.5, 0.8))
    assert float(rows[1]["X"]) == 0.1  # the inertial entry does not move
    assert float(rows[4]["P"]) == pytest.approx(
        2.0 * pressure(np.array([[0.1, 0.1, 0.1]]))[0], rel=1e-12
    )


def test_a_probe_column_that_a_step_cannot_fill_is_absent_from_the_table(tmp_path: Path) -> None:
    entry = ProbeEntry(np.array([[0.4, 0.4, 0.4]]), REFERENCE)
    volumes = [
        VolumeStep(0, None, make_grid()),
        VolumeStep(1, None, make_grid(with_pressure=False)),
    ]
    report = write_probe_table(
        tmp_path, "m", [entry], volumes, {"P": "p", "VX": lambda f: f["vel"][:, 0]}
    )
    assert report.absent_columns == ("P",) and report.listed == {"P": ["1"]}
    assert (
        report.file.read_text(encoding="utf-8").splitlines()[0] == "POL,PROBE,X,Y,Z,FRAME,STEP,VX"
    )


def test_a_vector_column_in_a_probe_table_and_a_bad_name_are_refused(tmp_path: Path) -> None:
    entry = ProbeEntry(np.array([[0.4, 0.4, 0.4]]), REFERENCE)
    volumes = [VolumeStep(None, None, make_grid())]
    with pytest.raises(VolumeError, match="vector"):
        write_probe_table(tmp_path, "v", [entry], volumes, {"V": "vel"})
    with pytest.raises(VolumeError, match="column name"):
        write_probe_table(tmp_path, "v", [entry], volumes, {"has space": "p"})
    with pytest.raises(VolumeError, match="format"):
        write_volume_section(tmp_path, "v", entry.points, REFERENCE, volumes, COLUMNS, format="csv")
    with pytest.raises(VolumeError, match="shape"):
        write_probe_table(tmp_path, "v", [entry], volumes, {"W": lambda f: f["p"][:0]})
