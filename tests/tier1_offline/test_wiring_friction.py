"""Sectional loads with a wall shear field: totals, and their pressure and friction parts."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import test_wiring_products as products
from test_wiring_products import collected

from cfdbender.core.contract import FreeStream, product_support
from cfdbender.matrix._point_products import point_products
from cfdbender.matrix.exports import CONTEXT, _cell
from cfdbender.post.sections import (
    Distribution,
    Reference,
    Surface,
    compute_sections,
    write_section_files,
)
from cfdbender.solvers import load_solver

P_INF = 1000.0
PARTS = ["Fx_pres", "Fz_pres", "Moment_pres", "Fx_fric", "Fz_fric", "Moment_fric"]
PARTS_INT = [
    "Fx_pres_int",
    "Fz_pres_int",
    "My_pres_int",
    "Fx_fric_int",
    "Fz_fric_int",
    "My_fric_int",
]
FRICTION = [*PARTS, *PARTS_INT]
#: The commit whose sections module wrote the files before a shear field existed.
BEFORE = "ba11d91"


def _surface(points, quads, shear=None, pressure=None):
    pts = np.asarray(points, dtype=np.float64)
    return Surface(
        pts,
        np.zeros((0, 3), np.int64),
        np.asarray(quads, dtype=np.int64),
        np.full(len(pts), P_INF) if pressure is None else np.asarray(pressure, dtype=np.float64),
        shear=None if shear is None else np.asarray(shear, dtype=np.float64),
    )


def _extrude(section, layers=3):
    """Extrude a closed loop of (x, z) points along y from 0 to 1 into quads."""
    n = len(section)
    points = [(x, j / layers, z) for j in range(layers + 1) for x, z in section]
    quads = [
        (j * n + i, j * n + (i + 1) % n, (j + 1) * n + (i + 1) % n, (j + 1) * n + i)
        for j in range(layers)
        for i in range(n)
    ]
    return points, quads


def _sections(surface, count=2):
    return compute_sections(
        surface,
        Reference(P_INF, 50.0),
        Distribution((), "MRP", ("XZ",), count, 1, True),
    )


def _plate(chord=2.0, half=0.01):
    return _extrude([(0.0, -half), (chord, -half), (chord, half), (0.0, half)])


def test_a_plate_with_uniform_shear_carries_the_shear_times_its_wetted_length():
    points, quads = _plate()
    result = _sections(_surface(points, quads, shear=[(5.0, 0.0, 0.0)] * len(points)))
    wetted = 2 * 2.0 + 2 * (2 * 0.01)
    assert result.sloads_columns[-12:] == tuple(FRICTION)
    for station in result.blocks[0].stations:
        assert station.fx_fric == pytest.approx(5.0 * wetted, rel=1e-12)
        assert station.fz_fric == pytest.approx(0.0, abs=1e-12)
        assert (station.fx_pres, station.fz_pres) == pytest.approx((0.0, 0.0), abs=1e-9)
        assert station.fx == pytest.approx(5.0 * wetted, rel=1e-12)  # the total
    total = sum(s.fx_fric * s.strip_length for s in result.blocks[0].stations)
    assert total == pytest.approx(5.0 * wetted * 1.0, rel=1e-12)


@pytest.mark.parametrize(
    ("shape", "fz", "moment"),
    [
        ("uniform", 0.0, 3.0 * 2 * math.pi * 0.5**2),
        ("cosine", -math.pi * 0.5 * 3.0, 0.0),
    ],
)
def test_a_cylinder_with_a_tangential_shear(shape, fz, moment):
    radius, tau, n = 0.5, 3.0, 256
    loop = [
        (radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    points, quads = _extrude(loop)
    shear = []
    for x, _, z in points:
        theta = math.atan2(z, x)
        size = tau if shape == "uniform" else tau * math.cos(theta)
        # tangential: y x r_hat = (sin, 0, -cos)
        shear.append((size * math.sin(theta), 0.0, -size * math.cos(theta)))
    result = _sections(_surface(points, quads, shear))
    for station in result.blocks[0].stations:
        assert station.fx_fric == pytest.approx(0.0, abs=1e-9)
        assert station.fz_fric == pytest.approx(fz, rel=1e-3, abs=1e-9)
        assert station.moment_fric == pytest.approx(moment, rel=1e-3, abs=1e-9)


def test_the_total_is_the_pressure_plus_the_friction(tmp_path):
    points, quads = _plate()
    pressure = [P_INF - 1000.0 * z + 10.0 * x for x, _, z in points]
    result = _sections(
        _surface(points, quads, shear=[(1.5, 0.0, 0.7)] * len(points), pressure=pressure)
    )
    _, loads = write_section_files(result, tmp_path, "P", "w")
    rows = list(csv.DictReader(io.StringIO(loads.read_text())))
    assert rows
    for row in rows:
        for total, stem in (("Fx", "Fx"), ("Fz", "Fz"), ("Moment", "Moment")):
            assert float(row[total]) == pytest.approx(
                float(row[stem + "_pres"]) + float(row[stem + "_fric"]), rel=1e-12, abs=1e-12
            )
        for total, stem in (("Fx_int", "Fx"), ("Fz_int", "Fz"), ("My_int", "My")):
            assert float(row[total]) == pytest.approx(
                float(row[stem + "_pres_int"]) + float(row[stem + "_fric_int"]),
                rel=1e-12,
                abs=1e-12,
            )


def test_zero_shear_gives_zero_friction_and_totals_equal_to_the_pressure():
    points, quads = _plate()
    pressure = [P_INF - 1000.0 * z for x, _, z in points]
    result = _sections(
        _surface(points, quads, shear=[(0.0, 0.0, 0.0)] * len(points), pressure=pressure)
    )
    for station in result.blocks[0].stations:
        assert (station.fx_fric, station.fz_fric, station.moment_fric) == (0.0, 0.0, 0.0)
        assert (station.fx, station.fz) == (station.fx_pres, station.fz_pres)


def test_without_shear_the_files_are_those_the_earlier_module_wrote(tmp_path):
    """Byte identity against the sections module before the shear field existed."""
    text = subprocess.run(
        ["git", "show", f"{BEFORE}:src/cfdbender/post/sections.py"],
        capture_output=True,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    ).stdout
    old = tmp_path / "sections_before.py"
    old.write_bytes(text)
    spec = importlib.util.spec_from_file_location("sections_before", old)
    assert spec is not None and spec.loader is not None
    before = importlib.util.module_from_spec(spec)
    sys.modules["sections_before"] = before
    spec.loader.exec_module(before)
    points, quads = _plate()
    pressure = [P_INF - 1000.0 * z + 10.0 * x for x, _, z in points]
    now = _sections(_surface(points, quads, pressure=pressure))
    then = before.compute_sections(
        before.Surface(
            np.asarray(points, dtype=np.float64),
            np.zeros((0, 3), np.int64),
            np.asarray(quads, dtype=np.int64),
            np.asarray(pressure, dtype=np.float64),
        ),
        before.Reference(P_INF, 50.0),
        before.Distribution((), "MRP", ("XZ",), 2, 1, True),
    )
    assert not set(FRICTION) & set(now.sloads_columns)
    a = write_section_files(now, tmp_path / "a", "P", "w")
    b = before.write_section_files(then, tmp_path / "b", "P", "w")
    assert [p.read_bytes() for p in a] == [p.read_bytes() for p in b]


def test_su2_shear_is_the_friction_coefficient_times_the_cp_dynamic_pressure():
    free = FreeStream(100.0, 50.0, 1.0, 10.0, "test")
    source = product_support(load_solver("su2"))["sections"].columns["shear"]
    found = source({"Skin_Friction_Coefficient": np.array([[0.002, 0.0, -0.001]])}, free)
    assert found[0].tolist() == pytest.approx([0.1, 0.0, -0.05])


def test_without_a_shear_field_the_part_columns_are_absent_and_named(tmp_path):
    post, name = collected(tmp_path)
    loads = post / f"sections/P1001/{name}_sloads_wing.csv"
    assert not set(FRICTION) & set(loads.read_text().splitlines()[0].split(","))
    data = json.loads((post / "products.json").read_text())
    entry = next(e for e in data["products"] if e["path"].endswith("_sloads_wing.csv"))
    assert entry["absent_columns"] == dict.fromkeys(
        FRICTION, "the adapter supplies no wall shear field"
    )


def test_su2_sections_carry_the_totals_and_the_parts(tmp_path):
    root, point, adapter = products._su2_window(tmp_path)
    found = point_products(
        root,
        "7001",
        "DP",
        point,
        adapter,
        lead=dict.fromkeys(CONTEXT, "NA"),
        moment_point=(0.0, 0.0, 0.0),
        cell=_cell,
        target=lambda relative: root / "post/rt" / relative,
    )
    loads = next(f for p, f in found.files.items() if p.name == "DP_sloads_B1.csv")
    rows = list(csv.DictReader(io.StringIO(loads["text"])))
    assert loads["absent_columns"] == {}
    assert set(PARTS) <= set(rows[0])
    for row in rows:
        assert float(row["Fx"]) == pytest.approx(float(row["Fx_pres"]) + float(row["Fx_fric"]))
