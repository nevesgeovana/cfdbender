"""Geometry: the section form, and sections read from CST coefficients or a file of ordinates.

Expected values come from closed forms written here, not from the code under test: the
NACA 4-digit thickness equation, the leading-edge radius 1.1019 t^2 of that family, and the
partition of unity of the Bernstein basis (equal coefficients give z = c sqrt(x) (1 - x)).
"""

from __future__ import annotations

import numpy as np
import pytest
from geom_inputs import stations as as_stations

from cfdbender._errors import InputValueError
from cfdbender.geometry import resolve_config
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._lifting import lifting_surface, section_models, section_points
from cfdbender.geometry._sections import (
    CstSection,
    SectionForm,
    read_ordinates,
    section_from_item,
    section_from_ordinates,
)
from cfdbender.geometry.config import load_config


def thickness_0012(x, te_open=True):
    """Half thickness of NACA 0012 (finite trailing edge, coefficient -0.1015)."""
    t = 0.12
    return (
        5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    )


def write_ordinates(path, count=121):
    beta = np.linspace(0, np.pi, count)
    x = (1 - np.cos(beta)) / 2
    yt = thickness_0012(x)
    pts = np.vstack([np.stack((x[::-1], yt[::-1]), 1), np.stack((x[1:], -yt[1:]), 1)])
    path.write_text("NACA 0012 test\n" + "\n".join(f"{a:.8f} {b:.8f}" for a, b in pts) + "\n")


def raw(items, **extra):
    return {
        "type": "lifting_surface",
        "planform": {"semispan_mm": 600.0, "taper": 0.5, "le_sweep_deg": 0.0},
        "stations": as_stations(items),
        **extra,
    }


def component(data, base):
    return unpack(resolve_config(data, base=base))["components"]["COMPONENT"]


def test_a_cst_section_is_a_section_form():
    section = CstSection((0.2, 0.2, 0.2), (-0.2, -0.2, -0.2), 0.0)
    assert isinstance(section, SectionForm)
    upper, lower = section.evaluate(np.array([0.25]))
    assert upper[0, 1] == pytest.approx(0.2 * 0.5 * 0.75, abs=1e-12)
    assert lower[0, 1] == pytest.approx(-0.2 * 0.5 * 0.75, abs=1e-12)
    assert section.nose_radius() == pytest.approx(0.2**2 / 2, rel=1e-12)  # z^2 = 2 r x
    assert section.te_thickness() == 0.0
    assert section.compat([section]) == 0.0
    thicker = CstSection((0.2, 0.2, 0.2), (-0.2, -0.2, -0.2), 0.01)
    assert section.compat([thicker]) == pytest.approx(0.005, abs=1e-12)  # half the base at x = 1


def test_the_nose_radius_of_a_fitted_naca_0012_is_the_published_one():
    section = section_from_item({"naca": "0012"}, degree=10)
    assert section.nose_radius() == pytest.approx(1.1019 * 0.12**2, rel=0.06)


def test_ordinates_of_a_file_become_a_cst_section(tmp_path):
    file = tmp_path / "foil.dat"
    write_ordinates(file)
    section = section_from_item({"file": str(file)}, degree=5)
    x = np.array([0.05, 0.3, 0.6, 0.9])
    upper, lower = section.evaluate(x)
    assert upper[:, 1] == pytest.approx(thickness_0012(x), abs=6e-4)
    assert lower[:, 1] == pytest.approx(-thickness_0012(x), abs=6e-4)
    assert section.te_thickness() == pytest.approx(2 * thickness_0012(1.0), abs=1e-4)


def test_a_surface_with_items_follows_the_cst_and_the_file(tmp_path):
    file = tmp_path / "foil.dat"
    write_ordinates(file)
    data = raw(
        [
            {"eta": 0.0, "file": "foil.dat"},
            {"eta": 1.0, "cst_upper": [0.2, 0.2, 0.2], "cst_lower": [-0.2, -0.2, -0.2], "te": 0.0},
        ]
    )
    comp = component(data, tmp_path)
    models = section_models(comp["sections"])
    root = section_points(comp, models, 0.0, np.array([0.3]), "upper")[0]
    tip = section_points(comp, models, 1.0, np.array([0.3]), "upper")[0]
    chord_root = comp["planform"]["root_chord"]
    chord_tip = comp["planform"]["tip_chord"]
    assert root[2] - 0.0 == pytest.approx(thickness_0012(0.3) * chord_root, abs=1e-3 * chord_root)
    # The skin's section is the stored B-spline, within the CST conversion limit of 2e-5 chords.
    assert tip[2] == pytest.approx(0.2 * np.sqrt(0.3) * 0.7 * chord_tip, abs=2e-5 * chord_tip)
    assert len(lifting_surface(comp)) > 0


def test_a_file_path_is_read_from_the_folder_of_the_input(tmp_path, monkeypatch):
    write_ordinates(tmp_path / "foil.dat")
    toml = tmp_path / "wing.toml"
    toml.write_text(
        '[components.WING]\ntype = "lifting_surface"\n'
        '[[components.WING.stations]]\neta = 0.0\nsection = {kind = "coords", file = "foil.dat"}\n'
        '[[components.WING.stations]]\neta = 1.0\nsection = {kind = "naca", code = "0012"}\n',
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    comp = unpack(load_config(toml))["components"]["WING"]
    assert len(lifting_surface(comp)) > 0


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("0 0\n1 0\n", "fewer than 8"),
        ("a b\nc d\n", "fewer than 8"),
    ],
)
def test_a_bad_ordinates_file_is_refused_with_its_name(tmp_path, content, message):
    file = tmp_path / "bad.dat"
    file.write_text(content, encoding="utf-8")
    with pytest.raises(InputValueError, match=message) as error:
        read_ordinates(file)
    assert "bad.dat" in str(error.value)


def test_ordinates_that_are_not_a_function_of_x_are_refused():
    t = np.linspace(0, 2 * np.pi, 41)
    circle = np.stack((np.cos(t), np.sin(t)), 1)[:-1]
    with pytest.raises(InputValueError, match="not a function of x|leading edge|residual"):
        section_from_ordinates(circle, "circle")


def test_a_missing_file_item_is_refused_at_the_input(tmp_path):
    with pytest.raises(InputValueError, match="file not found"):
        resolve_config(raw([{"eta": 0.0, "file": "nope.dat"}]), base=tmp_path)


def half_0012_closed(x):
    """Half thickness of NACA 0012 with the closed trailing edge (coefficient -0.1036)."""
    return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)


def test_a_sharp_file_takes_the_blunt_base_on_the_lifting_skin(tmp_path):
    """Rewritten for the one-skin lifting route.

    Before, the lifting route fitted CST to the file and kept its sharp edge. The skin is now
    built from the stored B-spline sections, and a file whose edge is sharp takes the base
    ``sections.trailing_edge_thickness`` by a linear wedge (blunt in the IGES, collapsed at mesh
    generation), with a warning in the resolved configuration.
    """
    from cfdbender.geometry._caps import lifting_model

    x = (1 - np.cos(np.linspace(0, np.pi, 121))) / 2
    yt = half_0012_closed(x)
    pts = np.vstack([np.stack((x[::-1], yt[::-1]), 1), np.stack((x[1:], -yt[1:]), 1)])
    rows = [f"{a:.10f} {b:.10f}" for a, b in pts]
    (tmp_path / "sharp.dat").write_text("\n".join(["NACA 0012 closed", *rows, ""]))
    resolved = resolve_config(raw([{"eta": 0.0, "file": "sharp.dat"}]), base=tmp_path)
    assert any("sharp trailing edge" in text for text in resolved["warnings"])
    comp = unpack(resolved)["components"]["COMPONENT"]
    te = comp["sections"]["trailing_edge_thickness"]
    assert te > 1e-3  # the default base is not zero
    model = lifting_model(comp)
    # The trailing-edge face is as wide as the base at the root: te chords.
    face = np.asarray(model.trailing_edge.controls)
    width = float(np.linalg.norm(face[1, 0] - face[0, 0]))
    assert width == pytest.approx(te * comp["planform"]["root_chord"], rel=1e-9)
    # The section meets the file plus the wedge, z = +-(half + x te / 2), measured as ordinates
    # aft of the nose, where an ordinate difference is close to a normal distance.
    section = model.ends["root"]
    assert section.te_thickness() == pytest.approx(te, abs=1e-9)
    dense = 0.02 + 0.98 * (1 - np.cos(np.linspace(0, np.pi, 801))) / 2
    upper, lower = section.evaluate(dense)
    half = half_0012_closed(dense) + dense * te / 2
    assert np.max(np.abs(upper[:, 1] - half)) <= 5e-4
    assert np.max(np.abs(lower[:, 1] + half)) <= 5e-4
