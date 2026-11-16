"""Geometry: the input reads the published key names and refuses the draft names."""

from __future__ import annotations

import json
import tomllib

import numpy as np
import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._lifting import cap_fraction
from cfdbender.geometry.config import resolve_config

KNOTS = [0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0]
BSPLINE = {
    "degree": 2,
    "knots": KNOTS,
    "poles": [[1.0, 0.0], [0.5, 0.1], [0.0, 0.0], [0.5, -0.1], [1.0, 0.0]],
}

SMI_WING = """
[surface]
name = "Wing"
model = "full"
[planform]
sref_m2 = 50.053
aspect_ratio = 7.9915
taper = 0.6979
root_le_mm = [8526.2, 0.0, 59.1]
le_sweep_deg = 0.0125
dihedral_deg = -0.0123
[cap]
type = "original"
[[stations]]
eta = 0.0
twist_deg = 3.531
section = { kind = "bspline", file = "sections/smi_root.json" }
[[stations]]
eta = 1.0
twist_deg = 3.260
section = { kind = "bspline", file = "sections/smi_tip.json" }
[[engine]]
name = "L1"
eta_axis = 0.356
[[control]]
name = "flap"
kind = "fowler"
chord_ratio = 0.30
eta = [0.13, 0.29]
"""

PA30_WING = """
[surface]
name = "Wing"
model = "full"
[planform]
sref_m2 = 16.537
aspect_ratio = 7.3
root_le_mm = [1662.6, 0.0, -503.0]
dihedral_deg = 5.0
[[bay]]
eta_end = 0.1185
chord_root_mm = 2159.0
chord_end_mm = 2159.0
le_sweep_deg = 0.0
[[bay]]
eta_end = 1.0
chord_end_mm = 990.6
le_sweep_deg = 0.0
[[stations]]
eta = 0.0
twist_deg = 2.0
section = { kind = "coords", file = "naca642a215_abbott.csv" }
[cap]
type = "rounded"
"""


def write_foil(path):
    x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    yt = 0.6 * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    pts = np.vstack([np.stack((x[::-1], yt[::-1]), 1), np.stack((x[1:], -yt[1:]), 1)])
    path.write_text("t\n" + "\n".join(f"{a:.10f} {b:.10f}" for a, b in pts) + "\n")


def resolve(text, tmp_path):
    (tmp_path / "sections").mkdir(exist_ok=True)
    for name in ("smi_root", "smi_tip"):
        (tmp_path / "sections" / f"{name}.json").write_text(json.dumps(BSPLINE), encoding="utf-8")
    write_foil(tmp_path / "naca642a215_abbott.csv")
    return resolve_config(tomllib.loads(text), base=tmp_path)


def wing(result, name="Wing"):
    return unpack(result)["components"][name]


def refuse(raw, match, kind="lifting_surface"):
    with pytest.raises(InputValueError, match=match):
        resolve_config({"components": {"W": {"type": kind, **raw}}})


def test_the_smi_wing_resolves_from_sref_aspect_ratio_and_taper(tmp_path):
    result = resolve(SMI_WING, tmp_path)
    component = wing(result)
    planform = component["planform"]
    assert planform["span"] == pytest.approx((50.053 * 7.9915) ** 0.5, rel=1e-9)
    assert planform["root_chord"] == pytest.approx(2 * 50.053 / (planform["span"] * 1.6979))
    assert planform["tip_chord"] == pytest.approx(0.6979 * planform["root_chord"])
    assert planform["dihedral"] == pytest.approx(-0.0123)
    assert component["placement"]["translation"] == pytest.approx([8.5262, 0.0, 0.0591])
    assert planform["incidence"] == pytest.approx(3.531)
    assert planform["tip_twist"] == pytest.approx(3.260 - 3.531)


def test_the_smi_section_files_are_read_through_the_bspline_input(tmp_path):
    items = wing(resolve(SMI_WING, tmp_path))["sections"]["items"]
    assert [i["bspline"]["knots"] for i in items] == [KNOTS, KNOTS]
    assert items[0]["bspline"]["poles"] == BSPLINE["poles"]


def test_the_smi_wing_skips_by_name_what_has_no_implementation_yet(tmp_path):
    warnings = resolve(SMI_WING, tmp_path)["warnings"]
    for text in (
        "Wing.cap.type",
        "Wing.control[0].cove",
        "Wing.engine[0].half_width_mm",
    ):
        assert any(w.startswith(text) for w in warnings), text
    assert not [w for w in warnings if "section.file" in w]


def test_surface_name_names_the_file_and_is_warned_only_when_unused(tmp_path):
    """In a [surface] file the name IS the component name; the reviewed code said skipped."""
    result = resolve(SMI_WING, tmp_path)
    assert "Wing" in unpack(result)["components"]
    assert not [w for w in result["warnings"] if "surface.name" in w]
    raw = {"type": "lifting_surface", "surface": {"name": "Other"}}
    warnings = resolve_config({"components": {"W": raw}})["warnings"]
    assert any(w.startswith("W.surface.name: the component is named by") for w in warnings)


def test_a_missing_or_malformed_section_file_is_refused_naming_the_station(tmp_path):
    text = SMI_WING.replace("sections/smi_root.json", "sections/none.json")
    with pytest.raises(InputValueError, match=r"Wing.stations\[0\].section.file: file not found"):
        resolve(text, tmp_path)
    (tmp_path / "sections" / "bad.json").write_text("[1, 2]", encoding="utf-8")
    text = SMI_WING.replace("sections/smi_root.json", "sections/bad.json")
    with pytest.raises(InputValueError, match=r"stations\[0\].section.file: expected degree"):
        resolve(text, tmp_path)


def test_the_pa30_wing_resolves_and_warns_that_bays_are_skipped(tmp_path):
    result = resolve(PA30_WING, tmp_path)
    component = wing(result)
    assert component["planform"]["span"] == pytest.approx((16.537 * 7.3) ** 0.5)
    assert component["planform"]["incidence"] == pytest.approx(2.0)
    assert component["tip_cap"]["kind"] == "rounded"
    assert component["sections"]["items"][0]["file"].endswith("naca642a215_abbott.csv")
    assert any(w.startswith("Wing.bay:") for w in result["warnings"])


def test_a_file_with_semispan_and_chords_is_set_a_and_the_semispan_is_half_the_span():
    text = (
        '[surface]\nname = "W"\n[planform]\n'
        "semispan_mm = 5000\nroot_chord_mm = 2000\ntip_chord_mm = 1000\n"
    )
    planform = wing(resolve_config(tomllib.loads(text)), "W")["planform"]
    assert planform["span"] == pytest.approx(10.0)
    assert planform["root_chord"] == pytest.approx(2.0)
    assert planform["taper"] == pytest.approx(0.5)
    # independent of the factor: the area of a trapezoid of span 10 m and chords 2 and 1 m
    assert planform["area"] == pytest.approx(15.0)
    assert planform["aspect_ratio"] == pytest.approx(100.0 / 15.0)


def test_a_single_fin_on_the_plane_has_sref_of_its_one_surface():
    text = (
        '[surface]\nname = "F"\nspan_axis = "z"\n[planform]\n'
        "sref_m2 = 0.4\naspect_ratio = 2.5\ntaper = 0.5\n"
    )
    component = wing(resolve_config(tomllib.loads(text)), "F")
    planform = component["planform"]
    assert planform["area"] == pytest.approx(0.4, rel=1e-12)
    assert planform["aspect_ratio"] == pytest.approx(2.5)
    assert planform["span"] == pytest.approx((2 * 0.4 * 2.5) ** 0.5)
    assert component["placement"]["rotation"] == [90.0, 0.0, 0.0]


def test_a_mirrored_fin_pair_has_sref_of_both_fins():
    """Off the symmetry plane a fin is mirrored: Sref and the aspect ratio are of the pair.

    Each fin is then half the span b = sqrt(AR Sref) high and carries half of Sref.
    """
    text = (
        '[surface]\nname = "F"\nspan_axis = "z"\n[planform]\n'
        "sref_m2 = 0.8\naspect_ratio = 2.5\ntaper = 0.5\nroot_le_mm = [9000.0, -1500.0, 0.0]\n"
    )
    planform = wing(resolve_config(tomllib.loads(text)), "F")["planform"]
    span = (0.8 * 2.5) ** 0.5
    assert planform["area"] == pytest.approx(0.8, rel=1e-12)
    assert planform["aspect_ratio"] == pytest.approx(2.5)
    assert planform["span"] == pytest.approx(span)
    # independent of the area rule: one fin, a trapezoid of height b / 2 with these chords
    one_fin = 0.5 * span * (planform["root_chord"] + planform["tip_chord"]) / 2
    assert one_fin == pytest.approx(0.4)


def test_a_wing_sref_is_the_published_area_of_both_halves():
    text = '[surface]\nname = "W"\n[planform]\nsref_m2 = 12.0\naspect_ratio = 9.0\ntaper = 0.6\n'
    planform = wing(resolve_config(tomllib.loads(text)), "W")["planform"]
    assert planform["area"] == pytest.approx(12.0)
    assert planform["span"] == pytest.approx((12.0 * 9.0) ** 0.5)


def station_wing(cap, tmp_path):
    write_foil(tmp_path / "foil.dat")
    return {
        "components": {
            "W": {
                "type": "lifting_surface",
                "planform": {"sref_m2": 4.0, "aspect_ratio": 9.0, "taper": 0.5},
                "cap": cap,
                "stations": [
                    {"eta": 0.0, "section": {"kind": "coords", "file": "foil.dat"}},
                    {"eta": 1.0, "section": {"kind": "coords", "file": "foil.dat"}},
                ],
                "control": [{"name": "a", "kind": "aileron", "eta": [0.5, 0.9]}],
                "engine": [{"name": "N", "eta_axis": 0.5, "half_width_mm": 100.0}],
            },
            "N": {"type": "body"},
        }
    }


def test_eta_is_a_fraction_of_the_cap_start_station(tmp_path):
    flat = wing(resolve_config(station_wing({"type": "flat"}, tmp_path), base=tmp_path), "W")
    assert flat["sections"]["eta"] == [0.0, 1.0]
    rounded_input = station_wing({"type": "rounded"}, tmp_path)
    component = wing(resolve_config(rounded_input, base=tmp_path), "W")
    share = cap_fraction(component)
    assert 0.5 < share < 0.995
    # eta runs over the span the skin covers, so the last station stays at 1 under a cap
    assert component["sections"]["eta"] == [0.0, 1.0]
    surface = component["control_surfaces"][0]
    assert (surface["eta_in"], surface["eta_out"]) == (0.5, 0.9)
    # a half width is a length: 100 mm over the skin's span, which is share of the semi-span
    reach = 0.1 / (component["planform"]["span"] / 2 * share)
    (bound,) = component["sections"]["nacelle_bounds"]
    assert bound["eta_in"] == pytest.approx(0.5 - reach)
    assert bound["eta_out"] == pytest.approx(0.5 + reach)
    assert (flat["control_surfaces"][0]["eta_in"], flat["control_surfaces"][0]["eta_out"]) == (
        0.5,
        0.9,
    )


def test_two_sets_in_one_surface_are_refused_naming_the_redundant_keys():
    refuse(
        {"planform": {"sref_m2": 10.0, "aspect_ratio": 8.0, "semispan_mm": 4000.0}},
        "planform.*semispan_mm.*sref_m2.*aspect_ratio.*redundant",
    )


@pytest.mark.parametrize(
    ("planform", "new"),
    [
        ({"span": 1.0}, "semispan_mm"),
        ({"height": 1.0}, "semispan_mm"),
        ({"area": 1.0}, "sref_m2"),
        ({"root_chord": 1.0}, "root_chord_mm"),
        ({"tip_chord": 1.0}, "tip_chord_mm"),
        ({"sweep": 1.0}, "le_sweep_deg"),
        ({"dihedral": 1.0}, "dihedral_deg"),
        ({"orientation": "vertical"}, "surface.span_axis"),
        ({"twist_axis": 0.25}, "surface.twist_axis"),
    ],
)
def test_an_old_planform_key_is_refused_with_its_new_name(planform, new):
    (old,) = planform
    refuse({"planform": planform}, f"planform.{old}: renamed to {new}.*refused")


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        ({"tip_cap": {"kind": "flat"}}, "tip_cap: renamed to cap"),
        ({"control_surfaces": []}, "control_surfaces: renamed to control"),
        ({"sections": {"items": []}}, "sections.items: renamed to stations"),
        ({"sections": {"airfoils": ["NACA0012"]}}, "sections.airfoils: renamed to stations"),
        ({"sections": {"eta": [0.0]}}, "sections.eta: renamed to stations"),
        ({"sections": {"nacelle_bounds": []}}, "sections.nacelle_bounds: renamed to engine"),
        ({"control": [{"name": "a", "eta_in": 0.1, "eta_out": 0.2}]}, "eta_in: renamed to eta"),
    ],
)
def test_an_old_table_name_is_refused_with_its_new_name(raw, text):
    refuse(raw, text)


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        ({"length": 1.0}, "length: renamed to length_mm"),
        ({"rear": "pole"}, "rear: renamed to tail"),
        ({"stations": [0.2, 0.5, 0.8]}, "stations: renamed to parts"),
        ({"nose": "pole"}, "nose: expected a table"),
        ({"sections": [{"at": 0.5, "width": 1.0}]}, r"sections\[0\].width: renamed to half_width"),
        ({"sections": [{"at": 0.5, "height": 1.0}]}, r"height: renamed to half_height_mm"),
        ({"sections": [{"at": 0.5, "centre_height": 1}]}, "centre_height: renamed to centre_z_mm"),
        ({"sections": [{"at": 0.5, "radius": 1.0}]}, "radius: renamed to radius_mm"),
    ],
)
def test_an_old_body_key_is_refused_with_its_new_name(raw, text):
    refuse(raw, text, kind="body")


def test_a_body_reads_the_published_names_in_millimetres():
    raw = {
        "type": "body",
        "length_mm": 7374.1,
        "nose": {"kind": "pole"},
        "tail": {"kind": "base"},
        "parts": [0.0, 0.2, 0.5, 0.8, 1.0],
        "sections": [
            {
                "at": 0.5,
                "shape": "superellipse",
                "half_width_mm": 121.1,
                "half_height_mm": 83.9,
                "centre_z_mm": -279.0,
                "n_top": 2.2,
                "n_bottom": 2.2,
            }
        ],
    }
    result = resolve_config({"components": {"B": raw}})
    body = unpack(result)["components"]["B"]
    assert body["length"] == pytest.approx(7.3741)
    assert body["rear"] == "base" and body["nose"] == "pole"
    assert body["stations"] == [0.2, 0.5, 0.8]
    section = body["sections"][0]
    assert (section["width"], section["height"], section["centre_height"]) == pytest.approx(
        (0.1211, 0.0839, -0.279)
    )


def test_a_body_key_without_an_implementation_warns_by_name():
    text = (
        '[body]\nname = "B"\ntype = "nacelle"\nlength_mm = 1000\nnose = { kind = "cap" }\n'
        'tail = { kind = "base", at = 0.9 }\nparts = [0.0, 0.5, 1.0]\n'
        '[[sections]]\nat = 0.5\nshape = "circle"\nradius_mm = 100\n'
    )
    warnings = resolve_config(tomllib.loads(text))["warnings"]
    for needle in ("B.type", "B.nose.kind", "B.tail.at", "B.parts"):
        assert any(w.startswith(needle) for w in warnings), needle


def test_a_message_names_the_key_as_the_user_wrote_it():
    bad = {"degree": 2, "knots": KNOTS[:-1], "poles": BSPLINE["poles"]}
    raw = {
        "type": "lifting_surface",
        "stations": [{"eta": 0.0, "section": {"kind": "bspline", **bad}}],
    }
    with pytest.raises(InputValueError, match=r"stations\[0\]\.section\.knots") as refused:
        resolve_config({"components": {"W": raw}})
    assert "items" not in str(refused.value)
    raw = {"type": "lifting_surface", "cap": {"type": "rounded", "lenght": 1.0}}
    warnings = resolve_config({"components": {"W": raw}})["warnings"]
    assert any("cap.lenght: unknown key" in w for w in warnings)
    assert not any("tip_cap" in w for w in warnings)
    refuse({"cap": {"type": "spiky"}}, r"cap\.type")
    refuse({"cap": {"type": "rounded", "extent": 1.0, "length": 2.0}}, "cap.length")


def test_an_old_key_is_refused_in_a_surface_file_too():
    with pytest.raises(InputValueError, match="Wing.planform.span: renamed"):
        resolve_config(tomllib.loads('[surface]\nname = "Wing"\n[planform]\nspan = 3.0\n'))


def test_a_resolved_configuration_still_round_trips_with_its_resolved_names(tmp_path):
    first = resolve(PA30_WING, tmp_path)
    again = resolve_config(first, base=tmp_path)
    assert wing(again)["planform"] == wing(first)["planform"]


@pytest.mark.parametrize(
    ("table", "text"),
    [
        ('[surface]\nspan_axis = "x"', "surface.span_axis"),
        ('[surface]\nmodel = "quarter"', "surface.model"),
        ("[planform]\nsref_m2 = -1", "planform.sref_m2"),
        ("[planform]\nroot_le_mm = [1, 2]", r"planform.root_le_mm"),
        ('[planform]\nroot_chord_mm = "x"', "planform.root_chord_mm"),
        ("[[stations]]\neta = 2.0", r"stations\[0\].eta"),
        ('[[stations]]\neta = 0.0\nsection = { kind = "cubic" }', "section.kind"),
        ("[[control]]\nname = 'a'\nkind = 'plain'", r"control\[0\].eta"),
    ],
)
def test_a_wrong_value_is_refused_naming_the_published_key(table, text):
    head = '[surface]\nname = "W"\n'
    body = table.removeprefix("[surface]\n") if table.startswith("[surface]") else table
    with pytest.raises(InputValueError, match=text):
        resolve_config(tomllib.loads(head + body + "\n"))


def test_a_fowler_flap_with_a_cove_table_still_needs_its_p3():
    refuse({"control": [{"name": "f", "kind": "fowler", "eta": [0.1, 0.4], "cove": {}}]}, "cove.p3")


def test_a_misspelt_key_warns_at_every_level_of_the_new_tables():
    text = (
        '[surface]\nname = "W"\nbogus = 1\n[planform]\nsref_m3 = 1\n[[stations]]\neta = 0.0\n'
        'bogus = 1\nsection = { kind = "naca", code = "0012" }\n[[control]]\nname = "a"\n'
        'kind = "plain"\neta = [0.1, 0.2]\nbogus = 1\n[[engine]]\nname = "N"\neta_axis = 0.3\n'
        "half_width_mm = 100\nbogus = 1\n"
    )
    warnings = resolve_config(tomllib.loads(text))["warnings"]
    for needle in (
        "W.surface.bogus: unknown key",
        "W.stations[0].bogus: unknown key",
        "W.control[0].bogus: unknown key",
        "W.engine[0].bogus: unknown key",
        "planform.sref_m3: unknown key",
    ):
        assert any(needle in w for w in warnings), needle


def test_an_engine_naming_no_component_is_warned_by_its_engine_key():
    text = (
        '[surface]\nname = "W"\n[planform]\nsref_m2 = 9.0\naspect_ratio = 9.0\ntaper = 0.6\n'
        '[[stations]]\neta = 0.0\nsection = { kind = "naca", code = "0012" }\n'
        '[[engine]]\nname = "N"\neta_axis = 0.3\nhalf_width_mm = 100\n'
    )
    warnings = resolve_config(tomllib.loads(text))["warnings"]
    assert any(w.startswith("W.engine.name: 'N' is no component") for w in warnings), warnings
    assert not any("nacelle_bounds" in w for w in warnings)
