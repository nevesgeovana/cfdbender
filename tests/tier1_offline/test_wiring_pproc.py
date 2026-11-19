"""The p file's product tables: pyflightstream's keys, defaults and refusals; warn and skip."""

from __future__ import annotations

import warnings

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.plan import PlanError, plan_matrix
from cfdbender.core.pproc import PprocError, read_post
from cfdbender.solvers import load_solver

ENTRY = {"families": "BLADE", "planes": ["YZ"]}


def read(post, supported=("sections", "volume_section", "probes")):
    said: list[str] = []
    found = read_post(post, supported=supported, warn=lambda key, text: said.append(text))
    return found, said


def test_sections_defaults_are_pyflightstreams():
    found, said = read({"sections": {"distributions": [ENTRY]}})
    assert said == []
    assert found.sections is not None
    entry = found.sections.distributions[0]
    assert (found.sections.count, found.sections.plot_direction) == (50, 1)
    assert (entry.frame, entry.count, entry.plot_direction, entry.integrate) == (
        "MRP",
        50,
        1,
        False,
    )
    assert (entry.families, entry.planes, entry.position) == ("BLADE", ("YZ",), 1)


def test_an_entry_falls_back_to_the_table_and_may_state_its_own():
    table = {"count": 20, "plot_direction": 2, "distributions": [ENTRY, {**ENTRY, "count": 7}]}
    found, _ = read({"sections": table})
    assert found.sections is not None
    first, second = found.sections.distributions
    assert (first.count, first.plot_direction, second.count, second.position) == (20, 2, 7, 2)


def test_volume_section_and_probes_defaults():
    found, _ = read(
        {
            "volume_section": {
                "shape": "circle",
                "plane": "YZ",
                "radii_m": [0, 1],
                "points": [3, 8],
            },
            "probes": [{"parameters": ["VX"], "lines": [{"start": [0, 0, 0], "end": [1, 0, 0]}]}],
        }
    )
    section, (probe,) = found.volume_section, found.probes
    assert section is not None
    assert (section.kind, section.frame, section.offset_m, section.format) == (
        "instantaneous",
        "MRP",
        0.0,
        "vtk",
    )
    assert (probe.frame, probe.points, probe.scale, probe.kind) == ("", 25, "m", "default")
    assert found.products == ("volume_section", "probes")


@pytest.mark.parametrize(
    ("post", "message"),
    [
        (
            {"sections": {"cont": 3}},
            r"\[sections\]\.cont: Extra inputs are not permitted\. Did you mean 'count'\?",
        ),
        (
            {"sections": {"count": 0}},
            r"\[sections\]\.count: Input should be greater than or equal to 1",
        ),
        (
            {"sections": {"distributions": [{"families": "B", "planes": []}]}},
            "planes: List should have at least 1 item",
        ),
        (
            {"sections": {"distributions": [{"families": "each_blade", "planes": ["XZ"]}]}},
            "families = 'each_blade' of a post-processing entry was renamed to frame = "
            "'LOCAL_AXIS'",
        ),
        (
            {"sections": {"distributions": [{"families": ["each", "B"], "planes": ["XZ"]}]}},
            "'each' and 'each_blade' expand an entry and stand alone",
        ),
        (
            {"sections": {"distributions": [{**ENTRY, "frame": " "}]}},
            "frame names one the run creates",
        ),
        ({"probes": {"parameters": ["VX"]}}, r"`\[probes\]` is a LIST of tables"),
        ({"probes": [{"scale": "propeller_radius"}]}, 'Write scale = "rotor_radius"'),
        ({"probes": [{"parameters": ["VQ"]}]}, r"fluid plot parameter\(s\) VQ are not among"),
        (
            {"probes": [{"lines": [{"start": [0, 0, 0], "end": [1, 0, 0]}]}]},
            "a probe entry drawing or citing points requires nonempty parameters",
        ),
        (
            {
                "probes": [
                    {
                        "parameters": ["VX"],
                        "rectangles": [
                            {
                                "origin": [0, 0, 0],
                                "along_u": [1, 0, 0],
                                "along_v": [2, 0, 0],
                                "points_u": 2,
                                "points_v": 2,
                            }
                        ],
                    }
                ]
            },
            "spans no plane",
        ),
        (
            {
                "probes": [
                    {
                        "parameters": ["VX"],
                        "circles": [
                            {
                                "center": [0, 0, 0],
                                "normal": [0, 0, 0],
                                "radius": 1.0,
                                "points_radial": 2,
                                "points_azimuth": 4,
                            }
                        ],
                    }
                ]
            },
            "which is no direction and so names no plane",
        ),
        ({"probes": [{"parameters": ["VX"], "kind": "every"}]}, "probe kind 'every' is not one of"),
        (
            {"volume_section": {"shape": "circle", "plane": "YZ", "radii_m": [0, 1]}},
            r"\[volume_section\] shape = 'circle' needs points",
        ),
        (
            {
                "volume_section": {
                    "shape": "rectangle",
                    "plane": "YZ",
                    "corners_m": [0, 0, 1, 1],
                    "radii_m": [0, 1],
                }
            },
            r"states radii_m, which is a circle's; a rectangle takes corners_m and "
            "refinement_layers",
        ),
        (
            {"volume_section": {"shape": "rectangle", "plane": "YZ", "corners_m": [0, 0, 0, 1]}},
            "enclose no area",
        ),
        (
            {
                "volume_section": {
                    "shape": "circle",
                    "plane": "YZ",
                    "radii_m": [1, 1],
                    "points": [2, 4],
                }
            },
            "the inner radius r1 is at least 0 and smaller than the outer r2",
        ),
        (
            {
                "volume_section": {
                    "shape": "circle",
                    "plane": "YZ",
                    "radii_m": [0, 1],
                    "points": [2, 2],
                }
            },
            "volume grid needs at least 2 samples per axis and 3 around a circle",
        ),
    ],
)
def test_a_malformed_table_is_refused_with_pyflightstreams_message(post, message):
    with pytest.raises(PprocError, match=message):
        read(post)


def test_a_malformed_table_is_refused_even_when_the_solver_does_not_supply_it():
    with pytest.raises(PprocError, match="Input should be greater than or equal to 1"):
        read({"sections": {"count": 0}}, supported=())


def test_a_table_the_adapter_does_not_supply_is_warned_and_skipped():
    found, said = read({"sections": {"distributions": [ENTRY]}}, supported=("probes",))
    assert found.sections is None and found.products == ()
    assert len(said) == 1 and "[sections] of the PPROC file is skipped" in said[0]
    assert "products it supplies: probes" in said[0]


def test_what_cfdbender_does_not_offer_is_warned_and_skipped():
    found, said = read(
        {
            "sections": {
                "include_symmetry": True,
                "distributions": [ENTRY, {**ENTRY, "attach": "body"}],
            },
            "probes": [
                {
                    "parameters": ["MACH"],
                    "field_formats": ["vtk"],
                    "lines": [{"start": [0, 0, 0], "end": [1, 0, 0]}],
                }
            ],
        }
    )
    assert found.sections is not None and not found.sections.include_symmetry
    assert len(found.sections.distributions) == 1
    assert found.probes[0].parameters == ("MACH", "VX", "VY", "VZ")
    assert any("include_symmetry = true is skipped" in text for text in said)
    assert any("attach = 'body' is skipped" in text for text in said)
    assert any("field_formats is skipped" in text for text in said)


def test_a_deprecated_kind_alias_warns_and_reads_as_its_replacement():
    found, said = read(
        {
            "probes": [
                {
                    "parameters": ["VX"],
                    "kind": "unsteady",
                    "lines": [{"start": [0, 0, 0], "end": [0, 0, 1]}],
                }
            ]
        }
    )
    assert found.probes[0].kind == "instantaneous"
    assert said == ["probe kind 'unsteady' is deprecated; use kind = 'instantaneous' instead."]


def _plan(root):
    return plan_matrix(root, "study", validators={"mock": load_solver("mock").validate_row})


def test_plan_warns_once_and_skips_a_table_the_solver_does_not_supply(tmp_path, monkeypatch):
    workspace(tmp_path)
    monkeypatch.delattr(load_solver("mock").validate_row, "post_products", raising=False)
    (tmp_path / "inputs/pproc/p001.toml").write_text(
        'history = true\n[[sections.distributions]]\nfamilies = "wing"\nplanes = ["XZ"]\n'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = _plan(tmp_path)
    texts = [str(w.message) for w in caught]
    assert len(plan.points) == 1
    assert [t for t in texts if "[sections]" in t] == [
        "POL 1001: [sections] of inputs/pproc/p001.toml is skipped; solver 'mock' does not "
        "supply the files it is written from (products it supplies: none)"
    ]
    assert not [t for t in texts if "does not read it" in t]


def test_plan_refuses_a_malformed_table_naming_the_row(tmp_path):
    workspace(tmp_path)
    (tmp_path / "inputs/pproc/p001.toml").write_text(
        "history = true\n[sections]\nplot_direction = 3\n"
    )
    with pytest.raises(
        PlanError,
        match=r"data row 1 \(POL 1001\): inputs/pproc/p001.toml "
        r"\[sections\]\.plot_direction: Input should be 1 or 2",
    ):
        _plan(tmp_path)
