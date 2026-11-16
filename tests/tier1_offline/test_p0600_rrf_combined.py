"""Read-back must measure a changed combined IGES with intact individual files.

Reproduction: retain a unit plate, put a plate of area 1.002 in combined.igs,
and keep both the component and isolated files at unit area.
"""

import pytest
from test_p0600_rrf_correspondence import stamp_faces

from cfdbender.geometry import _geom_export as export
from cfdbender.mesh._gmsh import open_kernel


@pytest.mark.parametrize(
    "file,factor,centred",
    [
        ("combined.igs", 1.002, False),
        ("combined.igs", 1.002, True),
        ("panel.igs", 1.002, True),
        ("panel.face-0.igs", 1.002, True),
        ("combined.igs", 1.0005, True),
        ("combined.igs", 1.0, True),
    ],
    ids=[
        "P0600-RRF-1-combined-native",
        "P0600-RRF-1-combined-centred",
        "P0600-RRF-1-component-0.2-percent",
        "P0600-RRF-1-isolated-0.2-percent",
        "P0600-RRF-1-within-limit",
        "P0600-RRF-1-control",
    ],
)
def test_p0600_rrf_combined_changed(tmp_path, file, factor, centred):
    kernel = open_kernel("combined face sabotage")
    try:
        original = kernel.add_planar_face([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        low, high = ((1 - factor) / 2, (1 + factor) / 2) if centred else (0, factor)
        changed = kernel.add_planar_face([(low, 0, 0), (high, 0, 0), (high, 1, 0), (low, 1, 0)])
        for output in ("panel.igs", "panel.face-0.igs", "combined.igs"):
            name = "panel_0" if output == "panel.face-0.igs" else "panel"
            kernel.export_iges({name: [changed if output == file else original]}, tmp_path / output)
    finally:
        kernel.close()
    for output in ("panel.igs", "panel.face-0.igs", "combined.igs"):
        stamp_faces(tmp_path / output, ["F0000001"])
    manifest = {
        "iges": "combined.igs",
        "face_orders": {
            file: ["F0000001"] for file in ("panel.igs", "panel.face-0.igs", "combined.igs")
        },
        "pieces": [
            {
                "component": "panel",
                "file": "panel.igs",
                "source_area_m2": 1.0,
                "free_edges": 4,
                "face_records": [
                    {
                        "name": "panel_0",
                        "iges": "panel.face-0.igs",
                        "area_m2": 1.0,
                        "area_error_m2": 0.0,
                        "iges_identity": "F0000001",
                        "centroid_m": [0.5, 0.5, 0],
                    }
                ],
            }
        ],
    }
    result = export.read_back(tmp_path, manifest, 0.1)
    check = next(c for c in result["checks"] if c["id"] == "area.read_back")
    assert "combined.igs" in result["outputs"]["inputs"]["files"]
    assert check["limit"] == 0.001
    assert check["status"] == ("failed" if factor > 1.001 else "passed"), check
    if factor > 1.001:
        assert check["where"]["file"] == file
        assert check["where"]["face"] == "panel_0"
        if centred:
            assert check["measured"] == pytest.approx(0.002, abs=1e-7)
            assert check["where"]["verification"] == "bounded, correspondence verified"


@pytest.mark.parametrize("damage", ["missing", "duplicate"], ids=lambda s: f"P0600-RRF-1-{s}")
def test_p0600_rrf_combined_requires_all_unique_identities(tmp_path, damage):
    import json

    from test_geom_kernel_export import _written_halves

    _written_halves(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    if damage == "missing":
        (tmp_path / manifest["iges"]).unlink()
    else:
        labels = manifest["face_orders"][manifest["iges"]]
        stamp_faces(tmp_path / manifest["iges"], [labels[0]] * len(labels))
    result = export.read_back(tmp_path, manifest, 0.1)
    check = next(c for c in result["checks"] if c["id"] == "area.read_back")
    assert check["status"] == "failed", check
    assert check["where"]["file"] == manifest["iges"]
    for text in ("plate", "face", str(check["measured"]), "0.001", "rebuild"):
        assert text in check["note"].lower()


@pytest.mark.parametrize("unit", ["mm"], ids=["P0600-RRF-1-mixed-export-units"])
def test_p0600_rrf_combined_correspondence_preserves_component_units(tmp_path, unit):
    import json

    import numpy as np

    from cfdbender.geometry._geom_assembly import Assembly
    from cfdbender.geometry._iges_names import read_iges_names
    from cfdbender.geometry.stages import GeneratedPiece, Patch

    pieces = {}
    for name, offset in (("metric", 0.0), ("rotor", 3.0)):
        controls = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], float)
        controls[..., 0] += offset
        patch = Patch("skin", controls, (0, 1), (0, 1), (2, 2), (2, 2), 1, 1)
        pieces[name] = GeneratedPiece.from_patches(name, "1.0.0", [patch])
    resolved = {
        "components": {
            "metric": {"type": "body"},
            "rotor": {"type": "propeller", "definition": {"output": {"units": unit}}},
        }
    }
    kernel = open_kernel("mixed export unit correspondence")
    try:
        assembly = Assembly(kernel, pieces, [], out=tmp_path)
        assembly.step("S10.D1")
        export.write_assembly(assembly, tmp_path / "combined.igs", resolved=resolved)
    finally:
        kernel.close()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert read_iges_names(tmp_path / "rotor.igs").unit == "MILLIMETER"
    assert read_iges_names(tmp_path / "combined.igs").unit == "METER"
    result = export.read_back(tmp_path, manifest, 0.1)
    assert all(c["status"] == "passed" for c in result["checks"]), result
