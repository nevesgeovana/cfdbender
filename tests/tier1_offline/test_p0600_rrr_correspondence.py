"""Refuse moved trims and renamed faces even when their areas are unchanged."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_p0600_rrf_correspondence import stamp_faces

from cfdbender.geometry import _geom_export as export
from cfdbender.geometry._geom_assembly import Assembly
from cfdbender.geometry.stages import GeneratedPiece, Patch
from cfdbender.mesh._gmsh import open_kernel


@pytest.mark.parametrize("kind", ["component", "isolated", "combined"])
@pytest.mark.parametrize("damage", ["move", "name", "none"])
def test_p0600_rrr_same_area_does_not_prove_correspondence(tmp_path, kind, damage):
    patch = Patch(
        "skin",
        np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], float),
        (0, 1),
        (0, 1),
        (2, 2),
        (2, 2),
        1,
        1,
    )
    piece = GeneratedPiece.from_patches("plate", "1.0.0", [patch])

    def trim(low):
        points = ((low, 0), (low + 0.25, 0), (low + 0.25, 1), (low, 1))
        return replace(
            piece.trims[0],
            loop=tuple((points[i], points[(i + 1) % 4]) for i in range(4)),
        )

    kernel = open_kernel("unchanged area correspondence regression")
    try:
        assembly = Assembly(kernel, {"plate": replace(piece, trims=(trim(0),))}, [], out=tmp_path)
        assembly.step("S10.D1")
        export.write_assembly(assembly, tmp_path / "combined.igs")
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        component = manifest["pieces"][0]
        face = component["face_records"][0]
        file = {
            "component": component["file"],
            "isolated": face["iges"],
            "combined": manifest["iges"],
        }[kind]
        name = face["name"] if kind == "isolated" else "plate"
        if damage == "move":
            base = kernel.add_bspline_surface(
                patch.controls,
                knots_u=[0, 1],
                knots_v=[0, 1],
                mult_u=[2, 2],
                mult_v=[2, 2],
                deg_u=1,
                deg_v=1,
            )
            moved = kernel.add_trimmed_face(base, [trim(0.5).loop])
            assert kernel.area(moved) == pytest.approx(face["area_m2"])
            assert np.max(np.abs(np.array(kernel.bbox(moved)) - face["bbox_m"])) < 1e-7
            kernel.export_iges({name: [moved]}, tmp_path / file)
        elif damage == "name":
            kernel.export_iges({"wrong": assembly.faces["plate"]}, tmp_path / file)
        if damage != "none":
            stamp_faces(tmp_path / file, [face["iges_identity"]])
    finally:
        kernel.close()
    result = export.read_back(tmp_path, manifest, 0.1)
    check = next(c for c in result["checks"] if c["id"] == "area.read_back")
    assert check["status"] == ("passed" if damage == "none" else "failed"), check
    if damage != "none":
        assert check["where"]["file"] == file
        for text in ("plate", "face", "0.001", "rebuild"):
            assert text in check["note"].lower()


@pytest.mark.parametrize("damage", ["pointer", "property-value"])
def test_p0600_rrr_name_pointer_damage_is_refused(tmp_path, damage):
    from test_geom_kernel_export import _written_halves

    from cfdbender.geometry._iges_write import read_iges, serialize_iges

    _written_halves(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    path = tmp_path / manifest["iges"]
    document = read_iges(path)
    face = next(e for e in document["entities"] if e["type"] == 144)
    if damage == "pointer":
        face["parameters"][-1] = "0"
    else:
        prop = next(e for e in document["entities"] if e["type"] == 406)
        prop["parameters"][2] = "5Hwrong"
    path.write_text(serialize_iges(document), encoding="ascii")
    result = export.read_back(tmp_path, manifest, 0.1)
    check = next(c for c in result["checks"] if c["id"] == "area.read_back")
    assert check["status"] == "failed", check
    assert "name" in check["note"].lower()
    assert "rebuild" in check["note"].lower()
