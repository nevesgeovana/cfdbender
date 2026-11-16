"""Reject an area whose numerical accuracy was not established.

Reproduction: export_area(kernel, Face(1)) with the review's wavy trim or
concentrated surface, or with an integrator that exhausts its budget.
"""

import numpy as np
import pytest
from test_p0600_rf3fix_trim import Rectangle

from cfdbender._errors import InputArtifactError
from cfdbender.geometry import _geom_area as area
from cfdbender.mesh._cad import Face


class ConcentratedSurface(Rectangle):
    def area(self, face):
        return 1.002

    def derivatives(self, face, uv):
        result = np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (len(uv), 1, 1))
        result[:, 0, 0] += 2 * np.maximum(0.0, 1 - np.abs(uv[:, 0] - 0.5) / 0.001)
        return result


@pytest.mark.parametrize(
    "kernel", [ConcentratedSurface()], ids=["P0600-RF3-3-review-convergence-probe"]
)
def test_p0600_rf3fix_gauss_agreement_cannot_certify_area(kernel):
    try:
        measured = area.export_area(kernel, Face(1))
    except InputArtifactError:
        return
    assert measured == pytest.approx(1.002, rel=1e-7)


@pytest.mark.parametrize("case", ["knots"], ids=["P0600-RF3-3-native-localized-feature"])
def test_p0600_rf3fix_native_knots_cannot_hide_area(case):
    from scipy.interpolate import BSpline

    from cfdbender.mesh._gmsh import open_kernel

    # x'(u) = 1 + a triangular feature of height 2 and width .002.
    # The integral is exactly 1.002; all three old even Gauss grids miss it.
    curve = BSpline([0, 0, 0.499, 0.5, 0.501, 1, 1], [1, 1, 3, 1, 1], 1).antiderivative()
    knots, counts = np.unique(curve.t, return_counts=True)
    kernel = open_kernel("localized area regression")
    try:
        face = kernel.add_bspline_surface(
            [[(float(x), 0.0, 0.0), (float(x), 1.0, 0.0)] for x in curve.c[: len(curve.t) - 3]],
            knots_u=knots.tolist(),
            mult_u=counts.tolist(),
            knots_v=[0, 1],
            mult_v=[2, 2],
            deg_u=2,
            deg_v=1,
        )
        assert area.export_area(kernel, face) == pytest.approx(1.002, rel=1e-7)
    finally:
        kernel.close()


@pytest.mark.parametrize("height", [5e-8], ids=["P0600-RF3-3-adjacent-knot-jacobian"])
def test_p0600_rf3fix_trim_error_uses_the_adjacent_patch_jacobian(height):
    class AdjacentPatch(Rectangle):
        def area(self, face):
            return 0.5 + 2 * 1e6 * height / 3

        def uv_bounds(self, face):
            return ((0.0, 0.0), (0.5 + height, 1.0))

        def reparametrize(self, edge, face, t):
            return {
                1: (0.5 * t, 0.0),
                2: (0.5 + 4 * height * t * (1 - t), t),
                3: (0.5 * (1 - t), 1.0),
                4: (0.0, 1 - t),
            }[edge]

        def derivatives(self, face, uv):
            result = np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (len(uv), 1, 1))
            result[:, 0, 0] = np.where(uv[:, 0] < 0.5, 1.0, 1e6)
            return result

        def _export_area_definition(self, face):
            from cfdbender.geometry._geom_area_brep import brep_definition

            vertices = "".join(
                f"Ve\n1e-7\n{x} {y} 0\n0 0\n0101101\n*\n"
                for x, y in [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]
            )
            edges = "".join(
                f"Ed\n1e-7 1 1 0\n2 {i} 1 0 0 {high}\n0\n0101000\n+{a} 0 -{b} 0 *\n"
                for i, a, b, high in [
                    (1, 11, 10, 0.5),
                    (2, 10, 9, 1),
                    (3, 9, 8, 0.5),
                    (4, 8, 11, 1),
                ]
            )
            # x'(u) is 1 to the left of u=.5, then 1e6. The right trim
            # bulges across that knot by 4*height*v*(1-v). Its area is known.
            return brep_definition(
                f"""CASCADE Topology V1,
Locations 0
Curve2ds 4
1 0 0 1 0
7 0 0 2 3 2 0.5 0 {0.5 + 2 * height} 0.5 0.5 1 0 3 1 3
1 0.5 1 -1 0
1 0 1 0 -1
Curves 0
Polygon3D 0
PolygonOnTriangulations 0
Surfaces 1
9 0 0 0 0 1 1 3 2 3 2
0 0 0 0 1 0 0.5 0 0 0.5 1 0 500000.5 0 0 500000.5 1 0
0 2 0.5 1 1 2
0 2 1 2
Triangulations 0
TShapes 11
"""
                + vertices
                + edges
                + """Wi
0101100
+7 0 +6 0 +5 0 +4 0 *
Fa
0 1e-10 1 0
0111000
+3 0 *
Co
1100000
+2 0 *
+1 0
"""
            )

    kernel = AdjacentPatch()
    try:
        measured = area.export_area(kernel, Face(1))
    except InputArtifactError:
        return
    assert measured == pytest.approx(kernel.area(Face(1)), rel=1e-7)


@pytest.mark.parametrize("difference", [0.0009998], ids=["P0600-RF3-3-error-crosses-limit"])
def test_p0600_rf3fix_error_allowance_is_part_of_acceptance(tmp_path, monkeypatch, difference):
    from cfdbender.geometry import _geom_export as export

    manifest = {
        "iges": "panel.igs",
        "face_orders": {file: ["F0000001"] for file in ("panel.igs", "panel.face-0.igs")},
        "pieces": [
            {
                "component": "panel",
                "file": "panel.igs",
                "source_area_m2": 1.0,
                "free_edges": 0,
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
    monkeypatch.setattr(
        export,
        "_inspect",
        lambda *args: {
            "areas": [1 + difference],
            "area_errors": [5e-7],
            "identities": ["F0000001"],
            "centroids": [[0.5, 0.5, 0]],
            "names": ["panel_0" if args[0].name == "panel.face-0.igs" else "panel"],
            "property_names": ["panel_0" if args[0].name == "panel.face-0.igs" else "panel"],
            "edges": [],
            "pid": 123,
        },
    )
    result = export.read_back(tmp_path, manifest, 0.1)
    check = next(c for c in result["checks"] if c["id"] == "area.read_back")
    assert check["limit"] == 0.001
    assert check["status"] == "failed", check
    assert check["measured"] == pytest.approx(difference + 5e-7)
    assert check["where"]["verification"] == "bounded, correspondence verified"


@pytest.mark.parametrize("factor", [1.002], ids=["P0600-RF3-3-native-0.2-percent-control"])
def test_p0600_rf3fix_native_area_sabotage_still_fails(tmp_path, factor):
    from test_p0600_rrf_correspondence import stamp_faces

    from cfdbender.geometry import _geom_export as export
    from cfdbender.mesh._gmsh import open_kernel

    kernel = open_kernel("isolated face sabotage control")
    try:
        original = kernel.add_planar_face([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        # Keep the centroid so this continues to test the numerical area gate
        # after the independent correspondence check was added.
        low, high = (1 - factor) / 2, (1 + factor) / 2
        changed = kernel.add_planar_face([(low, 0, 0), (high, 0, 0), (high, 1, 0), (low, 1, 0)])
        kernel.export_iges({"panel": [original]}, tmp_path / "panel.igs")
        kernel.export_iges({"panel_0": [changed]}, tmp_path / "panel.face-0.igs")
    finally:
        kernel.close()
    for file in ("panel.igs", "panel.face-0.igs"):
        stamp_faces(tmp_path / file, ["F0000001"])
    manifest = {
        "iges": "panel.igs",
        "face_orders": {file: ["F0000001"] for file in ("panel.igs", "panel.face-0.igs")},
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
    assert check["limit"] == 0.001
    assert check["status"] == "failed"
    assert check["measured"] == pytest.approx(0.002, abs=1e-7)
    assert check["where"]["face"] == "panel_0"


@pytest.mark.parametrize("height", [0.2], ids=["P0600-RF3-3-analytic-error-enclosure"])
def test_p0600_rf3fix_curved_area_encloses_an_independent_integral(height):
    from math import asinh, sqrt

    from cfdbender.mesh._gmsh import open_kernel

    kernel = open_kernel("analytic area enclosure control")
    try:
        face = kernel.add_bspline_surface(
            [[(0, 0, 0), (0, 1, 0)], [(0.5, 0, 0), (0.5, 1, 0)], [(1, 0, height), (1, 1, height)]],
            knots_u=[0, 1],
            mult_u=[3, 3],
            deg_u=2,
            knots_v=[0, 1],
            mult_v=[2, 2],
            deg_v=1,
        )
        # z = height*u**2; integrate sqrt(1 + (2*height*u)**2) analytically.
        exact = sqrt(1 + 4 * height**2) / 2 + asinh(2 * height) / (4 * height)
        value, error = area.measure_area(kernel, face)
        assert abs(value - exact) <= error
        assert 0 <= error <= 1e-6 * value
    finally:
        kernel.close()
