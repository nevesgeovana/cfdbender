"""Whole-curve rectangularity is required, as the review's trim probe shows."""

import numpy as np
import pytest

from cfdbender._errors import InputArtifactError
from cfdbender.geometry import _geom_area as area
from cfdbender.mesh._cad import Face


class Rectangle:
    def _export_area_definition(self, face):
        return {}

    def uv_bounds(self, face):
        return ((0.0, 0.0), (1.0, 1.0))

    def boundary(self, face):
        return (1, 2, 3, 4)

    def edge_bounds(self, edge):
        return (0.0, 1.0)

    def reparametrize(self, edge, face, t):
        return {1: (t, 0.0), 2: (1.0, t), 3: (t, 1.0), 4: (0.0, t)}[edge]

    def area(self, face):
        return 1.0

    def derivatives(self, face, uv):
        result = np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (len(uv), 1, 1))
        result[:, 0, 0] *= np.sqrt(len(uv))
        return result


class WavyTrim(Rectangle):
    def reparametrize(self, edge, face, t):
        return {
            1: (t, 0.1 * np.sin(16 * np.pi * t) ** 2),
            2: (1.0, t),
            3: (t, 1.0),
            4: (0.0, t),
        }[edge]

    def area(self, face):
        return 0.95

    def derivatives(self, face, uv):
        return np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (len(uv), 1, 1))


@pytest.mark.parametrize("kernel", [WavyTrim()], ids=["P0600-RF3-2-review-trim-probe"])
def test_p0600_rf3fix_samples_cannot_certify_rectangle(kernel):
    try:
        measured = area.export_area(kernel, Face(1))
    except InputArtifactError:
        return  # Refusal is sound when topology or an error estimate is unavailable.
    assert measured == pytest.approx(0.95, rel=1e-7)


@pytest.mark.parametrize("height", [0.3], ids=["P0600-RF3-2-native-curved-trim-control"])
def test_p0600_rf3fix_native_curved_trim_retains_its_domain(height):
    from cfdbender.mesh._gmsh import open_kernel

    kernel = open_kernel("curved trim area control")
    try:
        base = kernel.add_bspline_surface(
            [[(0, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 0)]],
            knots_u=[0, 1],
            knots_v=[0, 1],
            mult_u=[2, 2],
            mult_v=[2, 2],
            deg_u=1,
            deg_v=1,
        )
        face = kernel.add_trimmed_face(
            base,
            [
                [
                    [(0, 0), (0.5, height / 4), (1, 0)],
                    [(1, 0), (1, 1)],
                    [(1, 1), (0, 1)],
                    [(0, 1), (0, 0)],
                ]
            ],
        )
        # The omitted strip is height*u*(1-u), with integral height/6.
        assert area.export_area(kernel, face) == pytest.approx(1 - height / 6, rel=1e-7)
    finally:
        kernel.close()


@pytest.mark.parametrize("side", [0.2], ids=["P0600-RF3-2-native-hole-control"])
def test_p0600_rf3fix_native_hole_is_subtracted(side):
    from cfdbender.mesh._gmsh import open_kernel

    kernel = open_kernel("hole area control")
    try:
        base = kernel.add_bspline_surface(
            [[(0, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 0)]],
            knots_u=[0, 1],
            knots_v=[0, 1],
            mult_u=[2, 2],
            mult_v=[2, 2],
            deg_u=1,
            deg_v=1,
        )
        loops = []
        for low, high in [(0.0, 1.0), (side, 2 * side)]:
            corners = [(low, low), (high, low), (high, high), (low, high)]
            loops.append([[corners[i], corners[(i + 1) % 4]] for i in range(4)])
        face = kernel.add_trimmed_face(base, loops)
        assert area.export_area(kernel, face) == pytest.approx(1 - side**2, rel=1e-9)
    finally:
        kernel.close()


@pytest.mark.parametrize("case", ["range"], ids=["P0600-RF3-2-unrepresented-edge-range"])
def test_p0600_rf3fix_edge_ranges_cannot_silently_select_the_whole_curve(case):
    class InvalidRange(Rectangle):
        def derivatives(self, face, uv):
            return np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (len(uv), 1, 1))

        def _export_area_definition(self, face):
            from cfdbender.geometry._geom_area_brep import brep_definition

            # A native edge requests [-.1, 1.1] of a spline represented only
            # on [0, 1]. Substituting the full spline is not a measurement.
            edges = []
            for i in range(1, 5):
                low, high = (-0.1, 1.1) if i == 1 else (0, 1)
                edges.append(f"Ed\n1e-7 1 1 0\n2 {i} 1 0 {low} {high}\n0\n0101000\n*\n")
            text = (
                """CASCADE Topology V1,
Locations 0
Curve2ds 4
7 0 0 1 2 2 0 0 1 0 0 2 1 2
1 1 0 0 1
1 1 1 -1 0
1 0 1 0 -1
Curves 0
Polygon3D 0
PolygonOnTriangulations 0
Surfaces 1
1 0 0 0 0 0 1 1 0 0 0 1 0
Triangulations 0
TShapes 7
"""
                + "".join(edges)
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
            return brep_definition(text)

    with pytest.raises(InputArtifactError, match="range"):
        area.export_area(InvalidRange(), Face(1))


@pytest.mark.parametrize(
    "gap", [0.0, 2e-11], ids=["P0600-RF3-2-native-wire-order", "P0600-RF3-2-native-closure-error"]
)
def test_p0600_rf3fix_native_wire_independent_integral_uses_vertex_topology(gap):
    class Triangle(Rectangle):
        def area(self, face):
            return 0.5

        def boundary(self, face):
            return (1, 2, 3)

        def reparametrize(self, edge, face, t):
            return {1: (t, 0.0), 2: ((1 - gap) * (1 - t), t), 3: (0.0, 1 - t)}[edge]

        def derivatives(self, face, uv):
            return np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (len(uv), 1, 1))

        def _export_area_definition(self, face):
            from cfdbender.geometry._geom_area_brep import brep_definition

            # Oriented edges follow 6, 5, 4. Native storage is 6, 4, 5.
            vertices = "".join(
                f"Ve\n1e-7\n{x} {y} 0\n0 0\n0101101\n*\n" for x, y in [(0, 0), (1, 0), (0, 1)]
            )
            edges = "".join(
                f"Ed\n1e-7 1 1 0\n2 {i} 1 0 0 1\n0\n0101000\n+{a} 0 -{b} 0 *\n"
                for i, a, b in [(1, 9, 8), (2, 8, 7), (3, 7, 9)]
            )
            return brep_definition(
                f"""CASCADE Topology V1,
Locations 0
Curve2ds 3
1 0 0 1 0
7 0 0 1 2 2 {1 - gap} 0 0 1 0 2 1 2
1 0 1 0 -1
Curves 0
Polygon3D 0
PolygonOnTriangulations 0
Surfaces 1
1 0 0 0 0 0 1 1 0 0 0 1 0
Triangulations 0
TShapes 9
"""
                + vertices
                + edges
                + """Wi
0101100
+6 0 +4 0 +5 0 *
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

    kernel = Triangle()
    assert area.export_area(kernel, Face(1)) == pytest.approx((1 - gap) / 2, rel=1e-7)
    value, error = area.measure_area(kernel, Face(1))
    assert abs(value - (1 - gap) / 2) <= error
    assert error >= gap
