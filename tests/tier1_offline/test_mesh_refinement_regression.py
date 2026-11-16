"""Regression coverage grouped by subject."""
# ruff: noqa: E501

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy
import numpy as np
import pytest
from mesh_fixtures import Composite as _Workspace2RefineWingComposite
from mesh_fixtures import face_coordinates as _workspace_2_refine_wing__face_coordinates
from mesh_fixtures import read_mesh as _workspace_2_refine_wing__read_mesh
from mesh_fixtures import (
    sheet_with_strips as _workspace_2_refine_entry__sheet_with_strips,
)
from mesh_fixtures import (
    sheet_with_strips as _workspace_2_refine_level__sheet_with_strips,
)
from mesh_fixtures import write_source as _reviews2_refine_te__write_source
from mesh_fixtures import write_source as _workspace_2_refine_entry__write_source
from mesh_fixtures import write_source as _workspace_2_refine_level__write_source
from mesh_fixtures import write_source as _workspace_2_refine_wing__write_source
from test_mesh_refine import _json as _workspace_1_refine___json
from test_mesh_refine import _refused as _workspace_1_refine___refused
from test_mesh_refine import _source as _decided_docs_1_refine_tag___source
from test_mesh_refine import _source as _workspace_1_refine___source
from test_mesh_refine import _two_tubes as _workspace_1_refine___two_tubes

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh._demoted import UnknownKeyWarning
from cfdbender.mesh._refine import _config, audit_mesh, refine_mesh
from cfdbender.mesh._refine._config import RefineRequest, level_tag, resolve_request
from cfdbender.mesh._refine._geometry import order_like
from cfdbender.mesh._refine._grid import Grid, recover_grid, refine_grid
from cfdbender.mesh._refine._obj import read_obj, read_te
from cfdbender.mesh._refine._remesh import refine_group

_DECIDED_DOCS_1_REFINE_TAG__PAGE = (
    Path(__file__).resolve().parents[2] / "docs" / "mesh" / "reference.md"
)


def _decided_docs_1_refine_tag___numbers(node: object) -> list[float]:
    """Return every number of a parsed JSON tree."""
    if isinstance(node, dict):
        return [n for value in node.values() for n in _decided_docs_1_refine_tag___numbers(value)]
    if isinstance(node, list):
        return [n for value in node for n in _decided_docs_1_refine_tag___numbers(value)]
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return [float(node)]
    return []


class TestDecidedDocs1RefineTag:
    """Tier 1: the short refine tag and two factors it cannot tell apart.

    Pipeline role: quality gate on the level tag of ``refine`` (planning
    node ). A factor in a built tag shows 6 significant digits, so two
    factors that agree to 6 digits build one tag: the second level is refused as
    an existing one unless overwritten, and ``refine.json`` keeps the exact factor.
    The page docs/mesh/reference.md states all three facts.
    """

    @staticmethod
    def test_mesh_refinement_regression_refine_tag_close_factors_collide_and_refine_json_keeps_the_exact_one(
        tmp_path,
    ):
        # Verifies.
        """2.0000001 and 2.0000002 share; the second is refused, then replaced."""  # noqa: E501
        src = _decided_docs_1_refine_tag___source(tmp_path)
        first = refine_mesh(src, 2.0000001)
        assert first.folder.name.endswith("_R2"), first.folder
        record = json.loads((first.folder / f"{first.folder.name}.refine.json").read_text("utf-8"))
        assert 2.0000001 in _decided_docs_1_refine_tag___numbers(record["specs"]), record["specs"]
        with pytest.raises(InputArtifactError, match=r"the level exists; give overwrite=True"):
            refine_mesh(src, 2.0000002)
        second = refine_mesh(src, 2.0000002, overwrite=True)
        assert second.folder == first.folder


def _reviews2_refine_te___drop_last_edge(level) -> None:
    path = level.obj.with_suffix(".te.txt")
    lines = path.read_text("utf-8").splitlines()
    rows = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
    last = int(np.argmax(rows[:, 1]))
    kept = [row for i, row in enumerate(lines[1:]) if i != last]
    path.write_text("\n".join([lines[0], *kept]) + "\n", "utf-8")


class TestReviews2RefineTe:
    """(): the refined trailing edge of a [[perimeter]] list.

    Gate G4 compares the level's trailing-edge length with the source's, the
    source's taken from the exported list the record states, so a chain
    shortened by a dropped edge fails it while a refined level, whose edges are
    split but whose line is the same, passes.
    """

    @staticmethod
    def test_mesh_refinement_regression_techain_a_shortened_trailing_edge_chain_fails_g4(tmp_path):
        """removing the tip edge of the refined line fails G4, naming the length."""
        source = _reviews2_refine_te__write_source(
            tmp_path / "src", "wing", _workspace_2_refine_level__sheet_with_strips()
        )
        level = refine_mesh(source, 2, out_dir=tmp_path / "out")
        before = next(g for g in audit_mesh(level.obj, against=source).gates if g.name == "G4")
        assert before.passed, before
        _reviews2_refine_te___drop_last_edge(level)
        gate = next(g for g in audit_mesh(level.obj, against=source).gates if g.name == "G4")
        assert not gate.passed
        assert gate.values["length"] < gate.values["source_length"]


def _workspace_1_refine___one_row_sheet():
    """A sheet of 2 stations by 4 nodes, written row by row from station 0."""
    verts = numpy.array([[float(x), float(y), 0.0] for y in (0, 1) for x in range(4)])
    faces = [[i, i + 1, i + 5, i + 4] for i in range(3)]
    return verts, faces


class TestWorkspace1Refine:
    """the refinement request, the face order and a one-row grid.

    Each test is the regression of one defect and fails on the code before its fix.
    """

    @staticmethod
    @pytest.mark.parametrize("tag", [None, "empty"])
    def test_mesh_refinement_regression_wsrefine_level_tags_refuse_an_empty_selection_even_with_a_custom_tag(
        tag,
    ):
        """(): the independent level-tag guard refuses no work."""
        with pytest.raises(InputArtifactError, match="no family is selected"):
            level_tag(RefineRequest(specs={}, tag=tag), ["G"])

    @staticmethod
    def test_mesh_refinement_regression_wsrefine_a_refinement_file_that_is_not_utf8_is_refused(
        tmp_path,
    ):
        """(): a cp1252 byte is a refusal naming the file."""
        src = _workspace_1_refine___source(tmp_path)
        src.with_name("wing.refine.toml").write_bytes(b"# caf\xe9\n[families.G]\nfactor = 2\n")
        text = _workspace_1_refine___refused(tmp_path, src)
        assert "wing.refine.toml" in text and "cannot be read" in text

    @staticmethod
    def test_mesh_refinement_regression_wsrefine_the_body_keys_are_warned_and_skipped(tmp_path):
        """axial, circumferential, axis and origin are no longer read.

        Control: the level is the one the factor alone gives.
        """
        body = "axial = 2\ncircumferential = 3\naxis = [0, 1, 0]\norigin = [0, 0, 0]\n"
        src = _workspace_1_refine___source(
            tmp_path / "a", refine_toml="[families.G]\nfactor = 2\n" + body
        )
        with pytest.warns(GeometryWarning, match="axial is no longer read"):
            level = refine_mesh(src, out_dir=tmp_path / "x")
        plain = refine_mesh(
            _workspace_1_refine___source(tmp_path / "b", refine_toml="[families.G]\nfactor = 2\n"),
            out_dir=tmp_path / "y",
        )
        assert level.obj.name == plain.obj.name
        assert level.obj.read_bytes() == plain.obj.read_bytes()

    @staticmethod
    def test_mesh_refinement_regression_wsrefine_a_keep_first_row_only_table_is_warned_and_skipped(
        tmp_path,
    ):
        """keep_first_row is no longer read; a table stating it alone is skipped."""
        src = _workspace_1_refine___source(
            tmp_path, refine_toml="[families.G]\nkeep_first_row = true\nfactor = 2\n"
        )
        with pytest.warns(GeometryWarning, match="keep_first_row is no longer read"):
            level = refine_mesh(src, out_dir=tmp_path / "out")
        assert level.report["G"]["method"] == "grid"

    @staticmethod
    def test_mesh_refinement_regression_wsrefine_a_trailing_edge_file_that_is_not_utf8_is_refused(
        tmp_path,
    ):
        """(): the second reader refuses cp1252 by name."""
        path = tmp_path / "edge.txt"
        path.write_bytes(bytes([233]))
        with pytest.raises(InputArtifactError, match="edge.txt.*UTF-8"):
            read_te(path)

    @staticmethod
    def test_mesh_refinement_regression_wsrefine_a_selected_family_without_a_table_warns(tmp_path):
        """(): a copied family names the missing table."""
        src = _workspace_1_refine___source(tmp_path, refine_toml="[families.G]\nfactor = 2\n")
        with pytest.warns(GeometryWarning, match=r"add \[families.tail\].*give FACTOR"):
            request = resolve_request(src, ["G", "tail"])
        assert request.missing == ("tail",)

    @staticmethod
    def test_mesh_refinement_regression_wsrefine_missing_family_tables_are_recorded_beside_ignored_tables(
        tmp_path,
    ):
        """(): refine.json names copied selected families."""
        source = _workspace_1_refine___two_tubes(tmp_path, "[families.A]\nfactor = 2\n")
        with pytest.warns(GeometryWarning, match=r"add \[families.B\]"):
            level = refine_mesh(source)
        record = _workspace_1_refine___json(level.files[2])
        assert record["missing_families_tables"] == ["B"]
        assert record["ignored_families_tables"] == []

    @staticmethod
    def test_mesh_refinement_regression_wsrefine_the_face_order_does_not_depend_on_the_length_unit():
        """(): faces on one source face follow its vertices, any unit."""
        length = 10000.0
        verts = numpy.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0]])
        source = [[0, 1, 2]]
        near_second = numpy.array([[9850.0, 10.0, 0.0], [9950.0, 10.0, 0.0], [9900.0, 40.0, 0.0]])
        near_first = numpy.array(
            [[1000.0, 1000.0, 0.0], [1150.0, 1000.0, 0.0], [1000.0, 1150.0, 0.0]]
        )
        for scale in (1.0, 1.0e-4):
            pts = numpy.vstack([near_second, near_first]) * scale
            out = order_like(pts, [[0, 1, 2], [3, 4, 5]], verts * scale, source)
            assert sorted(out[0]) == [3, 4, 5], (scale, out)

    @staticmethod
    def test_mesh_refinement_regression_wsrefine_a_one_row_grid_is_written_in_its_rows_from_station_0():
        """(): refined spanwise, the sheet keeps its row order."""
        verts, faces = _workspace_1_refine___one_row_sheet()
        grid = recover_grid(verts, faces, (3, 7))
        assert isinstance(grid, Grid), grid
        level = refine_grid(verts, faces, grid, chordwise=1.0, spanwise=2.0)
        centroids = [tuple(numpy.round(level.points[f].mean(axis=0)[:2], 6)) for f in level.faces]
        assert centroids == [
            (0.5, 0.25),
            (1.5, 0.25),
            (2.5, 0.25),
            (0.5, 0.75),
            (1.5, 0.75),
            (2.5, 0.75),
        ], centroids

    @staticmethod
    def test_mesh_refinement_regression_wsrefine_a_one_column_grid_keeps_its_forward_grid_column_sweep():
        """(): a missing chordwise neighbour never reverses it."""
        verts = numpy.array([[float(x), float(y), 0.0] for y in range(4) for x in (0, 1)])
        faces = [[2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2] for k in range(3)]
        grid = recover_grid(verts, faces, (1, 3, 5, 7))
        assert isinstance(grid, Grid), grid
        level = refine_grid(verts, faces, grid, chordwise=2.0, spanwise=1.0)
        centroids = [tuple(numpy.round(level.points[f].mean(axis=0)[:2], 6)) for f in level.faces]
        assert centroids == [
            (0.75, 0.5),
            (0.75, 1.5),
            (0.75, 2.5),
            (0.25, 0.5),
            (0.25, 1.5),
            (0.25, 2.5),
        ], centroids


_WORKSPACE_2_REFINE_ENTRY__WITH_T = (("T", "tri", (0.03, 0.04, 0.05, 0.065, 0.08)),)


def _workspace_2_refine_entry___source(
    tmp_path: Path, strips=_WORKSPACE_2_REFINE_ENTRY__WITH_T
) -> Path:
    return _workspace_2_refine_entry__write_source(
        tmp_path / "src", "wing", _workspace_2_refine_entry__sheet_with_strips(strips)
    )


class TestWorkspace2RefineEntry:
    """refine_mesh takes its arguments as the API states them, and refuses alike."""  # noqa: E501

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def _optional_dependencies():
        pytest.importorskip("trimesh")

    @staticmethod
    def test_mesh_refinement_regression_refineentry_a_missing_obj_is_an_input_artifact_refusal(
        tmp_path,
    ):
        """(): a mesh path that is no file is refused as
        InputArtifactError, as audit_mesh refuses it, never a bare FileNotFoundError."""
        with pytest.raises(InputArtifactError) as caught:
            refine_mesh(tmp_path / "src" / "absent.obj", 2)
        assert "no such OBJ file" in str(caught.value)
        assert str(caught.value).endswith("Nothing was written.")
        assert list(tmp_path.iterdir()) == []

    @staticmethod
    def test_mesh_refinement_regression_refineentry_a_bare_family_name_is_one_family(tmp_path):
        """(): a string selects one whole family name."""
        src = _workspace_2_refine_entry___source(tmp_path)
        src.write_text(src.read_text("utf-8").replace("g G", "g GRID"), encoding="utf-8")
        level = refine_mesh(src, 2, families="GRID", out_dir=tmp_path / "out")
        assert level.report["GRID"]["method"] == "grid"

    @staticmethod
    def test_mesh_refinement_regression_refineentry_a_bare_file_name_goes_beside_its_folder(
        tmp_path,
    ):
        """(): a bare file name run in the mesh's own folder writes
        the level into the parent of that folder, the documented default, instead of refusing it
        as inside the source's folder."""
        src = _workspace_2_refine_entry___source(tmp_path)
        here = Path.cwd()
        os.chdir(src.parent)
        try:
            level = refine_mesh("wing.obj", 2, families=["G"])
        finally:
            os.chdir(here)
        assert level.folder.parent.resolve() == tmp_path.resolve()

    @staticmethod
    def test_mesh_refinement_regression_refineentry_the_rebuilt_whole_warning_points_at_the_caller(
        tmp_path,
    ):
        """(): the warning is attributed to the line that
        called refine_mesh, this file, not to a line of the library."""
        src = _workspace_2_refine_entry___source(tmp_path, (("T", "tri", (0.03, 0.04)),))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            refine_mesh(src, 2, families=["G"], out_dir=tmp_path / "out")
        rebuilt = [
            w
            for w in caught
            if issubclass(w.category, GeometryWarning) and "rebuilt whole" in str(w.message)
        ]
        assert len(rebuilt) == 1, [str(w.message) for w in caught]
        assert Path(rebuilt[0].filename).resolve() == Path(__file__).resolve(), rebuilt[0].filename


def _workspace_2_refine_level___record(level) -> dict:
    return json.loads(next(Path(level.folder).glob("*.refine.json")).read_text(encoding="utf-8"))


def _workspace_2_refine_level___te_rows(level) -> list[str]:
    return level.obj.with_suffix(".te.txt").read_text(encoding="utf-8").splitlines()[1:]


class TestWorkspace2RefineLevel:
    """what a level records and writes of its families and its trailing edge."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def _optional_dependencies():
        pytest.importorskip("trimesh")

    @staticmethod
    def test_mesh_refinement_regression_refinelevel_a_selected_family_at_factor_one_is_in_the_specs(
        tmp_path,
    ):
        """(): G at 2 and T at 1 are both selected; T is kept
        unchanged, and refine.json's specs still records it with its factor."""
        toml = "[families.G]\nfactor = 2\n[families.T]\nfactor = 1\n"
        mesh = _workspace_2_refine_level__sheet_with_strips(
            (("T", "tri", (0.03, 0.04, 0.05, 0.065, 0.08)),)
        )
        src = _workspace_2_refine_level__write_source(
            tmp_path / "src", "wing", mesh, refine_toml=toml
        )
        level = refine_mesh(src, config=src.with_suffix(".refine.toml"), out_dir=tmp_path / "out")
        record = _workspace_2_refine_level___record(level)
        assert record["specs"]["T"]["factor"] == 1.0, record["specs"]
        assert record["specs"]["G"]["factor"] == 2.0
        assert record["families"]["T"]["method"] == "unchanged"

    @staticmethod
    def test_mesh_refinement_regression_refinelevel_a_trailing_edge_between_two_interfaces_keeps_its_point(
        tmp_path,
    ):
        """(): T, a one-row quadrilateral strip between the grid G
        and the strip U, is remeshed; its trailing-edge edge joins two interface nodes, and its
        point is still in the level's points file: one row per source trailing-edge edge."""
        strips = (("T", "quad", (0.03,)), ("U", "tri", (0.04, 0.05)))
        mesh = _workspace_2_refine_level__sheet_with_strips(strips)
        toml = '[families.T]\nfactor = 1\nmethod = "remesh"\n'
        src = _workspace_2_refine_level__write_source(
            tmp_path / "src", "wing", mesh, refine_toml=toml
        )
        level = refine_mesh(src, config=src.with_suffix(".refine.toml"), out_dir=tmp_path / "out")
        assert _workspace_2_refine_level___record(level)["families"]["T"]["method"] == "remesh"
        assert len(_workspace_2_refine_level___te_rows(level)) == len(mesh.te)


#: Chordwise positions from the trailing edge (column 0) to the leading edge.
_WORKSPACE_2_REFINE_WING__X = (1.0, 0.995, 0.98, 0.94, 0.8, 0.6, 0.35, 0.12, 0.0)


#: Spanwise stations; the root is y = 0.
_WORKSPACE_2_REFINE_WING__Y = (0.0, 0.25, 0.5, 0.75, 1.0)


def _workspace_2_refine_wing___thickness(x: float) -> float:
    return 0.3 * x**0.5 * (1.0 - x)


def _workspace_2_refine_wing__wing(*, root: bool = True) -> _Workspace2RefineWingComposite:
    """Return the two-sheet wing, with the root band F when ``root``."""
    n, k = len(_WORKSPACE_2_REFINE_WING__X), len(_WORKSPACE_2_REFINE_WING__Y)
    verts: list[tuple[float, float, float]] = []
    upper: dict[tuple[int, int], int] = {}
    lower: dict[tuple[int, int], int] = {}
    for s, y in enumerate(_WORKSPACE_2_REFINE_WING__Y):
        for i, x in enumerate(_WORKSPACE_2_REFINE_WING__X):
            upper[s, i] = len(verts)
            verts.append((x, y, _workspace_2_refine_wing___thickness(x)))
        for i, x in enumerate(_WORKSPACE_2_REFINE_WING__X):
            if i in (0, n - 1):
                lower[s, i] = upper[s, i]
            else:
                lower[s, i] = len(verts)
                verts.append((x, y, -_workspace_2_refine_wing___thickness(x)))
    families: dict[str, list[list[int]]] = {"U": [], "L": []}
    for s in range(k - 1):
        for i in range(n - 1):
            families["U"].append(
                [upper[s, i], upper[s, i + 1], upper[s + 1, i + 1], upper[s + 1, i]]
            )
            families["L"].append(
                [lower[s, i], lower[s + 1, i], lower[s + 1, i + 1], lower[s, i + 1]]
            )
    if root:
        ring = [upper[0, i] for i in range(n)] + [lower[0, i] for i in range(n - 2, 0, -1)]
        below = []
        for v in ring:
            x, _, z = verts[v]
            below.append(len(verts))
            verts.append((x, -0.2, z))
        families["F"] = [
            [ring[j], below[j], below[(j + 1) % len(ring)], ring[(j + 1) % len(ring)]]
            for j in range(len(ring))
        ]
    points = numpy.asarray(verts, dtype=float)
    column = [upper[s, 0] for s in range(k)]
    te = 0.5 * (points[column[:-1]] + points[column[1:]])
    return _Workspace2RefineWingComposite(points, families, te, {})


def _workspace_2_refine_wing___source(tmp_path: Path, mesh: _Workspace2RefineWingComposite) -> Path:
    return _workspace_2_refine_wing__write_source(tmp_path / "src", "wing", mesh)


def _workspace_2_refine_wing___te_rows(level) -> list[tuple[float, ...]]:
    lines = level.obj.with_suffix(".te.txt").read_text(encoding="utf-8").splitlines()
    return [tuple(round(float(c), 6) for c in row.split(",")) for row in lines[1:]]


def _workspace_2_refine_wing___gate(level, name: str):
    return next(item for item in level.audit.gates if item.name == name)


class TestWorkspace2RefineWing:
    """a wing of two sheets sharing their trailing and leading edges, on a root.

    The synthetic source: ``U`` and ``L``, the upper and lower surfaces of a
    wing, each a sheet grid of five stations by nine chordwise nodes clustered
    at the trailing edge (x = 1, column 0); they share the trailing-edge and the
    leading-edge lines node for node. ``F``, a band of another component, closes
    the root station of both. The trailing-edge points file lists the trailing
    edge once, the edges U and L share.
    """

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def _optional_dependencies():
        pytest.importorskip("trimesh")

    @staticmethod
    def test_mesh_refinement_regression_refinewing_a_factor_one_grid_leaves_its_neighbour_byte_identical(
        tmp_path,
    ):
        """(): U at factor 1 keeps every node, so L, which shares its
        trailing and leading edges, is copied face for face; a root node of U beside the
        trailing-edge corner is no node of the shared trailing edge."""
        mesh = _workspace_2_refine_wing__wing()
        level = refine_mesh(
            _workspace_2_refine_wing___source(tmp_path, mesh),
            1,
            families=["U"],
            out_dir=tmp_path / "out",
        )
        verts, families = _workspace_2_refine_wing__read_mesh(level.obj)
        assert level.report["L"]["method"] == "unchanged", level.report["L"]
        assert "continuity" not in level.report["L"]
        assert level.report["L"].get("trailing_edge") in (None, "perimeter")
        assert _workspace_2_refine_wing__face_coordinates(
            verts, families["L"]
        ) == _workspace_2_refine_wing__face_coordinates(mesh.verts.round(9), mesh.families["L"])

    @staticmethod
    def test_mesh_refinement_regression_refinewing_a_factor_two_grid_welds_to_its_neighbours(
        tmp_path,
    ):
        """(): U at factor 2; L and F take U's new nodes on the curves
        they share and nothing else, so the level passes G1 and G3 (no crack, no edge of three)."""
        level = refine_mesh(
            _workspace_2_refine_wing___source(tmp_path, _workspace_2_refine_wing__wing()),
            2,
            families=["U"],
            out_dir=tmp_path / "out",
        )
        assert _workspace_2_refine_wing___gate(level, "G1").verdict == "pass", (
            _workspace_2_refine_wing___gate(level, "G1")
        )
        assert _workspace_2_refine_wing___gate(level, "G3").verdict == "pass", (
            _workspace_2_refine_wing___gate(level, "G3")
        )

    @staticmethod
    def test_mesh_refinement_regression_refinewing_a_trailing_edge_two_families_hold_is_written_once(
        tmp_path,
    ):
        """(, ): U refined chordwise only keeps its
        trailing-edge nodes and L is copied; both hold the trailing-edge edges, and the level's
        points file names each edge once, one row per source row."""
        mesh = _workspace_2_refine_wing__wing(root=False)
        level = refine_mesh(
            _workspace_2_refine_wing___source(tmp_path, mesh),
            chordwise=2,
            families=["U"],
            out_dir=tmp_path,
        )
        rows = _workspace_2_refine_wing___te_rows(level)
        assert len(rows) == len(set(rows)) == len(mesh.te), rows
        assert json.loads(next(level.folder.glob("*.refine.json")).read_text("utf-8"))

    @staticmethod
    @pytest.mark.parametrize("unit", ["MILLIMETER", None])
    def test_mesh_refinement_regression_shared_edge_targets_state_their_units(tmp_path, unit):
        """native lengths become metres only with a declared conversion."""
        from mesh_fixtures import sheet_with_strips

        mesh = sheet_with_strips((("T", "tri", (0.03, 0.04, 0.05, 0.065, 0.08)),))
        src = _workspace_2_refine_wing___source(tmp_path, mesh)
        sidecar = src.with_suffix(".boundaries.toml")
        if unit is not None:
            sidecar.write_text(
                sidecar.read_text("utf-8") + f'\n[import]\nunits = "{unit}"\n', encoding="utf-8"
            )
        te = src.with_suffix(".te.txt")
        te.write_text(te.read_text("utf-8").replace("METER", unit or "METER"), encoding="utf-8")
        level = refine_mesh(src, 2, families=["G"])
        entry = level.report["T"]
        if unit is None:
            assert "shared_edge_target_m" not in entry
            assert entry["shared_edge_target_mesh_units"]["max"] > 0
        else:
            assert 0 < entry["shared_edge_target_m"]["max"] < 0.001

    @staticmethod
    def test_mesh_refinement_regression_remesh_preserves_only_listed_trailing_edges(tmp_path):
        """two trailing edges in two lists do not make the cross sides edges.

        One list holding all four nodes would list the cross sides too (each joins
        two listed nodes); a list per trailing edge lists only its own side.
        """
        points = numpy.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        te = numpy.array([[0.5, 0, 0], [0.5, 1, 0]])
        folder = tmp_path / "src"
        folder.mkdir(parents=True)
        lists = ""
        for k, ids in enumerate(((0, 1), (2, 3))):
            rows = ["#  Id, Name, X , Y, Z"] + [
                f"{v + 1}, , {points[v][0]}, {points[v][1]}, {points[v][2]}" for v in ids
            ]
            (folder / f"te{k}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            lists += f'\n[[perimeter]]\nfile = "te{k}.txt"\nexport_te_per_family = true\n'
        src = _workspace_2_refine_wing__write_source(
            folder,
            "plate",
            _Workspace2RefineWingComposite(points, {"A": [[0, 1, 2, 3]]}, te, {}),
            refine_toml='[families.A]\nfactor = 1\nmethod = "remesh"\n' + lists,
        )
        level = refine_mesh(src)
        assert sorted(_workspace_2_refine_wing___te_rows(level)) == sorted(map(tuple, te.tolist()))

    @staticmethod
    def test_mesh_refinement_regression_assembled_grid_and_copied_family_share_one_trailing_edge(
        tmp_path,
    ):
        """.024: assembly deduplicates a grid part and its copied neighbour."""  # noqa: E501
        mesh = _workspace_2_refine_wing__wing(root=False)
        # Uniform chord spacing keeps both source and refined corner nodes away
        # from the neighboring spanwise edge on older refinement implementations.
        vertices = mesh.verts.copy()
        vertices[:, 0] = numpy.interp(
            vertices[:, 0],
            _WORKSPACE_2_REFINE_WING__X[::-1],
            numpy.linspace(0, 1, len(_WORKSPACE_2_REFINE_WING__X)),
        )
        mesh = _Workspace2RefineWingComposite(vertices, mesh.families, mesh.te, {})
        level = refine_mesh(
            _workspace_2_refine_wing___source(tmp_path, mesh),
            chordwise=2,
            families=["U"],
            out_dir=tmp_path / "out",
        )
        assert level.report["U"]["method"] == "grid"
        assert level.report["L"]["method"] == "unchanged"
        assert "continuity" not in level.report["L"]
        midpoints = _workspace_2_refine_wing___te_rows(level)
        assert len(midpoints) == len(mesh.te)
        assert sorted(midpoints) == sorted(map(tuple, mesh.te.tolist()))


class TestWorkspace2Remesh:
    """a remesh group's record is strict JSON."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def _optional_dependencies():
        pytest.importorskip("trimesh")

    @staticmethod
    def test_mesh_refinement_regression_remeshrecord_a_group_of_curve_edges_only_records_null(
        tmp_path,
    ):
        """(): a group whose every edge is a curve has no edge ratio;
        its record says null, which json.dumps(allow_nan=False) writes, instead of NaN."""
        obj = tmp_path / "one.obj"
        obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\ng A\nf 1 2 3\n", encoding="utf-8")
        done = refine_group(read_obj(obj), ["A"], set(), 1.0)
        assert done.info["edge_ratio_median"] is None
        json.dumps(done.info, allow_nan=False)


def test_mesh_refinement_regression_pc_refine_refusal_names_explicit_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """direct the user to the selected file without writing a level."""
    from mesh_fixtures import HAS_INSPECT, sheet_with_strips, write_source

    from cfdbender.mesh import cli

    source = write_source(tmp_path / "src", "wing", sheet_with_strips(()))
    config = tmp_path / "selected.toml"
    config.write_text('[refine]\ntag = "trial"\n', encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    code = cli.main(["refine", str(source), "--config", str(config)])

    captured = capsys.readouterr()
    diagnostic = captured.out + captured.err
    assert code != 0
    assert str(config) in diagnostic, diagnostic
    assert "factor" in diagnostic.lower(), diagnostic
    pdf = source.with_suffix(".refine-refusal.pdf")
    assert not HAS_INSPECT or f"A diagnostic PDF: {pdf}." in diagnostic
    assert not HAS_INSPECT or pdf.read_bytes().startswith(b"%PDF")
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p != pdf} == before


def _p0501_vars_read(tmp_path: Path, text: str, names=("G", "T", "U"), **kwargs) -> dict:
    """Read ``text`` as a refinement file against the families ``names``."""
    path = tmp_path / "w.refine.toml"
    path.write_text(text, encoding="utf-8")
    return _config.read_refine_file(path, list(names), **kwargs)


class TestP0501RefineVars:
    """variables declared once in [vars] and referenced as ${name}."""

    @staticmethod
    def test_mesh_refinement_regression_vars_a_whole_reference_keeps_the_variable_type(tmp_path):
        """A factor of [families.X] and of a component's governor take the float itself."""
        text = (
            "[vars]\nf_wing = 0.35013\nmembers = ['G', 'T']\nlead = true\n"
            '[families.G]\nmethod = "grid"\nfactor = "${f_wing}"\ngoverning_parallel = "${lead}"\n'
            '[families.T]\nfactor = "${f_wing}"\n'
            '[components]\nWing = "${members}"\n'
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stated = _p0501_vars_read(tmp_path, text)
        assert stated["specs"]["G"].factor == 0.35013
        assert stated["specs"]["G"].governing_parallel is True
        assert stated["specs"]["T"].factor == 0.35013
        assert stated["components"] == {"Wing": ["G", "T"]}

    @staticmethod
    def test_mesh_refinement_regression_vars_text_interpolation_and_chained_variables(tmp_path):
        """A reference inside a longer text is written as text; a variable may use another;
        a ``$`` in any other form stays text."""
        text = (
            '[vars]\nsuffix = "0p350"\nfull = "R${suffix}"\nbase = 0.35\nf = "${base}"\n'
            '[refine]\ntag = "${full}_x$1"\n[families.G]\nfactor = "${f}"\n'
        )
        stated = _p0501_vars_read(tmp_path, text)
        assert stated["tag"] == "R0p350_x$1"
        assert stated["specs"]["G"].factor == 0.35

    @staticmethod
    def test_mesh_refinement_regression_a_skipped_variable_is_named_when_a_reference_needs_it(
        tmp_path,
    ):
        """A variable skipped as invalid is not declared; the later reference says so."""
        text = '[vars]\nbad = [[1, 2]]\n[families.G]\nfactor = "${bad}"\n[families.T]\nfactor = 2\n'
        with pytest.warns(UnknownKeyWarning) as caught, pytest.raises(InputArtifactError):
            _p0501_vars_read(tmp_path, text)
        said = " ".join(str(w.message) for w in caught)
        assert "bad was skipped as invalid, see the earlier warning" in said, said

    @staticmethod
    def test_mesh_refinement_regression_a_table_key_is_not_substituted(tmp_path):
        """Only values are substituted: a ${name} written as a key stays text."""
        text = (
            '[vars]\nn = "G"\n[families.G]\nfactor = 2\n[families."${n}"]\nfactor = 1\n'
            "[families.T]\nfactor = 2\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stated = _p0501_vars_read(tmp_path, text, names=("G", "T", "${n}"))
            except InputArtifactError as error:
                assert "${n}" in str(error), error
            else:
                assert "${n}" in stated["specs"] and stated["specs"]["G"].factor == 2.0

    @staticmethod
    def test_mesh_refinement_regression_a_nested_value_in_vars_is_rejected_with_a_warning(tmp_path):
        """A nested table or an array of arrays is not a variable: warned and skipped."""
        text = "[vars]\ngood = 2\n[vars.nest]\na = 1\n[families.G]\nfactor = 1\n"
        with pytest.warns(UnknownKeyWarning, match="nest"):
            _p0501_vars_read(tmp_path, text)
        text = "[vars]\nrows = [[1, 2]]\n[families.G]\nfactor = 1\n"
        with pytest.warns(UnknownKeyWarning, match="rows"):
            _p0501_vars_read(tmp_path, text)

    @staticmethod
    def test_mesh_refinement_regression_a_variable_in_a_governing_family_factor_resolves(tmp_path):
        """The governor of a component states its factors through a variable."""
        text = (
            '[vars]\nf = 0.5\n[families.G]\ngoverning = true\nfactor = "${f}"\n'
            '[families.T]\nfactor = 2\n[components]\nWing = ["G", "T"]\n'
        )
        stated = _p0501_vars_read(tmp_path, text)
        assert stated["specs"]["G"].factor == 0.5 and stated["specs"]["G"].governing is True
        assert stated["components"] == {"Wing": ["G", "T"]}

    @staticmethod
    def test_mesh_refinement_regression_vars_a_level_matches_the_literal_file(tmp_path):
        """refine_mesh on a file using variables writes the same level as the literal file,
        and refine.json records the resolved values."""
        pytest.importorskip("trimesh")
        mesh = _workspace_2_refine_level__sheet_with_strips(
            (("T", "tri", (0.03, 0.04, 0.05, 0.065, 0.08)),)
        )
        literal = '[refine]\ntag = "Rv2"\n[families.G]\nfactor = 2\n[families.T]\nfactor = 1\n'
        using = (
            '[vars]\nfine = 2\nsame = 1\nname = "v${fine}"\n[refine]\ntag = "R${name}"\n'
            '[families.G]\nfactor = "${fine}"\n[families.T]\nfactor = "${same}"\n'
        )
        levels = []
        for root, toml in (("a", literal), ("b", using)):
            src = _workspace_2_refine_level__write_source(
                tmp_path / root / "src", "wing", mesh, refine_toml=toml
            )
            levels.append(refine_mesh(src, config=src.with_suffix(".refine.toml")))
        first, second = levels
        assert first.folder.name == second.folder.name == "wing_Rv2"
        obj = f"{first.folder.name}.obj"
        assert (first.folder / obj).read_bytes() == (second.folder / obj).read_bytes()
        record = (second.folder / f"{second.folder.name}.refine.json").read_text("utf-8")
        assert "${" not in record
        assert json.loads(record)["specs"]["G"]["factor"] == 2.0


def test_mesh_refinement_regression_f4_refine_mesh_docstring_names_the_governors_pdf():
    from cfdbender.mesh._refine import refine_mesh

    assert "refine-governors" in (refine_mesh.__doc__ or "")
