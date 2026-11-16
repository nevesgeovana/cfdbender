"""Tier 1, 0.34.0: the thin blade derived from a blade mesh (marker ).

Pipeline role: quality gate on the library core of ``cfd-mesh degenerate
--kind thin-blade``, :func:`cfdbender.mesh._degenerate.derive_thin_blade`.
The command line itself is a later package's; what is held
here is the geometry and the files.

THE ORACLE IS BUILT, NOT READ. Each test builds a cambered blade whose two sides
are mirror images about a known mean surface: a NACA-like section of 4 % camber
and 12 % thickness with a closed trailing edge, each side offset from the
camber line by the half thickness, extruded along the span in rings and closed
by a cap at each end. The mean surface the mesh carries is the camber line
through its own vertices, so the derived sheet must lie on it within round-off,
whatever stations the derivation chose. One solver-written saved simulation
(the half wing of the tier-3 library) is read too, so the saved-simulation
route is scored on a file the solver wrote.
"""
# ruff: noqa: E501

from __future__ import annotations

import math
from pathlib import Path

import numpy
import pytest

from cfdbender._digest import file_sha256 as _sha
from cfdbender._errors import InputArtifactError
from cfdbender.mesh._degenerate import (
    CHORD_PANELS,
    derive_thin_blade,
    thin_blade_path,
)
from cfdbender.mesh._sidecars import (
    inventory_sidecar,
    obj_boundary_names,
    read_inventory,
    read_mesh_import,
)

#: The test blade: chord, root and tip span coordinates, chordwise points per
#: side, and the number of rings.
CHORD, ROOT, TIP, SIDE_POINTS, RINGS = 0.2, 0.2, 1.0, 30, 12


def _camber(x: numpy.ndarray) -> numpy.ndarray:
    """The NACA 4-digit mean line of 4 % camber at 40 % chord, chord 1."""
    m, p = 0.04, 0.4
    return numpy.where(
        x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2)
    )


def _half_thickness(x: numpy.ndarray) -> numpy.ndarray:
    """The NACA 4-digit half thickness of 12 %, its trailing edge closed, chord 1."""
    return 0.6 * (
        0.2969 * numpy.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )


def _stations() -> numpy.ndarray:
    return 0.5 * (1 - numpy.cos(numpy.pi * numpy.arange(SIDE_POINTS + 1) / SIDE_POINTS))


def _section() -> numpy.ndarray:
    """One closed section (x, y), chord 1: leading edge, upper side, trailing edge, lower."""
    x = _stations()
    inner = x[1:-1]
    upper = numpy.column_stack([inner, _camber(inner) + _half_thickness(inner)])
    lower = numpy.column_stack([inner, _camber(inner) - _half_thickness(inner)])
    return numpy.vstack([[0.0, 0.0], upper, [1.0, 0.0], lower[::-1]])


def _chord(z: numpy.ndarray | float, taper: float) -> numpy.ndarray | float:
    """The local chord at span ``z``: CHORD at the root, less ``taper`` of it at the tip."""
    return CHORD * (1.0 - taper * (z - ROOT) / (TIP - ROOT))


def _blade(
    *, jitter: float = 0.0, closed: bool = True, taper: float = 0.0, unwelded: bool = False
) -> tuple[numpy.ndarray, list]:
    """Return the vertices and the triangles of the test blade, spanning z = ROOT to TIP.

    The section is scaled by CHORD and shifted so the quarter chord lies on the
    span axis. ``jitter`` moves each ring vertex along the span by up to that
    length, so the mesh is no longer built in rings. ``closed=False`` keeps the
    upper side only, an open sheet. ``taper`` shrinks the chord linearly along
    the span, each vertex scaled by the chord at its own span coordinate.
    ``unwelded=True`` writes the trailing edge twice, once per side, with no
    face between the two copies, as an export that does not weld it does.
    """
    section = _section() if closed else numpy.column_stack([_stations(), _camber(_stations())])
    te = SIDE_POINTS  # the trailing edge's index in the closed section
    if unwelded:
        section = numpy.insert(section, te + 1, section[te], axis=0)
    n = len(section)
    rng = numpy.random.default_rng(7)
    vertices = []
    for z in numpy.linspace(ROOT, TIP, RINGS):
        for x, y in section:
            at = z + (jitter * rng.uniform(-1, 1) if ROOT < z < TIP else 0.0)
            vertices.append([(x - 0.25) * _chord(at, taper), y * _chord(at, taper), at])
    faces = []
    count = n if closed else n - 1
    for k in range(RINGS - 1):
        for j in range(count):
            if unwelded and j == te:
                continue
            a, b = k * n + j, k * n + (j + 1) % n
            faces += [[a, b, b + n], [a, b + n, a + n]]
    if closed:
        for k, cap_z in ((0, ROOT), (RINGS - 1, TIP)):
            centre = len(vertices)
            middle = numpy.delete(section, te + 1, axis=0) if unwelded else section
            vertices.append([*(middle.mean(axis=0) - [0.25, 0.0]) * _chord(cap_z, taper), cap_z])
            faces += [
                [centre, k * n + (j + 1) % n, k * n + j]
                for j in range(n)
                if not (unwelded and j == te)
            ]
    return numpy.asarray(vertices), faces


def _write_obj(path: Path, vertices: numpy.ndarray, faces: list, group: str = "Blade1") -> Path:
    lines = [f"o {group}"]
    lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in vertices.tolist()]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _read_obj(path: Path) -> numpy.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    return numpy.asarray(
        [[float(v) for v in line.split()[1:4]] for line in lines if line.startswith("v ")]
    )


def _off_the_mean_surface(points: numpy.ndarray) -> numpy.ndarray:
    """Distance of each point, normal to the chord, from the mesh's own mean surface."""
    x = _stations()
    xi = points[:, 0] / CHORD + 0.25
    return numpy.abs(points[:, 1] - CHORD * numpy.interp(xi, x, _camber(x)))


def test_the_thin_blade_is_the_mean_surface_between_the_two_sides(tmp_path):
    """: every point of the derived sheet lies on the
    surface the blade's two mirror-image sides are symmetric about, within 1e-9 of
    the span, and strictly between the two sides away from the edges."""
    # , the mean surface lies between the two faces.
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces)
    result = derive_thin_blade(source, root_offset=0.05)
    sheet = _read_obj(result.mesh)
    assert len(sheet) == result.stations * (CHORD_PANELS + 1)
    assert _off_the_mean_surface(sheet).max() <= 1e-9 * (TIP - ROOT)
    xi = sheet[:, 0] / CHORD + 0.25
    inside = (xi > 1e-6) & (xi < 1 - 1e-6)
    x = _stations()
    upper = CHORD * numpy.interp(xi, x, _camber(x) + _half_thickness(x))
    lower = CHORD * numpy.interp(xi, x, _camber(x) - _half_thickness(x))
    assert (lower[inside] < sheet[inside, 1]).all()
    assert (sheet[inside, 1] < upper[inside]).all()
    # the sheet keeps the solid section's leading and trailing edge
    assert xi.min() == pytest.approx(0.0, abs=1e-12)
    assert xi.max() == pytest.approx(1.0, abs=1e-12)


def test_a_mesh_not_built_in_rings_and_a_rotated_blade_give_the_same_mean_surface(tmp_path):
    """: the derivation holds off the rings (sections cut
    between vertices) and in any frame (a blade turned about the origin)."""
    # , no dependence on rings or on the global axes.
    vertices, faces = _blade(jitter=0.01)
    result = derive_thin_blade(
        _write_obj(tmp_path / "loose.obj", vertices, faces), root_offset=0.05
    )
    assert _off_the_mean_surface(_read_obj(result.mesh)).max() <= 1e-9 * (TIP - ROOT)
    # a tapered blade off the rings: each section is cut between vertices, and
    # its chord (leading to trailing edge, both on edges along the span) is the
    # local chord at the station exactly, which a wrong cut point would move
    vertices, faces = _blade(jitter=0.01, taper=0.4)
    tapered = derive_thin_blade(
        _write_obj(tmp_path / "tapered.obj", vertices, faces), root_offset=0.05
    )
    rows = _read_obj(tapered.mesh).reshape(tapered.stations, CHORD_PANELS + 1, 3)
    chords = numpy.linalg.norm(rows[:, -1] - rows[:, 0], axis=1)
    assert numpy.abs(chords - _chord(rows[:, 0, 2], 0.4)).max() <= 1e-9 * (TIP - ROOT)
    turn = numpy.linalg.qr(numpy.random.default_rng(3).normal(size=(3, 3)))[0]
    vertices, faces = _blade()
    turned = derive_thin_blade(
        _write_obj(tmp_path / "turned.obj", vertices @ turn.T, faces), root_offset=0.05
    )
    back = _read_obj(turned.mesh) @ turn
    assert _off_the_mean_surface(back).max() <= 1e-9 * (TIP - ROOT)
    assert numpy.allclose(turned.span_direction, turn[:, 2], atol=1e-9)


def test_the_root_is_moved_outward_along_the_span_by_the_offset(tmp_path):
    """: the sheet starts the root offset above the
    blade's root, ends at its tip ring, and the sidecar states the offset used."""
    # , the root offset, along the span from root to tip.
    vertices, faces = _blade()
    for offset in (0.05, 0.13):
        source = _write_obj(tmp_path / f"blade_{offset}.obj", vertices, faces)
        result = derive_thin_blade(source, root_offset=offset)
        sheet = _read_obj(result.mesh)
        assert sheet[:, 2].min() == pytest.approx(ROOT + offset, abs=1e-12)
        assert sheet[:, 2].max() == pytest.approx(TIP, abs=1e-12)
        assert result.span_direction == (0.0, 0.0, 1.0)
        assert (result.root_span, result.root_offset) == (pytest.approx(ROOT + offset), offset)
        assert f"# root_offset = {offset!r}," in result.sidecar.read_text(encoding="utf-8")
    # the root is the end nearer the origin, whichever way the span points
    mirrored = _write_obj(tmp_path / "mirrored.obj", vertices * [1.0, 1.0, -1.0], faces)
    below = derive_thin_blade(mirrored, root_offset=0.05)
    assert below.span_direction == (0.0, 0.0, -1.0)
    assert _read_obj(below.mesh)[:, 2].max() == pytest.approx(-(ROOT + 0.05), abs=1e-12)
    # off the rings the root is still the lowest vertex plus the offset
    vertices, faces = _blade(jitter=0.01)
    loose = derive_thin_blade(_write_obj(tmp_path / "loose.obj", vertices, faces), root_offset=0.05)
    assert _read_obj(loose.mesh)[:, 2].min() == pytest.approx(
        vertices[:, 2].min() + 0.05, abs=1e-12
    )


def test_the_thin_blade_and_its_boundaries_are_written_beside_the_source(tmp_path):
    """: the OBJ under a name derived from the
    source's stem and its boundary inventory beside it, readable by the raw-mesh
    route, with the source's length unit; LF line ends."""
    # , the output and its boundaries file beside the source.
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "rotor_blade.obj", vertices, faces, group="Blade1")
    inventory_sidecar(source).write_text(
        'boundaries = ["Blade1"]\n\n[import]\nunits = "METER"\n', encoding="utf-8"
    )
    result = derive_thin_blade(source, root_offset=0.05)
    assert result.mesh == tmp_path / "rotor_blade_thin_blade.obj" == thin_blade_path(source)
    assert result.sidecar == tmp_path / "rotor_blade_thin_blade.boundaries.toml"
    assert obj_boundary_names(result.mesh) == ("Blade1",)
    assert read_inventory(result.sidecar) == ("Blade1",)
    stated = read_mesh_import(result.sidecar)
    assert stated is not None and stated.units == "METER" and result.unit == "METER"
    for written in (result.mesh, result.sidecar):
        assert b"\r" not in written.read_bytes()
    text = result.mesh.read_text(encoding="utf-8")
    triangles = [line for line in text.splitlines() if line.startswith("f ")]
    assert len(triangles) == 2 * (result.stations - 1) * CHORD_PANELS
    assert all(len(line.split()) == 4 for line in triangles)


def test_a_source_without_a_stated_unit_gets_no_import_table(tmp_path):
    """: a unit is never assumed; the sidecar says
    where to write it."""
    # , no [import] units invented.
    vertices, faces = _blade()
    result = derive_thin_blade(_write_obj(tmp_path / "b.obj", vertices, faces), root_offset=0.05)
    assert result.unit is None
    assert read_mesh_import(result.sidecar) is None
    assert "states no length unit" in result.sidecar.read_text(encoding="utf-8")


def test_the_source_is_never_modified_and_an_output_is_not_overwritten_unasked(tmp_path):
    """: the source keeps its bytes; an existing output
    is refused, naming it, unless overwrite is given."""
    # , never modifies the source; --overwrite to rewrite.
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces)
    before = _sha(source)
    first = derive_thin_blade(source, root_offset=0.05)
    assert _sha(source) == before
    written = _sha(first.mesh)
    with pytest.raises(InputArtifactError, match="already exists; pass overwrite") as refused:
        derive_thin_blade(source, root_offset=0.07)
    assert str(first.mesh) in str(refused.value) and "Nothing was written" in str(refused.value)
    assert _sha(first.mesh) == written
    second = derive_thin_blade(source, root_offset=0.07, overwrite=True)
    assert _sha(second.mesh) != written and _sha(source) == before


def test_a_mesh_whose_two_sides_cannot_be_separated_is_refused(tmp_path):
    """: an open sheet (a blade already without
    thickness) has no closed section, and a file of two groups no single blade."""
    # , sides that cannot be separated are refused by name.
    vertices, faces = _blade(closed=False)
    sheet = _write_obj(tmp_path / "sheet.obj", vertices, faces)
    with pytest.raises(
        InputArtifactError, match="is not one closed loop.*two sides cannot be separated"
    ) as refused:
        derive_thin_blade(sheet, root_offset=0.05)
    assert str(sheet) in str(refused.value)
    assert "the section at span 0.25 (along" in str(refused.value)  # a plain number
    vertices, faces = _blade()
    two = tmp_path / "two.obj"
    _write_obj(two, vertices, faces)
    two.write_text(two.read_text(encoding="utf-8") + "o Spinner\nf 1 2 3\n", encoding="utf-8")
    with pytest.raises(InputArtifactError, match=r"2 groups of faces \(Blade1, Spinner\)"):
        derive_thin_blade(two, root_offset=0.05)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sheet.obj", "two.obj"]


def test_a_root_offset_that_is_not_a_length_inside_the_blade_is_refused(tmp_path):
    """: the offset is required, positive and shorter
    than the blade; a blade whose root cannot be told from its tip is refused."""
    # , the offset refusals and the root rule.
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces)
    for bad in (0.0, -0.05, math.nan):
        with pytest.raises(InputArtifactError, match="is not a positive length"):
            derive_thin_blade(source, root_offset=bad)
    for word in ("0.05", True):  # a length is parsed before it is passed, never coerced here
        with pytest.raises(InputArtifactError, match="is not a number"):
            derive_thin_blade(source, root_offset=word)  # type: ignore[arg-type]
    with pytest.raises(InputArtifactError, match="reaches the blade's last section"):
        derive_thin_blade(source, root_offset=TIP - ROOT)
    middle = vertices - [0.0, 0.0, 0.5 * (ROOT + TIP)]
    centred = _write_obj(tmp_path / "centred.obj", middle, faces)
    with pytest.raises(InputArtifactError, match="cannot be told from its tip"):
        derive_thin_blade(centred, root_offset=0.05)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["blade.obj", "centred.obj"]


def _box(low: list, high: list, start: int) -> tuple[list, list]:
    """A closed box between two corners, its faces numbered from ``start``."""
    (x0, y0, z0), (x1, y1, z1) = low, high
    corners = [[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    faces = []
    for a, b, c, d in quads:
        faces += [[start + a, start + b, start + c], [start + a, start + c, start + d]]
    return corners, faces


def test_a_section_that_also_cuts_another_closed_body_is_refused(tmp_path):
    """: a blade and another closed body (a spinner
    exported in the same group) give a plane two closed loops; the blade's two
    sides cannot be separated from the other body's, so it is refused, by name."""
    # , one closed loop per section, never the largest of several.
    vertices, faces = _blade()
    corners, box = _box([0.4, -0.05, 0.3], [0.5, 0.05, 0.6], len(vertices))
    both = _write_obj(tmp_path / "both.obj", numpy.vstack([vertices, corners]), faces + box)
    with pytest.raises(InputArtifactError, match="its plane cuts 2 closed loops") as refused:
        derive_thin_blade(both, root_offset=0.05)
    assert str(both) in str(refused.value) and "Nothing was written" in str(refused.value)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["both.obj"]


def test_an_unwelded_trailing_edge_is_joined_before_the_sections_are_cut(tmp_path):
    """: an export that writes the trailing
    edge once per side gives the same thin blade as the welded mesh, rather than
    a refusal of a surface open along that edge."""
    # , coincident vertices are merged before cutting.
    vertices, faces = _blade()
    welded = derive_thin_blade(_write_obj(tmp_path / "w.obj", vertices, faces), root_offset=0.05)
    vertices, faces = _blade(unwelded=True)
    assert len(vertices) == RINGS * (2 * SIDE_POINTS + 1) + 2
    loose = derive_thin_blade(_write_obj(tmp_path / "u.obj", vertices, faces), root_offset=0.05)
    assert loose.stations == welded.stations
    assert numpy.allclose(_read_obj(loose.mesh), _read_obj(welded.mesh), rtol=0, atol=1e-12)


def test_the_source_import_table_is_carried_whole_to_the_thin_blade(tmp_path):
    """: the thin blade lies in the blade's own
    frame, so its sidecar states the source's unit and mesh operations, in
    order; a sidecar stating a CAD conversion is refused, nothing written."""
    # , units and [[import.operations]] carried, never dropped.
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces, group="naca")
    inventory_sidecar(source).write_text(
        'boundaries = ["naca"]\n\n[import]\nunits = "millimeter"\n\n'
        '[[import.operations]]\nop = "translate"\nvector = [0, 0, 250]\n\n'
        '[[import.operations]]\nop = "rotate"\naxis = "X"\nangle_deg = 90\n\n'
        '[[import.operations]]\nop = "rename"\nsurface = "naca"\nto = "Blade1"\n\n'
        '[[import.operations]]\nop = "scale"\nfactors = [0.5, 1.0, 2.0]\n',
        encoding="utf-8",
    )
    result = derive_thin_blade(source, root_offset=0.05)
    stated, carried = read_mesh_import(inventory_sidecar(source)), read_mesh_import(result.sidecar)
    assert stated is not None and carried == stated
    assert [operation.op for operation in carried.operations] == [
        "translate",
        "rotate",
        "rename",
        "scale",
    ]
    assert carried.names_after_renames(read_inventory(result.sidecar)) == ("Blade1",)
    cad = _write_obj(tmp_path / "cad.obj", vertices, faces)
    inventory_sidecar(cad).write_text(
        'boundaries = ["Blade1"]\n\n[import]\nunits = "FILE"\n\n'
        '[import.cad]\ntessellation_density = "LOW"\n',
        encoding="utf-8",
    )
    with pytest.raises(InputArtifactError, match=r"states \[import.cad\]") as refused:
        derive_thin_blade(cad, root_offset=0.05)
    assert str(cad) in str(refused.value) and not thin_blade_path(cad).exists()


@pytest.mark.parametrize("name", ["Blade #1", "Blade  one"])
def test_mesh_thin_blade_25_object_name_keeps_hash_and_spacing(tmp_path, name):
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces, group=name)
    try:
        result = derive_thin_blade(source, boundary=name, root_offset=0.05)
    except InputArtifactError as error:
        pytest.fail(f"the written object name must remain selectable: {error}")
    assert result is not None


def test_mesh_thin_blade_30_dense_sections_bound_temporary_memory(tmp_path, monkeypatch):
    import tracemalloc

    monkeypatch.setitem(globals(), "SIDE_POINTS", 1500)
    monkeypatch.setitem(globals(), "RINGS", 3)
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "dense.obj", vertices, faces)
    tracemalloc.start()
    try:
        result = derive_thin_blade(source, root_offset=0.05)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 40 * 1024**2, f"dense-section derivation allocated {peak} bytes"
    sheet = _read_obj(result.mesh)
    assert _off_the_mean_surface(sheet).max() < 1e-8
