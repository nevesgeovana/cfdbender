"""0.57.0 (G7R): the junction sub-steps on the kernel, by dominance in (u, v).

The assembly rule is that the dominant imprints on the dominated, the dominated loses the
inside, the dominant is split too. Every assembly here is synthetic (boxes of bilinear patches
and a fillet collar of planar ramps), built in the test; no research IGES is read. A YES fixture
runs every sub-step and passes; each NO fixture carries the defect named by its test
and must stop at that row's sub-step, naming its check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from kernel_support import kernel_or_skip

from cfdbender.geometry._patch import Patch


def quad(a, b, c, d, family="face"):
    """A bilinear patch with corners a, b (u), d (v), c opposite a."""
    net = np.array([[a, d], [b, c]], float)
    return Patch(family, net, (0.0, 1.0), (0.0, 1.0), (2, 2), (2, 2), 1, 1)


def box(lo, hi, skip=(), split_x=None):
    """The faces of an axis-aligned box as bilinear patches; ``split_x`` cuts the y faces."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    faces = {
        "xmin": ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)),
        "xmax": ((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)),
        "ymin": ((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)),
        "ymax": ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)),
        "zmin": ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)),
        "zmax": ((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)),
    }
    out = []
    for name, (a, b, c, d) in faces.items():
        if name in skip:
            continue
        if split_x is not None and name == "ymax":
            xs = split_x
            out.append(quad((x0, y1, z0), (xs, y1, z0), (xs, y1, z1), (x0, y1, z1), name))
            out.append(quad((xs, y1, z0), (x1, y1, z0), (x1, y1, z1), (xs, y1, z1), name))
            continue
        out.append(quad(a, b, c, d, name))
    return out


#: The synthetic aircraft: a body, a wing through its side (open root buried in the body) and a
#: fillet collar of four ramps from the body side (y = 1) to the wing (y = 1 + R).
R = 0.05
WING_X, WING_Z = (1.5, 2.5), (-0.1, 0.1)


def body():
    return box((0.0, -1.0, -1.0), (4.0, 1.0, 1.0), split_x=2.0)


def wing(skip=("ymin",), z=WING_Z, x=WING_X, root=0.5):
    return box((x[0], root, z[0]), (x[1], 3.0, z[1]), skip=skip)


def collar(sides=("top", "bottom", "lead", "trail")):
    (xa, xb), (za, zb) = WING_X, WING_Z
    outer = [(xa - R, za - R), (xb + R, za - R), (xb + R, zb + R), (xa - R, zb + R)]
    inner = [(xa, za), (xb, za), (xb, zb), (xa, zb)]
    names = ["bottom", "trail", "top", "lead"]
    out = []
    for k, name in enumerate(names):
        if name not in sides:
            continue
        o0, o1 = outer[k], outer[(k + 1) % 4]
        i0, i1 = inner[k], inner[(k + 1) % 4]
        out.append(
            quad(
                (o0[0], 1.0, o0[1]),
                (o1[0], 1.0, o1[1]),
                (i1[0], 1.0 + R, i1[1]),
                (i0[0], 1.0 + R, i0[1]),
                name,
            )
        )
    return out


CUT = {"dominant": ["BODY"], "dominated": ["WING"], "operation": "cut"}
PROJECT = {"dominant": ["FILLET"], "dominated": ["BODY", "WING"], "operation": "project"}


@pytest.fixture
def kernel():
    kernel_or_skip()
    from cfdbender.mesh._gmsh import open_kernel

    k = open_kernel("G7R assembly test")
    yield k
    k.close()


def assembly(kernel, tmp_path, pieces, rows, **kwargs):
    from cfdbender.geometry._geom_assembly import Assembly

    return Assembly(kernel, pieces, rows, out=tmp_path, **kwargs)


def checks_of(record):
    return {c["id"]: c for c in record["checks"]}


def failed(records):
    last = records[-1]
    bad = [c for c in last["checks"] if c["status"] == "failed"]
    return last["stage"], bad[0]["id"] if bad else None, bad[0] if bad else None


def wb(fillet=True):
    pieces = {"BODY": body(), "WING": wing()}
    rows = [CUT]
    if fillet:
        pieces["FILLET"] = collar()
        rows.append(PROJECT)
    return pieces, rows


# ---- YES: the whole sequence on the body, wing and fillet ----------------------------------


def test_geom_kernel_assembly_yes_every_substep_passes_on_a_body_wing_and_fillet(kernel, tmp_path):
    pieces, rows = wb()
    asm = assembly(kernel, tmp_path, pieces, rows)
    records = asm.run()
    stages = [r["stage"] for r in records]
    assert stages == [s for s, _ in asm.STEPS]
    for record in records:
        for check in record["checks"]:
            assert check["status"] in ("passed", "skipped"), (record["stage"], check)
    d4 = checks_of(records[stages.index("S10.D4")])
    assert d4["junction.curve_gap"]["measured"] <= 5e-6
    assert d4["junction.single_curve"]["measured"] == 0
    assert d4["junction.length"]["measured"] == pytest.approx(2 * (1.0 + 0.2), rel=1e-6)
    d5 = checks_of(records[stages.index("S10.D5")])
    assert d5["junction.projected_length"]["measured"] == pytest.approx(0.0, abs=1e-9)
    d8 = checks_of(records[stages.index("S10.D8")])
    # body: the wing footprint (1 x 0.2) and the collar ring; wing: the root inside the body
    # (0.5 deep) and the strip under the collar (R wide), on four sides of perimeter 2.4
    removed = d8["removal.area"]["where"]["removed_m2"]
    ring = (1.0 + 2 * R) * (0.2 + 2 * R) - 1.0 * 0.2
    assert removed["BODY"] == pytest.approx(0.2 + ring, rel=1e-6)
    assert removed["WING"] == pytest.approx(2.4 * (0.5 + R), rel=1e-6)
    assert d8["junction.no_triple_edge"]["measured"] == 0
    d10 = checks_of(records[stages.index("S10.D10")])
    assert d10["sew.free_introduced"]["measured"] == 0
    assert d10["sew.triple_edges"]["measured"] == 0
    cov = checks_of(records[stages.index("S10.D8b")])
    assert cov["coverage.skin"]["status"] == "passed"
    assert (tmp_path / "S10-assembly.igs").is_file()


def test_geom_kernel_assembly_the_dominant_is_split_and_no_area_constant_decides_a_removal(
    kernel, tmp_path
):
    """Design R6: the dominant body is split along the junction (its side face holds the hole),
    and a fixture scaled by 10 keeps the same faces (no area threshold decides)."""
    pieces, rows = wb(fillet=False)
    small = assembly(kernel, tmp_path / "a", pieces, rows)
    small.run()
    kept_small = {n: len(f) for n, f in small.faces.items()}
    scaled = {
        name: [
            Patch(
                p.family,
                p.controls * 10.0,
                p.knots_u,
                p.knots_v,
                p.multiplicities_u,
                p.multiplicities_v,
                p.degree_u,
                p.degree_v,
            )
            for p in patches
        ]
        for name, patches in pieces.items()
    }
    big = assembly(kernel, tmp_path / "b", scaled, rows)
    records = big.run()
    assert {n: len(f) for n, f in big.faces.items()} == kept_small
    assert records[-1]["stage"] == "S10.E2"
    # the dominant was split: both side faces it holds were cut along the junction, and the
    # footprint under the wing (1 x 0.2, scaled by 100 in area) left it
    removal = checks_of(records[[r["stage"] for r in records].index("S10.D8")])["removal.area"]
    assert removal["where"]["removed_m2"]["BODY"] == pytest.approx(0.2 * 100, rel=1e-6)
    # four sides, the upper and lower split where they cross the side's seam at x = 2
    assert len(big._junction("BODY", "WING")) == 6


# ---- NO fixtures, one per checked condition -----------------------------------------------


def test_geom_kernel_assembly_s1_no_a_piece_with_a_face_fewer_stops_at_load(kernel, tmp_path):
    pieces, rows = wb(fillet=False)
    asm = assembly(kernel, tmp_path, pieces, rows, expected_faces={"WING": 6})
    stage, check, entry = failed(asm.run())
    assert (stage, check) == ("S10.D1", "assembly.face_count")
    assert entry["where"]["component"] == "WING"


def test_geom_kernel_assembly_w2_no_pieces_of_a_row_that_do_not_touch_stop_at_load(
    kernel, tmp_path
):
    pieces = {"BODY": body(), "WING": wing(root=1.5)}  # the wing starts 0.5 off the body
    stage, check, _ = failed(assembly(kernel, tmp_path, pieces, [CUT]).run())
    assert (stage, check) == ("S10.D1", "placement.contact")


def test_geom_kernel_assembly_s2_no_a_wing_without_its_tip_face_has_unexplained_free_edges(
    kernel, tmp_path
):
    pieces = {"BODY": body(), "WING": wing(skip=("ymin", "ymax"))}
    stage, check, entry = failed(assembly(kernel, tmp_path, pieces, [CUT]).run())
    assert (stage, check) == ("S10.D2", "piece.free_edges")
    assert entry["where"]["component"] == "WING" and entry["measured"] == 4


def test_geom_kernel_assembly_bj7_no_a_seam_that_closes_only_at_0_01_mm_fails_at_0_001_mm(
    kernel, tmp_path
):
    shifted = body()
    top = shifted[-1]
    shifted[-1] = Patch(
        top.family,
        top.controls + np.array([0.0, 0.0, 5e-6]),
        top.knots_u,
        top.knots_v,
        top.multiplicities_u,
        top.multiplicities_v,
        1,
        1,
    )
    pieces = {"BODY": shifted, "WING": wing()}
    stage, check, entry = failed(assembly(kernel, tmp_path, pieces, [CUT]).run())
    assert (stage, check) == ("S10.C7", "junction.free_edges_classified")
    assert entry["where"]["component"] == "BODY"


def test_geom_kernel_assembly_bj6_no_a_half_body_off_the_symmetry_plane(kernel, tmp_path):
    half = box((0.0, 1e-5, -1.0), (4.0, 1.0, 1.0), skip=("ymin",))
    pieces = {"BODY": half, "WING": wing()}
    asm = assembly(kernel, tmp_path, pieces, [CUT], plane_axis=1)
    stage, check, _ = failed(asm.run())
    assert (stage, check) == ("S10.C6", "piece.symmetry_edges")


def test_geom_kernel_assembly_bj6_yes_a_half_body_on_the_plane_passes(kernel, tmp_path):
    half = box((0.0, 0.0, -1.0), (4.0, 1.0, 1.0), skip=("ymin",), split_x=2.0)
    asm = assembly(kernel, tmp_path, {"BODY": half, "WING": wing()}, [CUT], plane_axis=1)
    records = asm.run()
    assert records[-1]["stage"] == "S10.E2", failed(records)
    assert (
        checks_of(records[[r["stage"] for r in records].index("S10.C6")])["piece.symmetry_edges"][
            "status"
        ]
        == "passed"
    )


def test_geom_kernel_assembly_bj8_no_a_rim_of_the_dominated_outside_the_dominant(kernel, tmp_path):
    pieces = {"BODY": body(), "WING": wing(z=(0.95, 1.05))}  # the root's top edge leaves the body
    stage, check, _ = failed(assembly(kernel, tmp_path, pieces, [CUT]).run())
    assert (stage, check) == ("S10.C8", "join.rim_inside")


def test_geom_kernel_assembly_bj9_no_an_opening_of_the_dominant_20_mm_off_the_dominated(
    kernel, tmp_path
):
    """A body that carries its own opening for the wing; the wing is 20 mm thinner (the HT of
    the 22:04 file against the boom's opening)."""
    x0, x1, y1 = 0.0, 4.0, 1.0
    (xa, xb), (za, zb) = WING_X, WING_Z
    xs, zs = (x0, xa, xb, x1), (-1.0, za, zb, 1.0)
    side = [  # the side face as a 3 by 3 grid of quads, its middle one left open
        quad((xs[i], y1, zs[j]), (xs[i + 1], y1, zs[j]), (xs[i + 1], y1, zs[j + 1]),
             (xs[i], y1, zs[j + 1]))
        for i in range(3)
        for j in range(3)
        if (i, j) != (1, 1)
    ]  # fmt: skip
    framed = box((x0, -1.0, -1.0), (x1, y1, 1.0), skip=("ymax",)) + side
    thin = wing(z=(za + 0.02, zb - 0.02))
    stage, check, entry = failed(
        assembly(kernel, tmp_path, {"BODY": framed, "WING": thin}, [CUT]).run()
    )
    assert (stage, check) == ("S10.C9", "join.opening_on_partner")
    assert entry["measured"] == pytest.approx(0.02, rel=1e-3)


def test_geom_kernel_assembly_s4_no_an_open_dominated_piece_leaves_lonely_junction_ends(
    kernel, tmp_path
):
    pieces = {"BODY": body(), "WING": wing(skip=("ymin", "xmin"))}  # no leading-edge face
    asm = assembly(kernel, tmp_path, pieces, [CUT])
    asm.step("S10.D1")
    record = asm.step("S10.D4")
    assert checks_of(record)["junction.chain_ends"]["status"] == "failed"
    assert checks_of(record)["junction.chain_ends"]["measured"] == 2
    # the record names where the chain stops, so a failure on another machine can be read
    where = checks_of(record)["junction.chain_ends"]["where"]
    assert where["pair"] and where["count"] == len(where["curves"]) > 0
    assert len(where["lonely_ends"]) == 2
    for end in where["lonely_ends"]:
        assert end["curve"] in where["curves"] and len(end["xyz_m"]) == 3
        assert end["nearest_m"] > 2e-5


def test_geom_kernel_assembly_w4_no_a_junction_of_10_mm_is_refused_as_no_intersection(
    kernel, tmp_path
):
    tiny = wing(x=(2.0, 2.0025), z=(0.0, 0.0025))
    asm = assembly(kernel, tmp_path, {"BODY": body(), "WING": tiny}, [CUT])
    asm.step("S10.D1")
    entry = checks_of(asm.step("S10.D4"))["junction.length"]
    assert entry["status"] == "failed" and entry["measured"] == pytest.approx(0.01, rel=1e-6)


def scaled(pieces, factor):
    """The pieces with every pole multiplied by ``factor`` (a model of another size)."""
    return {
        name: [
            Patch(
                p.family,
                p.controls * factor,
                p.knots_u,
                p.knots_v,
                p.multiplicities_u,
                p.multiplicities_v,
                p.degree_u,
                p.degree_v,
            )
            for p in patches
        ]
        for name, patches in pieces.items()
    }


@pytest.mark.parametrize("factor", [0.01, 10.0])
def test_geom_kernel_assembly_w4_the_junction_length_limit_scales_with_the_pieces(
    kernel, tmp_path, factor
):
    """the shortest junction is relative to the pieces it judges, so the same
    model at 0.01 and 10 times its size keeps its verdicts. A 2.4 m junction of a wing 2.5 m
    across passes and a 20 mm one fails, whatever the scale (an absolute 0.1 m refused the
    first at 0.01 and accepted the second at 10)."""
    good = scaled({"BODY": body(), "WING": wing()}, factor)
    asm = assembly(kernel, tmp_path / "yes", good, [CUT])
    asm.step("S10.D1")
    entry = checks_of(asm.step("S10.D4"))["junction.length"]
    assert entry["status"] == "passed", entry
    assert entry["measured"] == pytest.approx(2.4 * factor, rel=1e-6)
    tiny = scaled({"BODY": body(), "WING": wing(x=(2.0, 2.005), z=(0.0, 0.005))}, factor)
    asm = assembly(kernel, tmp_path / "no", tiny, [CUT])
    asm.step("S10.D1")
    entry = checks_of(asm.step("S10.D4"))["junction.length"]
    assert entry["status"] == "failed", entry
    # at 0.01 the 0.2 mm junction meets the kernel's own resolution: within 0.1 percent
    assert entry["measured"] == pytest.approx(0.02 * factor, rel=1e-3)
    assert entry["limit"] == pytest.approx(0.04 * 2.5 * factor, rel=1e-9)


def test_geom_kernel_assembly_s5_no_a_fillet_ring_open_on_one_side_does_not_close_its_footprint(
    kernel, tmp_path
):
    pieces = {"BODY": body(), "WING": wing(), "FILLET": collar(("top", "lead", "trail"))}
    asm = assembly(kernel, tmp_path, pieces, [CUT, PROJECT])
    for stage in ("S10.D1", "S10.D4"):
        asm.step(stage)
    entry = checks_of(asm.step("S10.D5"))["junction.ends_on_cut"]
    assert entry["status"] == "failed"
    assert "class STRATEGY" in entry["note"]


def test_geom_kernel_assembly_a_substep_that_raises_fails_at_that_substep(
    kernel, tmp_path, monkeypatch
):
    """a kernel answering NaN for a curve projection made S10.D6 pass (the
    built-in max dropped the NaN, so a lost projection read 0). The NaN is now an infinite
    distance, its check cannot be built, and the sub-step is recorded failed where it stands,
    not raised out of the assembly (which a run reported as a refusal at S10.D1)."""
    from cfdbender.mesh._gmsh import GmshKernel

    def nowhere(self, edge, points):
        return np.full((len(points), 3), np.nan)

    monkeypatch.setattr(GmshKernel, "edge_project", nowhere)
    pieces, rows = wb()
    records = assembly(kernel, tmp_path, pieces, rows).run()
    stage, check, entry = failed(records)
    assert (stage, check) == ("S10.D6", "stage.raised")
    assert "junction.on_host_loops" in entry["note"] and "class BUG" in entry["note"]


def test_geom_kernel_assembly_coverage_and_sewing_catch_a_kept_face_deleted_after_removal(
    kernel, tmp_path
):
    """Design F.1 mutation: a kept face deleted after the removal; the edge audit still passes
    (no triple edge), coverage.skin fails with that face's area and sewing finds new free edges.
    Control: the unmutated fixture passes (the YES test above)."""
    pieces, rows = wb(fillet=False)
    asm = assembly(kernel, tmp_path, pieces, rows)
    for stage, _ in asm.STEPS:
        asm.step(stage)
        if stage == "S10.D8":
            break
    victim = max(asm.faces["BODY"], key=kernel.area)
    area = kernel.area(victim)
    kernel.remove([victim])
    asm.faces["BODY"].remove(victim)
    coverage = checks_of(asm.step("S10.D8b"))["coverage.skin"]
    assert coverage["status"] == "failed"
    assert coverage["where"]["missing_m2"] == pytest.approx(area, rel=2e-2)
    asm.step("S10.D9")
    sewn = checks_of(asm.step("S10.D10"))
    assert sewn["sew.free_introduced"]["status"] == "failed"
    assert sewn["sew.triple_edges"]["status"] == "passed"


def _tail():
    fuselage = box((0.0, -1.0, -1.0), (4.0, 1.0, 1.0))
    fin = box((2.5, -0.05, 0.5), (3.5, 0.05, 2.0), skip=("zmin",))
    tail = box((2.8, -1.5, 1.5), (3.4, 1.5, 1.6))
    pieces = {"BODY": fuselage, "FIN": fin, "HT": tail}
    rows = [
        {"dominant": ["BODY"], "dominated": ["FIN"], "operation": "cut"},
        {"dominant": ["FIN"], "dominated": ["HT"], "operation": "cut"},
    ]
    return pieces, rows


def test_geom_kernel_assembly_t1_yes_the_tail_group_is_one_fragment_with_every_junction(
    kernel, tmp_path
):
    pieces, rows = _tail()
    records = assembly(kernel, tmp_path, pieces, rows).run()
    assert records[-1]["stage"] == "S10.E2", failed(records)
    e1 = checks_of(records[[r["stage"] for r in records].index("S10.E1")])
    lengths = e1["group.junction_length"]["where"]["lengths_m"]
    assert lengths["BODY x FIN"] == pytest.approx(2 * (1.0 + 0.1), rel=1e-6)
    # the tail plane crosses both sides of the thin fin: two loops of 2 (0.6 + 0.1) m
    assert lengths["FIN x HT"] == pytest.approx(2 * 2 * (0.6 + 0.1), rel=1e-6)
    assert e1["group.single_curve"]["measured"] == 0


def test_geom_kernel_assembly_t1_pairs_cut_separately_are_measured_by_the_same_check(
    kernel, tmp_path
):
    """Record the junction count after sequential pair cuts at fuzzy 0.05 mm.

    On the reference kernel, pairs cut one after another stay one curve per junction.
    The check runs on that state and reads 0; this records the measurement.

    Measured on gmsh 4.15.2 (the tail of this file: BODY x FIN and FIN x HT
    cut one after another): group.single_curve 0, passed. The replacement NO, a pair of the
    group that does not meet, is the next test. A kernel on which this reads above 0 changes
    the numerical baseline, and this test fails to report that change."""
    pieces, rows = _tail()
    asm = assembly(kernel, tmp_path, pieces, rows)
    asm.step("S10.D1")
    asm.cut_pairs_separately()
    entry = checks_of(asm.step("S10.E1"))["group.single_curve"]
    assert entry["status"] == "passed" and entry["measured"] == 0


def test_geom_kernel_assembly_t1_no_a_pair_of_the_group_that_does_not_meet(kernel, tmp_path):
    """NO: the tail plane above the fin's top (0.1 m clear): the group finds no junction for
    that pair, so group.junction_length fails (and placement.contact stops a full run first)."""
    pieces, rows = _tail()
    pieces["HT"] = box((2.8, -1.5, 2.1), (3.4, 1.5, 2.2))
    asm = assembly(kernel, tmp_path, pieces, rows)
    assert checks_of(asm.step("S10.D1"))["placement.contact"]["status"] == "failed"
    entry = checks_of(asm.step("S10.E1"))["group.junction_length"]
    assert entry["status"] == "failed" and entry["where"]["lengths_m"]["FIN x HT"] == 0


def test_geom_kernel_assembly_t2_no_a_separate_piece_sunk_into_another_has_no_clearance(
    kernel, tmp_path
):
    pieces, rows = wb(fillet=False)
    pieces["CONTROL"] = box((3.5, 0.5, 0.9), (3.9, 0.9, 1.2))  # in no row, through the top
    stage, check, _ = failed(assembly(kernel, tmp_path, pieces, rows).run())
    assert (stage, check) == ("S10.E2", "assembly.clearance")


def test_geom_kernel_assembly_bj1_no_a_blister_0_015_mm_off_its_body_finds_no_shared_curve(
    kernel, tmp_path
):
    """A body join (blister on body) shares its crease after the cut; NO: the blister's rim sits
    0.015 mm below the body (the crease gap of the original left un-unified)."""
    join = {"dominant": ["BLISTER"], "dominated": ["BODY"], "operation": "cut"}
    good = box((1.0, 0.2, -1.3), (3.0, 0.6, -0.8), skip=("zmax",))
    records = assembly(
        kernel,
        tmp_path / "yes",
        {"BODY": body(), "BLISTER": good},
        [join],
        kinds={"BODY": "body", "BLISTER": "blister"},
    ).run()
    assert records[-1]["stage"] == "S10.E2", failed(records)
    c1 = checks_of(records[[r["stage"] for r in records].index("S10.C1")])
    assert c1["join.shared_pairs"]["measured"] == 4
    bad = box((1.0, 0.2, -1.3), (3.0, 0.6, -1.0 - 1.5e-5), skip=("zmax",))
    asm = assembly(
        kernel,
        tmp_path / "no",
        {"BODY": body(), "BLISTER": bad},
        [join],
        kinds={"BODY": "body", "BLISTER": "blister"},
    )
    stage, check, _ = failed(asm.run())
    assert (stage, check) in {("S10.D3", "join.crease_gap"), ("S10.C1", "join.shared_pairs")}
    assert stage == "S10.D3" or check == "join.shared_pairs"


def _steps_until(asm, last):
    """Run the sub-steps of ``asm`` in order up to and including ``last``."""
    for stage, _ in asm.STEPS:
        record = asm.step(stage)
        assert not [c for c in record["checks"] if c["status"] == "failed"], record
        if stage == last:
            return


def _edge_points(kernel, edge, count=9):
    low, high = kernel.edge_bounds(edge)
    return kernel.edge_values(edge, np.linspace(low, high, count).tolist())


def test_geom_kernel_assembly_bj3_no_an_edge_0_06_mm_off_its_own_face(
    kernel, tmp_path, monkeypatch
):
    """T-BJ3: an edge curve 0.06 mm off its face's surface (the original's own tolerance is
    0.05 mm). A kernel cannot be made to build that edge from patches, so the kernel the
    sub-step asks reports one edge of the joined blister 0.06 mm off its wall, normal to it:
    S10.C3 passes before and fails after, with that distance."""
    from cfdbender.mesh._gmsh import GmshKernel

    join = {"dominant": ["BLISTER"], "dominated": ["BODY"], "operation": "cut"}
    blister = box((1.0, 0.2, -1.3), (3.0, 0.6, -0.8), skip=("zmax",))
    asm = assembly(
        kernel,
        tmp_path,
        {"BODY": body(), "BLISTER": blister},
        [join],
        kinds={"BODY": "body", "BLISTER": "blister"},
    )
    _steps_until(asm, "S10.C2")
    assert checks_of(asm.step("S10.C3"))["join.edge_on_surface"]["status"] == "passed"
    # the bottom edge of the blister's wall x = 1 (along y at z = -1.3)
    bad = {
        abs(e)
        for face in asm.faces["BLISTER"]
        for e in kernel.boundary(face)
        if np.allclose(_edge_points(kernel, abs(e))[:, [0, 2]], [1.0, -1.3], atol=1e-9)
    }
    assert len(bad) == 1
    real = GmshKernel.edge_values

    def off(self, edge, parameters):
        values = real(self, edge, parameters)
        return values - np.array([6e-5, 0.0, 0.0]) if abs(edge) in bad else values

    monkeypatch.setattr(GmshKernel, "edge_values", off)
    entry = checks_of(asm.step("S10.C3"))["join.edge_on_surface"]
    assert entry["status"] == "failed"
    assert entry["measured"] == pytest.approx(6e-5, rel=1e-6)


def test_geom_kernel_assembly_s9_no_a_writer_that_drops_the_faces_with_inner_loops(
    kernel, tmp_path, monkeypatch
):
    """T-S9: a writer that loses what has inner loops reads back fewer faces. The wing passes
    through the middle of the body's side face, which keeps it as an inner loop; S10.D9 passes
    with the real writer and fails with that one, naming the body."""
    from cfdbender.geometry.stages import junctions
    from cfdbender.mesh._gmsh import GmshKernel

    def loops(k, face):
        ends = [tuple(_edge_points(k, abs(e), 2)) for e in k.boundary(face)]
        return len(junctions.chain(ends, junctions.CHAIN_END_GAP_M)[0])

    pieces = {"BODY": body(), "WING": wing(x=(0.5, 1.5))}
    asm = assembly(kernel, tmp_path, pieces, [CUT])
    _steps_until(asm, "S10.C4")
    assert any(loops(kernel, f) > 1 for f in asm.faces["BODY"])
    assert checks_of(asm.step("S10.D9"))["write.read_back"]["status"] == "passed"
    real = GmshKernel.export_iges

    def outer_only(self, groups, path, **options):
        kept = {name: [f for f in faces if loops(self, f) == 1] for name, faces in groups.items()}
        return real(self, kept, path, **options)

    monkeypatch.setattr(GmshKernel, "export_iges", outer_only)
    entry = checks_of(asm.step("S10.D9"))["write.read_back"]
    assert entry["status"] == "failed"
    assert entry["measured"] >= 1 and entry["where"]["component"] == "BODY"


def test_geom_kernel_assembly_bj4_one_combined_import_sews_the_join_and_file_by_file_does_not(
    kernel, tmp_path
):
    """T-BJ4: the join sewn from one combined file has no free crease; the same pair imported
    file by file leaves the crease free, so the check must use one import."""
    from cfdbender.geometry._geom_assembly import sewn_free_edges

    join = {"dominant": ["BLISTER"], "dominated": ["BODY"], "operation": "cut"}
    good = box((1.0, 0.2, -1.3), (3.0, 0.6, -0.8), skip=("zmax",))
    asm = assembly(
        kernel,
        tmp_path,
        {"BODY": body(), "BLISTER": good},
        [join],
        kinds={"BODY": "body", "BLISTER": "blister"},
    )
    records = asm.run()
    c4 = checks_of(records[[r["stage"] for r in records].index("S10.C4")])
    assert c4["join.group_free_edges"]["measured"] == 0
    kernel.export_iges({"BODY": asm.faces["BODY"]}, tmp_path / "a.igs")
    kernel.export_iges({"BLISTER": asm.faces["BLISTER"]}, tmp_path / "b.igs")
    apart = sum(
        len(sewn_free_edges(kernel, [p], 1e-5)) for p in (tmp_path / "a.igs", tmp_path / "b.igs")
    )
    assert apart >= 4


# ---- end to end at the verb a user calls ---------------------------------------------------


def _toml(tmp_path, rows):
    text = (
        "[components.FUS]\ntype = 'body'\nlength_mm = 4000.0\n"
        "[components.FAR]\ntype = 'body'\nlength_mm = 1000.0\n"
        "[components.FAR.placement]\ntranslation = [9.0, 0.0, 0.0]\n" + rows
    )
    path = tmp_path / "pair.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_geom_kernel_assembly_run_geometry_stops_at_the_substep_and_logs_it(tmp_path):
    """End to end through run_geometry: a cut row between two bodies that do not touch stops at
    S10.D1 placement.contact; the report names the sub-step and the log line holds the check,
    the measured value, the threshold and the verdict."""
    kernel_or_skip()
    from cfdbender.geometry import pipeline as _geom_run

    rows = "[[dominance]]\ndominant = 'FUS'\ndominated = 'FAR'\noperation = 'cut'\n"
    report = _geom_run.run_geometry(_toml(tmp_path, rows), out=tmp_path / "run", until="S10.E2")
    assert (report["status"], report["stage"], report["check"]) == (
        "failed",
        "S10.D1",
        "placement.contact",
    ), report
    log = (tmp_path / "run" / "geom-run.log").read_text(encoding="utf-8")
    assert "S10.D1" in log and "check placement.contact: measured" in log and "FAILED" in log
    record = json.loads((tmp_path / "run" / "S10.D1.json").read_text(encoding="utf-8"))
    assert record["outputs"]["inputs"]["rows"]


def test_geom_kernel_assembly_run_geometry_with_no_row_skips_the_substeps_with_their_reason(
    tmp_path,
):
    from cfdbender.geometry import pipeline as _geom_run

    report = _geom_run.run_geometry(_toml(tmp_path, ""), out=tmp_path / "run", until="S10.E2")
    record = json.loads((tmp_path / "run" / "S10.D4.json").read_text(encoding="utf-8"))
    assert all(c["status"] == "skipped" for c in record["checks"])
    assert "no dominance row" in record["checks"][0]["note"]
    assert not any(item.startswith("S10.D4 ") for item in report["not_run"])


# ---- through run_geometry, on synthetic pieces ------------------------------------------------


def _run_pieces(tmp_path, monkeypatch, pieces, rows, *, again=None):
    """Run run_geometry to S10.E2 on a TOML naming ``pieces`` (all bodies), the generation of
    the components replaced by the synthetic pieces; ``again`` is the second build (BJ5).

    The run stays in this process (the worker variable), so the replaced generation and any
    kernel defect a test plants are the ones the assembly meets."""
    kernel_or_skip()
    from cfdbender.geometry import pipeline as _geom_run
    from cfdbender.geometry import stages

    builds = []

    def generate(resolved, *args, **kwargs):
        builds.append(1)
        source = again if again is not None and len(builds) > 1 else pieces
        return {name: list(patches) for name, patches in source.items()}

    monkeypatch.setattr(stages, "generate_components", generate)
    monkeypatch.setenv("CFDBENDER_GMSH_WORKER", "1")
    text = "".join(
        f"[components.{name}]\ntype = 'body'\nlength_mm = 1000.0\n"
        f"[components.{name}.placement]\ntranslation = [{3.0 * k}, 0.0, 0.0]\n"
        for k, name in enumerate(pieces)
    )
    text += "".join(
        f"[[dominance]]\ndominant = {row['dominant']!r}\ndominated = {row['dominated']!r}\n"
        f"operation = {row['operation']!r}\n"
        for row in rows
    )
    path = tmp_path / "pieces.toml"
    path.write_text(text, encoding="utf-8")
    report = _geom_run.run_geometry(path, out=tmp_path / "run", until="S10.E2")
    return report, tmp_path / "run"


def test_geom_kernel_assembly_run_log_times_each_substep_by_its_own_wall_time(
    tmp_path, monkeypatch
):
    """the assembly runs every sub-step in the first call, so the runner's
    clock gave S10.D1 the whole assembly and every later sub-step about 0 s. Each sub-step's
    record and log line now carry the sub-step's own wall time."""
    pieces, rows = wb(fillet=False)
    report, run = _run_pieces(tmp_path, monkeypatch, pieces, rows)
    assert report["status"] != "failed", report
    assembly = json.loads((run / "S10-assembly.json").read_text(encoding="utf-8"))
    own = {r["stage"]: r["wall_s"] for r in assembly["records"]}
    for stage in ("S10.D1", "S10.D2", "S10.D8"):
        record = json.loads((run / f"{stage}.json").read_text(encoding="utf-8"))
        assert record["wall_s"] == own[stage], (stage, record["wall_s"], own[stage])
    log = (run / "geom-run.log").read_text(encoding="utf-8")
    assert f"(wall {own['S10.D2']:.3f} s)" in log


def _stopped(report):
    return report["status"], report["stage"], report["check"]


def test_geom_kernel_assembly_bj5_no_through_run_geometry_a_second_build_with_another_pole(
    tmp_path, monkeypatch
):
    """T-BJ5 through the verb: the second build of a piece from its own inputs moves one pole by
    1e-6 m (a generator that does not rebuild what it built); the run stops at S10.C5."""
    pieces, rows = wb(fillet=False)
    again = {name: list(patches) for name, patches in pieces.items()}
    first = again["WING"][0]
    again["WING"][0] = Patch(
        first.family,
        first.controls + np.array([0.0, 0.0, 1e-6]),
        first.knots_u,
        first.knots_v,
        first.multiplicities_u,
        first.multiplicities_v,
        first.degree_u,
        first.degree_v,
    )
    report, run = _run_pieces(tmp_path, monkeypatch, pieces, rows, again=again)
    assert _stopped(report) == ("failed", "S10.C5", "piece.regenerated"), report
    record = json.loads((run / "S10.C5.json").read_text(encoding="utf-8"))
    entry = checks_of(record)["piece.regenerated"]
    assert entry["measured"] == pytest.approx(1e-6, rel=1e-6)
    assert "class BUG" in entry["note"]


def test_geom_kernel_assembly_bj2_no_through_run_geometry_a_join_written_with_its_crease_apart(
    tmp_path, monkeypatch
):
    """T-BJ2 through the verb: the writer puts one side of the join 0.0205 mm off the other
    (the crease offset), so the two written curves are no longer one; the run stops
    at S10.C2 with that distance."""
    from cfdbender.mesh._gmsh import GmshKernel

    real = GmshKernel.export_iges

    def apart(self, groups, path, **options):
        if not Path(path).name.startswith("S10-join-"):
            return real(self, groups, path, **options)
        moved = self.copy(groups["BLISTER"])
        self.transform(moved, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2.05e-5], [0, 0, 0, 1]])
        try:
            return real(self, {**groups, "BLISTER": list(moved)}, path, **options)
        finally:
            self.remove(list(moved))

    monkeypatch.setattr(GmshKernel, "export_iges", apart)
    join = {"dominant": ["BLISTER"], "dominated": ["BODY"], "operation": "cut"}
    blister = box((1.0, 0.2, -1.3), (3.0, 0.6, -0.8), skip=("zmax",))
    report, run = _run_pieces(tmp_path, monkeypatch, {"BODY": body(), "BLISTER": blister}, [join])
    assert _stopped(report) == ("failed", "S10.C2", "join.written_pairs"), report
    record = json.loads((run / "S10.C2.json").read_text(encoding="utf-8"))
    assert checks_of(record)["join.written_pairs"]["measured"] == pytest.approx(2.05e-5, rel=1e-2)


def test_geom_kernel_assembly_s6_no_through_run_geometry_an_imprint_that_splits_no_host(
    tmp_path, monkeypatch
):
    """T-S6 through the verb: an imprint that leaves the hosts whole puts the projected junction
    on the host but on none of its loops; the run stops at S10.D6 with the distance."""
    from cfdbender.mesh._cad import FragmentMap
    from cfdbender.mesh._gmsh import GmshKernel

    def whole(self, faces, edges, *, fuzzy=0.0):
        return FragmentMap(faces=tuple(faces), from_a=tuple((f,) for f in faces), from_b=())

    monkeypatch.setattr(GmshKernel, "imprint", whole)
    pieces, rows = wb()
    report, run = _run_pieces(tmp_path, monkeypatch, pieces, rows)
    assert _stopped(report) == ("failed", "S10.D6", "junction.on_host_loops"), report
    record = json.loads((run / "S10.D6.json").read_text(encoding="utf-8"))
    assert checks_of(record)["junction.on_host_loops"]["measured"] > 5e-6


@pytest.mark.parametrize(
    ("kinds", "extent", "fuzzy"),
    [
        ({"host": "body", "panel": "lifting_surface"}, 2.5, 5e-5),
        ({"host": "body", "panel": "lifting_surface"}, 0.025, 5e-7),
        ({"BODY": "body", "BLISTER": "blister"}, 2.5, 0.0),
    ],
    ids=["P0570-wing-body-pair-fuzzy", "P0570-pair-fuzzy-scales", "P0570-body-join-pair-exact"],
)
def test_geom_kernel_assembly_a_wing_body_pair_is_cut_with_the_group_fuzzy_value(
    kinds, extent, fuzzy
):
    """CI run 43484118433: the exact fragment of a body and a wing whose blunt trailing-edge
    base lies in the plane of a body seam kept only the 2.52 mm junction across that base on one
    runner. A pair cut takes the group fuzzy value; only a body join stays exact."""
    import contextlib
    from types import SimpleNamespace

    from cfdbender.geometry._geom_assembly import Assembly

    calls = []
    names = list(kinds)

    def fragment(a, b, *, fuzzy=0.0):
        calls.append(fuzzy)
        return SimpleNamespace(from_a=[(f,) for f in a], from_b=[(f,) for f in b])

    stub = SimpleNamespace(
        cut_done=False,
        kinds=kinds,
        extents=dict.fromkeys(kinds, extent),
        faces={names[0]: [1, 2], names[1]: [3]},
        k=SimpleNamespace(fragment=fragment),
        _cut_groups=lambda: [names],
        _active_faces=contextlib.nullcontext,
        _regroup=lambda group, inputs, rows: None,
    )
    stub._group_fuzzy = lambda group: Assembly._group_fuzzy(stub, group)
    Assembly._cut_all(stub)
    assert calls == [pytest.approx(fuzzy, rel=1e-12)]
