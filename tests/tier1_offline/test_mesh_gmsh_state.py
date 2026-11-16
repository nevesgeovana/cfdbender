"""P0570 geometry kernel: a read does not depend on the unit an earlier session asked for.

OpenCASCADE keeps the target unit of its last import in a process-wide static that outlives
``gmsh.finalize``, so these tests hold on every version and platform, in any test order.
"""

import os
import subprocess
import sys

import pytest
from kernel_support import kernel_or_skip


def _leave_a_stale_unit(path):
    """Read ``path`` in metres in a session of its own, which outlives that session."""
    from cfdbender.mesh._gmsh import session

    with session("a test") as gmsh:
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.add("stale")
        gmsh.model.occ.importShapes(str(path))


def _write_plate(gmsh, path):
    gmsh.model.add("write")
    gmsh.model.occ.addRectangle(0, 0, 0, 1.2, 1.2)
    gmsh.model.occ.synchronize()
    gmsh.write(str(path))


def test_mesh_gmsh_state_a_read_after_a_session_that_asked_for_metres_stays_in_millimetres(
    tmp_path,
):
    kernel_or_skip()
    from cfdbender.mesh._gmsh import import_iges, session

    path = tmp_path / "plate.igs"
    with session("a test") as gmsh:
        _write_plate(gmsh, path)
    _leave_a_stale_unit(path)
    with session("a test") as gmsh:
        gmsh.model.add("read")
        faces = import_iges(gmsh, path)
        gmsh.model.occ.synchronize()
        assert gmsh.model.occ.getBoundingBox(2, faces[0])[3] == pytest.approx(1.2, abs=1e-3)


def test_mesh_gmsh_state_the_first_read_of_a_process_succeeds_in_millimetres(tmp_path):
    kernel_or_skip()
    from cfdbender.mesh._gmsh import session

    path = tmp_path / "plate.igs"
    with session("a test") as gmsh:
        _write_plate(gmsh, path)
    # Only a process that has read or written nothing yet can show this: the target unit does not
    # exist there, so this test runs in one of its own.
    code = (
        "import sys\n"
        "from cfdbender.mesh._gmsh import import_iges, session\n"
        "with session('a test') as gmsh:\n"
        "    gmsh.model.add('read')\n"
        "    faces = import_iges(gmsh, sys.argv[1])\n"
        "    gmsh.model.occ.synchronize()\n"
        "    print('XMAX', gmsh.model.occ.getBoundingBox(2, faces[0])[3])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
    )
    assert done.returncode == 0, done.stderr
    x_max = [line for line in done.stdout.splitlines() if line.startswith("XMAX")]
    assert float(x_max[0].split()[1]) == pytest.approx(1.2, abs=1e-3)


def test_mesh_gmsh_state_a_read_keeps_the_unit_the_session_asked_for(tmp_path):
    kernel_or_skip()
    from cfdbender.mesh._gmsh import import_iges, session

    path = tmp_path / "plate.igs"
    with session("a test") as gmsh:
        _write_plate(gmsh, path)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        import_iges(gmsh, path)
        assert gmsh.option.getString("Geometry.OCCTargetUnit") == "M"


def test_mesh_gmsh_state_a_generated_wing_identifies_at_its_span_after_a_stale_unit(tmp_path):
    kernel_or_skip()
    pytest.importorskip("contourpy")
    from test_geom_kernel_reverse import WING

    from cfdbender.geometry._geometry_generate import generate_iges
    from cfdbender.geometry._identify import identify_lifting_surface

    path = tmp_path / "wing.igs"
    generate_iges(WING, path)
    _leave_a_stale_unit(path)
    identified = identify_lifting_surface(path, "WING", stations=61, grid=129)
    assert identified.planform["span_m"] == pytest.approx(1.2, abs=1e-3)
