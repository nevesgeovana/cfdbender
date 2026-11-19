"""The core seams CFD++ needs, and that the other adapters do not see."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from cfdbender.core.convergence import StepRow, judge_cfdpp
from cfdbender.core.executables import check_builds, resolve_executable, resolve_fields
from cfdbender.core.neutral import neutral_values, wind_directions
from cfdbender.core.plan import resolve_geometry
from cfdbender.core.settings import WorkspaceError

FIELDS = {"serial": "path", "mpi": "path", "mpi_launcher": "tokens", "tools_dir": "path"}


def registry(root, declared: str, local: str | None = None):
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "inputs/executables.toml").write_text(declared, encoding="ascii")
    if local is not None:
        (root / "inputs/executables.local.toml").write_text(local, encoding="ascii")
    return root


ENTRY = (
    '[cfdpp."22.5.U3"]\nserial = "C:/vendor/serial.exe"\nmpi = "C:/vendor/mpi.exe"\n'
    'mpi_launcher = ["mpiexec", "-n", "{ncpus}"]\n'
)


# --- multi-field build entries --------------------------------------------------------


def test_cfdpp_seam_a_multi_field_entry_is_read_for_an_adapter_declaring_fields(tmp_path):
    vendor = (tmp_path / "vendor").as_posix()  # absolute on every platform
    root = registry(tmp_path / "ws", ENTRY.replace("C:/vendor", vendor))
    entry = resolve_fields(root, "cfdpp", "22.5.U3", FIELDS)
    assert entry["mpi_launcher"] == ["mpiexec", "-n", "{ncpus}"]
    assert entry["serial"].replace("\\", "/") == f"{vendor}/serial.exe"
    assert "tools_dir" not in entry


def test_cfdpp_seam_an_unknown_field_is_refused_with_the_known_ones(tmp_path):
    root = registry(tmp_path, ENTRY + 'launcher = ["x"]\n')
    with pytest.raises(WorkspaceError) as caught:
        resolve_fields(root, "cfdpp", "22.5.U3", FIELDS)
    assert str(caught.value) == (
        "inputs/executables.toml [cfdpp] '22.5.U3': unknown field 'launcher'; the fields are: "
        "serial, mpi, mpi_launcher, tools_dir"
    )


def test_cfdpp_seam_a_malformed_field_is_refused(tmp_path):
    root = registry(tmp_path, '[cfdpp."22.5.U3"]\nmpi_launcher = "mpiexec"\n')
    with pytest.raises(WorkspaceError, match="mpi_launcher must be a list of strings"):
        resolve_fields(root, "cfdpp", "22.5.U3", FIELDS)
    root = registry(tmp_path / "b", '[cfdpp]\n"22.5.U3" = "C:/one/path.exe"\n')
    with pytest.raises(WorkspaceError, match="give a table of the fields serial, mpi"):
        resolve_fields(root, "cfdpp", "22.5.U3", FIELDS)


def test_cfdpp_seam_the_local_file_replaces_fields(tmp_path):
    vendor = (tmp_path / "vendor").as_posix()  # absolute on every platform
    here = (tmp_path / "here").as_posix()
    local = f'[cfdpp."22.5.U3"]\nserial = "{here}/serial.exe"\ntools_dir = "{here}"\n'
    root = registry(tmp_path / "ws", ENTRY.replace("C:/vendor", vendor), local)
    entry = resolve_fields(root, "cfdpp", "22.5.U3", FIELDS)
    assert entry["serial"].replace("\\", "/") == f"{here}/serial.exe"
    assert entry["mpi"].replace("\\", "/") == f"{vendor}/mpi.exe"
    assert entry["tools_dir"].replace("\\", "/") == here


def test_cfdpp_seam_an_undeclared_build_is_refused_at_plan_for_a_fields_adapter(tmp_path):
    vendor = (tmp_path / "vendor").as_posix()  # absolute on every platform
    root = registry(tmp_path, ENTRY.replace("C:/vendor", vendor))
    adapter = SimpleNamespace(EXECUTABLE_FIELDS=FIELDS)
    check_builds(root, "cfdpp", adapter, ["22.5.U3"])
    with pytest.raises(WorkspaceError, match="SOLVER_BUILD '23.1' is not declared under"):
        check_builds(root, "cfdpp", adapter, ["23.1"])


def test_cfdpp_seam_the_two_field_form_is_unchanged(tmp_path):
    root = registry(tmp_path, '[su2]\n"8.3.0" = { path = "/opt/su2", launcher = ["mpirun"] }\n')
    found = resolve_executable(root, "su2", "8.3.0")
    assert found.launcher == ("mpirun",)
    with pytest.raises(WorkspaceError, match="give a path, or a table of path and launcher"):
        resolve_executable(
            registry(tmp_path / "b", ENTRY.replace("cfdpp", "su2")), "su2", "22.5.U3"
        )


# --- neutral coefficients -------------------------------------------------------------


@pytest.mark.parametrize(("alpha", "beta"), [(0.0, 0.0), (10.0, 0.0), (6.0, 4.0)])
def test_cfdpp_seam_neutral_three_dimensions_projects_on_the_wind_axes(alpha, beta):
    drag, lift = wind_directions(alpha, beta)
    cl, cd = 0.8, 0.03
    force = [cd * d + cl * up for d, up in zip(drag, lift, strict=True)]
    native = {"CFx": force[0], "CFy": force[1], "CFz": force[2], "CMx": 0.1, "CMy": -0.2,
              "CMz": 0.3}  # fmt: skip
    values = neutral_values("cfdpp", native, dimensions=3, condition={"ALPHA": alpha, "BETA": beta})
    assert values == pytest.approx({"CL": cl, "CD": cd, "CM": -0.2})


@pytest.mark.parametrize("alpha", [0.0, 10.0, -5.0])
def test_cfdpp_seam_neutral_two_dimensions_in_the_x_y_plane(alpha):
    a = math.radians(alpha)
    cl, cd = 1.1, 0.012
    # Drag along (cos a, sin a), lift along (-sin a, cos a), the moment about +z nose-down.
    native = {
        "CFx": cd * math.cos(a) - cl * math.sin(a),
        "CFy": cd * math.sin(a) + cl * math.cos(a),
        "CFz": 0.0,
        "CMz": 0.05,
    }
    values = neutral_values("cfdpp", native, dimensions=2, condition={"ALPHA": alpha})
    assert values == pytest.approx({"CL": cl, "CD": cd, "CM": -0.05})


def test_cfdpp_seam_neutral_leaves_the_other_solvers_unchanged():
    native = {"CL": 0.5, "CD": 0.02, "CMy": -0.1, "CFx": 9.0, "CFy": 9.0, "CFz": 9.0}
    assert neutral_values("su2", native) == {"CL": 0.5, "CD": 0.02, "CM": -0.1}


# --- the convergence rule --------------------------------------------------------------


def rows(residuals, lift=None, drag=None):
    lift = lift or [0.5] * len(residuals)
    drag = drag or [0.01] * len(residuals)
    return [
        StepRow(n + 1, r, cl, cd)
        for n, (r, cl, cd) in enumerate(zip(residuals, lift, drag, strict=True))
    ]


def test_cfdpp_seam_judge_r_holds_at_the_drop_below_the_run_maximum():
    # The residual first rises to 10, then falls one decade per step: 10 -> 1e-4 at step 7.
    residuals = [1.0, 10.0, 1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5]
    verdict = judge_cfdpp(rows(residuals, lift=[0.1 * n for n in range(1, 9)]), orders=5.0)
    assert verdict.criterion == "R" and verdict.iteration == 7
    assert verdict.first_iteration_rms == 7 and verdict.first_iteration_cauchy is None


def test_cfdpp_seam_judge_c_needs_settled_forces_and_three_orders():
    count = 400
    residuals = [10.0 ** (-min(n, 150) / 50.0) for n in range(count)]
    verdict = judge_cfdpp(rows(residuals), orders=12.0)
    assert verdict.criterion == "C"
    assert verdict.iteration == 251  # the first full Cauchy window, after the 3-order drop
    flat = judge_cfdpp(rows([1.0] * count), orders=12.0)
    assert flat.criterion is None and flat.drop_orders_last == 0.0


def test_cfdpp_seam_judge_neither_and_empty():
    assert judge_cfdpp([], orders=3.0).criterion is None
    verdict = judge_cfdpp(rows([1.0, 0.5, 0.4], lift=[0.1, 0.2, 0.3]), orders=3.0)
    assert verdict.criterion is None and verdict.rows == 3


# --- a grid folder as MESH ----------------------------------------------------------


def test_cfdpp_seam_a_folder_resolves_only_for_an_adapter_that_reads_folders(tmp_path):
    folder = tmp_path / "inputs/mesh/naca.cfdpp"
    folder.mkdir(parents=True)
    assert resolve_geometry(tmp_path, "naca.cfdpp", folders=True) == "inputs/mesh/naca.cfdpp"
    assert (
        resolve_geometry(tmp_path, "inputs/mesh/naca.cfdpp", folders=True)
        == "inputs/mesh/naca.cfdpp"
    )
    with pytest.raises(ValueError, match="names no file"):
        resolve_geometry(tmp_path, "naca.cfdpp")
    with pytest.raises(ValueError, match="names no file"):
        resolve_geometry(tmp_path, "inputs/mesh/naca.cfdpp")


def test_cfdpp_seam_judge_requires_a_criterion_at_the_last_step():
    verdict = judge_cfdpp(rows([1.0, 0.00001, 1.0]), orders=4.0, elems=1)
    assert not verdict.converged
    assert verdict.first_iteration_rms == 2

    verdict = judge_cfdpp(rows([1.0, 0.0001, 0.0001, 1.0]), orders=10.0, elems=1)
    assert not verdict.converged
    assert verdict.first_iteration_cauchy == 2


def test_cfdpp_seam_zero_final_residual_agrees_with_adapter():
    verdict = judge_cfdpp(rows([0.0, 0.0]), orders=3.0, elems=1)
    assert verdict.converged
    assert verdict.criterion == "R"
