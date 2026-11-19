"""CFD++ setup, reference and post files: what planning accepts and refuses.

Proved on the invented toy database, whose setup table binds the same keys the real
one will. The real database of 22.5.U3 is used only where its answer is fixed today:
the version gate and the refusal of a layout not yet established.
"""

from __future__ import annotations

import math
import shutil
import tomllib
import warnings
from pathlib import Path

import pytest
from cfdpp_workspace import (
    POST,
    SETUP,
    TOY,
    matrix,
    use_toy,
    with_keywords,
    workspace,
)

from cfdbender.core.flight import resolve_flow
from cfdbender.core.neutral import wind_directions
from cfdbender.core.plan import PlanError
from cfdbender.solvers.cfdpp import _config
from cfdbender.solvers.cfdpp._schema import schema_for
from cfdbender.workspace.planning import plan_workspace

NAMES = {1: "airfoil", 2: "farfield"}
REFERENCE = {"area": 1.0, "length": 1.0, "moment_origin": [0.25, 0.0, 0.0]}
POST_TABLE = tomllib.loads(POST)
CONDITION = {"MACH": 0.15, "REmi": 6.0, "TK": 300.0, "ALPHA": 10.0}


def toy():
    return schema_for("1.0", TOY)


def setup(**changes):
    table = tomllib.loads(SETUP)
    for dotted, value in changes.items():
        *path, leaf = dotted.split("__")
        target = table
        for name in path:
            target = target.setdefault(name, {})
        if value is None:
            target.pop(leaf, None)
        else:
            target[leaf] = value
    return table


def build(table=None, *, post=None, condition=None, names=None, dimension=2, reference=None):
    return _config.config(
        toy(),
        setup() if table is None else table,
        REFERENCE if reference is None else reference,
        POST_TABLE if post is None else post,
        CONDITION if condition is None else condition,
        mesh="inputs/mesh/naca.cfdpp",
        names=NAMES if names is None else names,
        dimension=dimension,
    )


def refusal(**kwargs) -> str:
    with pytest.raises(ValueError) as caught:
        build(**kwargs)
    return str(caught.value)


# --- the keys come from the database --------------------------------------------------


def test_cfdpp_config_required_keys_are_read_from_the_database(tmp_path):
    keys = _config.required_keys(toy())
    assert "iterations" in keys and "physics.method" in keys and "cfl_ramp_steps" in keys
    assert "physics.preconditioning_velocity" not in keys  # required only with its method
    assert _config.required_keys(toy(), "post") == ("output_frequency", "restart_frequency")
    assert not hasattr(_config, "REQUIRED_KEYS")
    # The data decides: a key the database stops requiring is no longer refused.
    data = tmp_path / "toy"
    shutil.copytree(TOY, data)
    path = data / "setup.toml"
    text = path.read_text(encoding="utf-8")
    entry = 'name = "numerics.limiter"\ntype = "word"\nrequired = true\n'
    assert entry in text
    path.write_text(text.replace(entry, entry.replace("true", "false")), encoding="utf-8")
    edited = schema_for("1.0", data)
    assert "numerics.limiter" not in _config.required_keys(edited)
    case = _config.config(
        edited,
        setup(numerics__limiter=None),
        REFERENCE,
        POST_TABLE,
        CONDITION,
        mesh="inputs/mesh/naca.cfdpp",
        names=NAMES,
        dimension=2,
    )
    assert "numerics.limiter" not in case.values


def test_cfdpp_config_a_complete_setup_is_accepted():
    case = build()
    assert case.iterations == 200 and case.convergence_orders == 6.0
    assert case.turbulence == "SpalartAllmaras" and case.variables == ("nut",)
    assert case.values["physics.method"] == "preconditioned"
    assert case.values["cfl_ramp_steps"] == (1, 100)
    assert [b.family for b in case.boundaries] == ["airfoil", "farfield"]
    assert case.force_selectors() == (1,)
    assert case.warnings == ()


def test_cfdpp_config_one_missing_numeric_is_refused_with_the_setups_hint():
    text = refusal(table=setup(cfl=None))
    assert text == (
        "setup file: missing required keys: cfl; cfd-workspace setups --template subsonic "
        "writes a complete setup"
    )


def test_cfdpp_config_several_missing_keys_are_listed_in_one_message():
    text = refusal(table=setup(cfl=None, numerics__limiter=None, iterations=None))
    assert "missing required keys: iterations, cfl, [numerics] limiter;" in text
    assert "cfd-workspace setups --template subsonic" in text


def test_cfdpp_config_a_key_required_by_a_choice_is_refused_when_missing():
    text = refusal(table=setup(physics__preconditioning_velocity=None))
    assert "missing required keys: [physics] preconditioning_velocity;" in text


def test_cfdpp_config_a_key_that_does_not_apply_is_warned_and_skipped():
    case = build(table=setup(physics__method="density-based"))
    assert "physics.preconditioning_velocity" not in case.values
    assert case.warnings == (
        "setup file: [physics] preconditioning_velocity applies only with "
        "[physics] method = preconditioned; skipped",
    )


def test_cfdpp_config_turbulence_values_follow_the_model():
    assert "viscosity_ratio" in refusal(table=setup(viscosity_ratio=None))
    sst = setup(turbulence="SST", turbulence_intensity=0.01)
    case = build(table=sst)
    assert case.variables == ("k", "omega")
    assert "turbulence_intensity" in refusal(table=setup(turbulence="SST"))
    laminar = build(table=setup(turbulence="laminar"))
    assert laminar.variables == ()
    assert laminar.warnings == (
        "setup file: viscosity_ratio is not used by turbulence laminar; skipped",
    )


def test_cfdpp_config_turbulence_takes_the_neutral_names_of_the_database():
    text = refusal(table=setup(turbulence="kOmega"))
    assert text.startswith("setup file: turbulence is 'kOmega'; the accepted values are: ")
    assert "SpalartAllmaras" in text and "laminar" in text and "inviscid" in text


def test_cfdpp_config_a_model_without_evidence_is_refused():
    text = refusal(table=setup(turbulence="RealizableKE", turbulence_intensity=0.01))
    assert "turbulence RealizableKE has no harvested or verified evidence" in text
    assert "documented only" in text


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"iterations": 0}, "iterations must be above 0; found 0"),
        ({"iterations": 1.5}, "iterations takes an integer"),
        ({"cfl": True}, "cfl takes a finite number"),
        ({"cfl_ramp_steps": [1]}, "cfl_ramp_steps takes a list of 2 integers"),
        ({"cfl_ramp_steps": [-1, 5]}, "cfl_ramp_steps must be at least 0"),
        ({"numerics__limiter": "superbee"}, "[numerics] limiter is 'superbee'"),
    ],
)
def test_cfdpp_config_each_value_is_typed_and_bounded(change, fragment):
    assert fragment in refusal(table=setup(**change))


def test_cfdpp_config_the_subsonic_template_states_every_required_key_of_the_real_version():
    from cfdbender.core import setup_templates

    text = setup_templates.read("cfdpp", "subsonic")
    assert "baseline case the solver's GUI saved for CFD++ 22.5.U3" in text.splitlines()[1]
    table = tomllib.loads(text)
    real = schema_for("22.5.U3")
    for name in _config.required_keys(real):
        head, _, leaf = name.rpartition(".")
        assert leaf in (table[head] if head else table), name
    case = _config.config(
        real,
        table,
        REFERENCE,
        POST_TABLE,
        {"MACH": 0.15, "REmi": 6.0, "TK": 300.0},
        mesh="inputs/mesh/naca.cfdpp",
        names=NAMES,
        dimension=2,
    )
    assert case.values["iterations"] == table["iterations"]


# --- what the setup file may not hold -----------------------------------------------


def test_cfdpp_config_a_solver_table_is_refused_naming_init_and_executables():
    table = setup()
    table["solver"] = {"executable": "x"}
    assert refusal(table=table) == (
        "setup file: CFD++ takes no [solver] table; the workspace's solver and its machine "
        "options (timeout_s, ram_estimate_gib, stop_grace_s) are in init.toml, and each "
        "build's executables are in inputs/executables.toml"
    )


def test_cfdpp_config_convergence_residual_is_refused_naming_convergence_orders():
    assert refusal(table=setup(convergence_residual=1e-6)) == (
        "setup file: unknown key 'convergence_residual'; this adapter reads "
        "convergence_orders (orders of residual drop)"
    )


def test_cfdpp_config_keywords_unchecked_is_refused_naming_keywords():
    table = setup()
    table["keywords_unchecked"] = {"a": 1}
    assert "raw keywords go in the one [keywords] table" in refusal(table=table)


def test_cfdpp_config_an_unknown_key_names_the_closest_known_one():
    text = refusal(table=setup(iteration=10))
    assert "unknown key 'iteration' (closest known key: iterations)" in text
    text = refusal(table=setup(numerics__multigrid=3))
    assert text.startswith("setup file [numerics]: unknown key 'multigrid'")


# --- boundaries -----------------------------------------------------------------------


def test_cfdpp_config_a_family_without_an_entry_is_refused_with_the_three_lists():
    table = setup()
    del table["boundaries"]["farfield"]
    assert refusal(table=table) == (
        "setup file [boundaries]: families of naca.cfdpp without an entry: farfield; "
        "declared: airfoil; the grid has: airfoil, farfield"
    )


def test_cfdpp_config_an_entry_for_a_family_the_grid_lacks_is_refused():
    table = setup()
    table["boundaries"]["flap"] = {"type": "symmetry"}
    assert refusal(table=table) == (
        "setup file [boundaries]: entries for families naca.cfdpp does not hold: flap; "
        "declared: airfoil, farfield, flap; the grid has: airfoil, farfield"
    )


def test_cfdpp_config_each_boundary_type_and_option():
    names = {1: "wing", 2: "farfield", 3: "sym"}
    table = setup()
    table["boundaries"] = {
        "wing": {"type": "wall", "treatment": "wall-function", "thermal": "isothermal",
                 "temperature": 280.0},
        "farfield": {"type": "farfield"},
        "sym": {"type": "symmetry"},
    }  # fmt: skip
    case = build(table=table, names=names, dimension=3, condition={**CONDITION, "BETA": 2.0})
    wing, far, sym = case.boundaries
    assert wing.neutral == {"type": "wall", "treatment": "wall-function", "thermal": "isothermal"}
    assert wing.temperature == 280.0 and wing.forces
    assert far.neutral == {"type": "farfield"} and not far.forces
    assert sym.neutral == {"type": "symmetry"} and not sym.forces
    slip = setup(boundaries__airfoil={"type": "wall", "slip": True, "thermal": "adiabatic"})
    case = build(table=slip)
    assert case.boundaries[0].neutral == {"type": "wall", "slip": "true"}
    assert case.warnings == (
        "setup file: [boundaries.airfoil] thermal does not apply to a slip wall; skipped",
    )


@pytest.mark.parametrize(
    ("airfoil", "fragment"),
    [
        ({"type": "inlet"}, "type 'inlet' is unknown; the types are: wall, farfield, symmetry"),
        ({"type": "wall"}, "missing required keys: treatment, thermal"),
        (
            {"type": "wall", "treatment": "solve-to-wall", "thermal": "isothermal"},
            "missing required keys: temperature",
        ),
        ({"type": "wall", "slip": "yes"}, "slip must be true or false"),
        ({"type": "farfield", "forces": True}, "unknown key 'forces'"),
    ],
)
def test_cfdpp_config_a_wrong_boundary_is_refused(airfoil, fragment):
    text = refusal(table=setup(boundaries__airfoil=airfoil))
    assert text.startswith("setup file [boundaries.airfoil]")
    assert fragment in text


def test_cfdpp_config_no_far_field_and_no_force_wall_are_refused():
    table = setup(boundaries__farfield={"type": "symmetry"})
    assert 'no family has type = "farfield"' in refusal(table=table)
    table = setup(
        boundaries__airfoil={
            "type": "wall", "treatment": "solve-to-wall", "thermal": "adiabatic", "forces": False
        }
    )  # fmt: skip
    assert "no wall has forces = true" in refusal(table=table)


def test_cfdpp_config_an_inviscid_flow_needs_slip_walls():
    text = refusal(table=setup(turbulence="inviscid"))
    assert "an inviscid flow takes slip = true on every wall" in text


def test_cfdpp_config_groups_name_wall_families():
    table = setup()
    table["groups"] = {"upper": ["airfoil"]}
    assert build(table=table).groups == {"upper": ("airfoil",)}
    table["groups"] = {"upper": ["farfield"]}
    assert "farfield is not a wall family; the walls are: airfoil" in refusal(table=table)


# --- flight, fluid and freestream ----------------------------------------------------


ROWS = {
    "altitude": {"MACH": 0.2, "ALTFT": 0.0, "TK": 290.0, "ALPHA": 0.0},
    "mach_temperature_pressure": {"MACH": 0.2, "TK": 290.0, "PPA": 101325.0, "ALPHA": 0.0},
    "reynolds_per_metre": CONDITION,
    "true_airspeed": {"TASmps": 60.0, "ALTFT": 0.0, "ALPHA": 0.0},
}
CUSTOM = [
    ("gas_constant", 400.0, 287.05287),
    ("gamma", 1.3, 1.4),
    ("sutherland_viscosity", 2.0e-5, 1.716e-5),
    ("sutherland_temperature", 300.0, 273.15),
    ("sutherland_constant", 120.0, 110.4),
]


@pytest.mark.parametrize("kind", sorted(ROWS))
@pytest.mark.parametrize(("leaf", "given", "used"), CUSTOM)
def test_cfdpp_config_custom_fluid_constants_are_refused_for_every_row(kind, leaf, given, used):
    text = refusal(table=setup(**{f"fluid__{leaf}": given}), condition=ROWS[kind])
    assert text.startswith(f"setup file: [fluid] {leaf} is {given!r}, but cfdbender uses {used!r}")
    assert text.endswith("custom fluid properties are not supported yet for this solver")


def test_cfdpp_config_review_reproducer_gas_constant_400_is_refused():
    row = {"MACH": 0.2, "ALTFT": 0.0, "TK": 290.0}
    assert "gas_constant is 400.0" in refusal(table=setup(fluid__gas_constant=400.0), condition=row)


@pytest.mark.parametrize("kind", sorted(ROWS))
def test_cfdpp_config_core_flight_fluid_constants_stay_accepted(kind):
    table = setup(fluid__gas_constant=287.05287 * (1 + 1e-12))
    assert build(table=table, condition=ROWS[kind]).values["fluid.gamma"] == 1.4


@pytest.mark.parametrize("alpha", [0.0, 10.0, -4.0])
def test_cfdpp_config_freestream_in_two_dimensions_lies_in_the_x_y_plane(alpha):
    case = build(condition={**CONDITION, "ALPHA": alpha})
    speed = case.flow.velocity
    a = math.radians(alpha)
    assert case.freestream.velocity == pytest.approx((speed * math.cos(a), speed * math.sin(a), 0))


@pytest.mark.parametrize(("alpha", "beta"), [(0.0, 0.0), (10.0, 0.0), (5.0, 3.0)])
def test_cfdpp_config_freestream_in_three_dimensions_follows_the_drag_direction(alpha, beta):
    names = {1: "airfoil", 2: "farfield"}
    case = build(condition={**CONDITION, "ALPHA": alpha, "BETA": beta}, names=names, dimension=3)
    drag = wind_directions(alpha, beta)[0]
    speed = case.flow.velocity
    assert case.freestream.velocity == pytest.approx(tuple(speed * d for d in drag))


def test_cfdpp_config_a_two_dimensional_grid_refuses_sideslip():
    text = refusal(condition={**CONDITION, "BETA": 1.0})
    assert text == "a two-dimensional grid flies at BETA 0; remove BETA from the row"


def test_cfdpp_config_freestream_state_and_turbulence_primitives():
    case = build()
    flow = resolve_flow(CONDITION, reference_length=1.0, label="x")
    stream = case.freestream
    assert (stream.pressure, stream.temperature, stream.density) == pytest.approx(
        (flow.pressure, flow.temperature, flow.density)
    )
    assert stream.dynamic_pressure == pytest.approx(0.5 * flow.density * flow.velocity**2)
    assert stream.turbulence["nut"] == pytest.approx(3.0 * flow.viscosity / flow.density)
    sst = build(table=setup(turbulence="SST", turbulence_intensity=0.01)).freestream
    k = 1.5 * (0.01 * flow.velocity) ** 2
    assert sst.turbulence["k"] == pytest.approx(k)
    assert sst.turbulence["omega"] == pytest.approx(k / (3.0 * flow.viscosity / flow.density))


def test_cfdpp_config_reference_and_post_files():
    assert refusal(reference={"area": 1.0}) == "REF file: missing length, moment_origin"
    assert "history = true is required" in refusal(post={"output_frequency": 1})
    text = refusal(post={"history": True, "output_frequency": 100})
    assert text.startswith("PPROC file: missing required keys: restart_frequency;")


def test_cfdpp_config_init_options_are_machine_options_only():
    assert _config.options({}) == {"timeout_s": 600.0, "ram_estimate_gib": 0.25,
                                   "stop_grace_s": 60.0}  # fmt: skip
    assert _config.options({"timeout_s": 5})["timeout_s"] == 5.0
    with pytest.raises(ValueError, match="unknown key 'line_ending'"):
        _config.options({"line_ending": "lf"})


def test_cfdpp_config_raw_keyword_values_are_flat():
    table = setup()
    table["keywords"] = {"zz": {"value": [[1, 2]]}}
    assert "a raw keyword takes a number, a word, or a flat list of them" in refusal(table=table)
    table["keywords"] = {"zz": True}
    assert "a raw keyword takes a number" in refusal(table=table)
    table["keywords"] = {"zz": {"value": 1, "place": "x"}}
    assert "an inline table holds value and an optional section" in refusal(table=table)


# --- through planning ----------------------------------------------------------------


def plan_refusal(root: Path) -> str:
    with pytest.raises(PlanError) as caught, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan_workspace(root, "study")
    return str(caught.value)


def test_cfdpp_plan_without_solver_build_is_refused_with_the_supported_list(tmp_path):
    root = workspace(tmp_path / "ws", table=matrix(build="-"), executables=False)
    assert plan_refusal(root).endswith(
        "SOLVER_BUILD is required for CFD++: state the solver version of this row "
        "(supported: 22.5.U3)"
    )


def test_cfdpp_plan_an_unsupported_version_is_refused(tmp_path):
    root = workspace(tmp_path / "ws", table=matrix(build="23.1"), executables=False)
    assert plan_refusal(root).endswith(
        "CFD++ 23.1: cfdbender writes mcfd.inp for 22.5.U3 only; no evidence exists for "
        "this version (see the version-onboarding page)"
    )


def test_cfdpp_plan_the_real_version_refuses_a_key_with_no_evidence_behind_it(tmp_path):
    # No harvested input ties a preconditioning velocity to a keyword of 22.5.U3.
    root = workspace(tmp_path / "ws", table=matrix(build="22.5.U3"), executables=False)
    assert plan_refusal(root).endswith(
        "inputs/setups/s001.toml [physics]: unknown key 'preconditioning_velocity'; "
        "known keys: method"
    )


def test_cfdpp_plan_a_file_named_as_geometry_is_refused_with_the_conversion_sentence(
    tmp_path, monkeypatch
):
    use_toy(monkeypatch)
    root = workspace(tmp_path / "ws", table=matrix(geometry="wing.cgns"), executables=False)
    (root / "inputs/mesh/wing.cgns").write_bytes(b"not a grid folder")
    assert plan_refusal(root).endswith(
        "MESH wing.cgns: CFD++ reads a native grid folder; convert the mesh once with the "
        "solver's own converter and name the folder"
    )


def test_cfdpp_plan_a_workflow_other_than_steady_is_refused(tmp_path, monkeypatch):
    use_toy(monkeypatch)
    root = workspace(tmp_path / "ws", table=matrix(workflow="unsteady"), executables=False)
    assert "WORKFLOW must be steady: CFD++ runs steady cases only" in plan_refusal(root)


def test_cfdpp_plan_validates_every_sweep_value(tmp_path, monkeypatch):
    use_toy(monkeypatch)
    root = workspace(
        tmp_path / "ws",
        table=matrix("MACH:0.15, REmi:6, TK:300, ALPHA:0, BETA:sweep", "0,2"),
        executables=False,
    )
    assert "a two-dimensional grid flies at BETA 0" in plan_refusal(root)


def test_cfdpp_plan_names_the_setup_file_in_a_refusal(tmp_path, monkeypatch):
    use_toy(monkeypatch)
    text = SETUP.replace("cfl = 1.0\n", "")
    root = workspace(tmp_path / "ws", setup=text, executables=False)
    assert "inputs/setups/s001.toml: missing required keys: cfl;" in plan_refusal(root)


def test_cfdpp_plan_a_harvested_combination_raises_no_warning(tmp_path, monkeypatch):
    use_toy(monkeypatch)
    text = SETUP.replace('method = "preconditioned"\npreconditioning_velocity = 50.0',
                         'method = "density-based"')  # fmt: skip
    root = workspace(tmp_path / "ws", setup=with_keywords("cflbot = 1e-4\n", text))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = plan_workspace(root, "study")
    assert len(plan.points) == 2


def test_missing_wall_numeric_names_the_setup_template_command():
    text = refusal(
        table=setup(
            boundaries__airfoil={
                "type": "wall",
                "treatment": "solve-to-wall",
                "thermal": "isothermal",
            }
        )
    )
    assert "temperature" in text
    assert "cfd-workspace setups" in text
