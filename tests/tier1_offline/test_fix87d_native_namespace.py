"""C-09 and the native passthrough namespace (Q1).

An unknown key of the s file is refused, except in a namespace an adapter declares as the
solver's own: SU2 forwards its upper-case native option names. Both sides are tested on
every adapter: the declaring one accepts the upper-case key, every other one refuses it.
"""

from __future__ import annotations

import pytest
from test_conformance_workspace import SOLVERS, case  # noqa: F401

from cfdbender.core.adapter import load_point
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

DECLARING = {"su2"}


def _with_key(root, line):
    setup = root / "inputs/setups/s001.toml"
    setup.write_bytes(line.encode("ascii") + b"\n" + setup.read_bytes())


def test_the_declaration_is_exactly_the_documented_set():
    declaring = {name for name in SOLVERS if getattr(load_solver(name), "NATIVE_PASSTHROUGH", None)}
    assert declaring == DECLARING


def test_an_upper_case_native_key_passes_only_where_declared(case):  # noqa: F811
    solver, root = case
    _with_key(root, "ZZ_NATIVE_OPTION = 1")
    if solver in DECLARING:
        plan = plan_workspace(root, "full")
        text = load_solver(solver).translate(load_point(root, plan.points[0], solver=solver))
        assert "ZZ_NATIVE_OPTION= 1" in text
    else:
        with pytest.raises(ValueError, match="ZZ_NATIVE_OPTION"):
            plan_workspace(root, "full")


def test_qprop_staged_config_is_the_file_the_solver_reads(tmp_path):
    """C-45 (Q5): the job's config is propeller.prop, which exists at staging."""
    import qprop_workspace
    from test_hpc_self_contained import submit

    qprop_workspace.workspace(
        tmp_path,
        setup=qprop_workspace.setup_text(blade="inputs/polars-2D/set1", n_sections=None),
        xfoil=False,
    )
    qprop_workspace.table_set(
        tmp_path,
        rows=[
            "0.06,0.030,50,a1.dat,p1.txt",
            "0.12,0.026,38,a1.dat,p2.txt",
            "0.19,0.021,22,a1.dat,p2.txt",
        ],
    )
    folder, config = submit(tmp_path)
    assert load_solver("qprop").STAGED_CONFIG == "propeller.prop"
    assert config == folder / "propeller.prop" and config.is_file()


def test_a_lower_case_unknown_key_is_refused_by_every_adapter(case):  # noqa: F811
    solver, root = case
    _with_key(root, "zz_native_option = 1")
    with pytest.raises(ValueError, match="zz_native_option"):
        plan_workspace(root, "full")
