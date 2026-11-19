"""The RESTART cell may hold the keyword MRF besides a source point or a dash."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from cfdbender.core.matrix import MatrixError, read_matrix
from cfdbender.core.plan import Point
from cfdbender.core.registry import RegistryError, _extras

HEADER = (
    "POL | RUN | DESCRIPTION | FLIGHT_CONDITION | SWEEP_VALUES | MESH | REF | SETUP | PPROC "
    "| WORKFLOW | RESTART"
)


def read(tmp_path: Path, cell: str):
    path = tmp_path / "m.fs"
    head = " 7001 | 1 | rotor | MACH:0.05, ALPHA:0 | - | rotor.su2 | 001 | 001 | 001 | steady"
    row = f"{head} | {cell} "
    path.write_text("\n".join([HEADER, "-" * 40, row]) + "\n", encoding="ascii")
    (found,) = read_matrix(path)
    return found


@pytest.mark.parametrize("cell", ["MRF", "mrf", "Mrf"])
def test_the_keyword_is_read_in_any_case_and_names_no_source(tmp_path, cell):
    found = read(tmp_path, cell)
    assert found.precursor is True and found.restart_from is None


@pytest.mark.parametrize("cell", ["-", "NA"])
def test_an_unstated_cell_is_a_cold_start(tmp_path, cell):
    found = read(tmp_path, cell)
    assert found.precursor is False and found.restart_from is None


def test_a_source_point_is_still_a_source_point(tmp_path):
    found = read(tmp_path, "7000/DP-M050AL+000")
    assert found.precursor is False and found.restart_from == "7000/DP-M050AL+000"


def test_a_word_that_is_neither_is_refused_naming_all_three_forms(tmp_path):
    with pytest.raises(MatrixError, match="POL/DP-name, MRF or -"):
        read(tmp_path, "FROZEN")


def point(**kwargs) -> Point:
    return Point(
        point_id="7001/DP-x",
        pol="7001",
        setup_id="s001",
        reference_id="r001",
        post_id="p001",
        condition=MappingProxyType({"MACH": 0.05}),
        walltime=None,
        directory="sims/sim_7001/datapoints/DP-x",
        actions=(),
        outputs=(),
        inputs_sha256=MappingProxyType({}),
        **kwargs,
    )


def test_a_point_without_the_keyword_has_the_flag_off():
    assert point().precursor is False
    assert point(precursor=True).precursor is True


def test_the_flag_is_read_back_from_a_record_and_a_wrong_type_is_refused():
    assert _extras({"precursor": True}) == {"precursor": True}
    with pytest.raises(RegistryError, match="precursor has an invalid value"):
        _extras({"precursor": "yes"})
