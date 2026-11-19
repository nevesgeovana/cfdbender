"""The time-domain functions agree with the numbers pyflightstream gives on the same inputs.

``tests/data/pyfs_unsteady_reference.json`` holds inputs and outputs recorded by
``tools/record_pyfs_unsteady_reference.py`` from pyflightstream on seeded histories and rows
(a startup transient, a periodic load, a slow drift, one to three rotors of different clocks,
partial revolutions, windows longer than the run, fractional windows at the halves, refused
rows). This test recomputes every entry with the cfdbender functions and compares without
importing pyflightstream: windows, step counts, refusals and warning texts with ``==``, means
and drifts to ``RTOL`` (1e-12, relative to the value, with a floor of ``RTOL`` times the
largest magnitude of the recorded result): they are numpy reductions, which are not bit
reproducible across numpy builds and CPUs. Clock steps and azimuths are plain IEEE
arithmetic and stay bit for bit. Regenerating the file is a deliberate act: the tool says how.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from cfdbender.core.settings import WorkspaceError
from cfdbender.core.unsteady_clock import (
    blade_azimuth_deg,
    blade_one_azimuth,
    revolution_steps,
    rotor_time_stepping,
    unsteady_time_stepping,
)
from cfdbender.core.unsteady_window import averaging_span, averaging_steps, stated_key
from cfdbender.post.revolutions import (
    FORCE_PLOT_PARAMETERS,
    is_force_or_moment_column,
    last_complete_revolution,
    per_revolution_table,
    revolution_drift_pct,
    revolution_drift_warnings,
    whole_revolution_steps,
)
from cfdbender.post.time_average import TimeHistory, step_clock, window_average

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "pyfs_unsteady_reference.json"


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


#: The relative tolerance of a float that is a numpy reduction (a mean or a drift).
RTOL = 1e-12


def _magnitude(value: object) -> float:
    """The largest finite magnitude of the floats inside a recorded value."""
    if isinstance(value, dict):
        return max((_magnitude(v) for v in value.values()), default=0.0)
    if isinstance(value, list | tuple):
        return max((_magnitude(v) for v in value), default=0.0)
    if isinstance(value, float) and math.isfinite(value):
        return abs(value)
    return 0.0


def _same(left: object, right: object, rtol: float = 0.0, floor: float = 0.0) -> bool:
    """Equal values, NaN equal to NaN, lists and tuples alike.

    A float is bit for bit (the sign of a zero included) when ``rtol`` is zero, and within
    ``rtol`` of the recorded value plus ``floor`` otherwise.
    """
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
        if rtol and math.isfinite(left) and math.isfinite(right):
            return abs(left - right) <= rtol * abs(right) + floor
        return left == right and math.copysign(1.0, left) == math.copysign(1.0, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return list(left) == list(right) and all(
            _same(left[k], right[k], rtol, floor) for k in left
        )
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(
            _same(a, b, rtol, floor) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right) and type(left) is type(right)


def _agrees(recorded: dict, call: Callable[[], object], *, rtol: float = 0.0) -> None:
    """The call returns the recorded value, or refuses with the recorded sentence."""
    if "error" in recorded:
        with pytest.raises(WorkspaceError) as refused:
            call()
        assert str(refused.value) == recorded["error"]
    else:
        got = call()
        floor = rtol * _magnitude(recorded["value"])
        assert _same(got, recorded["value"], rtol, floor), (got, recorded["value"])


def _history(recorded: dict) -> TimeHistory:
    return TimeHistory(
        steps=np.asarray(recorded["steps"], dtype=int),
        columns={k: np.asarray(v, dtype=float) for k, v in recorded["columns"].items()},
    )


def _tuple_or_none(value: tuple[int, int] | None) -> list[int] | None:
    return None if value is None else list(value)


def test_the_reference_is_not_empty_and_names_what_recorded_it(reference: dict) -> None:
    meta = reference["meta"]
    assert meta["pyflightstream_version"] and len(meta["pyflightstream_commit"]) == 40
    assert meta["pyflightstream_tree_dirty"] is False
    counts = {
        "windows": len(reference["windows"]),
        "averages": sum(len(e["windows"]) for e in reference["averages"]),
        "per_revolution": len(reference["per_revolution"]),
        "warnings": len(reference["warnings"]),
        "drift": len(reference["drift"]),
        "azimuth": len(reference["azimuth"]["azimuth"]),
        "clock": len(reference["clock"]["rotor"]),
    }
    assert all(count >= 20 for count in counts.values()), counts


def test_the_averaging_windows_agree(reference: dict) -> None:
    for entry in reference["windows"]:
        variables = entry["variables"]
        _agrees(
            entry["stated_key"],
            lambda v=variables: _tuple_or_none(stated_key(v)),
        )
        for case in entry["by_clock"]:
            _agrees(
                case["steps"],
                lambda v=variables, c=case["per_revolution"]: averaging_steps(v, per_revolution=c),
            )
            _agrees(
                case["span"],
                lambda v=variables, c=case["per_revolution"], n=case["last_step"]: _tuple_or_none(
                    averaging_span(v, last_step=n, per_revolution=c)
                ),
            )


def test_the_fractional_windows_round_halves_to_even(reference: dict) -> None:
    # the recording must hold the halves the rule is about
    halves = {
        (e["variables"].get("LAST_ITERS_AVG"), c["steps"].get("value"))
        for e in reference["windows"]
        for c in e["by_clock"]
        if c["per_revolution"] is None
    }
    assert (2.5, 2) in halves and (3.5, 4) in halves


def test_the_step_clock_agrees(reference: dict) -> None:
    for entry in reference["step_clock"]:
        _agrees(entry["steps"], lambda c=entry["cells"]: [int(s) for s in step_clock(c)])


def test_the_window_averages_agree(reference: dict) -> None:
    for entry in reference["averages"]:
        history = _history(entry["history"])
        for case in entry["windows"]:

            def run(h=history, w=tuple(case["window"])) -> dict:
                read: set[int] = set()
                result = window_average(h, window=w, read_steps=read)
                return {
                    "window": list(result.window),
                    "n_frames": result.n_frames,
                    "means": result.means,
                    "source_window": list(result.source_window),
                    "read_steps": sorted(read),
                }

            _agrees(case["result"], run, rtol=RTOL)


def test_the_per_revolution_tables_agree(reference: dict) -> None:
    for entry in reference["per_revolution"]:
        history = _history(entry["history"])

        def run(h=history, e=entry) -> dict:
            read: set[int] = set()
            windows, means, left = per_revolution_table(
                h, e["columns"], revolution_steps=e["revolution_steps"], read_steps=read
            )
            return {
                "windows": [list(w) for w in windows],
                "means": means,
                "left": left,
                "read_steps": sorted(read),
            }

        _agrees(entry["result"], run, rtol=RTOL)


def test_the_drifts_agree(reference: dict) -> None:
    for entry in reference["drift"]:
        _agrees(
            entry["result"],
            lambda e=entry: revolution_drift_pct(e["current"], e["previous"]),
            rtol=RTOL,
        )


def test_the_warnings_agree_word_for_word(reference: dict) -> None:
    warned = 0
    for entry in reference["warnings"]:
        _agrees(
            entry["clauses"],
            lambda e=entry: revolution_drift_warnings(
                e["previous"], e["current"], e["shown"], limit_pct=e["limit_pct"]
            ),
        )
        warned += len(entry["clauses"].get("value", []))
    assert warned > 50, "the recording must hold warnings, not only silences"


def test_the_load_columns_agree(reference: dict) -> None:
    for entry in reference["load_columns"]:
        assert is_force_or_moment_column(entry["name"]) is entry["result"], entry["name"]
    assert "FX" in FORCE_PLOT_PARAMETERS


def test_the_revolution_cuts_agree(reference: dict) -> None:
    cuts = reference["revolution_cuts"]
    for entry in cuts["whole"]:
        _agrees(entry["result"], lambda c=entry["clock"]: whole_revolution_steps(c))
    for entry in cuts["last_complete"]:
        _agrees(
            entry["result"],
            lambda e=entry: _tuple_or_none(
                last_complete_revolution(e["first"], e["last"], e["clock"])
            ),
        )


def test_the_rotor_clocks_agree(reference: dict) -> None:
    for entry in reference["clock"]["rotor"]:
        _agrees(
            entry["result"],
            lambda e=entry: rotor_time_stepping("7001", e["variables"], rpm=e["rpm"]).record(),
        )


def test_the_rotorless_clocks_agree(reference: dict) -> None:
    for entry in reference["clock"]["rotorless"]:
        _agrees(
            entry["result"],
            lambda e=entry: unsteady_time_stepping("7001", e["variables"]).record(),
        )


def test_the_azimuths_agree_bit_for_bit(reference: dict) -> None:
    azimuth = reference["azimuth"]
    for entry in azimuth["azimuth"]:
        got = blade_azimuth_deg(
            entry["datum_deg"],
            step=entry["step"],
            steps_per_revolution=entry["steps_per_revolution"],
            rpm=entry["rpm"],
        )
        assert _same(got, entry["result"]["value"]), entry
    for entry in azimuth["blade_one"]:
        _agrees(
            entry["result"],
            lambda e=entry: blade_one_azimuth(
                e["step"],
                datum_deg=e["datum_deg"],
                sense=e["sense"],
                steps_per_revolution=e["steps_per_revolution"],
            ),
        )
    for entry in azimuth["revolution_steps"]:
        _agrees(entry["result"], lambda e=entry: revolution_steps(e["number"], e["per_revolution"]))
