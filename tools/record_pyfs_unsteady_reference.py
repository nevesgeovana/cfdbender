"""Record the pyflightstream time-domain functions on seeded inputs, for the cross-check test.

A development tool, run by hand once in an environment where pyflightstream is importable::

    PYTHONPATH="src;<pyflightstream>/src" python tools/record_pyfs_unsteady_reference.py

It calls the pyflightstream functions that cfdbender's ``core.unsteady_window``,
``core.unsteady_clock`` and ``core.revolutions`` carry the semantics of, on deterministic inputs
(seeded numpy: a history with a startup transient, a periodic load and a slow drift; one to
three rotors of different steps per revolution; a partial last revolution; windows longer than
the run; fractional windows at the halves), and writes the inputs and outputs, with the
pyflightstream version and commit, to ``tests/data/pyfs_unsteady_reference.json``. The test
``test_unsteady_cross_check.py`` recomputes every entry with the cfdbender functions and compares
them without pyflightstream. Regenerating the file is a deliberate act; this tool is the only
code outside ``cfdbender._pyfs`` that imports pyflightstream.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

SEED = 20260918
OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "pyfs_unsteady_reference.json"


def _call(function, *args, **kwargs):  # type: ignore[no-untyped-def]
    """Return ``{"value": ...}`` or ``{"error": message}`` for one call."""
    try:
        return {"value": function(*args, **kwargs)}
    except Exception as error:  # the message is the datum
        return {"error": str(error), "type": type(error).__name__}


def _floats(values):  # type: ignore[no-untyped-def]
    return [float(v) for v in values]


# ------------------------------------------------------------------ window


def _record_windows(pf) -> list[dict]:  # type: ignore[no-untyped-def]
    windows = pf["windows"]
    rows: list[dict] = []
    variables_list: list[dict] = [
        {},
        {"LAST_REVS_AVG": "-"},
        {"LAST_ITERS_AVG": "NA"},
        {"LAST_REVS_AVG": ""},
        {"LAST_REVS_AVG": "2"},
        {"LAST_REVS_AVG": "0.5"},
        {"LAST_REVS_AVG": "1.5"},
        {"LAST_REVS_AVG": "2.5"},
        {"LAST_REVS_AVG": 3},
        {"LAST_REVS_AVG": "99"},
        {"LAST_REVS_AVG": "0"},
        {"LAST_REVS_AVG": "-1"},
        {"LAST_REVS_AVG": "nan"},
        {"LAST_REVS_AVG": "inf"},
        {"LAST_REVS_AVG": "abc"},
        {"LAST_REVS_AVG": True},
        {"LAST_REVS_AVG": "0.001"},
        {"LAST_ITERS_AVG": "100"},
        {"LAST_ITERS_AVG": 37},
        {"LAST_ITERS_AVG": 2.5},
        {"LAST_ITERS_AVG": 3.5},
        {"LAST_ITERS_AVG": 2.4},
        {"LAST_ITERS_AVG": 2.6},
        {"LAST_ITERS_AVG": "0.5"},
        {"LAST_ITERS_AVG": "0"},
        {"LAST_ITERS_AVG": "nope"},
        {"LAST_REVS_AVG": "1", "LAST_ITERS_AVG": "10"},
        {"LAST_REVS_AVG": "1", "LAST_ITERS_AVG": "-"},
        {"last_revs_avg": "1"},
    ]
    rng = np.random.default_rng(SEED)
    for _ in range(16):
        key = "LAST_REVS_AVG" if rng.random() < 0.5 else "LAST_ITERS_AVG"
        variables_list.append({key: float(np.round(rng.uniform(0.2, 9.0), 3))})
    clocks = [None, 0.0, math.nan, 5.0, 7.0, 36.0, 250.0, 0.5]
    for variables in variables_list:
        entry = {
            "variables": variables,
            "stated_key": _call(
                lambda v=variables: list(k) if (k := windows.stated_key(v)) is not None else None
            ),
            "by_clock": [],
        }
        for per_revolution in clocks:
            for last_step in (0, 10, 1000):
                spans = _call(
                    lambda v=variables, c=per_revolution, n=last_step: (
                        list(s)
                        if (s := windows.averaging_span(v, last_step=n, per_revolution=c))
                        is not None
                        else None
                    )
                )
                entry["by_clock"].append(
                    {
                        "per_revolution": per_revolution,
                        "last_step": last_step,
                        "steps": _call(
                            windows.averaging_steps, variables, per_revolution=per_revolution
                        ),
                        "span": spans,
                    }
                )
        rows.append(entry)
    return rows


def _record_step_clock(pf) -> list[dict]:  # type: ignore[no-untyped-def]
    read = pf["step_clock"]
    cases: list[list] = [
        ["3", "4", "5"],
        [10, 20, 30],
        ["1", "2", "4"],
        ["3", "3", "5"],
        ["5", "4", "6"],
        ["1.5", "2.5"],
        ["a", "2"],
        [1.0, math.nan],
        [1.0, math.inf],
        [None, 1],
        [],
        [7],
        ["1.0000000001", "2"],
        [101, 102, 103, 104],
    ]
    return [
        {"cells": cells, "steps": _call(lambda c=cells: [int(s) for s in read(c)])}
        for cells in cases
    ]


# ------------------------------------------------------------------ history


def _history(rng, rows: int, first: int, per_revolution: int, columns: int) -> dict:  # type: ignore[no-untyped-def]
    """Build a load history: startup transient, a periodic load of known frequency, a slow drift."""
    steps = np.arange(first, first + rows)
    k = np.arange(rows)
    out: dict[str, list[float]] = {}
    for index in range(columns):
        amplitude = float(rng.uniform(0.5, 40.0))
        base = float(rng.uniform(-400.0, 400.0))
        phase = float(rng.uniform(0, 2 * math.pi))
        harmonic = amplitude * np.sin(2 * math.pi * k / per_revolution + phase)
        transient = float(rng.uniform(5, 50)) * np.exp(-k / max(per_revolution, 1))
        drift = float(rng.uniform(-2e-3, 2e-3)) * k
        noise = rng.normal(0.0, 0.01 * amplitude, rows)
        out[f"col{index}"] = _floats(base + harmonic + transient + drift + noise)
    return {"steps": [int(s) for s in steps], "columns": out}


def _series(pf, history: dict):  # type: ignore[no-untyped-def]
    return pf["TimestepSeries"](
        steps=np.asarray(history["steps"], dtype=int),
        times_s=None,
        points=np.zeros((1, 3)),
        fields={
            name: np.asarray(values, dtype=float)[:, None]
            for name, values in history["columns"].items()
        },
        sources=(),
    )


def _record_averages(pf) -> list[dict]:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(SEED + 1)
    out: list[dict] = []
    for rows, first, per_revolution in (
        (12, 1, 4),
        (50, 1, 10),
        (100, 1, 24),
        (120, 1, 36),
        (97, 101, 24),
        (200, 1, 50),
        (33, 5, 7),
    ):
        history = _history(rng, rows, first, per_revolution, 4)
        last = first + rows - 1
        windows = [
            (first, last),
            (last - 9, last),
            (last, last),
            (first + 2, first + 4),
            (last - rows // 3, last),
            (first - 5, first + 3),
            (last - 3, last + 40),
            (last + 1, last + 9),
            (last, first),
            (first + 1, first + 1),
        ]
        out.append({"history": history, "windows": _windows_of(pf, history, windows)})
    # a sparse clock and a NaN column
    sparse = {
        "steps": [2, 4, 6, 8, 10, 12],
        "columns": {"a": [1.0, 3.0, 5.0, math.nan, 9.0, 11.0], "b": _floats(rng.normal(size=6))},
    }
    out.append(
        {
            "history": sparse,
            "windows": _windows_of(pf, sparse, [(3, 7), (2, 12), (4, 6), (7, 8), (13, 14)]),
        }
    )
    return out


def _windows_of(pf, history: dict, windows: list) -> list[dict]:  # type: ignore[no-untyped-def]
    """Average one history over each window and note the steps it read."""
    series = _series(pf, history)
    entries = []
    for window in windows:
        read: set[int] = set()

        def run(w=window, r=read):  # type: ignore[no-untyped-def]
            result = pf["blade_passage_average"](series, window=w, read_steps=r)
            return {
                "window": list(result.window),
                "n_frames": result.n_frames,
                "means": {k: float(v[0]) for k, v in result.fields.items()},
                "source_window": list(result.source_window),
                "read_steps": sorted(r),
            }

        entries.append({"window": list(window), "result": _call(run)})
    return entries


def _record_per_revolution(pf) -> list[dict]:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(SEED + 2)
    table = pf["per_revolution_table"]
    out: list[dict] = []
    # one to three rotors of different clocks over one run, a partial last revolution each
    for rotors in ((24,), (24, 36), (10, 36, 50)):
        for spr in rotors:
            for rows, first in (
                (spr * 3 + spr // 3, 1),
                (spr * 5, 1),
                (spr * 4 + 1, 201),
                (spr - 1, 1),
            ):
                history = _history(rng, rows, first, spr, 5)
                series = _series(pf, history)
                columns = list(history["columns"])

                def run(s=series, c=columns, n=spr):  # type: ignore[no-untyped-def]
                    read: set[int] = set()
                    windows, means, left = table(s, c, revolution_steps=n, read_steps=read)
                    return {
                        "windows": [list(w) for w in windows],
                        "means": means,
                        "left": left,
                        "read_steps": sorted(read),
                    }

                out.append(
                    {
                        "history": history,
                        "columns": columns,
                        "revolution_steps": spr,
                        "result": _call(run),
                    }
                )
    history = _history(rng, 30, 1, 5, 2)
    for bad in (0, -3):
        out.append(
            {
                "history": history,
                "columns": ["col0"],
                "revolution_steps": bad,
                "result": _call(
                    lambda b=bad: table(_series(pf, history), ["col0"], revolution_steps=b)
                ),
            }
        )
    return out


def _record_drift(pf) -> list[dict]:  # type: ignore[no-untyped-def]
    drift = pf["revolution_drift_pct"]
    rng = np.random.default_rng(SEED + 3)
    pairs: list[tuple[float, float]] = [
        (101.0, 100.0),
        (-99.0, -100.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (math.nan, 1.0),
        (1.0, math.nan),
        (math.inf, 1.0),
        (1.0, -math.inf),
        (1.7e308, -1.7e308),
        (-1.7e308, 1.7e308),
        (1e-320, 1e-310),
        (10.5, 10.0),
    ]
    for _ in range(60):
        previous = float(rng.normal(0, 50) * 10.0 ** rng.integers(-3, 4))
        pairs.append((previous * (1.0 + float(rng.normal(0, 0.02))), previous))
    return [{"current": c, "previous": p, "result": _call(drift, c, p)} for c, p in pairs]


def _record_warnings(pf) -> list[dict]:  # type: ignore[no-untyped-def]
    warn = pf["revolution_drift_warnings"]
    rng = np.random.default_rng(SEED + 4)
    names = [
        "FX_PROP_X",
        "FY_PROP_X",
        "FZ_PROP_X",
        "MX_PROP_X",
        "MY_PROP_X",
        "MZ_PROP_X",
        "FX_PROP_Blade1",
        "FY_PROP_Blade1",
        "CL_MRP_TOTAL",
        "CDI_MRP_TOTAL",
        "CDO_WING",
        "CD_WING",
        "CDP_WING",
        "CDV_WING",
        "MACH_P1",
        "VX_probe1",
        "FXX_TOTAL",
        "FX",
        "MZ_",
    ]
    cases: list[dict] = []
    for _ in range(24):
        previous = {n: float(rng.normal(0, 100) * 10.0 ** rng.integers(-3, 2)) for n in names}
        current = {
            n: v + float(rng.normal(0, 0.02)) * max(abs(v), 1e-3) * float(rng.choice([0.1, 1, 30]))
            for n, v in previous.items()
        }
        if rng.random() < 0.3:
            previous[names[int(rng.integers(len(names)))]] = math.nan
        if rng.random() < 0.3:
            current[names[int(rng.integers(len(names)))]] = math.nan
        if rng.random() < 0.2:
            previous[names[int(rng.integers(len(names)))]] = 0.0
        shown = {n: f"{n} [{'N' if n.startswith(('F', 'M')) else '-'}]" for n in names}
        for limit in (0.5, 1.0):
            cases.append(
                {
                    "previous": previous,
                    "current": current,
                    "shown": shown,
                    "limit_pct": limit,
                    "clauses": _call(warn, previous, current, shown, limit_pct=limit),
                }
            )
    # a change of exactly the limit does not warn; one step past it does
    for after in (101.0, 99.0, 101.0000001):
        edge_previous, edge_current = {"FX_G": 100.0, "FY_G": 50.0}, {"FX_G": after, "FY_G": 50.0}
        edge_shown = {"FX_G": "FX_G", "FY_G": "FY_G"}
        cases.append(
            {
                "previous": edge_previous,
                "current": edge_current,
                "shown": edge_shown,
                "limit_pct": 1.0,
                "clauses": _call(warn, edge_previous, edge_current, edge_shown, limit_pct=1.0),
            }
        )
    # the worked rotor of the product's own tests, three revolution means
    rotor = {
        "FX_PROP_X": [-350.0, -350.1, -350.2],
        "FY_PROP_X": [0.08, 0.025, 10.0],
        "FZ_PROP_X": [-0.05, 0.02, 0.006],
        "MX_PROP_X": [30.0, 30.01, 30.02],
        "MY_PROP_X": [0.01, 0.003, 0.0009],
        "MZ_PROP_X": [-0.004, 0.002, 1.0],
        "FX_PROP_Blade1": [-10.0, -10.0, -10.0],
        "FY_PROP_Blade1": [0.5, 0.5, 0.7],
        "MACH_P1": [0.1, 0.1, 0.2],
    }
    previous = {k: v[1] for k, v in rotor.items()}
    current = {k: v[2] for k, v in rotor.items()}
    shown = {k: k for k in rotor}
    for limit in (1.0, 3.5):
        cases.append(
            {
                "previous": previous,
                "current": current,
                "shown": shown,
                "limit_pct": limit,
                "clauses": _call(warn, previous, current, shown, limit_pct=limit),
            }
        )
    return cases


def _record_columns(pf) -> list[dict]:  # type: ignore[no-untyped-def]
    names = [
        "FX_MRP_TOTAL",
        "CDI_WING",
        "VX_probe1",
        "FXX_TOTAL",
        "MACH_P1",
        "FX",
        "",
        "_X",
        "MZ_A_B",
    ]
    return [{"name": n, "result": pf["is_force_or_moment_column"](n)} for n in names]


def _record_revolution_cuts(pf) -> dict:  # type: ignore[no-untyped-def]
    whole = pf["whole_revolution_steps"]
    last_complete = pf["last_complete_revolution"]
    clocks = [36.0, 36.0 + 1e-14, 36.5, 0.3, 1.0, 2.5, 100.0, 24.9999999, math.nan, math.inf, -4.0]
    spans = [(3, 13), (1, 8), (1, 4), (3, 5), (5, 1), (1, 1000), (101, 460)]
    return {
        "whole": [{"clock": c, "result": _call(whole, c)} for c in clocks],
        "last_complete": [
            {
                "first": a,
                "last": b,
                "clock": c,
                "result": _call(lambda a=a, b=b, c=c: _tup(last_complete(a, b, c))),
            }
            for c in (4.0, 24.0, 36.0, 36.4, 0.0)
            for a, b in spans
        ],
    }


def _tup(value):  # type: ignore[no-untyped-def]
    return None if value is None else list(value)


# ------------------------------------------------------------------ the clock


def _record_clock(pf) -> list[dict]:  # type: ignore[no-untyped-def]
    case_type, sweep = pf["SimCase"], pf["SweepAxis"]
    rotor, rotorless = pf["rotor_time_stepping"], pf["unsteady_time_stepping"]

    def make(variables: dict) -> object:
        return case_type(
            sim_id="7001",
            aircraft="RotorRig",
            recipe="unsteady_rotor",
            sweep=sweep(type="alpha", values=[0.0]),
            variables={k: str(v) for k, v in variables.items()},
        )

    def rotor_entry(variables: dict) -> dict:
        case = make(variables)
        result = _call(lambda: rotor(case).record())
        signed = None
        if "RPM" in variables:
            signed = float(variables["RPM"]) * (-1 if str(variables.get("RPM_SIGN")) == "-1" else 1)
        return {"variables": variables, "rpm": signed, "result": result}

    rows: list[dict] = []
    angular = [
        {"DELTA_THETA": "10.0", "REVOLUTIONS": "1.5", "RPM": "473.1778"},
        {"DELTA_THETA": "10.0", "REVOLUTIONS": "1.5", "RPM": "473.1778", "RPM_SIGN": "-1"},
        {"DELTA_THETA": "15", "REVOLUTIONS": "2", "RPM": "1200"},
        {"DELTA_THETA": "1.5", "REVOLUTIONS": "2.5", "RPM": "972"},
        {"DELTA_THETA": "20", "REVOLUTIONS": "2", "RPM": "1200"},
        {"DELTA_THETA": "7.0", "REVOLUTIONS": "1.5", "RPM": "473.1778"},
        {"DELTA_THETA": "20", "REVOLUTIONS": "0.01", "RPM": "1200"},
        # off a whole number of steps by 3.6e-9 (accepted) and by 3.6e-6 (refused)
        {"DELTA_THETA": "10", "REVOLUTIONS": "1.0000000001", "RPM": "1200"},
        {"DELTA_THETA": "10", "REVOLUTIONS": "1.0000001", "RPM": "1200"},
        {"DELTA_THETA": "0", "REVOLUTIONS": "1", "RPM": "1200"},
        {"DELTA_THETA": "-10", "REVOLUTIONS": "1", "RPM": "1200"},
        {"DELTA_THETA": "361", "REVOLUTIONS": "1", "RPM": "1200"},
        {"DELTA_THETA": "360", "REVOLUTIONS": "3", "RPM": "1200"},
        {"DELTA_THETA": "10", "REVOLUTIONS": "0", "RPM": "1200"},
        {"DELTA_THETA": "10", "REVOLUTIONS": "-1", "RPM": "1200"},
        {"DELTA_THETA": "10", "REVOLUTIONS": "1", "RPM": "0"},
        {"DELTA_THETA": "nan", "REVOLUTIONS": "1", "RPM": "1200"},
        {"DELTA_THETA": "10", "REVOLUTIONS": "inf", "RPM": "1200"},
        {"DELTA_THETA": "ten", "REVOLUTIONS": "1", "RPM": "1200"},
        {"DELTA_THETA": "10"},
        {"REVOLUTIONS": "1.5"},
        {
            "DELTA_THETA": "10",
            "REVOLUTIONS": "1.5",
            "DELTA_TIME": "0.01",
            "TIME_ITERATIONS": "5",
            "RPM": "1200",
        },
        {"DELTA_THETA": "1e-300", "REVOLUTIONS": "1", "RPM": "1e300"},
        {"DELTA_THETA": "1e-6", "REVOLUTIONS": "1e12", "RPM": "1200"},
        {"DELTA_THETA": "0.1", "REVOLUTIONS": "100000000000000000", "RPM": "1200"},
    ]
    explicit = [
        {"DELTA_TIME": "1e-4", "TIME_ITERATIONS": "720", "RPM": "1200"},
        {"DELTA_TIME": "1e-4", "TIME_ITERATIONS": "720", "RPM": "1200", "RPM_SIGN": "-1"},
        {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "10"},
        {"DELTA_TIME": "0.005", "TIME_ITERATIONS": "9007199254740993", "RPM": "1200"},
        {"DELTA_TIME": "0.00352", "TIME_ITERATIONS": "54", "RPM": "473.1723"},
        {"DELTA_TIME": "0.001"},
        {"TIME_ITERATIONS": "720"},
        {"DELTA_TIME": "0", "TIME_ITERATIONS": "5", "RPM": "1200"},
        {"DELTA_TIME": "-1", "TIME_ITERATIONS": "5", "RPM": "1200"},
        {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "0", "RPM": "1200"},
        {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "-3", "RPM": "1200"},
        {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "2.5", "RPM": "1200"},
        {"DELTA_TIME": "nan", "TIME_ITERATIONS": "5", "RPM": "1200"},
        {"DELTA_TIME": "abc", "TIME_ITERATIONS": "5"},
        {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "5", "RPM": "0"},
        {"DELTA_TIME": "1e-320", "TIME_ITERATIONS": "5", "RPM": "1e10"},
        {},
        {"LAST_REVS_AVG": "1"},
    ]
    for variables in (*angular, *explicit):
        rows.append(rotor_entry(variables))

    rotorless_cases = [
        {"DELTA_TIME": "0.002", "TIME_ITERATIONS": "40"},
        {"DELTA_TIME": "0.002"},
        {"TIME_ITERATIONS": "40"},
        {"DELTA_TIME": "0", "TIME_ITERATIONS": "40"},
        {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "0"},
        {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "1.5"},
        {},
        {"DELTA_THETA": "20", "REVOLUTIONS": "2"},
        {"DELTA_THETA": "20"},
    ]
    rotorless_rows: list[dict] = []
    for variables in rotorless_cases:
        case = make(variables)
        rotorless_rows.append(
            {"variables": variables, "result": _call(lambda c=case: rotorless(c).record())}
        )
    return [{"rotor": rows, "rotorless": rotorless_rows}]


# ------------------------------------------------------------------ azimuth and steps


def _record_azimuth(pf) -> dict:  # type: ignore[no-untyped-def]
    azimuth, one = pf["blade_azimuth_deg"], pf["blade_one_azimuth"]
    rng = np.random.default_rng(SEED + 5)
    cases: list[dict] = []
    stated = [
        (10.0, 18, 72, 2000.0),
        (10.0, 18, 72, -2000.0),
        (45.0, 0, 36, 1.0),
        (350.0, 3, 36, 1.0),
        (10.0, 36, 36, 1.0),
        (0.0, 36, 36, -1.0),
        (10.0, 5, 72, 5000.0),
        (10.0, 5, 72, 1.0),
        (10.0, 18, 72, None),
        (None, 18, 72, 1.0),
        (math.nan, 18, 72, 1.0),
        ("3", 18, 72, 1.0),
        (10.0, True, 72, 1.0),
        (10.0, math.inf, 72, 1.0),
        (10.0, 18, 0, 1.0),
        (10.0, 18, -72, 1.0),
        (10.0, 18, 72, 0.0),
        (10.0, 18, 72, math.nan),
        (-30.0, 7, 24, -1.0),
        (725.0, 7, 24, 1.0),
    ]
    for _ in range(60):
        stated.append(
            (
                float(np.round(rng.uniform(-400, 400), 3)),
                int(rng.integers(0, 600)),
                float(rng.choice([8, 24, 36, 50, 72, 36.5])),
                float(rng.choice([-1200.0, 1200.0, 973.5, -0.001])),
            )
        )
    for datum, step, per_revolution, rpm in stated:
        cases.append(
            {
                "datum_deg": datum,
                "step": step,
                "steps_per_revolution": per_revolution,
                "rpm": rpm,
                "result": _call(
                    azimuth, datum, step=step, steps_per_revolution=per_revolution, rpm=rpm
                ),
            }
        )
    ones: list[dict] = []
    for step, datum, sense, per_revolution in [
        (3, 10.0, -1.0, 8.0),
        (3, 10.0, 1.0, 8.0),
        (3, 10.0, 0.0, 8.0),
        (3, 10.0, 1.0, 0.0),
        (3, math.nan, 1.0, 8.0),
        (0, 0.0, 1.0, 36.0),
        (100, 30.0, -1.0, 24.0),
    ]:
        ones.append(
            {
                "step": step,
                "datum_deg": datum,
                "sense": sense,
                "steps_per_revolution": per_revolution,
                "result": _call(
                    one, step, datum_deg=datum, sense=sense, steps_per_revolution=per_revolution
                ),
            }
        )
    steps = pf["revolution_steps"]
    revolution = [
        {"number": n, "per_revolution": p, "result": _call(steps, n, p)}
        for n, p in [
            (1.5, 24.0),
            (0.1, 36.0),
            (0.3, 10.0),
            (2.0, 36.0),
            (1e300, 1e300),
            (math.inf, 2.0),
            (1e-9, 5.0),
            (0.5, 250.0),
            (3.0, 7.5),
        ]
    ]
    return {"azimuth": cases, "blade_one": ones, "revolution_steps": revolution}


# ------------------------------------------------------------------ main


def _meta(pf) -> dict:  # type: ignore[no-untyped-def]
    import pyflightstream

    root = Path(pyflightstream.__file__).resolve().parents[2]

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "pyflightstream_version": getattr(pyflightstream, "__version__", "unknown"),
        "pyflightstream_commit": git("rev-parse", "HEAD"),
        "pyflightstream_tree_dirty": bool(git("status", "--porcelain", "--", "src")),
        "numpy_version": np.__version__,
        "seed": SEED,
        "generator": "tools/record_pyfs_unsteady_reference.py",
        "sources": {
            "windows": "cases/windows.py: stated_key, averaging_steps, averaging_span",
            "step_clock": "post/_tables.py: _read_step_clock",
            "average": "post/unsteady.py: blade_passage_average",
            "per_revolution": "post/point_tables.py: per_revolution_table, revolution_drift_pct, "
            "is_force_or_moment_column",
            "warnings": "post/unsteady.py: revolution_drift_warnings",
            "revolution_cuts": "post/harmonics.py: whole_revolution_steps, "
            "last_complete_revolution",
            "clock": "cases/workflows/_timing.py: rotor_time_stepping, unsteady_time_stepping, "
            "revolution_steps",
            "azimuth": "post/axes.py: blade_azimuth_deg; post/unsteady.py: blade_one_azimuth",
        },
    }


def main() -> int:
    """Record every entry and write the JSON; print its path and sha256."""
    from pyflightstream.cases import SimCase, SweepAxis, windows
    from pyflightstream.cases.workflows import (
        rotor_time_stepping,
        unsteady_time_stepping,
    )
    from pyflightstream.cases.workflows._timing import revolution_steps
    from pyflightstream.post._tables import _read_step_clock
    from pyflightstream.post.axes import blade_azimuth_deg
    from pyflightstream.post.harmonics import last_complete_revolution, whole_revolution_steps
    from pyflightstream.post.point_tables import (
        is_force_or_moment_column,
        per_revolution_table,
        revolution_drift_pct,
    )
    from pyflightstream.post.unsteady import (
        TimestepSeries,
        blade_one_azimuth,
        blade_passage_average,
        revolution_drift_warnings,
    )

    pf = {
        "windows": windows,
        "SimCase": SimCase,
        "SweepAxis": SweepAxis,
        "rotor_time_stepping": rotor_time_stepping,
        "unsteady_time_stepping": unsteady_time_stepping,
        "revolution_steps": revolution_steps,
        "step_clock": _read_step_clock,
        "blade_azimuth_deg": blade_azimuth_deg,
        "blade_one_azimuth": blade_one_azimuth,
        "last_complete_revolution": last_complete_revolution,
        "whole_revolution_steps": whole_revolution_steps,
        "is_force_or_moment_column": is_force_or_moment_column,
        "per_revolution_table": per_revolution_table,
        "revolution_drift_pct": revolution_drift_pct,
        "TimestepSeries": TimestepSeries,
        "blade_passage_average": blade_passage_average,
        "revolution_drift_warnings": revolution_drift_warnings,
    }
    document = {
        "meta": _meta(pf),
        "windows": _record_windows(pf),
        "step_clock": _record_step_clock(pf),
        "averages": _record_averages(pf),
        "per_revolution": _record_per_revolution(pf),
        "drift": _record_drift(pf),
        "warnings": _record_warnings(pf),
        "load_columns": _record_columns(pf),
        "revolution_cuts": _record_revolution_cuts(pf),
        "clock": _record_clock(pf)[0],
        "azimuth": _record_azimuth(pf),
    }
    text = json.dumps(document, separators=(",", ":"), allow_nan=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(OUT)
    print("sha256", hashlib.sha256(text.encode("utf-8")).hexdigest())
    print("bytes", len(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
