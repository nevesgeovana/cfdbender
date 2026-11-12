"""Resolver numerics that the example-based tests leave unpinned.

Each case kills a sabotage of core/flight.py that the matrix-column tests let
survive: the stratosphere, the viscosity of a pinned temperature, the MUPas,
ASMPS, RHOkgm3 and PPA pins, the REF length in a Reynolds solve, and the sign
refusals. Reference values are ISO 2533 table entries and Sutherland's law
evaluated by hand.
"""

from __future__ import annotations

import pytest

from cfdbender.core.flight import FlightConditionError, resolve_flow

FEET_PER_METRE = 1 / 0.3048


@pytest.mark.parametrize(
    ("altitude_m", "temperature", "pressure"),
    [
        (5000.0, 255.65, 54019.9),
        (11000.0, 216.65, 22632.1),
        (15000.0, 216.65, 12044.6),
        (20000.0, 216.65, 5474.9),
    ],
)
def test_standard_atmosphere_matches_the_iso_table(altitude_m, temperature, pressure):
    state = resolve_flow(
        {"MACH": 0.5, "ALTFT": altitude_m * FEET_PER_METRE}, reference_length=None, label="P"
    )
    assert state.temperature == pytest.approx(temperature, abs=1e-6)
    assert state.pressure == pytest.approx(pressure, rel=2e-5)


def test_a_temperature_pin_moves_the_viscosity_and_the_reynolds_density():
    # Sutherland at 250 K: 1.716e-5 (250/273.15)^1.5 (383.55/360.4) = 1.5989e-5 Pa s.
    state = resolve_flow(
        {"TASmps": 100.0, "TK": 250.0, "REmi": 1.0}, reference_length=2.0, label="P"
    )
    assert state.viscosity == pytest.approx(1.5989e-5, rel=1e-4)
    assert state.density == pytest.approx(1e6 * 1.5989e-5 / (100.0 * 2.0), rel=1e-4)


def test_reynolds_density_uses_the_pinned_viscosity_and_the_ref_length():
    state = resolve_flow(
        {"TASmps": 50.0, "REmi": 2.0, "MUPas": 2e-5}, reference_length=0.5, label="P"
    )
    assert state.viscosity == 2e-5
    assert state.density == pytest.approx(2e6 * 2e-5 / (50.0 * 0.5))
    assert state.reynolds_length == 0.5


def test_sonic_pin_density_pin_and_pressure_pin_are_used():
    sonic = resolve_flow({"MACH": 0.5, "ASMPS": 300.0}, reference_length=None, label="P")
    assert sonic.velocity == pytest.approx(150.0)
    pinned = resolve_flow(
        {"MACH": 0.3, "RHOkgm3": 1.0, "PPA": 90000.0}, reference_length=None, label="P"
    )
    assert (pinned.density, pinned.pressure, pinned.density_source) == (1.0, 90000.0, "pinned")


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ({"TASmps": -5.0}, "cannot be negative"),
        ({"MACH": 0.2, "TK": 0.0}, "TK must be positive"),
        ({"MACH": 0.2, "RHOkgm3": -1.0}, "RHOkgm3 must be positive"),
        ({"MACH": 0.2, "REmi": -1.0}, "REmi must be positive"),
    ],
)
def test_signs_are_refused(condition, message):
    with pytest.raises(FlightConditionError, match=message):
        resolve_flow(condition, reference_length=1.0, label="P")
