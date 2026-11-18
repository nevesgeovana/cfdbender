"""The Trefftz numerics: trace of section loads, probe-plane integrals and rotor channels."""

from __future__ import annotations

import math

import numpy as np
import pytest
from farfield_support import build_lattice

from cfdbender.core.farfield import (
    FarfieldInputError,
    cylindrical_components,
    lattice_fields,
    plane_fields,
    plane_induced_drag,
    plane_vorticity_drag,
    rotor_channels,
    strip_edges,
    trace_coefficients,
    trace_forces,
)


def _trace():
    y = np.linspace(-1.0, 1.0, 5)
    return y, np.zeros(5), 1.0 - y**2


def test_f102_a_station_end_rule_refuses_instead_of_returning_nan():
    y, z, gamma = _trace()
    # A trace that sheds nothing at its ends induces nothing there: a finite answer.
    assert math.isfinite(
        trace_forces(y, z=z, gamma=gamma, rho=1.2, v_inf=30.0, ends="station").lift
    )
    # Circulation that does not vanish at the end puts a nonzero filament under a trace point.
    with pytest.raises(FarfieldInputError, match="shed filament"):
        trace_forces(y, z=z, gamma=gamma + 0.5, rho=1.2, v_inf=30.0, ends="station")


@pytest.mark.parametrize("rho", [-1.2, 0.0, float("nan")])
def test_f103_an_invalid_density_is_refused(rho):
    y, z, gamma = _trace()
    with pytest.raises(FarfieldInputError, match="density"):
        trace_forces(y, z=z, gamma=gamma, rho=rho, v_inf=30.0)


@pytest.mark.parametrize("chunk", [-1, 0, 2.5, math.inf, math.nan, "a", True])
def test_f099_a_nonpositive_chunk_is_refused(chunk):
    """every chunk that is not a positive integer is the documented refusal."""
    axis = np.linspace(0.0, 1.0, 5)
    plane = plane_fields(axis, z=axis, fields={"v": np.ones((5, 5)), "w": np.ones((5, 5))})
    with pytest.raises(FarfieldInputError, match="chunk"):
        plane_vorticity_drag(plane, rho=1.2, chunk=chunk)
    assert math.isfinite(plane_vorticity_drag(plane, rho=1.2, chunk=np.int64(2)))


def test_f104_rotor_coverage_counts_every_channel():
    lattice = build_lattice(tip_radius=1.0, stations=(0.0, 1.0), lateral_radius=None)
    shape = (2, lattice.n_r, lattice.n_psi)
    values = {k: np.ones(shape) for k in ("u", "v", "w", "p_prime")}
    values["p_prime"][1, 0, 0] = np.nan
    ds = cylindrical_components(lattice_fields(lattice, fields=values))
    assert rotor_channels(ds, rho=1.2, v_inf=1.0, station=1.0).coverage < 1.0


def test_plane_induced_drag_is_the_crossflow_energy_integral():
    axis = np.linspace(0.0, 2.0, 5)
    plane = plane_fields(axis, z=axis, fields={"v": np.full((5, 5), 0.5), "w": np.zeros((5, 5))})
    drag = plane_induced_drag(plane, rho=1.2, v_inf=30.0)
    # (rho / 2) * integral of (v^2 + w^2) over the 2 m by 2 m plane.
    assert drag.induced_drag == pytest.approx(0.5 * 1.2 * 0.25 * 4.0)
    assert drag.coverage == 1.0


_FARFIELD_TREFFTZ__RHO = 1.2


_FARFIELD_TREFFTZ__V = 30.0


_FARFIELD_TREFFTZ__SPAN = 10.0


_FARFIELD_TREFFTZ__G0 = 5.0


_FARFIELD_TREFFTZ__Q = 0.5 * _FARFIELD_TREFFTZ__RHO * _FARFIELD_TREFFTZ__V * _FARFIELD_TREFFTZ__V


def _farfield_trefftz___uniform_stations(
    per_side: int, span: float = _FARFIELD_TREFFTZ__SPAN
) -> np.ndarray:
    """``2 n`` stations at the centres of ``2 n`` equal strips across the span."""
    h = (span / 2.0) / per_side
    return -span / 2.0 + h * (np.arange(2 * per_side) + 0.5)


def _farfield_trefftz___elliptic(
    y: np.ndarray, span: float = _FARFIELD_TREFFTZ__SPAN, peak: float = _FARFIELD_TREFFTZ__G0
) -> np.ndarray:
    return peak * np.sqrt(1.0 - (2.0 * y / span) ** 2)


def _farfield_trefftz___efficiency(
    lift: float, drag: float, span: float = _FARFIELD_TREFFTZ__SPAN
) -> float:
    """``e = L^2 / (pi q b^2 D)``, the span efficiency of a planar wing of span ``b``."""
    return lift * lift / (math.pi * _FARFIELD_TREFFTZ__Q * span * span * drag)


def test_f7_0203_a_zero_strength_filament_under_another_sheets_point_induces_nothing():
    """a silent filament that coincides with a point of a second sheet is finite."""
    y = np.array([-1.0, 0.0, 1.0, -0.5, 5.0])
    result = trace_forces(
        y, z=np.zeros(5), gamma=np.array([1.0, 1.0, 1.0, 0.5, 0.5]), rho=1.0, v_inf=1.0,
        sheets=[0, 0, 0, 1, 1],
    )  # fmt: skip
    assert np.all(np.isfinite(result.normalwash))
    assert math.isfinite(result.induced_drag)


class TestF7TrefftzDiagnostics:
    @staticmethod
    def test_x_f_0308_coincident_trace_points_are_refused_by_name():
        y = np.array([0.0, 0.5, 0.5, 1.0])
        with pytest.raises(FarfieldInputError, match="coincident trace points"):
            trace_forces(
                y,
                z=np.zeros_like(y),
                gamma=np.ones_like(y),
                rho=1.2,
                v_inf=30.0,
                ends="station",
            )


class TestFarfieldTrefftz:
    """the discrete trace of the section loads, on analytic loadings.

    Every loading is written here in closed form: an elliptic circulation at
    uniform stations (the strip centres a section distribution has), the same at
    cosine stations, a uniform circulation (two tip filaments), and two sheets at
    two heights. The expected numbers are computed in the test from the formula,
    never read back from the code under test. The post-side clauses (the lift
    density from the section forces, the named skips, the files) are in
    ``test_p0400_farfield_products.py``.
    """

    @staticmethod
    @pytest.mark.parametrize(("per_side", "bias_pct"), [(40, 1.05), (80, 0.52), (160, 0.26)])
    def test_p0400_trefftza_an_elliptic_loading_shows_the_measured_strip_bias(per_side, bias_pct):
        """E_T exceeds 1 by the definitions page's bias, halving."""
        y = _farfield_trefftz___uniform_stations(per_side)
        result = trace_forces(
            y,
            z=np.zeros_like(y),
            gamma=_farfield_trefftz___elliptic(y),
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
        )
        e = _farfield_trefftz___efficiency(result.lift, result.induced_drag)
        assert 100.0 * (e - 1.0) == pytest.approx(bias_pct, abs=0.02)

    @staticmethod
    def test_p0400_trefftza_the_station_rule_is_not_the_far_field_rule():
        """edges at the end stations shorten the span by one spacing.

        The control of the mutant "the station rule used for the trace": with the
        integrated sectional loads' rule the traced span is one spacing short, the
        end filaments sit on the end stations, and the bias of the test above is not
        met (the trace is not even finite).
        """
        y = _farfield_trefftz___uniform_stations(40)
        extended = strip_edges(y, ends="extended")
        station = strip_edges(y, ends="station")
        spacing = y[1] - y[0]
        assert extended[0] == pytest.approx(-_FARFIELD_TREFFTZ__SPAN / 2.0) and extended[
            -1
        ] == pytest.approx(_FARFIELD_TREFFTZ__SPAN / 2.0)
        assert (station[-1] - station[0]) == pytest.approx(_FARFIELD_TREFFTZ__SPAN - spacing)
        with pytest.raises(FarfieldInputError, match="shed filament"):
            trace_forces(
                y,
                z=np.zeros_like(y),
                gamma=_farfield_trefftz___elliptic(y),
                rho=_FARFIELD_TREFFTZ__RHO,
                v_inf=_FARFIELD_TREFFTZ__V,
                ends="station",
            )

    @staticmethod
    def test_p0400_trefftza_a_uniform_circulation_is_two_tip_filaments():
        """lift rho V Gamma b exactly, drag of the two
        filaments.
        """
        y = _farfield_trefftz___uniform_stations(20)
        gamma = np.full_like(y, _FARFIELD_TREFFTZ__G0)
        result = trace_forces(
            y,
            z=np.zeros_like(y),
            gamma=gamma,
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
        )
        assert result.lift == pytest.approx(
            _FARFIELD_TREFFTZ__RHO
            * _FARFIELD_TREFFTZ__V
            * _FARFIELD_TREFFTZ__G0
            * _FARFIELD_TREFFTZ__SPAN,
            rel=1e-14,
        )
        # Closed form: the two filaments at +-b/2 shed +Gamma (left) and -Gamma
        # (right); the downwash at the lifting line, positive down, is
        # (1/4 pi) sum dGamma / (y - e) and the drag rho sum Gamma w dy.
        dy = _FARFIELD_TREFFTZ__SPAN / len(y)
        w = (_FARFIELD_TREFFTZ__G0 / (4.0 * math.pi)) * (
            1.0 / (y + _FARFIELD_TREFFTZ__SPAN / 2.0) - 1.0 / (y - _FARFIELD_TREFFTZ__SPAN / 2.0)
        )
        expected = _FARFIELD_TREFFTZ__RHO * float(np.sum(_FARFIELD_TREFFTZ__G0 * w * dy))
        assert result.induced_drag == pytest.approx(expected, rel=1e-12)
        assert result.downwash == pytest.approx(w, rel=1e-12)

    @staticmethod
    def test_p0400_trefftza_a_mirrored_half_span_equals_the_full_span():
        """the mirror rebuilds the full sheet, no root filament."""
        y = _farfield_trefftz___uniform_stations(40)
        z = 0.02 * np.abs(y)
        gamma = _farfield_trefftz___elliptic(y)
        full = trace_forces(
            y, z=z, gamma=gamma, rho=_FARFIELD_TREFFTZ__RHO, v_inf=_FARFIELD_TREFFTZ__V
        )
        right = y > 0.0
        half = trace_forces(
            y[right],
            z=z[right],
            gamma=gamma[right],
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
            mirror=True,
        )
        assert half.lift == pytest.approx(full.lift, rel=1e-12)
        assert half.induced_drag == pytest.approx(full.induced_drag, rel=1e-12)
        assert half.original.sum() == right.sum()
        left = trace_forces(
            y[~right],
            z=z[~right],
            gamma=gamma[~right],
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
            mirror=True,
        )
        assert left.induced_drag == pytest.approx(full.induced_drag, rel=1e-12)

    @staticmethod
    def test_p0400_trefftza_a_half_trace_not_mirrored_is_not_the_full_span():
        """the control of the mutant "the mirror skipped"."""
        y = _farfield_trefftz___uniform_stations(40)
        right = y > 0.0
        full = trace_forces(
            y,
            z=np.zeros_like(y),
            gamma=_farfield_trefftz___elliptic(y),
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
        )
        half = trace_forces(
            y[right],
            z=np.zeros(right.sum()),
            gamma=_farfield_trefftz___elliptic(y)[right],
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
        )
        assert half.induced_drag != pytest.approx(full.induced_drag / 2.0, rel=1e-3)

    @staticmethod
    def test_p0400_trefftza_a_straddling_half_model_is_refused():
        """stations on both sides cannot say which half was meant."""
        y = np.array([-0.5, 0.5, 1.5, 2.5])
        with pytest.raises(FarfieldInputError, match="straddle"):
            trace_forces(
                y,
                z=np.zeros_like(y),
                gamma=np.ones_like(y),
                rho=_FARFIELD_TREFFTZ__RHO,
                v_inf=_FARFIELD_TREFFTZ__V,
                mirror=True,
            )

    @staticmethod
    def test_p0400_trefftza_cosine_stations_converge_to_the_elliptic_efficiency():
        """uneven stations converge to e = 1 as they double."""
        errors = []
        for per_side in (20, 40, 80, 160):
            theta = (np.arange(2 * per_side) + 0.5) * math.pi / (2 * per_side)
            y = -_FARFIELD_TREFFTZ__SPAN / 2.0 * np.cos(theta)
            result = trace_forces(
                y,
                z=np.zeros_like(y),
                gamma=_farfield_trefftz___elliptic(y),
                rho=_FARFIELD_TREFFTZ__RHO,
                v_inf=_FARFIELD_TREFFTZ__V,
            )
            errors.append(
                abs(_farfield_trefftz___efficiency(result.lift, result.induced_drag) - 1.0)
            )
        assert all(b < 0.6 * a for a, b in zip(errors, errors[1:], strict=False)), errors
        assert errors[-1] < 0.005

    @staticmethod
    def test_p0400_trefftza_family_shares_add_up_and_isolated_values_are_each_alone():
        """the share of the total and the family alone, both."""
        wing_y = _farfield_trefftz___uniform_stations(40)
        tail_y = _farfield_trefftz___uniform_stations(10, span=3.0)
        y = np.concatenate([wing_y, tail_y])
        z = np.concatenate([np.zeros_like(wing_y), np.full_like(tail_y, 0.6)])
        gamma = np.concatenate(
            [_farfield_trefftz___elliptic(wing_y), _farfield_trefftz___elliptic(tail_y, 3.0, -0.8)]
        )
        sheets = np.concatenate([np.zeros(len(wing_y), int), np.ones(len(tail_y), int)])
        both = trace_forces(
            y,
            z=z,
            gamma=gamma,
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
            sheets=sheets,
        )
        assert both.sheet_drag_share.sum() == pytest.approx(both.induced_drag, rel=1e-14)
        wing = trace_forces(
            wing_y,
            z=np.zeros_like(wing_y),
            gamma=_farfield_trefftz___elliptic(wing_y),
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
        )
        tail = trace_forces(
            tail_y,
            z=np.full_like(tail_y, 0.6),
            gamma=_farfield_trefftz___elliptic(tail_y, 3.0, -0.8),
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
        )
        assert both.sheet_drag_isolated[0] == pytest.approx(wing.induced_drag, rel=1e-12)
        assert both.sheet_drag_isolated[1] == pytest.approx(tail.induced_drag, rel=1e-12)
        interference = both.induced_drag - both.sheet_drag_isolated.sum()
        assert abs(interference) > 1e-6 * both.induced_drag
        assert both.sheet_drag_share[0] != pytest.approx(both.sheet_drag_isolated[0], rel=1e-6)

    @staticmethod
    def test_p0400_trefftza_the_trace_follows_the_quarter_chord_height():
        """a dihedral sheet is traced in (y, z), with side force zero."""  # noqa: E501
        y = _farfield_trefftz___uniform_stations(40)
        z = 0.1 * np.abs(y)
        result = trace_forces(
            y,
            z=z,
            gamma=_farfield_trefftz___elliptic(y),
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
        )
        flat = trace_forces(
            y,
            z=np.zeros_like(y),
            gamma=_farfield_trefftz___elliptic(y),
            rho=_FARFIELD_TREFFTZ__RHO,
            v_inf=_FARFIELD_TREFFTZ__V,
        )
        assert result.side_force == pytest.approx(0.0, abs=1e-9 * result.lift)
        assert result.induced_drag != pytest.approx(flat.induced_drag, rel=1e-6)

    @staticmethod
    def test_p0400_trefftza_the_coefficients_and_the_na_rules():
        """q SREF normalisation, E_T NA at zero drag."""
        c = trace_coefficients(
            120.0, side_force=3.0, induced_drag=4.0, q=_FARFIELD_TREFFTZ__Q, s_ref=8.0, b_ref=10.0
        )
        assert c["CL_T"] == pytest.approx(120.0 / (_FARFIELD_TREFFTZ__Q * 8.0))
        assert c["CY_T"] == pytest.approx(3.0 / (_FARFIELD_TREFFTZ__Q * 8.0))
        assert c["AR"] == pytest.approx(12.5)
        assert c["E_T"] == pytest.approx(c["CL_T"] ** 2 / (math.pi * 12.5 * c["CDI_T"]))
        assert math.isnan(
            trace_coefficients(
                1.0, side_force=0.0, induced_drag=0.0, q=_FARFIELD_TREFFTZ__Q, s_ref=8.0, b_ref=10.0
            )["E_T"]
        )
        assert math.isnan(
            trace_coefficients(
                1.0, side_force=0.0, induced_drag=1.0, q=_FARFIELD_TREFFTZ__Q, s_ref=0.0, b_ref=10.0
            )["CL_T"]
        )
