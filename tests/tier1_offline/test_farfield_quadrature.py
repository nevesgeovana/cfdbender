"""Far-field quadrature and harmonic layer: coverage, units, harmonic guards and strip edges."""

from __future__ import annotations

import math

import numpy as np
import pytest
from farfield_support import build_lattice

from cfdbender.core.farfield import (
    FarfieldInputError,
    azimuthal_harmonics,
    grid_coverage,
    grid_integral,
    lattice_fields,
    plane_integral,
    ring_sample_weights,
    sample_area_coverage,
    sample_coverage,
    strip_edges,
    symmetry_floor,
)


def _core_small_farfield___ring(
    n: int, values: np.ndarray | None = None, psi_0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    psi = psi_0 + 2.0 * np.pi * np.arange(n) / n
    data = np.cos(psi) if values is None else values
    return data, psi


def test_p0460_g3_excludes_every_blade_multiple():
    psi = np.arange(32) * 2 * np.pi / 32
    values = 2 * 0.4828 * np.cos(6 * psi) + 4 * np.cos(12 * psi) + 2 * 3.47e-5 * np.cos(psi)
    assert symmetry_floor(values, psi=psi, m_max=13, excluded_orders=[6, 12]) == pytest.approx(
        3.47e-5
    )
    assert symmetry_floor(values, psi=psi, m_max=5) == pytest.approx(3.47e-5)
    assert symmetry_floor(values, psi=psi, m_max=13) == pytest.approx(2.0)
    assert math.isnan(symmetry_floor(values, psi=psi, m_max=2, excluded_orders=[1, 2]))
    values[0] = np.nan
    assert math.isnan(symmetry_floor(values, psi=psi, m_max=13, excluded_orders=[6, 12]))


def _fields(tip_radius=1.0, value=np.pi):
    lattice = build_lattice(tip_radius=tip_radius, stations=(0.0,), lateral_radius=None)
    field = np.full((1, lattice.n_r, lattice.n_psi), value)
    return lattice, field, lattice_fields(lattice, fields={"u": field})


def _holed(field, ring):
    holed = field.copy()
    holed[0, ring, :] = np.nan
    return holed


def test_p0420_a_hole_at_the_rim_costs_more_coverage_than_a_hole_at_the_hub():
    """one missing ring at either end of the same lattice, same sample count."""
    lattice, field, ds = _fields()
    hub = lattice_fields(lattice, fields={"u": _holed(field, 0)})
    rim = lattice_fields(lattice, fields={"u": _holed(field, lattice.n_r - 1)})
    by_count_hub = float(sample_coverage(hub["u"])[0])
    by_count_rim = float(sample_coverage(rim["u"])[0])
    assert by_count_hub == pytest.approx(by_count_rim), "the count cannot tell them apart"
    area_hub = float(sample_area_coverage(hub["u"], hub)[0])
    area_rim = float(sample_area_coverage(rim["u"], rim)[0])
    assert area_rim < area_hub < 1.0
    weights = ring_sample_weights(ds)
    assert area_hub == pytest.approx(1.0 - weights[0] / weights.sum())
    assert area_rim == pytest.approx(1.0 - weights[-1] / weights.sum())


def test_p0420_the_area_coverage_is_the_share_of_the_integral_a_hole_removes():
    """for a constant field the coverage and the integral lose the same share."""
    lattice, field, whole = _fields()
    rim = lattice_fields(lattice, fields={"u": _holed(field, lattice.n_r - 1)})
    coverage = float(sample_area_coverage(rim["u"], rim)[0])
    finite_integral = float(plane_integral(rim, np.nan_to_num(rim["u"], nan=0.0))[0])
    assert finite_integral / float(plane_integral(whole, whole["u"])[0]) == pytest.approx(coverage)


def test_p0420_a_complete_plane_has_coverage_one_both_ways():
    """Control: a complete plane has coverage one."""
    _lattice, _field, ds = _fields()
    assert float(sample_area_coverage(ds["u"], ds)[0]) == pytest.approx(1.0)
    assert float(sample_coverage(ds["u"])[0]) == 1.0


def test_p0420_a_lattice_of_another_ring_count_is_refused():
    """the area each sample stands for is undefined for another lattice."""
    _lattice, _field, ds = _fields()
    with pytest.raises(FarfieldInputError, match="rings"):
        sample_area_coverage(ds["u"][:, :-1, :], ds)


def test_p0420_the_grid_coverage_weights_a_corner_by_its_quarter_cell():
    """a missing corner stands for a quarter cell, an inner one for a cell."""
    y = np.linspace(0.0, 4.0, 5)
    z = np.linspace(0.0, 4.0, 5)
    integrand = np.ones((5, 5))
    corner = integrand.copy()
    corner[0, 0] = np.nan
    inner = integrand.copy()
    inner[2, 2] = np.nan
    assert grid_coverage(corner) == pytest.approx(grid_coverage(inner))
    assert grid_coverage(corner, y=y, z=z) == pytest.approx(1.0 - 0.25 / 16.0)
    assert grid_coverage(inner, y=y, z=z) == pytest.approx(1.0 - 1.0 / 16.0)
    assert grid_coverage(integrand, y=y, z=z) == 1.0
    lost = grid_integral(y, z=z, integrand=np.nan_to_num(inner, nan=0.0))
    assert lost / grid_integral(y, z=z, integrand=integrand) == pytest.approx(
        grid_coverage(inner, y=y, z=z)
    )


def test_p0420_one_grid_axis_alone_is_refused():
    """half an axis pair does not define the weights."""
    with pytest.raises(FarfieldInputError, match="both axes"):
        grid_coverage(np.ones((3, 3)), y=np.arange(3.0))


def test_p0420_a_ledger_scales_with_the_square_of_the_length_unit_of_tip_radius():
    """the statement the docstrings make, pinned as the behaviour it describes."""
    _lattice, _field, metres = _fields(tip_radius=1.0)
    _lattice, _field, millimetres = _fields(tip_radius=1000.0)
    in_metres = float(plane_integral(metres, metres["u"])[0])
    in_millimetres = float(plane_integral(millimetres, millimetres["u"])[0])
    assert in_millimetres == pytest.approx(in_metres * 1000.0**2)


def test_last_symmetry_floor_needs_an_order_above_zero():
    """m_max=0 left an empty slice and numpy's bare ValueError."""
    psi = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    values = np.cos(psi)
    with pytest.raises(FarfieldInputError, match="m_max=0 keeps no harmonic above order zero"):
        symmetry_floor(values, psi=psi, m_max=0)
    assert symmetry_floor(values, psi=psi, m_max=2) == pytest.approx(0.5)


class TestCoreSmallFarfield:
    """the azimuthal harmonics guard.

    The anti-aliasing cap ``m_max >= n // 2`` refused an odd grid of
    ``n = 2 m_max + 1`` azimuths, which carries order ``m_max`` unaliased, with a
    message that contradicted itself; a negative or non-integer ``m_max`` was
    accepted and gave an empty or meaningless ``m`` axis.
    """

    @staticmethod
    def test_p0400_ffharmonics_odd_grid_carries_its_highest_order() -> None:
        """n = 13 with m_max = 6 is accepted; n = 9 with m_max = 4 too."""
        values, psi = _core_small_farfield___ring(13)
        assert azimuthal_harmonics(values, psi=psi, m_max=6).shape[-1] == 7
        values, psi = _core_small_farfield___ring(9)
        assert azimuthal_harmonics(values, psi=psi, m_max=4).shape[-1] == 5

    @staticmethod
    def test_p0400_ffharmonics_even_grid_nyquist_is_still_refused() -> None:
        """n = 12 with m_max = 6 is the Nyquist order and is refused."""
        values, psi = _core_small_farfield___ring(12)
        with pytest.raises(FarfieldInputError, match="needs more than 12 uniform azimuths"):
            azimuthal_harmonics(values, psi=psi, m_max=6)

    @staticmethod
    @pytest.mark.parametrize("m_max", [-1, 2.5, True])
    def test_p0400_ffharmonics_m_max_must_be_a_non_negative_integer(m_max: object) -> None:
        """m_max = -1 or 2.5 on a 16-azimuth lattice is refused."""  # noqa: E501
        values, psi = _core_small_farfield___ring(16)
        with pytest.raises(FarfieldInputError, match="is not an azimuthal order"):
            azimuthal_harmonics(values, psi=psi, m_max=m_max)  # type: ignore[arg-type]

    @staticmethod
    def test_p0400_ffharmonics_origin_offset_grid_gives_unrotated_coefficients() -> None:
        """psi starting at 0.3 rad gives c_1 = 1/2, real.

        The FFT computes c_m only for psi_k = 2 pi k / N; on a grid offset by psi_0 every order
        came back multiplied by exp(i m psi_0).
        """
        n, psi_0 = 16, 0.3
        psi = psi_0 + 2.0 * np.pi * np.arange(n) / n
        field = np.cos(psi) + 0.25 * np.sin(2.0 * psi)
        c = azimuthal_harmonics(field, psi=psi, m_max=3)
        np.testing.assert_allclose(c, [0.0, 0.5, -0.125j, 0.0], atol=1e-12)


class TestRfixR5Quadrature:
    """Exercise the strip-edge API contract on flat, descending and nonfinite coordinates."""

    @staticmethod
    def test_p0400_rfixr5_strip_edges_contract():
        """document trace order, actual refusals and NaN propagation."""
        np.testing.assert_array_equal(strip_edges(np.array([0, 0])), [0, 0, 0])
        np.testing.assert_array_equal(strip_edges(np.array([2, 1])), [2.5, 1.5, 0.5])
        assert np.isnan(strip_edges(np.array([0, np.nan]))).all()
        doc = strip_edges.__doc__ or ""
        assert "constant or descending" in doc
        assert "Nonfinite coordinates propagate" in doc
        assert "one-dimensional" in doc and "ends" in doc
        assert "strictly increasing" not in doc
