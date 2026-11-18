"""Far-field ledger gate: synthetic fields with exact answers.

Every ledger equation is validated against pencil-and-paper fields: uniform flow, an
analytic actuator-disk jump, a solid-body swirl patch, and an imposed pure 1P cosine
loading. No solver, fast, deterministic.
"""

import dataclasses
import math
import warnings
from inspect import Parameter, signature

import numpy as np
import pytest
from farfield_support import build_lattice

from cfdbender.post.farfield import (
    FarfieldInputError,
    axial_flux,
    axial_force,
    azimuthal_harmonics,
    crossflow_kinetic_energy,
    cylindrical_components,
    grid_integral,
    in_plane_moment,
    lateral_flux,
    lattice_fields,
    mass_closure,
    mass_flux,
    plane_integral,
    sample_coverage,
    shaft_torque,
    spurious_diagnostic,
    symmetry_floor,
    to_counts,
    transverse_flux,
    transverse_force,
)

RHO = 1.2
V_INF = 30.0

LATTICE = build_lattice(
    tip_radius=1.0,
    stations=(-2.0, 0.5, 1.0, 2.0),
    lateral_radius=None,
)
SHAPE = (len(LATTICE.stations), LATTICE.n_r, LATTICE.n_psi)
R = LATTICE.ring_centers[None, :, None]
PSI = LATTICE.psi[None, None, :]
R_IN = LATTICE.ring_edges[0]
R_OUT = LATTICE.ring_edges[-1]


def uniform_fields():
    return {
        "u": np.full(SHAPE, V_INF),
        "v": np.zeros(SHAPE),
        "w": np.zeros(SHAPE),
        "p_prime": np.zeros(SHAPE),
    }


def make_dataset(fields):
    return cylindrical_components(lattice_fields(LATTICE, fields=fields))


def test_g0_uniform_flow_closes_every_ledger():
    ds = make_dataset(uniform_fields())
    closure = mass_closure(ds, RHO)
    area = np.pi * (R_OUT**2 - R_IN**2)
    assert float(closure.relative_spread) == 0.0
    assert float(mass_flux(ds, RHO)[0]) == pytest.approx(RHO * V_INF * area, rel=1e-12)
    assert axial_force(ds, RHO, V_INF, inlet=-2.0, outlet=2.0) == pytest.approx(0.0, abs=1e-9)
    fz = transverse_force(ds, RHO, inlet=-2.0, outlet=2.0)
    assert float(fz.total) == pytest.approx(0.0, abs=1e-9)
    assert float(shaft_torque(ds, RHO).max()) == pytest.approx(0.0, abs=1e-12)
    my = in_plane_moment(ds, RHO, V_INF, inlet=-2.0, outlet=2.0)
    assert float(my.total) == pytest.approx(0.0, abs=1e-9)
    ke = crossflow_kinetic_energy(ds, RHO)
    assert float(np.abs(ke.total).max()) == 0.0
    assert symmetry_floor(ds["u"], psi=ds.psi) < 1e-11 * V_INF


def test_g0_actuator_disk_jump_recovers_the_analytic_thrust():
    # Pressure jump dp inside a disk whose edge coincides with a ring
    # edge, applied on the outlet plane only: the ring partition makes
    # the quadrature exact, so the analytic value is met to precision.
    dp = 120.0
    edge_index = int(np.argmin(np.abs(np.asarray(LATTICE.ring_edges) - 1.0)))
    r_disk = LATTICE.ring_edges[edge_index]
    fields = uniform_fields()
    outlet = list(LATTICE.stations).index(1.0)
    inside = (LATTICE.ring_centers < r_disk)[None, :, None]
    fields["p_prime"][outlet] = np.where(inside[0], dp, 0.0)
    ds = make_dataset(fields)
    analytic = dp * np.pi * (r_disk**2 - R_IN**2)
    assert axial_force(ds, RHO, V_INF, inlet=-2.0, outlet=1.0) == pytest.approx(analytic, rel=1e-12)
    assert axial_force(ds, RHO, V_INF, inlet=-2.0, outlet=2.0) == pytest.approx(0.0, abs=1e-9)


def test_g0_solid_body_swirl_recovers_torque_and_fills_only_the_swirl_channel():
    omega = 8.0
    edge_index = int(np.argmin(np.abs(np.asarray(LATTICE.ring_edges) - 1.0)))
    r_core = LATTICE.ring_edges[edge_index]
    v_theta = np.where(R < r_core, omega * R, 0.0) * np.ones(SHAPE)
    fields = uniform_fields()
    fields["v"] = v_theta * np.cos(PSI)
    fields["w"] = -v_theta * np.sin(PSI)
    ds = make_dataset(fields)
    assert np.allclose(ds["v_theta"], v_theta, atol=1e-12)
    assert np.allclose(ds["v_r"], 0.0, atol=1e-12)
    torque = float(shaft_torque(ds, RHO)[0])
    analytic = RHO * V_INF * omega * 0.5 * np.pi * (r_core**4 - R_IN**4)
    assert torque == pytest.approx(analytic, rel=5e-3)
    ke = crossflow_kinetic_energy(ds, RHO)
    assert float(ke.swirl[0]) == pytest.approx(float(ke.total[0]), rel=1e-12)
    assert abs(float(ke.induced[0])) < 1e-10 * abs(float(ke.total[0]))


def test_g0_pure_1p_cosine_loading_puts_the_moment_in_the_harmonic_term():
    amplitude = 3.0
    fields = uniform_fields()
    outlet = list(LATTICE.stations).index(0.5)
    fields["u"][outlet] = V_INF + amplitude * (R[0] * np.cos(PSI[0]))
    ds = make_dataset(fields)
    my = in_plane_moment(ds, RHO, V_INF, inlet=-2.0, outlet=0.5)
    analytic = RHO * V_INF * amplitude * np.pi * (R_OUT**4 - R_IN**4) / 4.0
    assert float(my.total) == pytest.approx(analytic, rel=5e-3)
    # The 1P case is pure disk distortion: the moment is entirely the
    # order-1 harmonic (loading) term, the moment-arm term is zero.
    assert float(my.moment_arm_term) == pytest.approx(0.0, abs=1e-9)
    assert float(my.loading_term) == pytest.approx(float(my.total), rel=1e-12)
    # Two independent code paths, one test.
    harmonic = in_plane_moment(ds, RHO, V_INF, inlet=-2.0, outlet=0.5, method="harmonic")
    assert float(harmonic.total) == pytest.approx(float(my.total), rel=1e-10)
    # The order-1 coefficient of u is a(r)/2 on the loaded plane.
    loaded_u = ds["u"][ds.station_index(0.5)]
    c1 = azimuthal_harmonics(loaded_u, psi=ds.psi, m_max=2)[..., 1]
    assert np.allclose(np.real(c1), amplitude * LATTICE.ring_centers / 2.0, rtol=1e-10)
    c3 = azimuthal_harmonics(loaded_u, psi=ds.psi, m_max=3)[..., 3]
    assert float(np.abs(c3).max()) < 1e-12


def test_transverse_flux_harmonic_path_matches_the_direct_quadrature():
    fields = uniform_fields()
    fields["u"] = V_INF * (1.0 + 0.1 * np.cos(PSI) + 0.05 * np.sin(2.0 * PSI)) * np.ones(SHAPE)
    fields["w"] = (0.2 + 0.03 * np.cos(PSI)) * R * np.ones(SHAPE)
    ds = make_dataset(fields)
    direct = transverse_flux(ds, RHO, method="quadrature")
    spectral = transverse_flux(ds, RHO, method="harmonic")
    assert np.allclose(direct, spectral, rtol=1e-10)


def test_g3_symmetry_floor_is_recorded_at_machine_precision():
    ds = make_dataset(uniform_fields())
    floor = symmetry_floor(ds["u"], psi=ds.psi)
    assert floor < 1e-11 * V_INF


def test_nonuniform_azimuths_are_refused_with_the_physical_cause():
    ds = make_dataset(uniform_fields())
    warped = np.sort(np.random.default_rng(0).uniform(0, 2 * np.pi, LATTICE.n_psi))
    with pytest.raises(ValueError, match="uniform"):
        azimuthal_harmonics(ds["u"], psi=warped)
    with pytest.raises(ValueError, match="uniform"):
        plane_integral(dataclasses.replace(ds, psi=warped), ds["u"])


def test_spurious_diagnostic_reports_counts():
    counts = spurious_diagnostic(101.0, 100.0, rho_inf=RHO, v_inf=V_INF, s_ref=1.0)
    assert counts == pytest.approx(2.0 * 1.0 / (RHO * V_INF**2) * 1e4)


# Two defects of one class in two places: a reduction that silently discarded part of its
# input and returned a plausible number for the rest.


def test_a_missing_sample_no_longer_shrinks_the_integral():
    """Uniform pi with one of four samples NaN gives NaN, not 3*pi/4.

    A sum that skipped the absent sample while its ring weight stayed in the
    geometry would come back short in exact proportion to the missing data,
    with no error, no NaN and no coverage indicator, which is
    indistinguishable from a real physical reduction of flux.
    """
    lattice = build_lattice(tip_radius=1.0, stations=(0.0,), lateral_radius=None)
    shape = (1, lattice.n_r, lattice.n_psi)
    field = np.full(shape, np.pi)
    ds = lattice_fields(lattice, fields={"u": field})
    complete = float(plane_integral(ds, ds["u"])[0])

    holed = field.copy()
    holed[0, 0, 0] = np.nan
    ds_holed = lattice_fields(lattice, fields={"u": holed})
    result = float(plane_integral(ds_holed, ds_holed["u"])[0])

    assert np.isfinite(complete)
    assert np.isnan(result), (
        f"a missing sample produced the finite value {result!r} instead of NaN; "
        f"the complete field integrates to {complete!r}"
    )


def test_sample_coverage_says_how_much_is_missing():
    """NaN says the plane is unusable; this says how far from usable.

    Without it the fix trades a silently wrong number for an opaque one.
    """
    lattice = build_lattice(tip_radius=1.0, stations=(0.0,), lateral_radius=None)
    shape = (1, lattice.n_r, lattice.n_psi)
    field = np.full(shape, np.pi)
    ds = lattice_fields(lattice, fields={"u": field})
    assert float(sample_coverage(ds["u"])[0]) == 1.0

    holed = field.copy()
    holed[0, 0, 0] = np.nan
    ds_holed = lattice_fields(lattice, fields={"u": holed})
    total = lattice.n_r * lattice.n_psi
    assert float(sample_coverage(ds_holed["u"])[0]) == pytest.approx((total - 1) / total)


def _nyquist_dataset(n_psi_target):
    """A one-station lattice carrying a pure alternating azimuthal signal.

    The Nyquist mode is the highest frequency a discrete azimuth grid can
    represent: +1, -1, +1, -1 around the ring.
    """
    lattice = build_lattice(tip_radius=1.0, stations=(0.0,), lateral_radius=None)
    n_psi = lattice.n_psi
    shape = (1, lattice.n_r, n_psi)
    sign = np.cos(np.arange(n_psi) * np.pi)  # +1, -1, +1, ... exactly
    signal = np.broadcast_to(sign, shape).copy()
    ds = lattice_fields(lattice, fields={"u": signal, "v": np.zeros(shape), "w": signal})
    return cylindrical_components(ds), n_psi


def test_the_harmonic_path_no_longer_returns_zero_on_the_nyquist_mode():
    """The Nyquist signal: quadrature pi, harmonic pi, never 0.0.

    The two paths are documented as INDEPENDENT VERIFICATIONS of the same
    number. A harmonic path that summed orders 0..n/2-1 and omitted Nyquist
    would return zero on the one class of signal where the two disagree and
    report nothing. A verification that cannot fail is not a verification.
    """
    ds, _ = _nyquist_dataset(None)
    quadrature = transverse_flux(ds, RHO, component="w", method="quadrature")
    harmonic = transverse_flux(ds, RHO, component="w", method="harmonic")
    assert float(quadrature[0]) != pytest.approx(0.0, abs=1e-12), (
        "the fixture no longer carries a Nyquist signal, so this proves nothing"
    )
    assert float(harmonic[0]) == pytest.approx(float(quadrature[0]), rel=1e-10)


def test_the_two_paths_still_agree_on_an_ordinary_mode():
    """The control, and the guard against overcorrecting.

    Giving the Nyquist order weight 2 rather than 1 would fix the test above
    and break this one, because it double counts an order that is its own
    conjugate. Both directions have to be pinned.
    """
    lattice = build_lattice(tip_radius=1.0, stations=(0.0,), lateral_radius=None)
    shape = (1, lattice.n_r, lattice.n_psi)
    psi = lattice.psi[None, None, :]
    signal = np.broadcast_to(np.cos(psi), shape).copy()
    # u carries the same mode, so the product mean is NONZERO. With u = 1 the
    # product mean is zero at every order and the assertion below degenerates
    # to zero equals zero, which would pass against a harmonic path that
    # returns 0.0 unconditionally and against a doubled-Nyquist mutant.
    ds = cylindrical_components(
        lattice_fields(
            lattice,
            fields={"u": 1.0 + signal, "v": np.zeros(shape), "w": signal},
        )
    )
    quadrature = float(transverse_flux(ds, RHO, component="w", method="quadrature")[0])
    harmonic = float(transverse_flux(ds, RHO, component="w", method="harmonic")[0])
    assert abs(quadrature) > 1.0, (
        f"the fixture is degenerate: quadrature={quadrature!r}, so an agreement "
        "assertion would prove nothing"
    )
    assert harmonic == pytest.approx(quadrature, rel=1e-10)


# The missing-sample rule across the WHOLE public ledger surface. With one NaN sample in an
# otherwise uniform field, a reduction that skipped it would return a finite value:
# transverse_flux(method="harmonic") a number where the quadrature returns nan,
# symmetry_floor 0.0, meaning "perfectly axisymmetric", from a plane with a hole in it, and
# mass_closure a relative_spread of 0.0 while its own mdot was nan.


def _holed(*, holes=("u", "v", "w", "p_prime"), stations=(-1.0, 0.0, 1.0), station=1):
    """A lattice with exactly one missing sample, in the named fields only.

    THREE stations by default and a hole in ONE of them. A single station with
    every field holed at the same index would not guard four of the reductions:
    an all-NaN array reduces to NaN whether or not skipna is set, so
    mass_closure's mean and max, the in_plane_moment loading sum, and the
    v_theta azimuthal mean would all pass against a body that skips the hole.
    A multi-station plane with one dead station is what distinguishes
    "skipped it" from "propagated it".

    ``holes`` selects which fields lose the sample, so a caller can leave the
    axial velocity finite and reach the transverse-only channels; with every
    field holed, plane_integral alone forces NaN and the sibling reductions
    are never exercised.
    """
    lattice = build_lattice(tip_radius=1.0, stations=stations, lateral_radius=None)
    shape = (len(stations), lattice.n_r, lattice.n_psi)
    fields = {
        "u": np.full(shape, V_INF),
        "v": np.zeros(shape),
        "w": np.full(shape, 1.0),
        "p_prime": np.zeros(shape),
    }
    for name in holes:
        fields[name] = fields[name].copy()
        fields[name][station, 0, 0] = np.nan
    return cylindrical_components(lattice_fields(lattice, fields=fields))


@pytest.mark.parametrize(
    "name,call",
    [
        ("mass_flux", lambda ds: mass_flux(ds, RHO)),
        ("axial_flux", lambda ds: axial_flux(ds, RHO, V_INF)),
        (
            "transverse_flux quadrature",
            lambda ds: transverse_flux(ds, RHO, component="w", method="quadrature"),
        ),
        (
            "transverse_flux harmonic",
            lambda ds: transverse_flux(ds, RHO, component="w", method="harmonic"),
        ),
        ("shaft_torque", lambda ds: shaft_torque(ds, RHO)),
        ("crossflow_kinetic_energy", lambda ds: crossflow_kinetic_energy(ds, RHO).total),
        ("symmetry_floor", lambda ds: symmetry_floor(ds["u"], psi=ds.psi)),
        ("mass_closure", lambda ds: mass_closure(ds, RHO).relative_spread),
        # The public entry points a user calls.
        ("axial_force", lambda ds: axial_force(ds, RHO, V_INF, inlet=-1.0, outlet=0.0)),
        (
            "transverse_force",
            lambda ds: transverse_force(ds, RHO, inlet=-1.0, outlet=0.0).total,
        ),
        (
            "in_plane_moment total",
            lambda ds: in_plane_moment(ds, RHO, V_INF, inlet=-1.0, outlet=0.0).total,
        ),
        (
            "in_plane_moment loading",
            lambda ds: in_plane_moment(ds, RHO, V_INF, inlet=-1.0, outlet=0.0).loading_term,
        ),
    ],
)
def test_no_ledger_reports_a_number_from_an_incomplete_plane(name, call):
    """One missing sample must reach the answer, in every channel.

    A ledger that silently drops it returns a value smaller in exact
    proportion to the missing data, which is indistinguishable from a real
    physical reduction. That is the whole finding, and it is a property of
    the SURFACE rather than of plane_integral, which is why this is
    parametrized over the public functions rather than asserted once.
    """
    result = np.asarray(call(_holed()))
    assert np.isnan(result).any(), f"{name} returned {result!r} from a plane with a missing sample"


def test_the_same_ledgers_are_finite_on_a_complete_plane():
    """The control.

    Without it the parametrization above is satisfied by a module that
    returns NaN unconditionally.
    """
    lattice = build_lattice(tip_radius=1.0, stations=(0.0,), lateral_radius=None)
    shape = (1, lattice.n_r, lattice.n_psi)
    ds = cylindrical_components(
        lattice_fields(
            lattice,
            fields={
                "u": np.full(shape, V_INF),
                "v": np.zeros(shape),
                "w": np.full(shape, 1.0),
                "p_prime": np.zeros(shape),
            },
        )
    )
    assert np.isfinite(np.asarray(mass_flux(ds, RHO))).all()
    assert np.isfinite(np.asarray(symmetry_floor(ds["u"], psi=ds.psi)))
    assert np.isfinite(np.asarray(mass_closure(ds, RHO).relative_spread))
    assert np.isfinite(np.asarray(transverse_flux(ds, RHO, component="w", method="harmonic"))).all()


def test_the_two_transverse_paths_agree_on_incompleteness():
    """They are documented as independent verifications of one number.

    Before this fix the quadrature path returned nan and the harmonic path
    returned a finite 24.0298 from the same holed plane, so the cross-check
    disagreed silently in exactly the situation it exists to catch.
    """
    ds = _holed()
    quadrature = np.asarray(transverse_flux(ds, RHO, component="w", method="quadrature"))
    harmonic = np.asarray(transverse_flux(ds, RHO, component="w", method="harmonic"))
    # Per station: the holed one dies in both paths, the live ones survive in
    # both. Asserting NaN everywhere would demand the fix destroy good data.
    assert np.isnan(quadrature[1]) and np.isnan(harmonic[1])
    assert np.isfinite(quadrature[0]) and np.isfinite(harmonic[0])
    assert np.isfinite(quadrature[2]) and np.isfinite(harmonic[2])


def test_the_two_paths_agree_at_an_odd_azimuth_count():
    """The odd-n branch of the Parseval weights, which had no test.

    At odd n there is no self-conjugate order, so every order above zero is
    paired and the Nyquist special case must NOT be taken. The review
    measured the pre-fix body returning 3.7e-30 against 1.5707963 here.
    """
    lattice = build_lattice(tip_radius=1.0, stations=(0.0,), lateral_radius=None, n_psi=15)
    shape = (1, lattice.n_r, lattice.n_psi)
    psi = lattice.psi[None, None, :]
    signal = np.broadcast_to(np.cos(7.0 * psi), shape).copy()
    ds = cylindrical_components(
        lattice_fields(
            lattice,
            fields={"u": 1.0 + signal, "v": np.zeros(shape), "w": signal},
        )
    )
    quadrature = float(transverse_flux(ds, RHO, component="w", method="quadrature")[0])
    harmonic = float(transverse_flux(ds, RHO, component="w", method="harmonic")[0])
    assert abs(quadrature) > 1.0
    assert harmonic == pytest.approx(quadrature, rel=1e-10)


def test_the_swirl_channel_sees_a_hole_that_is_only_in_the_transverse_field():
    """The residual the first fixture could not reach.

    crossflow_kinetic_energy["swirl"] is built on an azimuthal MEAN of
    v_theta, not on plane_integral, so it is the one channel whose
    correctness depends on that reduction alone. With u left finite, the
    pre-fix body returned swirl = 2.4e-32, a confident "no swirl", from a
    plane with a dead sample, while the sibling "total" was NaN. The two
    numbers disagreed in the same returned object and nothing said so.
    """
    ds = _holed(holes=("v", "w"))
    energy = crossflow_kinetic_energy(ds, RHO)
    assert np.isnan(np.asarray(energy.swirl)).any(), (
        f"swirl came back {np.asarray(energy.swirl)!r} from a holed plane"
    )


def test_mass_closure_sees_one_dead_station_among_live_ones():
    """The measurement the commit message quoted, now reproducible.

    relative_spread is the scalar gate G1 judges. With one dead station of
    three, the pre-fix mean and max skipped it and the gate reported a
    plausible spread over the survivors; the quoted 0.0 is only reachable
    with at least two stations, which the first fixture did not have.
    """
    ds = _holed(holes=("u",))
    closure = mass_closure(ds, RHO)
    mdot = np.asarray(closure.mdot)
    assert np.isnan(mdot[1]), "the holed station should carry NaN"
    assert np.isfinite(mdot[0]) and np.isfinite(mdot[2]), (
        "the live stations must stay live, or this proves nothing about skipping"
    )
    assert np.isnan(np.asarray(closure.relative_spread)), (
        "the closure judgment was taken over the surviving stations"
    )


def _lateral_fixture():

    lattice = build_lattice(tip_radius=1.0, stations=(-2.0, 0.5, 1.0, 2.0), lateral_radius=2.6)
    n_lat = len(lattice.lateral_stations)
    shape = (n_lat, lattice.n_psi)
    lateral = {
        "v": np.full(shape, 0.5),
        "w": np.full(shape, 0.25),
        "v_r": np.full(shape, 1.0),
        "p_prime": np.full(shape, 2.0),
    }
    base = {name: np.zeros(_shape_of(lattice)) for name in ("u", "v", "w", "p_prime")}
    return lattice_fields(lattice, fields=base, lateral=lateral)


def _shape_of(lattice):
    return (len(lattice.stations), lattice.n_r, lattice.n_psi)


def test_lateral_flux_reversed_bounds_are_refused_never_a_silent_value():
    """A swapped inlet and outlet is refused, neither 0 nor a guessed sign."""
    fields = _lateral_fixture()
    first, last = float(fields.lateral_stations[0]), float(fields.lateral_stations[-1])
    forward = lateral_flux(fields, rho=RHO, inlet=first, outlet=last)
    assert forward != 0.0
    with pytest.raises(FarfieldInputError, match="upstream of the inlet"):
        lateral_flux(fields, rho=RHO, inlet=last, outlet=first)
    assert lateral_flux(fields, rho=RHO, inlet=first, outlet=first) == 0.0


def test_lateral_flux_accepts_bounds_within_roundoff_like_the_annular_lookup():
    """A station off by roundoff is found, as station_index finds it."""
    fields = _lateral_fixture()
    first, last = float(fields.lateral_stations[0]), float(fields.lateral_stations[-1])
    exact = lateral_flux(fields, rho=RHO, inlet=first, outlet=last)
    nudged = lateral_flux(fields, rho=RHO, inlet=first, outlet=last * (1 + 4e-16) + 1e-15)
    assert nudged == pytest.approx(exact, rel=1e-12)


def test_mass_closure_states_a_zero_mean_instead_of_an_unlabelled_nan():
    """Zero mean with zero spread differs from zero mean with nonzero spread."""
    fields = uniform_fields()
    fields["u"] = np.zeros(SHAPE)
    closed = mass_closure(make_dataset(fields), RHO)
    assert closed.status == "zero-mean-zero-spread"
    assert closed.absolute_spread == 0.0
    assert np.isnan(closed.relative_spread)

    signs = np.array([1.0, -1.0, 1.0, -1.0])[:, None, None]
    fields["u"] = np.full(SHAPE, V_INF) * signs
    open_ = mass_closure(make_dataset(fields), RHO)
    assert open_.status == "zero-mean-nonzero-spread"
    assert open_.absolute_spread == pytest.approx(float(np.abs(open_.mdot).max()))
    assert open_.absolute_spread > 0.0
    assert mass_closure(make_dataset(uniform_fields()), RHO).status == "ok"
    # The zero mean is stated, quietly: no numpy warning rides on the NaN.
    fields["u"] = np.zeros(SHAPE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(mass_closure(make_dataset(fields), RHO).relative_spread)


# ---- shape and finiteness checks of the far-field core ----


def test_f094_f095_f096_mismatched_grids_are_refused():
    lattice = build_lattice(tip_radius=1.0, stations=(0.0,), lateral_radius=None)
    ds = lattice_fields(lattice, fields={"u": np.ones((1, lattice.n_r, lattice.n_psi))})
    with pytest.raises(FarfieldInputError, match="trailing axes"):
        plane_integral(ds, np.ones((1, lattice.n_r, lattice.n_psi - 1)))
    y = np.linspace(0.0, 1.0, 5)
    with pytest.raises(FarfieldInputError, match="partial column"):
        grid_integral(y, z=y, integrand=np.ones((5, 4)))
    with pytest.raises(FarfieldInputError, match="azimuth"):
        azimuthal_harmonics(np.ones(16), psi=np.linspace(0.0, 2 * np.pi, 8, endpoint=False))


def test_f101_infinite_samples_give_nan_integrals():
    y = np.linspace(0.0, 1.0, 5)
    field = np.ones((5, 5))
    field[2, 2] = np.inf
    assert math.isnan(grid_integral(y, z=y, integrand=field))
    lattice = build_lattice(tip_radius=1.0, stations=(0.0,), lateral_radius=None)
    ds = lattice_fields(lattice, fields={"u": np.ones((1, lattice.n_r, lattice.n_psi))})
    bad = np.ones((1, lattice.n_r, lattice.n_psi))
    bad[0, 0, 0] = np.inf
    assert np.isnan(plane_integral(ds, bad)[0])


@pytest.mark.parametrize("function,args", [(to_counts, (1.0,)), (spurious_diagnostic, (2.0, 1.0))])
def test_a_missing_reference_area_is_refused_with_the_remedy(function, args):
    with pytest.raises(TypeError, match="s_ref is required") as caught:
        function(*args, rho_inf=1.2, v_inf=20.0)
    assert "reference area" in str(caught.value)
    assert "m2" in str(caught.value)
    assert "no safe default" in str(caught.value)
    assert signature(function).parameters["s_ref"].default is Parameter.empty
