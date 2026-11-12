"""P0000: QPROP section polar fit, propeller file, and output table parsing.

Oracles are hand arithmetic on analytic polars and on a stand-in QPROP table,
never the production formulas.
"""

from __future__ import annotations

import math

import pytest

from cfdbender.solvers.qprop import _output, _polar, _prop

ALPHAS = [float(a) for a in range(-8, 17)]
CL0, SLOPE, CD0, CD2, CLCD0 = 0.1, 0.11, 0.010, 0.02, 0.32


def synthetic_polar():
    rows = []
    for alpha in ALPHAS:
        cl = CL0 + SLOPE * alpha
        rows.append((alpha, cl, CD0 + CD2 * (cl - CLCD0) ** 2))
    return rows


def xfoil_text(rows):
    lines = [
        " XFOIL         Version 6.99",
        "",
        "   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr",
        "  ------ -------- --------- --------- -------- -------- --------",
    ]
    lines += [
        f" {a:7.3f} {c:8.4f} {d:9.5f}   0.00046  -0.0000   0.6870   0.6870" for a, c, d in rows
    ]
    return "\n".join(lines) + "\n"


def test_p0000_qprop_polar_rows_are_read_by_value_not_position():
    rows = _polar.read_polar_rows(xfoil_text(synthetic_polar()))
    assert len(rows) == len(ALPHAS)
    assert rows[0] == pytest.approx(
        (-8.0, CL0 - 8 * SLOPE, CD0 + CD2 * (CL0 - 8 * SLOPE - CLCD0) ** 2), abs=1e-4
    )
    assert _polar.read_polar_rows("no table here\n") == []


def test_p0000_qprop_polar_fit_recovers_the_analytic_parameters():
    fit = _polar.fit_polar(synthetic_polar())
    assert fit is not None
    assert fit.cl0 == pytest.approx(CL0, abs=1e-9)
    assert fit.cla == pytest.approx(SLOPE * 180 / math.pi, rel=1e-9)
    assert fit.clmin == pytest.approx(CL0 - 8 * SLOPE) and fit.clmax == pytest.approx(
        CL0 + 16 * SLOPE
    )
    assert fit.cd0 == pytest.approx(CD0) and fit.clcd0 == pytest.approx(CLCD0)
    assert fit.cd2u == pytest.approx(CD2, rel=1e-9) and fit.cd2l == pytest.approx(CD2, rel=1e-9)


def test_p0000_qprop_polar_fit_needs_three_linear_range_points():
    assert _polar.fit_polar([(-8.0, -0.8, 0.02), (10.0, 1.2, 0.03), (12.0, 1.4, 0.04)]) is None
    assert _polar.fit_polar([]) is None


def test_p0000_qprop_polar_script_is_deterministic_xfoil_keystrokes():
    text = _polar.polar_script(
        reynolds=97221.4,
        mach=0.1234,
        alpha_start=-8,
        alpha_end=16,
        alpha_step=1,
        ncrit=9,
        iterations=150,
    )
    assert text == (
        "PLOP\nG\n\nLOAD airfoil.dat\n\nPANE\nOPER\nVISC 97221\nMACH 0.123\nITER 150\n"
        "PACC\npolar.txt\n\nASEQ -8 16 1\n\nQUIT\n"
    )
    assert "VPAR\nN 7\n\nPACC" in _polar.polar_script(
        reynolds=1e5, mach=0.1, alpha_start=-4, alpha_end=10, alpha_step=0.5, ncrit=7, iterations=80
    )


def station(r, chord, beta):
    fit = _polar.fit_polar(synthetic_polar())
    return _prop.Station(r, chord, beta, fit, 100000.0)


def test_p0000_qprop_prop_file_layout_and_beta_offset():
    stations = [station(0.05, 0.03, 50.0), station(0.1, 0.028, 35.0), station(0.2, 0.02, 20.0)]
    text = _prop.render_prop("study", 3, 0.2, stations, beta_add=2.5)
    lines = text.splitlines()
    assert lines[0] == "study"
    assert lines[2].split()[:2] == ["3", "0.2000"]
    assert any(line.split()[:3] == ["0.0", "0.0", "2.50"] for line in lines)
    data = [line.split() for line in lines if line and line[0] != "#" and len(line.split()) == 13]
    assert [row[:3] for row in data] == [
        ["0.05000", "0.03000", "50.000"],
        ["0.10000", "0.02800", "35.000"],
        ["0.20000", "0.02000", "20.000"],
    ]
    assert data[0][3:5] == ["0.1000", f"{SLOPE * 180 / math.pi:.4f}"]
    assert data[0][11:] == ["100000", "-0.5"]
    assert text == _prop.render_prop("study", 3, 0.2, stations, beta_add=2.5)


def test_p0000_qprop_qcon_file_carries_the_flight_condition():
    assert _prop.render_qcon(1.2, 1.8e-5, 340.0).split("!")[0].split() == ["1.2"]
    lines = _prop.render_qcon(1.2, 1.8e-5, 340.0).splitlines()
    assert [line.split()[0] for line in lines] == ["1.2", "1.80000e-05", "340"]


HEADER = (
    "#  V(m/s)    rpm      Dbeta      T(N)       Q(N-m)    Pshaft(W)    Volts     Amps    "
    "effmot   effprop   adv       CT          CP        DV(m/s)   eff     Pelec       Pprop"
    "        cl_avg  cd_avg"
)


def qprop_text(order=None):
    values = {
        "V(m/s)": "10.000", "rpm": "6000.", "Dbeta": "0.000", "T(N)": "18.25",
        "Q(N-m)": "0.5477", "Pshaft(W)": "344.1", "Volts": "0.", "Amps": "0.",
        "effmot": "1.0000", "effprop": "0.5304", "adv": "0.09474", "CT": "0.3016E-01",
        "CP": "0.5387E-02", "DV(m/s)": "10.8819", "eff": "0.5304", "Pelec": "344.1",
        "Pprop": "182.5", "cl_avg": "1.2755", "cd_avg": "0.3354E-01",
    }  # fmt: skip
    names = list(values) if order is None else order
    head = "#  " + "   ".join(names)
    row = "#  " + "   ".join(values[n] for n in names)
    return (
        f"# QPROP Version 1.22\n#\n{head}\n{row}\n#\n#  radius   chord   beta\n  0.03 0.02 50.5\n"
    )


@pytest.mark.parametrize("swap", [False, True])
def test_p0000_qprop_table_is_parsed_by_header_names(swap):
    names = (
        "V(m/s) rpm Dbeta T(N) Q(N-m) Pshaft(W) Volts Amps effmot effprop adv CT CP "
        "DV(m/s) eff Pelec Pprop cl_avg cd_avg"
    ).split()
    if swap:
        i, j = names.index("CT"), names.index("CP")
        names[i], names[j] = names[j], names[i]
    table = _output.parse_table(qprop_text(names))
    assert table["CT"] == pytest.approx(0.03016) and table["CP"] == pytest.approx(0.005387)
    assert table["T(N)"] == 18.25 and table["rpm"] == 6000.0
    assert _output.version(qprop_text()) == "1.22"


def test_p0000_qprop_table_refuses_missing_or_malformed_rows():
    assert _output.parse_table("no table\n") == {}
    broken = qprop_text().replace("0.3016E-01", "*****")
    assert _output.parse_table(broken) == {}


def test_p0000_qprop_coefficients_in_both_normalisations():
    table = _output.parse_table(qprop_text())
    radius, rho = 0.168, 1.225
    got = _output.coefficients(table, rho=rho, radius=radius)
    n, diameter = 6000.0 / 60.0, 2 * radius
    assert got["J"] == pytest.approx(10.0 / (n * diameter))
    assert got["CT"] == pytest.approx(18.25 / (rho * n**2 * diameter**4))
    assert got["CP"] == pytest.approx(344.1 / (rho * n**3 * diameter**5))
    assert got["CT_QPROP"] == pytest.approx(0.03016) and got["CP_QPROP"] == pytest.approx(0.005387)
    assert got["EFFICIENCY"] == pytest.approx(18.25 * 10.0 / 344.1)
    assert (got["T"], got["Q"], got["P"]) == (18.25, 0.5477, 344.1)


def test_p0000_qprop_mutant_swapped_ct_cp_is_detected(monkeypatch):
    """A parser that swaps the CT and CP columns must fail the oracle comparison."""
    expected = {"CT_QPROP": 0.03016, "CP_QPROP": 0.005387}

    def check(table):
        got = _output.coefficients(table, rho=1.225, radius=0.168)
        assert got["CT_QPROP"] == pytest.approx(expected["CT_QPROP"])
        assert got["CP_QPROP"] == pytest.approx(expected["CP_QPROP"])

    check(_output.parse_table(qprop_text()))
    real = _output.parse_table

    def mutant(text):
        table = real(text)
        table["CT"], table["CP"] = table["CP"], table["CT"]
        return table

    monkeypatch.setattr(_output, "parse_table", mutant)
    with pytest.raises(AssertionError):
        check(_output.parse_table(qprop_text()))
