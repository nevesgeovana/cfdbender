"""Offline QPROP stand-in: prints an operating-point table from analytic coefficients.

Thrust and power follow CT = 0.12 - 0.06 J and CP = 0.04 + 0.01 J in the standard
normalisation, using the radius from the propeller file and rho from qcon.def.
CFDBENDER_QPROP_FAKE_MODE: converged (default), failed (no table) or nonzero.
"""

import math
import os
import sys
import time
from pathlib import Path


def main():
    prop, motor, speed, rpm = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    assert motor == "none"
    lines = Path(prop).read_text(encoding="ascii").splitlines()
    blades, radius = lines[2].split("!")[0].split()
    stations = [x for x in lines if x and x[0] != "#" and len(x.split()) == 13]
    assert int(blades) >= 1 and len(stations) >= 1
    rho = float(Path("qcon.def").read_text(encoding="ascii").splitlines()[0].split("!")[0])
    print(" QPROP Version 1.22 (stand-in)")
    mode = os.environ.get("CFDBENDER_QPROP_FAKE_MODE", "converged")
    if mode == "timeout":
        time.sleep(30)
    if mode == "failed":
        print(" Convergence failed")
        return 0
    if mode == "nonzero":
        return 3
    n, d = rpm / 60.0, 2.0 * float(radius)
    j = speed / (n * d)
    thrust = (0.12 - 0.06 * j) * rho * n**2 * d**4
    power = (0.04 + 0.01 * j) * rho * n**3 * d**5
    torque = power / (2.0 * math.pi * n)
    ct = thrust / (
        0.5 * rho * (2 * math.pi * n * float(radius)) ** 2 * math.pi * float(radius) ** 2
    )
    cp = power / (0.5 * rho * (2 * math.pi * n * float(radius)) ** 3 * math.pi * float(radius) ** 2)
    adv = speed / (2 * math.pi * n * float(radius))
    names = (
        "V(m/s) rpm Dbeta T(N) Q(N-m) Pshaft(W) Volts Amps effmot effprop adv CT CP "
        "DV(m/s) eff Pelec Pprop cl_avg cd_avg"
    ).split()
    values = [speed, rpm, 0.0, thrust, torque, power, 0.0, 0.0, 1.0, thrust * speed / power, adv]
    values += [ct, cp, 0.0, thrust * speed / power, power, thrust * speed, 1.0, 0.03]
    print("#")
    print("#  " + "   ".join(names))
    print("#  " + "   ".join(f"{v:.6g}" for v in values))
    print("#")
    print("#  radius   chord   beta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
