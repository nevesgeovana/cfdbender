"""The rotor table of a matrix: the whole rotor's quantities, one row per rotating-frame point.

``post/<matrix>/rotor.csv`` is written only when a point's result carries the ``rotor``
mapping an SU2 steady rotating-frame point states
(:func:`cfdbender.solvers.su2._config.rotor_point`).
It is a product of its own, beside the polar files, so the documented columns of those files
stay as they are and a matrix without a rotor writes none. The columns are::

    POL  POINT  ROTOR  BLADES  MONITORED_BLADES  D  VINF  RHO  QINF  RPM  N  J  T  Q
    CT  CQ  CP  ETA

with ``D`` the rotor diameter in m, ``VINF`` the free-stream speed in m/s, ``RHO`` the
density in kg/m3, ``QINF`` the dynamic pressure SU2 divides by in Pa, ``RPM`` the rotation
speed in rev/min and ``N`` in rev/s, ``J = VINF / (N D)``, ``T`` the thrust along the shaft in
N and ``Q`` the shaft torque in N m of the whole rotor, ``CT = T / (RHO N^2 D^4)``,
``CQ = Q / (RHO N^2 D^5)``, ``CP = 2 pi CQ`` and ``ETA = J CT / CP``. A value that does not
exist is the token NA.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

#: The columns of rotor.csv and the key of the result's rotor mapping each one reads.
ROTOR_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ROTOR", "rotor"),
    ("BLADES", "blades"),
    ("MONITORED_BLADES", "monitored_blades"),
    ("D", "diameter_m"),
    ("VINF", "velocity_m_s"),
    ("RHO", "density_kg_m3"),
    ("QINF", "dynamic_pressure_pa"),
    ("RPM", "rpm"),
    ("N", "n_rps"),
    ("J", "advance_ratio"),
    ("T", "thrust_n"),
    ("Q", "torque_nm"),
    ("CT", "ct"),
    ("CQ", "cq"),
    ("CP", "cp"),
    ("ETA", "efficiency"),
)


def rotor_table(
    folder: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    write: Callable[[Path, str], Path],
    cell: Callable[[object], str],
) -> tuple[Path, ...]:
    """Write ``rotor.csv`` for the points that state a rotor, or nothing.

    Parameters
    ----------
    folder : Path
        ``post/<matrix>``.
    rows : sequence of mapping
        The campaign summary rows, in record order.
    write, cell : callable
        The writer of a derived table and the cell formatter of the polar files.

    Returns
    -------
    tuple of Path
        The one written path, or an empty tuple when no point has a rotor.
    """
    lines: list[str] = []
    for row in rows:
        result = cast(Mapping[str, object], row.get("result") or {})
        rotor = result.get("rotor")
        if not isinstance(rotor, dict):
            continue
        name = str(row["point_id"]).split("/", 1)[1]
        cells = [str(row["pol"]), name, *(cell(rotor.get(key)) for _, key in ROTOR_COLUMNS)]
        lines.append(",".join(cells))
    if not lines:
        return ()
    header = ",".join(["POL", "POINT", *(name for name, _ in ROTOR_COLUMNS)])
    text = chr(10).join([header, *lines]) + chr(10)
    return (write(folder / "rotor.csv", text),)
