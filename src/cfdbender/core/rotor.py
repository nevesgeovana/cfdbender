"""The rotor of a matrix row: its reference declaration, its speed and its loads.

The REF file declares a rotor as a pyflightstream reference file does, a table of its own
with ``kind = "rotor"`` (the table name is the rotor's alias)::

    [ROTOR]
    kind = "rotor"
    alias = "ROTOR"            # optional; when stated it equals the table name
    x_m = 0.0                  # the hub in the mesh frame, m (default 0)
    y_m = 0.0
    z_m = 0.0
    axis = "X"                 # the shaft: the direction thrust is counted positive along
    rpm_sign = -1              # 1: turns right-handed about the shaft; -1: left-handed
    diameter_m = 0.4064        # rotor diameter D, m
    families_general = ["SPINNER"]
    families_blades = ["Blade1", "Blade2", "Blade3", "Blade4", "Blade5", "Blade6"]
    blade1 = { azimuth_deg = 0.0, zero = "Z" }

The blade count is the length of ``families_blades``. ``blade1`` says where blade one sits
at step zero, ``azimuth_deg`` measured from the axis ``zero`` (a letter with at most one
sign, one of the two axes that are not the shaft) right-handed about the shaft; blade ``k``
of ``B`` sits ``(k - 1) 360 / B`` further on.

The signs are pyflightstream's (``docs/definitions/rotor-coefficients.md``): with ``a`` the
unit shaft, ``F`` the whole-rotor force and ``M`` its moment about the hub
(``M_hub = M_ref + (r_ref - r_hub) x F``), ``n`` the speed in rev/s signed by ``rpm_sign``::

    T = F . a        Q = M_hub . a
    CT = T / (rho n^2 D^4)   CQ = Q / (rho n^2 D^5)   CP = 2 pi CQ sign(n)   ETA = J CT / CP

The in-plane coefficients ``CN``, ``CS``, ``CMN`` and ``CMS`` are the force and the hub
moment along the rotor's normal ``N`` (the part of the mesh frame's up, ``+z``, square to the
shaft) and side ``S = N x a``, over the same scales; all four are NA for a shaft along up.

The whole-rotor loads of a point solved on one of ``m`` periodic sectors are the sum over the
``m`` copies of the sector's force and hub moment, copy ``j`` turned about the shaft through
the hub by ``360 j / m`` degrees; they are never the sector's loads times ``m``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from cfdbender.core.rotor_coefficients import (
    advance_ratio,
    force_scale,
    moment_scale,
    power_from_torque,
    rotor_coefficients,
)
from cfdbender.core.run_types import CONTROLLED_KEYS
from cfdbender.core.unsteady_clock import BladeDatum, blade_datum, blade_one_azimuth

Vector = tuple[float, float, float]

#: The keys of a rotor declaration of the REF file.
READ_KEYS = (
    "kind",
    "alias",
    "x_m",
    "y_m",
    "z_m",
    "axis",
    "rpm_sign",
    "diameter_m",
    "families_general",
    "families_blades",
    "blade1",
)
#: The mesh frame's up direction, which the rotor's normal axis is taken from.
REFERENCE_UP: Vector = (0.0, 0.0, 1.0)
#: A shaft whose part square to up is shorter than this has no normal and no side axis.
ROTOR_NORMAL_FLOOR = 1e-9
_AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


@dataclass(frozen=True)
class RotorBlock:
    """One validated rotor declaration of the REF file.

    Attributes
    ----------
    name : str
        The declaration's table name, which is its alias: ``ROTOR`` in ``[ROTOR]``.
    diameter_m : float
        The rotor diameter D.
    blades : int or None
        The length of ``families_blades``, None when the table has none.
    origin : tuple of float
        The hub in the mesh frame.
    axis : tuple of float
        The unit shaft; thrust is counted positive along it.
    rpm_sign : int
        1 when the rotor turns right-handed about the shaft, -1 when left-handed.
    blade1 : BladeDatum
        Where blade one sits at step zero.
    """

    name: str
    diameter_m: float
    blades: int | None
    origin: Vector
    axis: Vector
    rpm_sign: int
    blade1: BladeDatum = field(default_factory=BladeDatum)

    @property
    def alias(self) -> str:
        """The word every column of the rotor table carries, the table's name."""
        return self.name

    def blade_azimuth_deg(self, blade: int, *, step: float, steps_per_revolution: float) -> float:
        """Return where blade ``blade`` (1-based) is at ``step``, in degrees in [0, 360).

        ``(blade1.azimuth_deg + sense step 360 / steps_per_revolution + (blade - 1) 360 / B)``
        modulo 360, the sense being ``rpm_sign``.
        """
        count = self.blades or 1
        if not 1 <= blade <= count:
            raise ValueError(f"rotor {self.name} has {count} blades; blade {blade} is not one")
        one = blade_one_azimuth(
            step,
            datum_deg=self.blade1.azimuth_deg,
            sense=float(self.rpm_sign),
            steps_per_revolution=steps_per_revolution,
        )
        return (one + (blade - 1) * 360.0 / count) % 360.0 + 0.0


def _real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"REF file {label} must be a finite number")
    return float(value)


def _family_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(i, str) and i.strip() for i in value):
        raise ValueError(f"REF file {label} must be a list of family names")
    return value


#: |shaft . datum| above which blade one's datum lies along the shaft (about 5 degrees).
DATUM_ALIGNMENT_LIMIT = 0.9961946980917455
#: The largest angle between an installed shaft and the axis it names, in degrees.
SHAFT_TILT_LIMIT_DEG = 45.0


def shaft_axis(axis: object, name: str) -> tuple[str, Vector]:
    """Read a rotor's ``axis``: a letter, or ``{ shaft, vector_wrt_global }``, as pyflightstream.

    The letter X, Y or Z is the positive global axis. The table names the shaft letter and the
    installed direction, which is normalized and must lie within 45 degrees of the positive
    axis the letter names. Either way the unit vector is the direction thrust is counted
    positive along, and ``rpm_sign`` is the right-hand rule about it.

    Returns
    -------
    tuple
        The shaft letter and the unit vector.
    """
    if isinstance(axis, str):
        token = axis.strip().upper()
        if token not in _AXES:
            raise ValueError(
                f"REF file {name}: axis = {axis!r} is not an axis; write X, Y or Z, or a table "
                "with shaft and vector_wrt_global for a rotor installed at an angle"
            )
        return token, _AXES[token]
    if isinstance(axis, list):
        nearest = "XYZ"[max(range(3), key=lambda i: abs(float(axis[i])))] if len(axis) == 3 else "X"
        raise ValueError(
            f"REF file {name}: axis = {axis!r} did not say which frame axis is the shaft; "
            f'write axis = {{ shaft = "{nearest}", vector_wrt_global = {axis!r} }}'
        )
    if not isinstance(axis, Mapping):
        raise ValueError(f"REF file {name}: axis = {axis!r} is not an axis")
    unknown = sorted(set(axis) - {"shaft", "vector_wrt_global"})
    if unknown or "shaft" not in axis or "vector_wrt_global" not in axis:
        raise ValueError(
            f"REF file {name}: the axis table states shaft and vector_wrt_global, and only them"
        )
    shaft = axis["shaft"]
    if shaft not in _AXES:
        raise ValueError(f'REF file {name}: axis.shaft must be "X", "Y" or "Z"')
    stated = axis["vector_wrt_global"]
    if not isinstance(stated, list) or len(stated) != 3:
        raise ValueError(f"REF file {name}: axis.vector_wrt_global must be three numbers")
    components = [_real(value, f"{name}.axis.vector_wrt_global") for value in stated]
    scale = max(abs(value) for value in components)
    if scale == 0:
        raise ValueError("vector_wrt_global has zero length; state the shaft direction")
    scaled = [value / scale for value in components]
    length = math.hypot(*scaled)
    component = scaled["XYZ".index(shaft)] / length
    angle = math.degrees(math.acos(max(-1.0, min(1.0, component))))
    if angle > SHAFT_TILT_LIMIT_DEG + 1e-10:
        raise ValueError(
            f"REF file {name}: vector_wrt_global is {angle:.2f} degrees from shaft {shaft}; "
            "at most 45 degrees is accepted. Choose the matching shaft letter"
        )
    x, y, z = (value / length for value in scaled)
    return shaft, (x, y, z)


def rotor_block(reference: Mapping[str, object]) -> RotorBlock | None:
    """Return the REF file's rotor declaration, or None when it holds none.

    Parameters
    ----------
    reference : mapping
        The parsed REF file.

    Returns
    -------
    RotorBlock or None
        The validated declaration: the one table whose ``kind`` is ``"rotor"``.

    Raises
    ------
    ValueError
        More than one rotor is declared, a required key is missing, a key is unknown or of
        the wrong type, the diameter is not positive, the axis is invalid, the alias differs
        from the table name, or blade one's datum is not a usable axis.
    """
    found = {
        key: value
        for key, value in reference.items()
        if isinstance(value, dict) and value.get("kind") == "rotor"
    }
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            f"REF file declares {len(found)} rotors ({', '.join(sorted(found))}): multi-rotor "
            "rotating-frame runs are not supported yet; declare one"
        )
    ((name, block),) = found.items()
    unknown = sorted(
        key for key in block if key not in READ_KEYS and re.fullmatch(r"blade[0-9]+", key) is None
    )
    if unknown:
        raise ValueError(
            f"REF file {name} has unknown keys: {', '.join(unknown)} "
            f"(the keys are {', '.join(READ_KEYS)})"
        )
    for key in ("diameter_m", "axis"):
        if key not in block:
            raise ValueError(f"REF file {name} needs {key}")
    alias = block.get("alias", name)
    if not isinstance(alias, str) or alias.strip().casefold() != name.casefold():
        raise ValueError(
            f"REF file {name}.alias is {alias!r}: a rotor's alias is its table name; write "
            f'alias = "{name}" or leave it out'
        )
    diameter = _real(block["diameter_m"], f"{name}.diameter_m")
    if diameter <= 0.0:
        raise ValueError(f"REF file {name}.diameter_m must be positive")
    shaft, vector = shaft_axis(block["axis"], name)
    sign = block.get("rpm_sign", 1)
    if type(sign) is not int or sign not in (1, -1):
        raise ValueError(f"REF file {name}.rpm_sign must be 1 or -1")
    if "families_general" in block:
        _family_list(block["families_general"], f"{name}.families_general")
    blades: int | None = None
    if "families_blades" in block:
        blades = len(_family_list(block["families_blades"], f"{name}.families_blades"))
        if not blades:
            raise ValueError(f"REF file {name}.families_blades must be a list of family names")
    datum = blade_datum(block)
    if datum.zero.lstrip("+-") == shaft:
        raise ValueError(
            f"REF file {name}: blade1.zero {datum.zero} equals shaft {shaft}; "
            "choose one of the two remaining axes"
        )
    datum_axis = _AXES[datum.zero.lstrip("+-")]
    alignment = abs(_dot(vector, datum_axis))
    if alignment > DATUM_ALIGNMENT_LIMIT:
        degrees = math.degrees(math.acos(min(1.0, alignment)))
        raise ValueError(
            f"REF file {name}: blade1 measures its azimuth from {datum.zero}, which is "
            f"{degrees:.2f} degrees from the shaft direction {vector}. An azimuth measured from "
            "a datum that nearly lies along the shaft locates nothing, so it is refused rather "
            "than reported. Choose a datum square to the disk"
        )
    origin = (
        _real(block.get("x_m", 0.0), f"{name}.x_m"),
        _real(block.get("y_m", 0.0), f"{name}.y_m"),
        _real(block.get("z_m", 0.0), f"{name}.z_m"),
    )
    return RotorBlock(name, diameter, blades, origin, vector, sign, datum)


def rotor_speed(
    block: RotorBlock, condition: Mapping[str, float], velocity: float
) -> tuple[float, str]:
    """Return the rotor's speed in rev/s, a magnitude, and the key it came from.

    ``RPM / 60`` when the condition states ``RPM``; else ``V / (J D)`` with ``J`` the
    condition's ``ADVANCE_RATIO``, ``V`` the free-stream speed and ``D`` the diameter.

    Raises
    ------
    ValueError
        The condition states both or neither, or a value that is not positive.
    """
    stated = [key for key in ("ADVANCE_RATIO", "RPM") if key in condition]
    if len(stated) != 1:
        raise ValueError(
            "FLIGHT_CONDITION states its rotor speed twice: ADVANCE_RATIO and RPM; keep one"
            if stated
            else "FLIGHT_CONDITION states no rotor speed: state 'ADVANCE_RATIO: <J>' or "
            "'RPM: <rev/min>'"
        )
    (key,) = stated
    value = float(condition[key])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"FLIGHT_CONDITION {key} must be positive: a rotor speed is a magnitude, and its "
            "hand is the rotor's rpm_sign"
        )
    if key == "RPM":
        return value / 60.0, key
    return velocity / (value * block.diameter_m), key


def rotation_rate(block: RotorBlock, advance_ratio: float, velocity: float) -> tuple[Vector, float]:
    """Return the rotation vector in rad/s and the speed n in rev/s of an advance ratio.

    Parameters
    ----------
    block : RotorBlock
        The rotor.
    advance_ratio : float
        J = V / (n D), positive.
    velocity : float
        The free-stream speed V in m/s, positive.

    Returns
    -------
    tuple
        ``rpm_sign 2 pi n axis_hat`` and ``n = V / (J D)``.
    """
    n = velocity / (advance_ratio * block.diameter_m)
    return rotation_vector(block, n), n


def rotation_vector(block: RotorBlock, rps: float) -> Vector:
    """Return ``rpm_sign 2 pi n axis_hat`` in rad/s for a speed ``n`` in rev/s."""
    rate = 2.0 * math.pi * rps * block.rpm_sign
    x, y, z = (rate * c + 0.0 for c in block.axis)
    return (x, y, z)


def _cross(a: Sequence[float], b: Sequence[float]) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def turn(vector: Sequence[float], axis: Sequence[float], angle: float) -> Vector:
    """Return a vector turned by ``angle`` radians right-handed about a unit axis (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)
    cross = _cross(axis, vector)
    along = _dot(axis, vector) * (1.0 - c)
    x, y, z = (v * c + k * s + a * along for v, k, a in zip(vector, cross, axis, strict=True))
    return (x, y, z)


def periodic_sum(
    force: Sequence[float],
    moment: Sequence[float],
    axis: Sequence[float],
    copies: int,
) -> tuple[Vector, Vector]:
    """Return the force and moment of ``copies`` periodic copies of a sector, summed.

    Parameters
    ----------
    force, moment : sequence of float
        The sector's force and its moment about a point on the axis.
    axis : sequence of float
        The unit vector of the rotation axis through that point.
    copies : int
        The number of identical sectors that make the whole; copy ``j`` is the sector turned
        by ``360 j / copies`` degrees about the axis.

    Returns
    -------
    tuple
        The summed force and the summed moment about the same point.
    """
    f = [0.0, 0.0, 0.0]
    m = [0.0, 0.0, 0.0]
    for j in range(copies):
        angle = 2.0 * math.pi * j / copies
        for total, vector in ((f, force), (m, moment)):
            turned = turn(vector, axis, angle)
            for i in range(3):
                total[i] += turned[i]
    return (f[0], f[1], f[2]), (m[0], m[1], m[2])


NOT_APPLICABLE = "NA"


@dataclass(frozen=True)
class RotorLoads:
    """What an adapter supplies for one rotor of one point.

    Attributes
    ----------
    force_n : tuple of float
        The monitored force in N, in the mesh frame.
    moment_nm : tuple of float
        The monitored moment in N m about ``about_m``, in the mesh frame.
    about_m : tuple of float
        The point the moment is about, in the mesh frame.
    speed_m_s : float
        The free-stream speed of the point as the solver ran it.
    density_kg_m3 : float or None
        The free-stream density as the solver ran it; None when unknown.
    """

    force_n: Vector
    moment_nm: Vector
    about_m: Vector
    speed_m_s: float
    density_kg_m3: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the loads as a record stores them."""
        return {
            "force_n": list(self.force_n),
            "moment_nm": list(self.moment_nm),
            "about_m": list(self.about_m),
            "speed_m_s": self.speed_m_s,
            "density_kg_m3": self.density_kg_m3,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RotorLoads:
        """Read the loads back from a record."""

        def vector(key: str) -> Vector:
            x, y, z = (float(v) for v in data[key])  # type: ignore[attr-defined]
            return (x, y, z)

        density = data.get("density_kg_m3")
        return cls(
            vector("force_n"),
            vector("moment_nm"),
            vector("about_m"),
            float(data["speed_m_s"]),  # type: ignore[arg-type]
            None if density is None else float(density),  # type: ignore[arg-type]
        )


def in_plane_axes(shaft: Sequence[float]) -> tuple[Vector, Vector] | None:
    """Return the rotor's normal ``N`` and side ``S`` unit vectors, or None for a shaft along up.

    ``N`` is the part of :data:`REFERENCE_UP` square to the shaft, normalized; ``S = N x a``.
    """
    along = _dot(REFERENCE_UP, shaft)
    square = tuple(u - along * a for u, a in zip(REFERENCE_UP, shaft, strict=True))
    reach = math.sqrt(_dot(square, square))
    if reach <= ROTOR_NORMAL_FLOOR:
        return None
    nx, ny, nz = (c / reach for c in square)
    normal = (nx, ny, nz)
    return normal, _cross(normal, shaft)


#: The rotor row's coefficient columns, before the alias suffix, in the table's order.
ROTOR_ROW_COLUMNS = ("RPM", "DIAMETER", "J", "CT", "CQ", "CP", "ETA", "CN", "CS", "CMN", "CMS")


def rotor_row(
    block: RotorBlock, loads: RotorLoads, *, rps: float, copies: int = 1
) -> dict[str, float | str]:
    """Return the rotor-table quantities of one rotor from the loads an adapter supplies.

    Parameters
    ----------
    block : RotorBlock
        The rotor.
    loads : RotorLoads
        The monitored force and moment, the speed and the density.
    rps : float
        The speed in rev/s, a magnitude; its hand is ``block.rpm_sign``.
    copies : int, optional
        The periodic copies the monitored surface stands for, summed as vectors (1: the
        monitored surface is the reported rotor).

    Returns
    -------
    dict
        :data:`ROTOR_ROW_COLUMNS`, then ``T`` and ``Q`` in N and N m; NA where a value does
        not exist (no density, a static point for ``ETA``, a shaft along up for the in-plane
        four).
    """
    signed = block.rpm_sign * rps
    arm = tuple(r - o for r, o in zip(loads.about_m, block.origin, strict=True))
    about_hub = tuple(
        m + c for m, c in zip(loads.moment_nm, _cross(arm, loads.force_n), strict=True)
    )
    force, moment = periodic_sum(loads.force_n, about_hub, block.axis, copies)
    thrust = _dot(force, block.axis) + 0.0
    torque = _dot(moment, block.axis) + 0.0
    row: dict[str, float | str] = {
        "RPM": 60.0 * signed,
        "DIAMETER": block.diameter_m,
        "J": advance_ratio(loads.speed_m_s, signed, block.diameter_m) if rps else NOT_APPLICABLE,
    }
    density = loads.density_kg_m3
    if density is None or not density > 0.0 or not rps:
        row.update(dict.fromkeys(ROTOR_ROW_COLUMNS[3:], NOT_APPLICABLE))
    else:
        row.update(
            rotor_coefficients(
                thrust_n=thrust,
                torque_nm=torque,
                rps=signed,
                diameter_m=block.diameter_m,
                density_kg_m3=density,
                speed_m_s=loads.speed_m_s,
            )
        )
        row["CP"] = power_from_torque(float(row["CQ"]), signed)
        axes = in_plane_axes(block.axis)
        if axes is None:
            row.update(dict.fromkeys(("CN", "CS", "CMN", "CMS"), NOT_APPLICABLE))
        else:
            normal, side = axes
            forces = force_scale(signed, density, block.diameter_m)
            moments = moment_scale(signed, density, block.diameter_m)
            row["CN"] = _dot(force, normal) / forces + 0.0
            row["CS"] = _dot(force, side) / forces + 0.0
            row["CMN"] = _dot(moment, normal) / moments + 0.0
            row["CMS"] = _dot(moment, side) / moments + 0.0
    row["T"], row["Q"] = thrust, torque
    return row


#: The row controls of the symmetry and of the run types: never solver keys. The run types'
#: own vocabulary is :data:`cfdbender.core.run_types.CONTROLLED_KEYS`.
ROW_CONTROLS = tuple(dict.fromkeys((*CONTROLLED_KEYS, "CLOCK_MOTION", "MOTIONS")))
TIME_ACCURATE_PREFIX = "EXPORT_"


def is_row_control(key: str) -> bool:
    """Whether a row variable is a control of the matrix and never a solver key."""
    return key in ROW_CONTROLS or key.startswith(TIME_ACCURATE_PREFIX)


@dataclass(frozen=True)
class PeriodicState:
    """What the row states about the symmetry of the meshed model.

    Attributes
    ----------
    symmetry : str
        ``NONE`` or ``PERIODIC``: what was meshed.
    copies : int
        The periodic copies the sector stands for (1 without symmetry).
    loads : bool
        Whether the reported loads are the whole rotor's (the default) or the sector's.
    """

    symmetry: str
    copies: int
    loads: bool

    @property
    def multiplier(self) -> int:
        """The copies summed into the whole-rotor loads: ``copies`` or 1."""
        return self.copies if self.symmetry == "PERIODIC" and self.loads else 1


def periodic_state(setup: Mapping[str, object]) -> PeriodicState:
    """Read SYMMETRY, SYMMETRY_LOADS and PERIODIC_COPIES as the FlightStream side reads them.

    ``SYMMETRY`` says what was meshed (``NONE``, the default, or ``PERIODIC``);
    ``PERIODIC_COPIES`` is required with ``PERIODIC`` and refused otherwise;
    ``SYMMETRY_LOADS`` (the row's, else the setup's ``symmetry_loads``, else true) says whether
    the loads are the whole rotor's or the sector's. The run types' own variables are
    checked by :func:`cfdbender.core.run_types.skip_unread_variables`.

    Raises
    ------
    ValueError
        A symmetry other than NONE or PERIODIC, PERIODIC without a positive whole number of
        copies, copies without PERIODIC, or a non-boolean SYMMETRY_LOADS.
    """
    symmetry = str(setup.get("SYMMETRY", "NONE")).strip().upper()
    if symmetry not in {"NONE", "PERIODIC"}:
        raise ValueError(f"SYMMETRY {symmetry} is not read: the rotor block reads NONE or PERIODIC")
    copies = setup.get("PERIODIC_COPIES")
    if symmetry != "PERIODIC":
        if copies is not None:
            raise ValueError("PERIODIC_COPIES needs SYMMETRY: PERIODIC and is refused otherwise")
        count = 1
    else:
        if copies is None:
            raise ValueError(
                "SYMMETRY: PERIODIC needs PERIODIC_COPIES, the copies the sector stands for"
            )
        if (
            isinstance(copies, bool)
            or not isinstance(copies, int | float)
            or copies != int(copies)
            or copies < 1
        ):
            raise ValueError("PERIODIC_COPIES must be a positive whole number")
        count = int(copies)
    stated = setup.get("SYMMETRY_LOADS", setup.get("symmetry_loads", True))
    if isinstance(stated, str) and stated.strip().lower() in {"true", "false"}:
        stated = stated.strip().lower() == "true"
    if not isinstance(stated, bool):
        raise ValueError("SYMMETRY_LOADS must be true or false")
    return PeriodicState(symmetry, count, stated)
