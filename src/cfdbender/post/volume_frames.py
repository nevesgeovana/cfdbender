"""The frames a volume cut or a probe is declared in, and where they place their points.

A layout is laid in the local coordinates of a frame (see :mod:`cfdbender.post.volume_layouts`).
:meth:`Frame.place` turns those points into inertial (reference frame) coordinates, which is
where the volume file is sampled and where the product files state them.

The frames are the reference frame itself (``REFERENCE``), a fixed frame (a body or a rotor's
hub frame: an origin and the directions of its axes) and a rotating frame, which is a fixed frame
that turns about the rotor axis through its origin. A rotating frame follows the rotor: at a step
it is turned by the azimuth of the rotor's first blade at that step,
``(datum + sense * step * 360 / steps_per_revolution) mod 360`` degrees
(:func:`cfdbender.core.unsteady_clock.blade_one_azimuth`), positive about the axis by the right-hand
rule, so a point fixed in the frame stays on the same blade-fixed position as the blade turns.
Its local zero direction is then blade one's direction at azimuth zero of the datum. The datum is
the initial position of the first blade that the reference file carries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cfdbender.core.unsteady_clock import blade_one_azimuth

__all__ = ["REFERENCE", "Frame", "FrameError", "RotorSpin"]


class FrameError(ValueError):
    """A frame that is not defined or cannot place a point."""


@dataclass(frozen=True)
class RotorSpin:
    """How a rotating frame turns.

    Attributes
    ----------
    axis : tuple of three floats
        The rotor axis in inertial coordinates, through the frame's origin; any length but zero.
    datum_deg : float
        Where the first blade sits at step zero, in degrees about the axis.
    sense : float
        The sign of the rotor's speed: positive turns by the right-hand rule about ``axis``.
    steps_per_revolution : float
        The rotor's clock.
    """

    axis: tuple[float, float, float]
    datum_deg: float
    sense: float
    steps_per_revolution: float

    def angle_deg(self, step: float) -> float:
        """Return the azimuth of the first blade at ``step``, in degrees in [0, 360)."""
        return blade_one_azimuth(
            step,
            datum_deg=self.datum_deg,
            sense=self.sense,
            steps_per_revolution=self.steps_per_revolution,
        )


@dataclass(frozen=True)
class Frame:
    """A named frame: its origin and axes in inertial coordinates, and its spin if it rotates.

    Attributes
    ----------
    name : str
        The frame's name, as the declaration states it.
    origin : tuple of three floats
        The frame's origin in inertial coordinates.
    axes : tuple of three tuples of three floats
        The frame's x, y and z directions in inertial coordinates, as rows; unit and
        orthogonal. The identity by default.
    spin : RotorSpin, optional
        Present for a rotating frame.
    """

    name: str = "REFERENCE"
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axes: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    spin: RotorSpin | None = None

    @property
    def rotating(self) -> bool:
        """Whether the frame turns with a rotor."""
        return self.spin is not None

    def place(
        self, points: Sequence[Sequence[float]] | NDArray[np.float64], step: float | None = None
    ) -> NDArray[np.float64]:
        """Return the inertial coordinates of local points, at a step for a rotating frame.

        Parameters
        ----------
        points : array_like
            Shape ``(n, 3)``, in the frame's local coordinates.
        step : float, optional
            The solver step; required by a rotating frame, ignored by a fixed one.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 3)``.

        Raises
        ------
        FrameError
            If the frame rotates and no step is given, or its axes are not three directions.
        """
        local = np.asarray(points, dtype=np.float64)
        if local.ndim != 2 or local.shape[1] != 3:
            raise FrameError(f"points must have shape (n, 3), got {local.shape}")
        basis = np.asarray(self.axes, dtype=np.float64)
        if basis.shape != (3, 3):
            raise FrameError(f"the axes of frame {self.name!r} are not three directions")
        origin = np.asarray(self.origin, dtype=np.float64)
        offset = local @ basis
        if self.spin is None:
            return np.asarray(origin + offset, dtype=np.float64)
        if step is None:
            raise FrameError(f"frame {self.name!r} rotates; place its points at a step")
        axis = np.asarray(self.spin.axis, dtype=np.float64)
        length = float(np.linalg.norm(axis))
        if length == 0.0 or not math.isfinite(length):
            raise FrameError(f"frame {self.name!r} has a rotor axis of zero length")
        axis = axis / length
        angle = math.radians(self.spin.angle_deg(step))
        cos, sin = math.cos(angle), math.sin(angle)
        rotated = (
            offset * cos
            + np.cross(axis, offset) * sin
            + np.outer(offset @ axis, axis) * (1.0 - cos)
        )
        return np.asarray(origin + rotated, dtype=np.float64)


#: The inertial frame.
REFERENCE = Frame()
