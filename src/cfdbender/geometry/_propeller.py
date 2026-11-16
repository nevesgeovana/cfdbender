"""The public propeller design value and immutable parameter modifiers.

Pipeline role: pure Python API for the same strict input read by TOML. The
workspace runner writes its surfaces and records through the guarded kernel.
"""

from __future__ import annotations

import json
import tomllib
from copy import deepcopy
from functools import cached_property
from pathlib import Path

import numpy as np

from cfdbender._errors import InputValueError
from cfdbender._textio import write_text

from ._propeller_build import blade_geometry, rotor_geometry
from ._propeller_checks import design_report
from ._propeller_input import validate_definition


def _toml_literal(value):
    """Render one schema value with TOML's inline-table syntax."""
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(f"{json.dumps(k)} = {_toml_literal(v)}" for k, v in value.items())
            + " }"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    return json.dumps(value)


class Propeller:
    """Classical planar propeller in mm, with degree-five sections and a cubic C2 loft.

    Parameters
    ----------
    config : dict, optional
        Strict schema 1 mapping. Alternatively supply radius, blades, stations
        and airfoils as keyword arguments. Station rows use r_R, c_R,
        beta_deg, t_c and airfoil, with optional stacking and displacement.
    radius, hub_radius : float
        Radius and hub radius in mm.
    blades : int
        Positive blade count.
    stations : list of dict or dict
        Radial station rows or schema 1 column arrays.
    airfoils : dict
        Named profiles, each containing exactly one of naca, file or cst.

    Examples
    --------
    >>> from cfdbender.geometry import Propeller
    >>> Propeller({})
    Traceback (most recent call last):
    ...
    cfdbender._errors.InputValueError: schema_version: got None, expected 1
    """

    def __init__(
        self,
        config=None,
        *,
        radius=None,
        blades=None,
        stations=None,
        airfoils=None,
        hub_radius=0.0,
        pitch_setting=30.0,
        pitch_reference_r_R=0.7,
        rotation="positive_x",
        options=None,
        output=None,
        frame=None,
        base_dir=None,
    ):
        if config is None:
            if isinstance(stations, list) and stations:
                keys = set(stations[0])
                if any(set(row) != keys for row in stations):
                    raise InputValueError("stations: every row must have the same keys")
                stations = {key: [row[key] for row in stations] for key in keys}
            config = {
                "schema_version": 1,
                "propeller": {
                    "radius": radius,
                    "blades": blades,
                    "hub_radius": hub_radius,
                    "pitch_setting": pitch_setting,
                    "pitch_reference_r_R": pitch_reference_r_R,
                    "rotation": rotation,
                },
                "stations": stations,
                "airfoils": airfoils,
                "options": options or {},
                "output": output or {},
                "frame": frame or {},
            }
        self._data = validate_definition(config, base_dir or Path())

    @property
    def config(self):
        """Independent schema 1 mapping; changing it does not change this design."""
        return deepcopy(self._data)

    @classmethod
    def from_toml(cls, path):
        """Read schema 1 TOML, resolving coordinate files beside that input."""
        file = Path(path)
        with file.open("rb") as stream:
            return cls(tomllib.load(stream), base_dir=file.parent)

    def with_pitch(self, setting):
        """Return an independent design with the reference pitch setting in degrees."""
        data = self.config
        data["propeller"]["pitch_setting"] = setting
        return type(self)(data)

    def _with_distribution(self, key, value, units):
        data = self.config
        r = np.asarray(data["stations"]["r_R"])
        values = value(r) if callable(value) else value
        values = np.full(len(r), values) if np.ndim(values) == 0 else np.asarray(values)
        if values.shape != r.shape or values.dtype.kind not in "fiu":
            raise InputValueError(f"stations.{key}: expected one finite number per station")
        data["stations"][key] = values.astype(float).tolist()
        data["stations"].setdefault("units", {})[key] = units
        for option in list(data["options"]):
            if option.startswith(f"{key}_tip_") or option == f"{key}_power":
                del data["options"][option]
        return type(self)(data)

    def with_skew(self, value, *, units="deg"):
        """Return a design with scalar, station array or callable skew, in deg or s/R."""
        return self._with_distribution("skew", value, units)

    def with_rake(self, value, *, units="x/R"):
        """Return a design with scalar, station array or callable aft rake, in x/R or deg."""
        return self._with_distribution("rake", value, units)

    @cached_property
    def _blade(self):
        return blade_geometry(self._data)

    def blade(self):
        """Build one blade, returning patches and station information in mm."""
        return deepcopy(self._blade)

    def rotor(self):
        """Build every blade at equal azimuths about the shaft."""
        return rotor_geometry(self._data, self._blade)

    def report(self):
        """Return numerical checks and design measures; kernel checks live in run_geometry."""
        return design_report(self._data, self._blade)

    def component(self, *, blades=None):
        """Return the component mapping accepted by resolve_config and run_geometry."""
        data = self.config
        if blades is not None:
            data["output"]["blades"] = blades
        return {"type": "propeller", "definition": data}

    def to_toml(self, path):
        """Write schema 1 TOML that reads back with identical parameters and surfaces."""
        write_text(
            path,
            "\n".join(f"{key} = {_toml_literal(value)}" for key, value in self._data.items())
            + "\n",
        )
