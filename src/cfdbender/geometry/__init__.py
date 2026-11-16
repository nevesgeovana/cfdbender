"""Parametric geometry: component records, configuration, sections, surfaces and propellers.

Pipeline role: pure numerical layer below the workspace. It describes aircraft components by a
few parameters and rebuilds their skin as B-spline patches, with no geometry kernel and no
solver; it needs numpy, scipy and pydantic. The stages that need a kernel (IGES writing,
meshing) import the types defined here.

The public names are :class:`ComponentRecord` (the parameter contract), :func:`load_config`
and :func:`resolve_config` (the TOML input boundary), :class:`Propeller` (the propeller
builder), and :func:`run_geometry`, :func:`describe_config`, :func:`list_configurations` and
``STAGE_ROWS`` (the stage runner and its table). The sections, distributions, lifting
surfaces, caps and B-spline patches behind them are private modules of this package;
:mod:`cfdbender.geometry.stages` gathers them for the stages.

Examples
--------
>>> from cfdbender.geometry import resolve_config
>>> resolved = resolve_config({"type": "lifting_surface"})
>>> resolved["components"]["COMPONENT"]["type"]["source"]
'given'
"""

from cfdbender.geometry._propeller import Propeller
from cfdbender.geometry.component import ComponentRecord
from cfdbender.geometry.config import load_config, resolve_config
from cfdbender.geometry.pipeline import describe_config, list_configurations, run_geometry
from cfdbender.geometry.stage_table import STAGE_ROWS, StageRow

__all__ = [
    "STAGE_ROWS",
    "ComponentRecord",
    "Propeller",
    "StageRow",
    "describe_config",
    "list_configurations",
    "load_config",
    "resolve_config",
    "run_geometry",
]
