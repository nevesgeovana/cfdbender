"""The typed part of the adapter contract for the products the post writes from field files.

An adapter that supplies the files a product is written from declares it with a
``PRODUCTS`` mapping, product table name to :class:`ProductSupport`: which file the
product is read from (a surface with connectivity, or a volume), the product column each
field of that file fills (a field name, or a callable on the fields and the free stream),
and the columns it cannot fill, each with its reason. A column the adapter cannot fill is
absent from the product file and named in ``products.json``. The adapter's
``product_inputs(point)`` then states, for one recorded point, the files of each step, the
free stream of the Cp columns, the sign of the surface normal and the surface's family
(marker) names. The agnostic layer computes and writes; it never asks which solver it is.

An adapter that declares nothing supplies no product: a p file declaring one is warned and
skipped at plan. Standard library only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cfdbender.core.pproc import PRODUCT_TABLES

__all__ = [
    "SURFACE",
    "VOLUME",
    "ColumnSource",
    "FreeStream",
    "ProductInputs",
    "ProductSupport",
    "StepFile",
    "product_support",
]

#: A product read from a surface file: points, cells, wall families and point fields.
SURFACE = "surface"
#: A product read from a volume file: points, cells and point fields.
VOLUME = "volume"

#: A product column's source: the field's name in the file, or a callable that takes the
#: point fields (name to array) and the point's :class:`FreeStream` and returns the column.
ColumnSource = str | Callable[[Mapping[str, Any], "FreeStream"], Any]


@dataclass(frozen=True)
class FreeStream:
    """The free stream the Cp columns of one point refer to.

    Attributes
    ----------
    p_inf : float
        Free-stream static pressure, Pa.
    q_ref : float
        The dynamic pressure a Cp column divides by, Pa.
    density : float
        Free-stream density, kg/m3.
    speed : float
        Free-stream speed, m/s.
    cp_reference : str
        The rule that gave ``p_inf`` and ``q_ref``, as the product's provenance states it.
    """

    p_inf: float
    q_ref: float
    density: float
    speed: float
    cp_reference: str


@dataclass(frozen=True)
class ProductSupport:
    """What an adapter supplies for one product.

    Attributes
    ----------
    source : str
        :data:`SURFACE` or :data:`VOLUME`.
    columns : mapping of str to ColumnSource
        The product columns the adapter fills and where each comes from. For a surface, the
        key ``pressure`` is the static pressure every load is integrated from.
    unfillable : mapping of str to str
        The product columns it cannot fill, each with the reason the manifest states.
    """

    source: str
    columns: Mapping[str, ColumnSource]
    unfillable: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StepFile:
    """One file a product is read from.

    Attributes
    ----------
    step : int or None
        The 1-based solver step of a time-accurate point; None for a steady one.
    time_s : float or None
        The physical time of the step, s.
    path : str
        The workspace-relative file.
    zone : int or None
        The zone a multizone point's file holds; None for a single-zone point. The files of
        one step are listed together, in zone order.
    """

    step: int | None
    time_s: float | None
    path: str
    zone: int | None = None


@dataclass(frozen=True)
class ProductInputs:
    """What one recorded point supplies to the products the post writes.

    Attributes
    ----------
    surface, volume : tuple of StepFile
        The files present, in step order; one with step None for a steady point.
    free_stream : FreeStream or None
        The Cp reference; None when the run states none.
    normal_sign : int
        ``+1`` when a surface cell's winding normal points from the wall into the fluid.
    families : tuple of str
        The family (marker) names the surface file holds.
    families_separable : bool
        Whether the surface file says which cell is of which family; when it does not, a
        distribution cuts either every family of the file or none.
    family_field : str or None
        The cell field holding each cell's index into ``families``, when separable.
    iteration : int or None
        The solver iteration a steady point's files are of.
    steps_per_revolution : float or None
        The rotor clock of a time-accurate point.
    time_accurate : bool
        Whether the point marches in time.
    windowed : bool
        Whether the files are those of an export window, one per step; without one a
        time-accurate point supplies its last step only.
    missing : tuple of (str, str)
        Each file the point should hold and does not, with the reason.
    notes : mapping of str to str
        Further provenance the manifest states (for example the velocity's frame).
    zones : int
        The number of zones of a multizone point, whose files are one per zone and step; 1 for
        a single-zone point.
    family_zones : mapping of str to int
        The zone each family belongs to, of a multizone point; empty when the zones of the
        families are not known.
    """

    surface: tuple[StepFile, ...] = ()
    volume: tuple[StepFile, ...] = ()
    free_stream: FreeStream | None = None
    normal_sign: int = 1
    families: tuple[str, ...] = ()
    families_separable: bool = False
    family_field: str | None = None
    iteration: int | None = None
    steps_per_revolution: float | None = None
    time_accurate: bool = False
    windowed: bool = False
    missing: tuple[tuple[str, str], ...] = ()
    notes: Mapping[str, str] = field(default_factory=dict)
    zones: int = 1
    family_zones: Mapping[str, int] = field(default_factory=dict)

    @property
    def zoned(self) -> bool:
        """Whether the point is multizone."""
        return self.zones > 1


def product_support(adapter: object) -> Mapping[str, ProductSupport]:
    """Return an adapter's validated ``PRODUCTS`` declaration; empty when it declares none.

    Raises
    ------
    ValueError
        A product name outside the p-file tables, a source other than surface or volume, or
        a column both filled and declared unfillable (a developer error).
    """
    declared: object = getattr(adapter, "PRODUCTS", {})
    label = getattr(adapter, "__name__", type(adapter).__name__)
    if not isinstance(declared, Mapping):
        raise ValueError(f"{label}: PRODUCTS must map product names to ProductSupport")
    for name, support in declared.items():
        if name not in PRODUCT_TABLES or not isinstance(support, ProductSupport):
            raise ValueError(
                f"{label}: PRODUCTS[{name!r}] must be a ProductSupport of one of "
                f"{', '.join(PRODUCT_TABLES)}"
            )
        if support.source not in (SURFACE, VOLUME):
            raise ValueError(f"{label}: PRODUCTS[{name!r}].source must be surface or volume")
        if both := sorted(set(support.columns) & set(support.unfillable)):
            raise ValueError(f"{label}: PRODUCTS[{name!r}] both fills and lacks {both}")
    return declared
