"""Check the setup, reference and post-processing files of a CFD++ point; build the case.

The setup file states the case only (no ``[solver]`` table): the run control, the
turbulence, the ``[physics]``, ``[fluid]`` and ``[numerics]`` tables, one
``[boundaries.<family>]`` table per family of the grid, optional ``[groups]`` and the
one raw-keyword door, ``[keywords]``. Which keys exist, their types, which are required
and which input keyword each one sets come from the ``setup`` table of the interface
database for the row's version, never from this module: the keys of a version are its
schema's (:func:`setup_keys`, :func:`required_keys`). No key has a default; a missing
one is refused with every missing key in one message.

The setup table binds a key with these fields: ``type`` (``int``, ``real``, ``word``,
``ints``, ``reals``; ``length`` for a list), ``required``, ``when`` (required and read
only when other keys hold the values stated), ``needed_by`` (required and read only
when the turbulence model transports one of the variables named), ``above`` or ``min``
(bounds), ``keyword`` and ``index`` (where the value goes), ``values`` (a word's value
in the input), ``models`` (the words also include the database's turbulence models),
``source = "case"`` (a value cfdbender states itself) and ``file = "post"`` (a key of
the post-processing file; the output and restart frequencies live there).

Nothing here writes a file. The freestream follows ``core.flight``; the turbulence
values of the freestream are cfdbender's own (a working assumption until a saved
input confirms them): ``nut = viscosity_ratio * mu / rho``,
``k = 1.5 (turbulence_intensity * V)^2``, ``omega = k / nut`` and
``epsilon = 0.09 k omega``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from cfdbender.core.flight import (
    GAMMA,
    GAS_CONSTANT,
    SUTHERLAND_CONSTANT,
    SUTHERLAND_TEMPERATURE,
    SUTHERLAND_VISCOSITY,
    FlowState,
    resolve_flow,
)
from cfdbender.core.neutral import wind_directions
from cfdbender.core.results import References
from cfdbender.solvers._host._config import number, unknown_keys
from cfdbender.solvers.cfdpp._schema import EMITTED, Scalar, Schema, schema_for

SOLVER_LABEL = "CFD++"
SETUPS_HINT = "cfd-workspace setups --template subsonic writes a complete setup"
SETUP = "setup file"
REFERENCE = "REF file"
POST = "PPROC file"
SETUP_TABLE = "setup"
REFERENCE_KEYS = ("area", "length", "moment_origin")
#: Setup keys and tables the adapter reads itself; every other key comes from the database.
OWN_KEYS = ("mesh",)
OWN_TABLES = ("boundaries", "groups", "keywords")
#: Keys refused with a pointer to the key this adapter reads instead.
INSTEAD = {
    "convergence_residual": "this adapter reads convergence_orders (orders of residual drop)",
    "keywords_unchecked": "raw keywords go in the one [keywords] table",
}
#: Machine options of init.toml [solver]; not solver numerics, so they have values.
OPTIONS: Mapping[str, float] = MappingProxyType(
    {"timeout_s": 600.0, "ram_estimate_gib": 0.25, "stop_grace_s": 60.0}
)
BOUNDARY_TYPES = ("wall", "farfield", "symmetry")
TREATMENTS = ("solve-to-wall", "wall-function")
THERMALS = ("adiabatic", "isothermal")
WALL_KEYS = ("type", "treatment", "thermal", "temperature", "slip", "forces")
KEY_TYPES = ("int", "real", "word", "ints", "reals")
#: Freestream variables cfdbender can state; an information set names its own in order.
FLOW_VARIABLES = ("p", "t", "rho", "u", "v", "w", "nut", "k", "omega", "epsilon")
#: The fluid constants a row's REmi is resolved with (core.flight).
FLIGHT_FLUID = {
    "fluid.gas_constant": GAS_CONSTANT,
    "fluid.gamma": GAMMA,
    "fluid.sutherland_viscosity": SUTHERLAND_VISCOSITY,
    "fluid.sutherland_temperature": SUTHERLAND_TEMPERATURE,
    "fluid.sutherland_constant": SUTHERLAND_CONSTANT,
}
_FLUID_LAW = ("fluid.viscosity", "sutherland")


@dataclass(frozen=True)
class Key:
    """One key of the setup or post-processing file, as the database binds it."""

    name: str
    file: str
    type: str | None
    length: int | None
    required: bool
    when: Mapping[str, Scalar]
    needed_by: tuple[str, ...]
    above: float | None
    min: float | None
    keyword: str | None
    index: int
    values: Mapping[str, Scalar | tuple[Scalar, ...]]
    models: bool
    source: str | None
    #: A list key written one element per keyword, in order.
    keywords: tuple[str, ...] = ()
    #: The variable of the equation-set information set the value goes to.
    variable: str | None = None
    #: The keyword whose fixed value is divided by the key's value to give the written one.
    divides: str | None = None
    #: For a value cfdbender states itself: the value by grid dimension, absent when none.
    dimension: Mapping[str, Scalar] = field(default_factory=dict)

    @property
    def table(self) -> str | None:
        """The table of a dotted name, or None for a top-level key."""
        return self.name.split(".", 1)[0] if "." in self.name else None

    @property
    def leaf(self) -> str:
        """The key as written inside its table."""
        return self.name.split(".", 1)[-1]


@dataclass(frozen=True)
class Boundary:
    """One grid family and its declared condition; ``neutral`` selects the boundary code."""

    family: str
    selector: int
    type: str
    neutral: Mapping[str, str]
    temperature: float | None
    forces: bool


@dataclass(frozen=True)
class RawKeyword:
    """One ``[keywords]`` entry: the keyword, its values and the optional section hint."""

    name: str
    values: tuple[Scalar, ...]
    section: str | None


@dataclass(frozen=True)
class Freestream:
    """The freestream of a point in SI units, with the turbulence values of its model."""

    pressure: float
    temperature: float
    density: float
    viscosity: float
    speed: float
    velocity: tuple[float, float, float]
    dynamic_pressure: float
    turbulence: Mapping[str, float]

    def value(self, variable: str) -> float:
        """Return one variable of the freestream by its neutral name."""
        known = {
            "p": self.pressure,
            "t": self.temperature,
            "rho": self.density,
            "u": self.velocity[0],
            "v": self.velocity[1],
            "w": self.velocity[2],
            **self.turbulence,
        }
        if variable not in known:
            raise ValueError(
                f"freestream variable {variable!r} is not stated for this case; "
                f"the variables are: {', '.join(known)}"
            )
        return known[variable]


@dataclass(frozen=True)
class Case:
    """Everything the input of one point is built from; immutable and file-free."""

    schema: Schema
    values: Mapping[str, object]
    turbulence: str
    variables: tuple[str, ...]
    boundaries: tuple[Boundary, ...]
    groups: Mapping[str, tuple[str, ...]]
    raw: tuple[RawKeyword, ...]
    mesh: str
    flow: FlowState
    freestream: Freestream
    alpha: float
    beta: float
    dimension: int
    references: References
    restart: bool = False
    warnings: tuple[str, ...] = field(default=())

    @property
    def iterations(self) -> int:
        """The steps of the run, from the setup key that sets them."""
        value = self.values.get("iterations")
        return value if isinstance(value, int) else 0

    @property
    def convergence_orders(self) -> float:
        """Orders of residual drop that end the run."""
        value = self.values.get("convergence_orders")
        return float(value) if isinstance(value, int | float) else 0.0

    def force_selectors(self) -> tuple[int, ...]:
        """Selectors of the walls whose forces make the total."""
        return tuple(sorted(b.selector for b in self.boundaries if b.forces))


def database(build: str | None) -> Schema:
    """Return the schema of a row's SOLVER_BUILD, refusing an absent or unknown version."""
    return schema_for(build)


def options(solver_options: Mapping[str, object], schema: Schema | None = None) -> dict[str, float]:
    """Return the machine options of init.toml [solver], refusing an unknown one."""
    known = {*OPTIONS, "line_endings"}
    if unknown := set(solver_options) - known:
        raise unknown_keys("init.toml [solver]", unknown, known)
    if solver_options.get("line_endings", "lf") not in ("lf", "crlf"):
        raise ValueError('init.toml [solver] line_endings must be "lf" or "crlf"')
    if schema is not None and schema.layout is not None and "line_endings" in solver_options:
        required = "crlf" if schema.require_grammar().line_ending == "\r\n" else "lf"
        if solver_options["line_endings"] != required:
            raise ValueError(
                f"init.toml [solver] line_endings must be {required} for "
                f"{schema.solver} {schema.version}: the established layout fixes the ending"
            )
    result = dict(OPTIONS)
    for name, value in solver_options.items():
        if name == "line_endings":
            continue
        result[name] = number(value, name, "init.toml [solver]")
        if result[name] < 0 or (name == "timeout_s" and result[name] == 0):
            raise ValueError(f"init.toml [solver] {name} must be positive")
    return result


def setup_keys(schema: Schema) -> tuple[Key, ...]:
    """Return the keys the database binds for this version, checking each entry."""
    keys = []
    for entry in schema.entries.get(SETUP_TABLE, ()):
        attrs = entry.attrs
        label = f"{schema.solver} {schema.version} setup key {entry.name}"
        known = {
            "file", "type", "length", "required", "when", "needed_by", "above", "min",
            "keyword", "index", "values", "models", "source", "keywords", "variable",
            "divides", "dimension",
        }  # fmt: skip
        if unknown := sorted(set(attrs) - known):
            raise ValueError(f"{label}: unknown fields {', '.join(unknown)}")
        kind = attrs.get("type")
        source = attrs.get("source")
        if source is None and kind not in KEY_TYPES:
            raise ValueError(f"{label}: type {kind!r}; the types are: {', '.join(KEY_TYPES)}")
        if source not in (None, "case"):
            raise ValueError(f"{label}: source must be case")
        keyword = attrs.get("keyword")
        targets = tuple(str(name) for name in attrs.get("keywords") or ())
        for name in (keyword, attrs.get("divides"), *targets):
            if name is not None and name not in schema.keywords:
                raise ValueError(f"{label} sets keyword {name}, which the database does not hold")
        if keyword is not None and targets:
            raise ValueError(f"{label}: keyword and keywords exclude each other")
        divides = attrs.get("divides")
        if divides is not None and not isinstance(schema.keywords[divides].fixed, int | float):
            raise ValueError(f"{label} divides keyword {divides}, which holds no fixed number")
        variable = attrs.get("variable")
        if variable is not None and not any(
            variable in (tag.variables or ()) for tag in schema.infosets.values()
        ):
            raise ValueError(f"{label} sets variable {variable}, which no information set holds")
        values = attrs.get("values") or {}
        keys.append(
            Key(
                name=entry.name,
                file=str(attrs.get("file", "setup")),
                type=None if kind is None else str(kind),
                length=attrs.get("length"),
                required=bool(attrs.get("required", False)),
                when=MappingProxyType(dict(attrs.get("when") or {})),
                needed_by=tuple(attrs.get("needed_by") or ()),
                above=attrs.get("above"),
                min=attrs.get("min"),
                keyword=keyword,
                index=int(attrs.get("index", 0)),
                values=MappingProxyType(
                    {
                        str(word): tuple(value) if isinstance(value, list) else value
                        for word, value in values.items()
                    }
                ),
                models=bool(attrs.get("models", False)),
                source=source,
                keywords=targets,
                variable=None if variable is None else str(variable),
                divides=None if divides is None else str(divides),
                dimension=MappingProxyType(
                    {str(k): v for k, v in (attrs.get("dimension") or {}).items()}
                ),
            )
        )
    return tuple(keys)


def required_keys(schema: Schema, file: str = "setup") -> tuple[str, ...]:
    """Return the keys a file of this version must always state, as the database says."""
    return tuple(
        key.name for key in setup_keys(schema) if key.file == file and key.required and not key.when
    )


def post_keys(schema: Schema) -> tuple[str, ...]:
    """Return the post-processing keys this adapter reads: history and the bound ones."""
    return ("history", *(key.name for key in setup_keys(schema) if key.file == "post"))


def display(name: str) -> str:
    """Write a dotted key as the user writes it: ``[physics] method``."""
    table, _, leaf = name.rpartition(".")
    return f"[{table}] {leaf}" if table else name


def choices(key: Key, schema: Schema) -> tuple[str, ...]:
    """Return the words a word key accepts."""
    models = tuple(schema.turbulence) if key.models else ()
    return (*models, *key.values)


def config(
    schema: Schema,
    setup: Mapping[str, object],
    reference: Mapping[str, object],
    post: Mapping[str, object],
    condition: Mapping[str, float],
    *,
    mesh: str,
    names: Mapping[int, str],
    dimension: int,
    restart: bool = False,
) -> Case:
    """Check the three files and the row's condition, and return the case.

    Parameters
    ----------
    schema : Schema
        The interface database of the row's version.
    setup, reference, post : mapping
        The setup, reference and post-processing files. A ``solver`` table in the
        setup is ignored here: planning refuses one the user wrote (the run itself
        adds the build's executables there).
    condition : mapping
        One point's flight condition.
    mesh : str
        The workspace-relative grid folder.
    names : mapping of int to str
        The grid's selectors and their family names.
    dimension : int
        2 or 3, from the grid's node type.
    restart : bool, optional
        True when the point starts from a saved solution.
    """
    keys = setup_keys(schema)
    _check_unknown(setup, keys)
    values, missing, skipped = _read_keys(schema, keys, setup, post)
    turbulence = str(values.get("turbulence", ""))
    model = schema.turbulence.get(turbulence)
    variables = tuple(model.variables or ()) if model is not None else ()
    for key in keys:
        if key.needed_by and set(key.needed_by) & set(variables):
            if _stated(key, setup, post) is None:
                missing.append(key.name)
            else:
                values[key.name] = _value(key, _stated(key, setup, post), schema)
        elif key.needed_by and _stated(key, setup, post) is not None:
            skipped.append(f"{display(key.name)} is not used by turbulence {turbulence or 'none'}")
    setup_missing = [name for name in missing if _file_of(keys, name) == "setup"]
    post_missing = [name for name in missing if _file_of(keys, name) == "post"]
    if setup_missing:
        raise ValueError(
            f"{SETUP}: missing required keys: {', '.join(display(n) for n in setup_missing)}; "
            f"{SETUPS_HINT}"
        )
    if post.get("history") is not True:
        raise ValueError(
            f"{POST}: history = true is required: cfdbender reads the residual and force "
            "history of every CFD++ run"
        )
    if post_missing:
        raise ValueError(
            f"{POST}: missing required keys: {', '.join(post_missing)}; CFD++ writes them "
            "into its input and no value is assumed"
        )
    if model is not None and model.status not in EMITTED:
        raise ValueError(
            f"{SETUP}: turbulence {turbulence} has no harvested or verified evidence for "
            f"{schema.solver} {schema.version} ({model.status.value} only); cfdbender does "
            "not write it yet, so choose another turbulence model"
        )
    alpha = float(condition.get("ALPHA", 0.0))
    beta = float(condition.get("BETA", 0.0))
    if dimension == 2 and beta != 0.0:
        raise ValueError("a two-dimensional grid flies at BETA 0; remove BETA from the row")
    references = _references(reference)
    _check_fluid(values, condition)
    flow = resolve_flow(condition, reference_length=references.length, label="FLIGHT_CONDITION")
    boundaries = _boundaries(setup, names, turbulence, mesh, skipped)
    groups = _groups(setup, boundaries)
    raw = _raw(setup)
    return Case(
        schema=schema,
        values=MappingProxyType(values),
        turbulence=turbulence,
        variables=variables,
        boundaries=boundaries,
        groups=MappingProxyType(groups),
        raw=raw,
        mesh=mesh,
        flow=flow,
        freestream=freestream(flow, alpha, beta, dimension, values, variables),
        alpha=alpha,
        beta=beta,
        dimension=dimension,
        references=references,
        restart=restart,
        warnings=tuple(f"{SETUP}: {message}; skipped" for message in skipped),
    )


def freestream(
    flow: FlowState,
    alpha: float,
    beta: float,
    dimension: int,
    values: Mapping[str, object],
    variables: Sequence[str],
) -> Freestream:
    """Return the freestream: velocity along the drag direction, and turbulence values.

    Two dimensions: the flow lies in the x-y plane, velocity ``V (cos a, sin a, 0)``.
    Three dimensions: ``V`` times the drag direction of ``core.neutral.wind_directions``.
    """
    speed = flow.velocity
    if dimension == 2:
        a = math.radians(alpha)
        direction = (math.cos(a), math.sin(a), 0.0)
    else:
        direction = wind_directions(alpha, beta)[0]
    velocity = (speed * direction[0], speed * direction[1], speed * direction[2])
    turbulence: dict[str, float] = {}
    nut = None
    ratio = values.get("viscosity_ratio")
    if isinstance(ratio, int | float):
        nut = float(ratio) * flow.viscosity / flow.density
    intensity = values.get("turbulence_intensity")
    k = 1.5 * (float(intensity) * speed) ** 2 if isinstance(intensity, int | float) else None
    for variable in variables:
        if variable == "nut" and nut is not None:
            turbulence["nut"] = nut
        elif variable == "k" and k is not None:
            turbulence["k"] = k
        elif variable in ("omega", "epsilon") and k is not None and nut is not None:
            omega = k / nut
            turbulence[variable] = omega if variable == "omega" else 0.09 * k * omega
        else:
            raise ValueError(
                f"{SETUP}: the turbulence variable {variable!r} cannot be stated from "
                "viscosity_ratio and turbulence_intensity"
            )
    return Freestream(
        pressure=flow.pressure,
        temperature=flow.temperature,
        density=flow.density,
        viscosity=flow.viscosity,
        speed=speed,
        velocity=velocity,
        dynamic_pressure=0.5 * flow.density * speed**2,
        turbulence=MappingProxyType(turbulence),
    )


def _file_of(keys: Sequence[Key], name: str) -> str:
    return next((key.file for key in keys if key.name == name), "setup")


def _stated(key: Key, setup: Mapping[str, object], post: Mapping[str, object]) -> object:
    table = post if key.file == "post" else setup
    if key.table is None:
        return table.get(key.name)
    inner = table.get(key.table)
    return inner.get(key.leaf) if isinstance(inner, Mapping) else None


def _check_unknown(setup: Mapping[str, object], keys: Sequence[Key]) -> None:
    mine = [key for key in keys if key.file == "setup" and key.source is None]
    top = {key.name for key in mine if key.table is None} | set(OWN_KEYS)
    tables = {key.table for key in mine if key.table is not None}
    if "solver" in setup:
        raise ValueError(
            f"{SETUP}: {SOLVER_LABEL} takes no [solver] table; the workspace's solver and its "
            "machine options (timeout_s, ram_estimate_gib, stop_grace_s) are in init.toml, "
            "and each build's executables are in inputs/executables.toml"
        )
    for name in setup:
        if name in INSTEAD:
            raise ValueError(f"{SETUP}: unknown key {name!r}; {INSTEAD[name]}")
    unknown = set(setup) - top - tables - set(OWN_TABLES)
    if unknown:
        raise unknown_keys(SETUP, unknown, top | tables | set(OWN_TABLES))
    for table in sorted(tables):
        inner = setup.get(table)
        if inner is None:
            continue
        if not isinstance(inner, Mapping):
            raise ValueError(f"{SETUP}: [{table}] must be a table")
        known = {key.leaf for key in mine if key.table == table}
        if extra := set(inner) - known:
            raise unknown_keys(f"{SETUP} [{table}]", extra, known)


def _read_keys(
    schema: Schema, keys: Sequence[Key], setup: Mapping[str, object], post: Mapping[str, object]
) -> tuple[dict[str, object], list[str], list[str]]:
    values: dict[str, object] = {}
    missing: list[str] = []
    skipped: list[str] = []
    for key in keys:
        if key.source is not None or key.needed_by:
            continue
        stated = _stated(key, setup, post)
        if key.when:
            holds = all(_raw_of(name, setup, post, keys) == want for name, want in key.when.items())
            if not holds:
                if stated is not None:
                    condition = ", ".join(f"{display(n)} = {v}" for n, v in key.when.items())
                    skipped.append(f"{display(key.name)} applies only with {condition}")
                continue
            if stated is None:
                missing.append(key.name)
                continue
        elif stated is None:
            if key.required:
                missing.append(key.name)
            continue
        values[key.name] = _value(key, stated, schema)
    return values, missing, skipped


def _raw_of(
    name: str, setup: Mapping[str, object], post: Mapping[str, object], keys: Sequence[Key]
) -> object:
    key = next((key for key in keys if key.name == name), None)
    if key is None:
        table, _, leaf = name.rpartition(".")
        inner = setup.get(table) if table else setup
        return inner.get(leaf) if isinstance(inner, Mapping) else None
    return _stated(key, setup, post)


def _value(key: Key, stated: object, schema: Schema) -> object:
    where = f"{POST if key.file == 'post' else SETUP}: {display(key.name)}"
    if key.type in ("ints", "reals"):
        if not isinstance(stated, list) or (key.length is not None and len(stated) != key.length):
            count = f"{key.length} " if key.length is not None else ""
            kind = "integers" if key.type == "ints" else "numbers"
            raise ValueError(f"{where} takes a list of {count}{kind}; found {stated!r}")
        scalar = "int" if key.type == "ints" else "real"
        return tuple(_scalar(item, scalar, key, where) for item in stated)
    if key.type == "word":
        allowed = choices(key, schema)
        if not isinstance(stated, str) or stated not in allowed:
            raise ValueError(
                f"{where} is {stated!r}; the accepted values are: {', '.join(allowed)}"
            )
        return stated
    return _scalar(stated, str(key.type), key, where)


def _scalar(value: object, kind: str, key: Key, where: str) -> int | float:
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} takes an integer; found {value!r}")
        result: int | float = value
    else:
        result = _real(value, where)
    if key.above is not None and not result > key.above:
        raise ValueError(f"{where} must be above {key.above:g}; found {value!r}")
    if key.min is not None and result < key.min:
        raise ValueError(f"{where} must be at least {key.min:g}; found {value!r}")
    return result


def _real(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{where} takes a finite number; found {value!r}")
    return float(value)


def _check_fluid(values: Mapping[str, object], condition: Mapping[str, float]) -> None:
    """Refuse fluid constants other than core.flight's, for every kind of row.

    The flow resolution (velocity, density, Reynolds number) and the coefficient
    normalization use core.flight's air constants, so a setup stating others would write
    a state that is not the row's. The refusal stands until the resolution takes the setup's
    fluid; ``condition`` is kept so the call site does not change.
    """
    del condition
    law, wanted = _FLUID_LAW
    if law in values and values[law] != wanted:
        raise ValueError(
            f"{SETUP}: {display(law)} is {values[law]!r}, but cfdbender uses {wanted!r}; "
            "custom fluid properties are not supported yet for this solver"
        )
    for name, constant in FLIGHT_FLUID.items():
        stated = values.get(name)
        if isinstance(stated, int | float) and not math.isclose(stated, constant, rel_tol=1e-9):
            raise ValueError(
                f"{SETUP}: {display(name)} is {stated!r}, but cfdbender uses {constant!r}; "
                "custom fluid properties are not supported yet for this solver"
            )


def _references(reference: Mapping[str, object]) -> References:
    missing = [name for name in REFERENCE_KEYS if name not in reference]
    if missing:
        raise ValueError(f"{REFERENCE}: missing {', '.join(missing)}")
    origin = reference["moment_origin"]
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError(f"{REFERENCE}: moment_origin needs three finite coordinates")
    return References(
        number(reference["area"], "area", REFERENCE),
        number(reference["length"], "length", REFERENCE),
        (
            number(origin[0], "moment_origin", REFERENCE),
            number(origin[1], "moment_origin", REFERENCE),
            number(origin[2], "moment_origin", REFERENCE),
        ),
        "geometry",
        ("x aft", "y right (2D: y up)", "z up"),
    )


def _boundaries(
    setup: Mapping[str, object],
    names: Mapping[int, str],
    turbulence: str,
    mesh: str,
    skipped: list[str],
) -> tuple[Boundary, ...]:
    table = setup.get("boundaries", {})
    if not isinstance(table, Mapping):
        raise ValueError(f"{SETUP}: [boundaries] must hold one table per family")
    grid = mesh.rsplit("/", 1)[-1]
    families = {name: selector for selector, name in names.items()}
    declared = sorted(table)
    absent = sorted(set(families) - set(declared))
    foreign = sorted(set(declared) - set(families))
    if absent or foreign:
        parts = []
        if absent:
            parts.append(f"families of {grid} without an entry: {', '.join(absent)}")
        if foreign:
            parts.append(f"entries for families {grid} does not hold: {', '.join(foreign)}")
        raise ValueError(
            f"{SETUP} [boundaries]: {'; '.join(parts)}; declared: {', '.join(declared) or 'none'}; "
            f"the grid has: {', '.join(sorted(families))}"
        )
    result = []
    for family in declared:
        entry = table[family]
        where = f"{SETUP} [boundaries.{family}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} must be a table with a type")
        kind = entry.get("type")
        if kind not in BOUNDARY_TYPES:
            raise ValueError(
                f"{where}: type {kind!r} is unknown; the types are: {', '.join(BOUNDARY_TYPES)}"
            )
        if kind != "wall":
            if extra := set(entry) - {"type"}:
                raise unknown_keys(where, extra, {"type"})
            result.append(Boundary(family, families[family], kind, {"type": kind}, None, False))
            continue
        result.append(_wall(family, families[family], entry, turbulence, where, skipped))
    if not any(item.type == "farfield" for item in result):
        raise ValueError(
            f'{SETUP} [boundaries]: no family has type = "farfield"; one far field carries '
            "the freestream"
        )
    if not any(item.forces for item in result):
        raise ValueError(
            f"{SETUP} [boundaries]: no wall has forces = true; the total force is the sum "
            "over those walls"
        )
    return tuple(sorted(result, key=lambda item: item.selector))


def _wall(
    family: str,
    selector: int,
    entry: Mapping[str, object],
    turbulence: str,
    where: str,
    skipped: list[str],
) -> Boundary:
    if extra := set(entry) - set(WALL_KEYS):
        raise unknown_keys(where, extra, WALL_KEYS)
    for name in ("forces", "slip"):
        if name in entry and not isinstance(entry[name], bool):
            raise ValueError(f"{where}: {name} must be true or false")
    forces = entry.get("forces", True) is True
    slip = entry.get("slip", False) is True
    if turbulence == "inviscid" and not slip:
        raise ValueError(f"{where}: an inviscid flow takes slip = true on every wall")
    if slip:
        for name in ("treatment", "thermal", "temperature"):
            if name in entry:
                skipped.append(f"[boundaries.{family}] {name} does not apply to a slip wall")
        return Boundary(family, selector, "wall", {"type": "wall", "slip": "true"}, None, forces)
    missing = [name for name in ("treatment", "thermal") if name not in entry]
    if missing:
        raise ValueError(
            f"{where}: missing required keys: {', '.join(missing)} (a viscous wall states its "
            f"treatment, one of {', '.join(TREATMENTS)}, and its thermal condition, one of "
            f"{', '.join(THERMALS)}); {SETUPS_HINT}"
        )
    treatment, thermal = entry["treatment"], entry["thermal"]
    if treatment not in TREATMENTS:
        raise ValueError(f"{where}: treatment {treatment!r}; one of: {', '.join(TREATMENTS)}")
    if thermal not in THERMALS:
        raise ValueError(f"{where}: thermal {thermal!r}; one of: {', '.join(THERMALS)}")
    temperature = None
    if thermal == "isothermal":
        if "temperature" not in entry:
            raise ValueError(
                f"{where}: missing required keys: temperature (an isothermal wall); {SETUPS_HINT}"
            )
        temperature = number(entry["temperature"], "temperature", where)
        if temperature <= 0:
            raise ValueError(f"{where}: temperature must be above 0 K")
    elif "temperature" in entry:
        skipped.append(f"[boundaries.{family}] temperature does not apply to an adiabatic wall")
    neutral = {"type": "wall", "treatment": str(treatment), "thermal": str(thermal)}
    return Boundary(family, selector, "wall", neutral, temperature, forces)


def _groups(
    setup: Mapping[str, object], boundaries: Sequence[Boundary]
) -> dict[str, tuple[str, ...]]:
    table = setup.get("groups", {})
    if not isinstance(table, Mapping):
        raise ValueError(f"{SETUP}: [groups] maps a group name to a list of wall families")
    walls = sorted(item.family for item in boundaries if item.type == "wall")
    groups = {}
    for name, members in table.items():
        if (
            not isinstance(members, list)
            or not members
            or not all(isinstance(m, str) for m in members)
        ):
            raise ValueError(f"{SETUP} [groups] {name}: give a list of wall family names")
        if unknown := [m for m in members if m not in walls]:
            raise ValueError(
                f"{SETUP} [groups] {name}: {', '.join(unknown)} is not a wall family; "
                f"the walls are: {', '.join(walls)}"
            )
        groups[str(name)] = tuple(members)
    return groups


def _raw(setup: Mapping[str, object]) -> tuple[RawKeyword, ...]:
    table = setup.get("keywords", {})
    if not isinstance(table, Mapping):
        raise ValueError(f"{SETUP}: [keywords] maps a solver keyword to its value")
    result = []
    for name, entry in table.items():
        if not name or any(character.isspace() for character in name):
            raise ValueError(
                f"{SETUP} [keywords]: keyword name {name!r} must be one non-empty word "
                "without whitespace"
            )
        where = f"{SETUP} [keywords] {name}"
        section = None
        value = entry
        if isinstance(entry, Mapping):
            if "value" not in entry or set(entry) - {"value", "section"}:
                raise ValueError(f"{where}: an inline table holds value and an optional section")
            value = entry["value"]
            hint = entry.get("section")
            if hint is not None and (not isinstance(hint, str) or not hint):
                raise ValueError(f"{where}: section names a section of the input")
            section = hint
        items = value if isinstance(value, list) else [value]
        if not all(_is_scalar(item) for item in items):
            raise ValueError(
                f"{where}: a raw keyword takes a number, a word, or a flat list of them; "
                f"found {value!r}"
            )
        result.append(RawKeyword(str(name), tuple(items), section))
    return tuple(result)


def _is_scalar(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float | str)
        and (not isinstance(value, float) or math.isfinite(value))
    )
