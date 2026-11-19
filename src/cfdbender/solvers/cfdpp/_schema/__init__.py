"""The interface database of the solver input: versions, entries and their evidence.

The database is data only, TOML files under ``data/``: ``_meta.toml`` lists the
supported solver versions in vendor release order, and every other file holds
entries (sections of the input, keywords, information-set tags, boundary codes,
turbulence models, interdependency rules, tools, files, output layouts and physics
combinations). This module loads it, checks it, and gives the view of one version
as a :class:`Schema`.

Every entry carries one status row per registered version, and may carry a meaning and a
locator (where its fact was read). A status is promoted only by measured evidence:
``documented`` needs nothing more; ``harvested``, ``verified``, ``broken`` and ``removed`` each
need a committed report. No entry holds a default value: the setup file states
every value, so a ``default`` or ``value`` field fails the load.

The engine holds no keyword name of the real solver: everything real comes from the
data. A test may point any function here at another database folder (the toy
database of the tests) with the ``data`` argument.
"""

from __future__ import annotations

import functools
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

DATA = Path(__file__).resolve().parent / "data"
META = "_meta.toml"

Scalar = int | float | str


class Status(StrEnum):
    """Evidence class of an entry for one solver version."""

    DOCUMENTED = "documented"
    HARVESTED = "harvested"
    VERIFIED = "verified"
    BROKEN = "broken"
    REMOVED = "removed"


EMITTED = frozenset({Status.HARVESTED, Status.VERIFIED})
NEEDS_REPORT = frozenset({Status.HARVESTED, Status.VERIFIED, Status.BROKEN, Status.REMOVED})
VALUE_TYPES = ("int", "real", "word", "choice")
ENTRY_KINDS = ("pairs", "infosets", "boundaries", "lines")
RULE_KINDS = ("requires", "excludes", "not_greater", "count")
LEVELS = ("hard", "advisory")
LINE_ENDINGS = {"lf": "\n", "crlf": "\r\n"}
#: ``setup`` binds the keys of the setup and post-processing files to the input: one
#: entry per key, read by the adapter's configuration (``_config``), not by this loader.
GENERIC_TABLES = ("tools", "files", "outputs", "combinations", "setup")
FORBIDDEN_FIELDS = frozenset({"default", "defaults", "value"})
_COMMON = frozenset({"meaning", "locator", "status"})
_TABLE_KEYS = {
    "sections": "name",
    "keywords": "name",
    "infosets": "tag",
    "boundary_codes": "code",
    "turbulence": "name",
    "rules": "id",
    **dict.fromkeys(GENERIC_TABLES, "name"),
}
_FIELDS = {
    "sections": {"name", "order", "kind", "open", "close", "counted", "head", "header"},
    "keywords": {"name", "section", "order", "arity", "types", "choices", "range", "fixed"}
    | {"repeat"},
    "infosets": {"tag", "variables", "types", "fixed", "turbulence", "role"},
    "boundary_codes": {"code", "params", "neutral"},
    "turbulence": {"name", "codes", "variables", "equation_set"},
    "rules": {"id", "kind", "level", "keyword", "index", "other", "other_index", "section"}
    | {"message"},
}


@dataclass(frozen=True)
class Version:
    """One registered solver version of ``_meta.toml``."""

    canonical: str
    aliases: tuple[str, ...]
    prints: str | None
    inherits_base: bool | None
    help_edition: str | None


@dataclass(frozen=True)
class Section:
    """One section of the input: its place, its opening and closing lines, its entry kind.

    A section is enclosed (``open`` and ``close`` lines), headed (``head``: a line of
    its head word and its entry count, then the optional ``header`` line) or bare (none
    of them: a run of its own keywords).
    """

    name: str
    order: int | None
    kind: str
    open: str | None
    close: str | None
    counted: bool
    meaning: str
    locator: str
    status: Status
    report: str | None
    head: str | None = None
    header: str | None = None


@dataclass(frozen=True)
class Keyword:
    """One keyword of the input; ``fixed`` is a constant of the supported scope, not a default.

    A last type ending in ``...`` takes any number of values of that type; ``repeat``
    lets the keyword appear more than once in its section.
    """

    name: str
    section: str | None
    order: int | None
    arity: int | None
    types: tuple[str, ...] | None
    choices: tuple[Scalar, ...] | None
    range: tuple[float | None, float | None] | None
    fixed: Scalar | tuple[Scalar, ...] | None
    meaning: str
    locator: str
    status: Status
    report: str | None
    repeat: bool = False


@dataclass(frozen=True)
class InfoSetTag:
    """One information-set tag and the order of its variables.

    ``types`` gives each variable's type (real when absent); ``fixed`` holds the
    constants of the supported scope by variable; ``turbulence`` says the variables of
    the turbulence model follow; ``role`` names what cfdbender uses the set for.
    """

    tag: str
    variables: tuple[str, ...] | None
    meaning: str
    locator: str
    status: Status
    report: str | None
    types: tuple[str, ...] | None = None
    fixed: Mapping[str, Scalar] = MappingProxyType({})
    turbulence: bool = False
    role: str | None = None


@dataclass(frozen=True)
class Param:
    """One parameter of a boundary code; ``fixed`` is a constant of the supported scope."""

    name: str
    type: str
    fixed: Scalar | None = None


@dataclass(frozen=True)
class BoundaryCode:
    """One numeric boundary-condition code, its parameters and its neutral meaning."""

    code: int
    params: tuple[Param, ...] | None
    neutral: Mapping[str, str]
    meaning: str
    locator: str
    status: Status
    report: str | None


@dataclass(frozen=True)
class Turbulence:
    """One turbulence model: its neutral name, its codes and its transported variables.

    ``equation_set`` holds the values the model writes into the variables of the
    equation-set information set, by variable name.
    """

    name: str
    codes: tuple[int, ...] | None
    variables: tuple[str, ...] | None
    meaning: str
    locator: str
    status: Status
    report: str | None
    equation_set: Mapping[str, int] = MappingProxyType({})


@dataclass(frozen=True)
class Rule:
    """One interdependency rule over an input model, ``hard`` or ``advisory``."""

    id: str
    kind: str
    level: str
    keyword: str
    index: int
    other: str | None
    other_index: int
    section: str | None
    message: str
    locator: str
    status: Status
    report: str | None


@dataclass(frozen=True)
class Entry:
    """One entry of a table the engine does not interpret (tools, files, outputs, combinations)."""

    table: str
    name: str
    attrs: Mapping[str, Any]
    meaning: str
    locator: str
    status: Status
    report: str | None


@dataclass(frozen=True)
class NumberFormat:
    """How reals are printed (``shortest`` or a format specification) and the line ending."""

    real: str
    line_ending: str


@dataclass(frozen=True)
class Layout:
    """How the lines of the input look beyond keyword pairs.

    ``comment`` is the prefix of a comment line; ``infoset_header`` the title line of
    a multi-line information set (with ``{index}``, ``{count}`` and ``{tag}``),
    ``infoset_values`` the word that opens each of its value lines,
    ``infoset_per_line`` the values per line and ``infoset_trailing_space`` whether a
    value line ends with a space; ``boundary_width`` the right-aligned width of the
    numeric columns of a boundary row and ``boundary_family`` whether the row ends
    with the family name; ``every_section`` says a generated input writes every
    enclosed section, an empty one as its opening and closing lines.
    """

    comment: str | None = None
    infoset_header: str | None = None
    infoset_values: str | None = None
    infoset_per_line: int | None = None
    infoset_trailing_space: bool = False
    boundary_width: int | None = None
    boundary_family: bool = False
    every_section: bool = False


@dataclass(frozen=True)
class Schema:
    """The view of the interface database for one solver version."""

    solver: str
    input_file: str
    version: str
    sections: tuple[Section, ...]
    keywords: Mapping[str, Keyword]
    infosets: Mapping[str, InfoSetTag]
    boundary_codes: Mapping[int, BoundaryCode]
    turbulence: Mapping[str, Turbulence]
    rules: tuple[Rule, ...]
    entries: Mapping[str, tuple[Entry, ...]]
    number_format: NumberFormat | None
    raw_section: str | None
    layout: Layout | None = None

    def section(self, name: str) -> Section:
        """Return the section called ``name``, refusing an unknown one with the list."""
        for section in self.sections:
            if section.name == name:
                return section
        known = ", ".join(section.name for section in self.sections)
        raise ValueError(
            f"{self.solver} {self.version}: no section {name!r} in the input; "
            f"the sections are: {known}"
        )

    def require_grammar(self) -> NumberFormat:
        """Return the number format, refusing a schema whose layout is not yet established."""
        missing = [section.name for section in self.sections if not _placed(section)]
        if self.number_format is None or missing or not self.sections:
            raise ValueError(
                f"{self.solver} {self.version}: the layout of {self.input_file} is not "
                "established in the interface database yet (number format, section order "
                "and opening lines), so cfdbender can neither read nor write it"
            )
        return self.number_format

    def emittable(self, name: str) -> bool:
        """Say whether cfdbender may generate keyword ``name`` for this version.

        A ``harvested`` or ``verified`` keyword may be generated; a ``documented``
        one may not (a user may still write it raw). A ``broken`` or ``removed``
        keyword, and a name the database does not hold, are refused.
        """
        keyword = self.keywords.get(name)
        if keyword is None:
            raise ValueError(
                f"{self.solver} {self.version}: keyword {name} is not in the interface "
                "database, so cfdbender cannot generate it"
            )
        if keyword.status is Status.BROKEN:
            raise ValueError(
                f"{self.solver} {self.version}: keyword {name} is broken for this version "
                f"(report {keyword.report}): measured evidence shows the solver does not "
                "behave as its help says"
            )
        if keyword.status is Status.REMOVED:
            raise ValueError(
                f"{self.solver} {self.version}: keyword {name} is removed in this version "
                f"(report {keyword.report}): measured evidence shows this version does not "
                "take it"
            )
        return keyword.status in EMITTED


def _placed(section: Section) -> bool:
    """Say whether a section's place and shape in the input are established."""
    if section.order is None:
        return False
    if section.head is not None:
        return section.open is None and section.close is None
    if section.open is not None or section.close is not None:
        return section.open is not None and section.close is not None
    return section.kind == "pairs"


def supported_versions(data: Path | None = None) -> tuple[str, ...]:
    """Return the canonical names of the registered versions, in vendor release order."""
    return tuple(version.canonical for version in _meta(_root(data))[2])


def canonical(version: str | None, data: Path | None = None) -> str:
    """Return the canonical name of ``version``: itself, an alias, or either with ``-suffix``."""
    solver, input_file, versions = _meta(_root(data))
    supported = ", ".join(entry.canonical for entry in versions)
    given = (version or "").strip()
    if not given:
        raise ValueError(
            f"SOLVER_BUILD is required for {solver}: state the solver version of this row "
            f"(supported: {supported})"
        )
    for entry in versions:
        if given == entry.canonical or given in entry.aliases:
            return entry.canonical
    for entry in versions:
        for name in (entry.canonical, *entry.aliases):
            if given.startswith(name + "-"):
                return entry.canonical
    raise ValueError(
        f"{solver} {given}: cfdbender writes {input_file} for {supported} only; no evidence "
        "exists for this version (see the version-onboarding page)"
    )


def schema_for(version: str | None, data: Path | None = None) -> Schema:
    """Return the schema of ``version``, refusing an unknown or unstated version.

    A version registered without ``inherits_base`` is refused: whether a version
    inherits the evidence of the previous one is decided before anything is loaded.
    """
    root = _root(data)
    name = canonical(version, root)
    solver, _, versions = _meta(root)
    entry = next(item for item in versions if item.canonical == name)
    if entry.inherits_base is None:
        raise ValueError(
            f"{solver} {name} is registered in {META} without an inheritance statement: "
            "write inherits_base (true or false) for it before its schema can load"
        )
    return _schema(root, name)


def locator_index(schema: Schema) -> dict[str, tuple[str, ...]]:
    """List the locators shared by more than one entry, each with its ``table name`` labels."""
    seen: dict[str, list[str]] = {}
    for table, name, locator in _labelled(schema):
        seen.setdefault(locator, []).append(f"{table} {name}")
    return {locator: tuple(sorted(names)) for locator, names in seen.items() if len(names) > 1}


def _labelled(schema: Schema) -> Iterable[tuple[str, str, str]]:
    for section in schema.sections:
        yield "sections", section.name, section.locator
    for keyword in schema.keywords.values():
        yield "keywords", keyword.name, keyword.locator
    for tag in schema.infosets.values():
        yield "infosets", tag.tag, tag.locator
    for code in schema.boundary_codes.values():
        yield "boundary_codes", str(code.code), code.locator
    for model in schema.turbulence.values():
        yield "turbulence", model.name, model.locator
    for rule in schema.rules:
        yield "rules", rule.id, rule.locator
    for table, entries in schema.entries.items():
        for entry in entries:
            yield table, entry.name, entry.locator


def _root(data: Path | None) -> Path:
    return (data if data is not None else DATA).resolve()


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path.name}: not valid TOML: {error}") from None


@functools.cache
def _meta(root: Path) -> tuple[str, str, tuple[Version, ...]]:
    raw = _read(root / META)
    unknown = sorted(set(raw) - {"solver", "input_file", "versions"})
    if unknown:
        raise ValueError(f"{META}: unknown keys {', '.join(unknown)}")
    solver = _text(raw.get("solver"), f"{META}: solver")
    input_file = _text(raw.get("input_file"), f"{META}: input_file")
    versions: list[Version] = []
    for item in raw.get("versions") or []:
        if not isinstance(item, dict):
            raise ValueError(f"{META}: each version is a mapping")
        name = _text(item.get("canonical"), f"{META}: canonical")
        extra = sorted(
            set(item) - {"canonical", "aliases", "prints", "inherits_base", "help_edition"}
        )
        if extra:
            raise ValueError(f"{META}: version {name} has unknown keys {', '.join(extra)}")
        inherits = item.get("inherits_base")
        if inherits is not None and not isinstance(inherits, bool):
            raise ValueError(f"{META}: inherits_base of {name} must be true or false")
        versions.append(
            Version(
                canonical=name,
                aliases=tuple(str(alias) for alias in item.get("aliases") or ()),
                prints=_optional_text(item.get("prints"), f"{META}: prints of {name}"),
                inherits_base=inherits,
                help_edition=_optional_text(
                    item.get("help_edition"), f"{META}: help_edition of {name}"
                ),
            )
        )
    if not versions:
        raise ValueError(f"{META}: no version is registered")
    names = [name for entry in versions for name in (entry.canonical, *entry.aliases)]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{META}: version names registered twice: {', '.join(repeated)}")
    return solver, input_file, tuple(versions)


@functools.cache
def _tables(root: Path) -> tuple[dict[str, list[tuple[str, dict[str, Any]]]], dict[str, Any]]:
    tables: dict[str, list[tuple[str, dict[str, Any]]]] = {name: [] for name in _TABLE_KEYS}
    grammar: dict[str, Any] = {}
    for path in sorted(root.glob("*.toml")):
        if path.name == META:
            continue
        raw = _read(path)
        for key, value in raw.items():
            if key == "grammar":
                if grammar:
                    raise ValueError(f"{path.name}: the grammar is stated in two files")
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}: grammar must be a mapping")
                grammar = dict(value)
                continue
            if key not in tables:
                known = ", ".join(["grammar", *tables])
                raise ValueError(f"{path.name}: unknown table {key!r}; the tables are: {known}")
            for item in value or []:
                if not isinstance(item, dict):
                    raise ValueError(f"{path.name}: each entry of {key} is a mapping")
                tables[key].append((path.name, item))
    return tables, grammar


@functools.cache
def _schema(root: Path, version: str) -> Schema:
    solver, input_file, versions = _meta(root)
    registered = tuple(entry.canonical for entry in versions)
    tables, grammar = _tables(root)
    for table, items in tables.items():
        _check_entries(table, items, registered)

    def evidence(where: str, item: dict[str, Any]) -> tuple[Status, str | None]:
        row = item["status"][version]
        return Status(row["status"]), row.get("report")

    sections = []
    for where, item in tables["sections"]:
        status, report = evidence(where, item)
        kind = item.get("kind")
        if kind not in ENTRY_KINDS:
            raise ValueError(
                f"{where}: section {item['name']} has kind {kind!r}; "
                f"the kinds are: {', '.join(ENTRY_KINDS)}"
            )
        sections.append(
            Section(
                name=item["name"],
                order=_optional_int(item.get("order"), f"{where}: order of {item['name']}"),
                kind=kind,
                open=_optional_text(item.get("open"), f"{where}: open of {item['name']}"),
                close=_optional_text(item.get("close"), f"{where}: close of {item['name']}"),
                counted=bool(item.get("counted", False)),
                meaning=item.get("meaning", ""),
                locator=item.get("locator", ""),
                status=status,
                report=report,
                head=_optional_text(item.get("head"), f"{where}: head of {item['name']}"),
                header=_optional_text(item.get("header"), f"{where}: header of {item['name']}"),
            )
        )
    sections.sort(key=lambda section: (section.order is None, section.order or 0))
    section_names = {section.name for section in sections}

    keywords: dict[str, Keyword] = {}
    places: dict[tuple[str, int], str] = {}
    for where, item in tables["keywords"]:
        keyword = _keyword(where, item, section_names, *evidence(where, item))
        if keyword.section is not None and keyword.order is not None:
            place = (keyword.section, keyword.order)
            if place in places:
                raise ValueError(
                    f"{where}: keywords {places[place]} and {keyword.name} share order "
                    f"{keyword.order} in section {keyword.section}"
                )
            places[place] = keyword.name
        keywords[keyword.name] = keyword

    infosets = {}
    for where, item in tables["infosets"]:
        infosets[item["tag"]] = _infoset(where, item, *evidence(where, item))

    codes = {}
    for where, item in tables["boundary_codes"]:
        code = item["code"]
        codes[code] = BoundaryCode(
            code=code,
            params=_params(item.get("params"), f"{where}: params of code {code}"),
            neutral=MappingProxyType(
                {str(key): str(value) for key, value in (item.get("neutral") or {}).items()}
            ),
            meaning=item.get("meaning", ""),
            locator=item.get("locator", ""),
            status=evidence(where, item)[0],
            report=evidence(where, item)[1],
        )

    turbulence = {}
    for where, item in tables["turbulence"]:
        codes_raw = item.get("codes")
        if codes_raw is not None and not all(
            isinstance(code, int) and not isinstance(code, bool) for code in codes_raw
        ):
            raise ValueError(f"{where}: codes of {item['name']} must be integers")
        written = item.get("equation_set") or {}
        if not isinstance(written, dict) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in written.values()
        ):
            raise ValueError(f"{where}: equation_set of {item['name']} maps names to integers")
        turbulence[item["name"]] = Turbulence(
            equation_set=MappingProxyType({str(k): int(v) for k, v in written.items()}),
            name=item["name"],
            codes=tuple(codes_raw) if codes_raw is not None else None,
            variables=_names(item.get("variables"), f"{where}: variables of {item['name']}"),
            meaning=item.get("meaning", ""),
            locator=item.get("locator", ""),
            status=evidence(where, item)[0],
            report=evidence(where, item)[1],
        )

    rules = tuple(
        _rule(where, item, keywords, section_names, *evidence(where, item))
        for where, item in tables["rules"]
    )

    entries = {
        table: tuple(
            Entry(
                table=table,
                name=str(item["name"]),
                attrs=MappingProxyType(
                    {key: value for key, value in item.items() if key not in _COMMON | {"name"}}
                ),
                meaning=item.get("meaning", ""),
                locator=item.get("locator", ""),
                status=evidence(where, item)[0],
                report=evidence(where, item)[1],
            )
            for where, item in tables[table]
        )
        for table in GENERIC_TABLES
    }

    number_format, raw_section, layout = _grammar(grammar, sections)
    return Schema(
        solver=solver,
        input_file=input_file,
        version=version,
        sections=tuple(sections),
        keywords=MappingProxyType(keywords),
        infosets=MappingProxyType(infosets),
        boundary_codes=MappingProxyType(codes),
        turbulence=MappingProxyType(turbulence),
        rules=rules,
        entries=MappingProxyType(entries),
        number_format=number_format,
        raw_section=raw_section,
        layout=layout,
    )


def _infoset(where: str, item: dict[str, Any], status: Status, report: str | None) -> InfoSetTag:
    tag = item["tag"]
    label = f"{where}: information set {tag}"
    variables = _names(item.get("variables"), f"{label} variables")
    types = _names(item.get("types"), f"{label} types")
    if types is not None:
        if variables is None or len(types) != len(variables):
            raise ValueError(f"{label}: types must list one type per variable")
        if bad := sorted(set(types) - {"int", "real"}):
            raise ValueError(f"{label}: variable types are int or real; found {', '.join(bad)}")
    fixed_raw = item.get("fixed") or {}
    if not isinstance(fixed_raw, dict):
        raise ValueError(f"{label}: fixed is a table of variable names and values")
    if unknown := sorted(set(fixed_raw) - set(variables or ())):
        raise ValueError(f"{label}: fixed names unknown variables {', '.join(unknown)}")
    turbulence = item.get("turbulence", False)
    if not isinstance(turbulence, bool):
        raise ValueError(f"{label}: turbulence must be true or false")
    return InfoSetTag(
        tag=tag,
        variables=variables,
        meaning=item.get("meaning", ""),
        locator=item.get("locator", ""),
        status=status,
        report=report,
        types=types,
        fixed=MappingProxyType({str(k): _scalar(v, label) for k, v in fixed_raw.items()}),
        turbulence=turbulence,
        role=_optional_text(item.get("role"), f"{label} role"),
    )


def _check_entries(
    table: str, items: list[tuple[str, dict[str, Any]]], registered: tuple[str, ...]
) -> None:
    key = _TABLE_KEYS[table]
    names = [str(item.get(key)) for _, item in items]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"duplicate {table} entries: {', '.join(repeated)}")
    for where, item in items:
        name = item.get(key)
        if name is None or (isinstance(name, str) and not name.strip()):
            raise ValueError(f"{where}: an entry of {table} has no {key}")
        label = f"{where}: {table} entry {name}"
        held = sorted(FORBIDDEN_FIELDS & set(item))
        if held:
            raise ValueError(
                f"{label} holds {', '.join(held)}; the interface database holds no default "
                "or value: the setup file states every value"
            )
        allowed = _FIELDS.get(table)
        if allowed is not None:
            unknown = sorted(set(item) - allowed - _COMMON)
            if unknown:
                raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")
        # A rule's message is its meaning, in our own words. A meaning and a locator are
        # optional; when present they are text.
        if table == "rules" and (
            not isinstance(item.get("message"), str) or not item["message"].strip()
        ):
            raise ValueError(f"{label} has no message")
        for field in ("meaning", "locator"):
            if field in item and not isinstance(item[field], str):
                raise ValueError(f"{label} has a {field} that is not text")
        _check_status(label, item.get("status"), registered)


def _check_status(label: str, rows: Any, registered: tuple[str, ...]) -> None:
    if not isinstance(rows, dict):
        raise ValueError(f"{label} has no status rows; one per version: {', '.join(registered)}")
    missing = [version for version in registered if version not in rows]
    if missing:
        raise ValueError(f"{label} has no status row for version {', '.join(missing)}")
    extra = sorted(str(version) for version in rows if version not in registered)
    if extra:
        raise ValueError(
            f"{label} has a status row for {', '.join(extra)}, which is not registered in {META}"
        )
    for version, row in rows.items():
        if not isinstance(row, dict) or set(row) - {"status", "report"}:
            raise ValueError(f"{label}: the {version} row holds status and report only")
        try:
            status = Status(str(row.get("status")))
        except ValueError:
            known = ", ".join(item.value for item in Status)
            raise ValueError(
                f"{label}: status {row.get('status')!r} for {version} is unknown; "
                f"the statuses are: {known}"
            ) from None
        report = row.get("report")
        if status in NEEDS_REPORT and (not isinstance(report, str) or not report.strip()):
            raise ValueError(
                f"{label} is {status.value} for {version} without a report; a status above "
                "documented is promoted only by a committed report"
            )


def _keyword(
    where: str,
    item: dict[str, Any],
    sections: set[str],
    status: Status,
    report: str | None,
) -> Keyword:
    name = item["name"]
    label = f"{where}: keyword {name}"
    section = _optional_text(item.get("section"), f"{label} section")
    if section is not None and section not in sections:
        raise ValueError(f"{label} names section {section!r}, which the database does not hold")
    arity = _optional_int(item.get("arity"), f"{label} arity")
    if arity is not None and arity < 1:
        raise ValueError(f"{label}: arity must be at least 1")
    types = _names(item.get("types"), f"{label} types")
    if types is not None:
        if any(kind.endswith("...") for kind in types[:-1]):
            raise ValueError(f"{label}: only the last type may take any number of values")
        bad = [kind for kind in types if kind.removesuffix("...") not in VALUE_TYPES]
        if bad:
            raise ValueError(
                f"{label}: unknown value types {', '.join(bad)}; "
                f"the types are: {', '.join(VALUE_TYPES)}"
            )
        if arity is None or len(types) != arity:
            raise ValueError(f"{label}: types must list one type per value (arity {arity})")
    choices_raw = item.get("choices")
    choices = None if choices_raw is None else tuple(_scalar(v, label) for v in choices_raw)
    if types is not None and ("choice" in types) != (choices is not None):
        raise ValueError(f"{label}: a choice type and a choices list go together")
    bounds = item.get("range")
    value_range: tuple[float | None, float | None] | None = None
    if bounds is not None:
        if not isinstance(bounds, dict) or set(bounds) - {"min", "max"}:
            raise ValueError(f"{label}: range is a table of min and max, an open end omitted")
        value_range = (_bound(bounds.get("min"), label), _bound(bounds.get("max"), label))
    fixed_raw = item.get("fixed")
    fixed: Scalar | tuple[Scalar, ...] | None
    if isinstance(fixed_raw, list):
        fixed = tuple(_scalar(v, label) for v in fixed_raw)
    else:
        fixed = None if fixed_raw is None else _scalar(fixed_raw, label)
    order = _optional_int(item.get("order"), f"{label} order")
    repeat = item.get("repeat", False)
    if not isinstance(repeat, bool):
        raise ValueError(f"{label}: repeat must be true or false")
    if status in EMITTED and (section is None or order is None or types is None):
        raise ValueError(
            f"{label} is {status.value} without its section, order, arity and types; "
            "evidence of a keyword includes its place in the input"
        )
    return Keyword(
        name=name,
        section=section,
        order=order,
        arity=arity,
        types=types,
        choices=choices,
        range=value_range,
        fixed=fixed,
        meaning=item.get("meaning", ""),
        locator=item.get("locator", ""),
        status=status,
        report=report,
        repeat=repeat,
    )


def _rule(
    where: str,
    item: dict[str, Any],
    keywords: Mapping[str, Keyword],
    sections: set[str],
    status: Status,
    report: str | None,
) -> Rule:
    label = f"{where}: rule {item['id']}"
    kind = item.get("kind")
    if kind not in RULE_KINDS:
        raise ValueError(f"{label}: kind {kind!r}; the kinds are: {', '.join(RULE_KINDS)}")
    level = item.get("level")
    if level not in LEVELS:
        raise ValueError(f"{label}: level {level!r}; the levels are: {', '.join(LEVELS)}")
    keyword = _text(item.get("keyword"), f"{label} keyword")
    other = _optional_text(item.get("other"), f"{label} other")
    section = _optional_text(item.get("section"), f"{label} section")
    for name in (keyword, other):
        if name is not None and name not in keywords:
            raise ValueError(f"{label} names keyword {name}, which the database does not hold")
    if kind == "count" and section not in sections:
        raise ValueError(f"{label}: a count rule names a section of the database")
    if kind != "count" and other is None:
        raise ValueError(f"{label}: a {kind} rule names an other keyword")
    return Rule(
        id=str(item["id"]),
        kind=kind,
        level=level,
        keyword=keyword,
        index=_optional_int(item.get("index"), f"{label} index") or 0,
        other=other,
        other_index=_optional_int(item.get("other_index"), f"{label} other_index") or 0,
        section=section,
        message=_text(item.get("message"), f"{label} message"),
        locator=item.get("locator", ""),
        status=status,
        report=report,
    )


_GRAMMAR_KEYS = {"number_format", "line_ending", "raw_section", "comment", "infosets", "boundaries"}
_GRAMMAR_KEYS |= {"every_section"}


def _layout(grammar: dict[str, Any]) -> Layout | None:
    comment = _optional_text(grammar.get("comment"), "grammar: comment")
    every = grammar.get("every_section", False)
    if not isinstance(every, bool):
        raise ValueError("grammar: every_section must be true or false")
    sets = grammar.get("infosets") or {}
    rows = grammar.get("boundaries") or {}
    if not isinstance(sets, dict) or set(sets) - {"header", "values", "per_line", "trailing_space"}:
        raise ValueError("grammar: infosets holds header, values, per_line and trailing_space")
    if not isinstance(rows, dict) or set(rows) - {"width", "family"}:
        raise ValueError("grammar: boundaries holds width and family")
    if comment is None and not sets and not rows and not every:
        return None
    header = _optional_text(sets.get("header"), "grammar: infosets header")
    if header is not None and not {"{index}", "{count}", "{tag}"} <= set(header.split()):
        raise ValueError("grammar: infosets header names {index}, {count} and {tag}")
    if (header is None) != (sets.get("values") is None):
        raise ValueError("grammar: infosets header and values go together")
    per_line = _optional_int(sets.get("per_line"), "grammar: infosets per_line")
    width = _optional_int(rows.get("width"), "grammar: boundaries width")
    if (per_line is not None and per_line < 1) or (width is not None and width < 1):
        raise ValueError("grammar: per_line and width are positive")
    for name, value in (("infosets trailing_space", sets.get("trailing_space", False)),
                        ("boundaries family", rows.get("family", False))):  # fmt: skip
        if not isinstance(value, bool):
            raise ValueError(f"grammar: {name} must be true or false")
    return Layout(
        comment=comment,
        infoset_header=header,
        infoset_values=_optional_text(sets.get("values"), "grammar: infosets values"),
        infoset_per_line=per_line,
        infoset_trailing_space=bool(sets.get("trailing_space", False)),
        boundary_width=width,
        boundary_family=bool(rows.get("family", False)),
        every_section=every,
    )


def _grammar(
    grammar: dict[str, Any], sections: list[Section]
) -> tuple[NumberFormat | None, str | None, Layout | None]:
    unknown = sorted(set(grammar) - _GRAMMAR_KEYS)
    if unknown:
        raise ValueError(f"grammar: unknown keys {', '.join(unknown)}")
    ending = grammar.get("line_ending", "lf")
    if ending not in LINE_ENDINGS:
        raise ValueError(f"grammar: line_ending {ending!r}; use lf or crlf")
    raw_format = grammar.get("number_format")
    number_format = None
    if raw_format is not None:
        if not isinstance(raw_format, dict) or set(raw_format) != {"real"}:
            raise ValueError("grammar: number_format holds real only")
        real = _text(raw_format["real"], "grammar: number_format real")
        if real != "shortest":
            try:
                format(1.5, real)
            except ValueError:
                raise ValueError(
                    f"grammar: number_format real {real!r} is neither shortest nor a "
                    "format specification"
                ) from None
        number_format = NumberFormat(real=real, line_ending=LINE_ENDINGS[ending])
    raw_section = _optional_text(grammar.get("raw_section"), "grammar: raw_section")
    if raw_section is not None:
        kinds = {section.name: section.kind for section in sections}
        if kinds.get(raw_section) != "pairs":
            raise ValueError(
                f"grammar: raw_section {raw_section!r} must name a section of keyword pairs"
            )
    return number_format, raw_section, _layout(grammar)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty text")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _names(value: Any, label: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of names")
    return tuple(value)


def _params(value: Any, label: str) -> tuple[Param, ...] | None:
    if value is None:
        return None
    params = []
    for item in value:
        if not isinstance(item, dict) or not {"name", "type"} <= set(item) <= {
            "name",
            "type",
            "fixed",
        }:
            raise ValueError(f"{label}: each parameter is {{name, type}} and optionally fixed")
        if item["type"] not in ("int", "real", "word"):
            raise ValueError(f"{label}: parameter type {item['type']!r} is not int, real or word")
        fixed = item.get("fixed")
        params.append(
            Param(
                name=str(item["name"]),
                type=item["type"],
                fixed=None if fixed is None else _scalar(fixed, label),
            )
        )
    return tuple(params)


def _scalar(value: Any, label: str) -> Scalar:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{label}: {value!r} is not an integer, a real or a word")
    scalar: Scalar = value
    return scalar


def _bound(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label}: a range bound is a number")
    bound: float = value
    return bound
