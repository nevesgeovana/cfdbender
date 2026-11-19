"""Turn a checked case into a model of the solver input: the one place that maps choices.

Every keyword, information-set tag and boundary code comes from the interface database
of the row's version; this module only follows its bindings:

* each keyword the database fixes for the supported scope is written with that value;
* each setup key bound to a keyword writes its value there (a word through the
  key's ``values``, a turbulence model through its codes);
* one information set holds the freestream, its tag chosen as the one whose
  variables are exactly the case's (pressure, temperature, the velocity components
  of the grid's dimension and the model's turbulence variables); it is the initial
  state and every far field's state;
* each family's boundary block takes the code whose neutral description equals the
  family's declaration, with its parameters by name (``temperature`` from the wall,
  ``set`` the freestream set);
* the ``[keywords]`` entries are placed by what the database knows of each.

cfdbender generates a keyword, a tag or a code only when its status for the version is
harvested or verified; a needed one with documentation only is refused, naming it.
What a user writes in ``[keywords]`` is the user's responsibility: placed and
type-checked where the database knows the keyword, written as given otherwise, and
never warned about; plan
``--verbose`` lists each with its evidence class.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from cfdbender.solvers.cfdpp._config import (
    SETUP,
    Case,
    RawKeyword,
    display,
    setup_keys,
)
from cfdbender.solvers.cfdpp._schema import (
    EMITTED,
    InfoSetTag,
    Keyword,
    Scalar,
    Schema,
    Status,
)
from cfdbender.solvers.cfdpp._schema._model import Block, Boundary, InfoSet, InputModel, Pair
from cfdbender.solvers.cfdpp._schema._model import render as render_model
from cfdbender.solvers.cfdpp._schema._rules import check

#: Values cfdbender states itself for a ``source = "case"`` key, by key name.
CASE_SOURCES = ("restart", "set_count", "initial_set")
#: The freestream is the first and only information set.
FREESTREAM_SET = 1
COMBINATION_FIELDS = ("method", "turbulence", "treatment", "dimension")


@dataclass(frozen=True)
class Built:
    """A built input: its model, the free keywords, plan lines and the silent fields."""

    model: InputModel
    free: frozenset[str]
    lines: tuple[str, ...]
    raw_keywords: tuple[Mapping[str, object], ...]
    schema_status: str
    advisories: tuple[str, ...]


def build(case: Case) -> Built:
    """Return the input model of ``case``, refusing what the database cannot support."""
    schema = case.schema
    schema.require_grammar()
    where = f"{schema.solver} {schema.version}"
    pairs: dict[str, list[Scalar | None]] = {}
    origin: dict[str, str] = {}

    def need(name: str, why: str) -> Keyword:
        try:
            emitted = schema.emittable(name)
        except ValueError as error:
            raise ValueError(f"{why}: {error}") from None
        keyword = schema.keywords[name]
        if not emitted:
            raise ValueError(
                f"{where}: {why} needs keyword {name}, which has no harvested or verified "
                f"evidence for this version ({keyword.status.value} only); cfdbender "
                "does not write it yet, so choose a setting that does not need it"
            )
        return keyword

    for keyword in schema.keywords.values():
        if keyword.fixed is not None:
            need(keyword.name, "the supported scope")
            fixed = keyword.fixed if isinstance(keyword.fixed, tuple) else (keyword.fixed,)
            pairs[keyword.name] = list(fixed)
            origin[keyword.name] = "fixed"

    equation_tag = _role(schema, "equation_set")
    # The freestream follows the equation set when the input defines one.
    freestream_set = FREESTREAM_SET + (1 if equation_tag is not None else 0)
    equation: dict[str, Scalar] = {}
    for key in setup_keys(schema):
        if key.keyword is None and not key.keywords and key.variable is None:
            continue
        if key.source is not None:
            if key.dimension:
                if str(case.dimension) not in key.dimension:
                    continue
                value: object = key.dimension[str(case.dimension)]
            elif key.name not in CASE_SOURCES:
                raise ValueError(
                    f"{where}: setup key {key.name} has source case, which cfdbender states "
                    f"only for: {', '.join(CASE_SOURCES)}, or by grid dimension"
                )
            else:
                value = {
                    "restart": int(case.restart),
                    "set_count": 1,
                    "initial_set": freestream_set,
                }[key.name]
        elif key.name in case.values:
            value = case.values[key.name]
        else:
            continue
        if isinstance(value, str) and key.source is None:
            known = schema.turbulence.get(value) if key.models else None
            if known is not None:
                if known.codes is None:
                    raise ValueError(f"{where}: turbulence {value} has no codes in the database")
                value = known.codes
            else:
                value = key.values[value]
        if key.divides is not None:
            dividend = schema.keywords[key.divides].fixed
            assert isinstance(dividend, int | float) and isinstance(value, int | float)
            need(key.divides, display(key.name))
            value = float(dividend) / float(value)
        if key.variable is not None:
            assert isinstance(value, int | float | str)
            equation[key.variable] = value
            continue
        if key.keywords:
            sequence = value if isinstance(value, tuple) else (value,)
            if len(sequence) != len(key.keywords):
                raise ValueError(
                    f"{where}: setup key {key.name} holds {len(sequence)} values for the "
                    f"{len(key.keywords)} keywords {', '.join(key.keywords)}"
                )
            for name, item in zip(key.keywords, sequence, strict=True):
                need(name, display(key.name))
                pairs[name] = [item]
                origin.setdefault(name, key.name)
            continue
        assert key.keyword is not None
        sequence = value if isinstance(value, tuple) else (value,)
        keyword = need(key.keyword, display(key.name))
        arity = keyword.arity or len(sequence)
        slots = pairs.setdefault(keyword.name, [None] * arity)
        if key.index + len(sequence) > arity:
            raise ValueError(
                f"{where}: setup key {key.name} writes {len(sequence)} values from position "
                f"{key.index + 1} of {keyword.name}, which takes {arity}"
            )
        for offset, item in enumerate(sequence):
            slots[key.index + offset] = item
        origin.setdefault(keyword.name, key.name)
    for name, slots in pairs.items():
        if any(item is None for item in slots):
            raise ValueError(
                f"{where}: keyword {name} takes {len(slots)} values and the setup keys bound "
                "to it state only some of them"
            )

    blocks: dict[str, list[object]] = {}

    def place(section: str, item: object) -> None:
        blocks.setdefault(section, []).append(item)

    for name, slots in pairs.items():
        keyword = schema.keywords[name]
        assert keyword.section is not None
        place(keyword.section, Pair(name, tuple(item for item in slots if item is not None)))

    infosets = _section_of_kind(schema, "infosets", where)
    lines = [f"write {schema.input_file} for {where}"]
    if equation_tag is not None:
        place(infosets, _equation_set(case, equation_tag, equation, where))
        lines.append(f"information set {equation_tag.tag} defines the equation set")
    tag = _infoset(case, where)
    variables = _variables_of(case, tag)
    place(infosets, InfoSet(tag.tag, tuple(case.freestream.value(v) for v in variables)))
    lines.append(f"information set {tag.tag} ({', '.join(variables)}) is the freestream")
    count = freestream_set
    walls: dict[str, int] = {}
    for boundary in case.boundaries:
        if boundary.temperature is None:
            continue
        wall_tag = _role(schema, "wall_temperature")
        if wall_tag is None or wall_tag.status not in EMITTED:
            break  # the boundary code then names what it lacks
        count += 1
        walls[boundary.family] = count
        place(infosets, InfoSet(wall_tag.tag, (boundary.temperature,)))
        lines.append(f"information set {wall_tag.tag} is the temperature of {boundary.family}")
    bcs = _section_of_kind(schema, "boundaries", where)
    named = schema.layout is not None and schema.layout.boundary_family
    for boundary in case.boundaries:
        code, values = _code(case, boundary, where, freestream_set, walls.get(boundary.family))
        family = boundary.family if named else None
        place(bcs, Boundary(boundary.selector, code, values, family))
        lines.append(f"boundary {boundary.family} (selector {boundary.selector}) -> code {code}")
    for entry in schema.entries.get("tools", ()):
        section = entry.attrs.get("section")
        if section is not None and entry.status in EMITTED:
            args = [str(arg) for arg in entry.attrs.get("args", ())]
            place(str(section), " ".join([entry.name, *args]))
    if schema.layout is not None and schema.layout.every_section:
        for section_entry in schema.sections:
            if section_entry.open is not None and section_entry.kind in ("pairs", "lines"):
                blocks.setdefault(section_entry.name, [])

    free, raw_lines, records = _raw(case, pairs, origin, place)
    lines += raw_lines
    model = InputModel(
        tuple(Block(section, tuple(items)) for section, items in blocks.items())  # type: ignore[arg-type]
    )
    violations = check(model, schema)
    hard = [v for v in violations if v.level == "hard"]
    if hard:
        raise ValueError(
            f"{where}: the input breaks " + "; ".join(f"rule {v.rule}: {v.message}" for v in hard)
        )
    advisories = tuple(f"{where}: rule {v.rule}: {v.message}" for v in violations if v not in hard)
    return Built(
        model=model,
        free=frozenset(free),
        lines=tuple(lines),
        raw_keywords=tuple(records),
        schema_status=_schema_status(case, where),
        advisories=advisories,
    )


def render(case: Case) -> str:
    """Return the text of the input of ``case``: deterministic, ASCII, the schema's endings."""
    built = build(case)
    return render_model(built.model, case.schema, free=built.free)


def describe(built: Built) -> tuple[str, ...]:
    """Return the plan lines of a built input (one per raw keyword, besides the blocks)."""
    return built.lines


def _section_of_kind(schema: Schema, kind: str, where: str) -> str:
    for section in schema.sections:
        if section.kind == kind:
            return section.name
    raise ValueError(f"{where}: the database holds no section of {kind}")


def _role(schema: Schema, role: str) -> InfoSetTag | None:
    """Return the information-set tag the database gives ``role``, if any."""
    return next((tag for tag in schema.infosets.values() if tag.role == role), None)


def _variables_of(case: Case, tag: InfoSetTag) -> tuple[str, ...]:
    assert tag.variables is not None
    return tuple(tag.variables) + (tuple(case.variables) if tag.turbulence else ())


def _equation_set(case: Case, tag: InfoSetTag, bound: Mapping[str, Scalar], where: str) -> InfoSet:
    """Build the equation set from the scope constants, the model's values and the bound keys."""
    if tag.status not in EMITTED:
        raise ValueError(
            f"{where}: the equation set needs information-set tag {tag.tag}, which has no "
            f"harvested or verified evidence for this version ({tag.status.value} only)"
        )
    values: dict[str, Scalar] = dict(tag.fixed)
    model = case.schema.turbulence.get(case.turbulence)
    if model is not None:
        values.update(model.equation_set)
    values.update(bound)
    assert tag.variables is not None
    missing = [name for name in tag.variables if name not in values]
    if missing:
        raise ValueError(
            f"{where}: the equation set {tag.tag} needs {', '.join(missing)}, which neither the "
            f"turbulence model {case.turbulence or 'none'} nor a setup key states"
        )
    return InfoSet(tag.tag, tuple(values[name] for name in tag.variables))


def _infoset(case: Case, where: str):  # type: ignore[no-untyped-def]
    schema = case.schema
    chosen = _role(schema, "freestream")
    if chosen is not None:
        if chosen.status not in EMITTED:
            raise ValueError(
                f"{where}: the freestream needs information-set tag {chosen.tag}, which has no "
                f"harvested or verified evidence for this version ({chosen.status.value} only)"
            )
        return chosen
    needed = ["p", "t", "u", "v", *(["w"] if case.dimension == 3 else []), *case.variables]
    for tag in schema.infosets.values():
        if tag.variables is not None and sorted(tag.variables) == sorted(needed):
            if tag.status not in EMITTED:
                raise ValueError(
                    f"{where}: the freestream needs information-set tag {tag.tag}, which has no "
                    f"harvested or verified evidence for this version ({tag.status.value} only)"
                )
            return tag
    known = ", ".join(
        f"{tag.tag} ({', '.join(tag.variables)})"
        for tag in schema.infosets.values()
        if tag.variables is not None
    )
    raise ValueError(
        f"{where}: no information-set tag holds the variables {', '.join(needed)}; "
        f"the tags with a known order are: {known or 'none'}"
    )


def _code(
    case: Case,
    boundary: object,
    where: str,
    freestream_set: int = FREESTREAM_SET,
    temperature_set: int | None = None,
) -> tuple[int, tuple[Scalar, ...]]:
    from cfdbender.solvers.cfdpp._config import Boundary as Declared

    assert isinstance(boundary, Declared)
    schema = case.schema
    wanted = dict(boundary.neutral)
    found = [code for code in schema.boundary_codes.values() if dict(code.neutral) == wanted]
    label = ", ".join(f"{k} = {v}" for k, v in wanted.items())
    if not found:
        supported = "; ".join(
            ", ".join(f"{key} = {value}" for key, value in code.neutral.items())
            for code in schema.boundary_codes.values()
            if code.status in EMITTED
        )
        raise ValueError(
            f"{SETUP} [boundaries.{boundary.family}]: {where} has no boundary code for {label}; "
            f"supported conditions: {supported or 'none'}"
        )
    if len(found) > 1:
        raise ValueError(
            f"{where}: boundary codes {', '.join(str(c.code) for c in found)} all describe "
            f"{label}; the database must hold one"
        )
    code = found[0]
    if code.status not in EMITTED:
        raise ValueError(
            f"{SETUP} [boundaries.{boundary.family}]: {label} needs boundary code {code.code}, "
            f"which has no harvested or verified evidence for {where} "
            f"({code.status.value} only)"
        )
    values: list[Scalar] = []
    for param in code.params or ():
        if param.fixed is not None:
            values.append(param.fixed)
        elif param.name == "temperature" and boundary.temperature is not None:
            values.append(boundary.temperature)
        elif param.name == "set":
            values.append(freestream_set)
        elif param.name == "temperature_set" and temperature_set is not None:
            values.append(temperature_set)
        else:
            raise ValueError(
                f"{where}: boundary code {code.code} takes {param.name}, which cfdbender "
                "cannot state"
            )
    return code.code, tuple(values)


def _show(values: tuple[Scalar, ...]) -> str:
    return " ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values)


def _same(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        numbers = all(isinstance(x, int | float) and not isinstance(x, bool) for x in (a, b))
        if not (float(a) == float(b) if numbers else a == b):  # type: ignore[arg-type]
            return False
    return True


def _typed(raw: RawKeyword, keyword: Keyword) -> None:
    where = f"{SETUP} [keywords] {raw.name}"
    assert keyword.types is not None
    kinds = keyword.types
    variadic = bool(kinds and kinds[-1].endswith("..."))
    minimum = len(kinds) - int(variadic)
    expected = f"{'at least ' if variadic else ''}{minimum} value(s): {', '.join(kinds)}"
    if len(raw.values) < minimum or (not variadic and len(raw.values) != minimum):
        raise ValueError(f"{where} takes {expected}; found {_show(raw.values)}")
    if variadic:
        kinds = kinds[:-1] + (kinds[-1][:-3],) * (len(raw.values) - minimum)
    for value, kind in zip(raw.values, kinds, strict=True):
        good = (
            (kind == "int" and isinstance(value, int))
            or (kind == "real" and isinstance(value, int | float))
            or (kind == "word" and isinstance(value, str))
            or (kind == "choice" and keyword.choices is not None and value in keyword.choices)
        )
        if not good:
            raise ValueError(f"{where} takes {expected}; found {_show(raw.values)}")


def _raw(
    case: Case,
    pairs: Mapping[str, list[Scalar | None]],
    origin: Mapping[str, str],
    place: Callable[[str, object], None],
) -> tuple[set[str], list[str], list[Mapping[str, object]]]:
    schema = case.schema
    free: set[str] = set()
    lines: list[str] = []
    records: list[Mapping[str, object]] = []
    pair_sections = [s.name for s in schema.sections if s.kind == "pairs"]
    for raw in case.raw:
        where = f"{SETUP} [keywords] {raw.name}"
        keyword = schema.keywords.get(raw.name)
        hinted = False
        if keyword is not None and keyword.status in (Status.BROKEN, Status.REMOVED):
            try:
                schema.emittable(raw.name)
            except ValueError as error:
                raise ValueError(f"{where}: {error}") from None
        if keyword is not None and keyword.fixed is not None:
            fixed = keyword.fixed if isinstance(keyword.fixed, tuple) else (keyword.fixed,)
            raise ValueError(
                f"{where}: {raw.name} is fixed by cfdbender's supported scope ({_show(fixed)}); "
                "it cannot be set"
            )
        if keyword is not None and raw.name in origin:
            generated = tuple(pairs[raw.name])
            if not _same(generated, raw.values):
                raise ValueError(
                    f"{SETUP}: keywords {raw.name} conflicts with {display(origin[raw.name])}; "
                    "state it in one place"
                )
            evidence = keyword.status.value
            section = keyword.section
        elif keyword is not None and keyword.status in EMITTED:
            _typed(raw, keyword)
            section = keyword.section
            assert section is not None  # an entry with evidence has its place
            place(section, Pair(raw.name, raw.values))
            evidence = keyword.status.value
        elif keyword is not None and keyword.section is not None:
            if keyword.types is not None:
                _typed(raw, keyword)
            else:
                free.add(raw.name)
            section = keyword.section
            place(section, Pair(raw.name, raw.values))
            evidence = "documented"
        else:
            section = raw.section or schema.raw_section
            hinted = raw.section is not None
            if raw.section is not None and raw.section not in pair_sections:
                raise ValueError(
                    f"{where}: section {raw.section!r} is not a keyword section of "
                    f"{schema.solver} {schema.version}; the sections are: "
                    f"{', '.join(pair_sections)}"
                )
            if section is None:
                raise ValueError(
                    f"{where}: {schema.solver} {schema.version} has no section for raw keywords; "
                    'name one with section = "<name>"'
                )
            free.add(raw.name)
            place(section, Pair(raw.name, raw.values))
            evidence = "documented" if keyword is not None else "unknown"
        text = {
            "harvested": f"harvested for {schema.version}",
            "verified": f"verified for {schema.version}",
            "documented": f"documented only, no evidence for {schema.version}",
            "unknown": "unknown to cfdbender",
        }[evidence]
        lines.append(
            f"keywords {raw.name} = {_show(raw.values)} -> section {section}"
            f"{' (hint)' if hinted else ''}; {text}"
        )
        records.append(
            {
                "keyword": raw.name,
                "value": list(raw.values) if len(raw.values) != 1 else raw.values[0],
                "section": section,
                "evidence": evidence,
            }
        )
    return free, lines, records


def _schema_status(case: Case, where: str) -> str:
    walls = {
        b.neutral.get("treatment")
        for b in case.boundaries
        if b.type == "wall" and b.forces and "treatment" in b.neutral
    }
    facts: dict[str, object] = {
        "method": case.values.get("physics.method"),
        "turbulence": case.turbulence,
        "treatment": next(iter(walls)) if len(walls) == 1 else None,
        "dimension": case.dimension,
    }
    for entry in case.schema.entries.get("combinations", ()):
        stated = {k: v for k, v in entry.attrs.items() if k in COMBINATION_FIELDS}
        if stated and all(facts[k] == v for k, v in stated.items()):
            if entry.status in (Status.BROKEN, Status.REMOVED):
                raise ValueError(
                    f"{where}: the physics combination {entry.name} is {entry.status.value} for "
                    f"this version (report {entry.report})"
                )
            if entry.status is Status.VERIFIED:
                return Status.VERIFIED.value
    return Status.HARVESTED.value
