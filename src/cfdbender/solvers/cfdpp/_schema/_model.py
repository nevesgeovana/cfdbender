"""The grammar engine: an ordered, typed model of the solver input and its two directions.

The engine is version-neutral and holds no solver keyword: the sections, their
opening and closing lines, the keywords with their types, the information-set
tags, the boundary codes, the number format and the layout of the multi-line
entries all come from a :class:`Schema`.

The input is read as a sequence of sections. A section is one of three shapes:

- **enclosed**: it opens and closes with a line of its own (``open``, ``close``) and
  may start with the count of its entries on the next line (``counted``);
- **headed**: it opens with a line holding its head word and the count of its
  entries (``head``), optionally followed by a fixed column-title line
  (``header``), and ends after that many entries;
- **bare**: a run of keyword pairs whose keywords belong to the section, with no
  opening or closing line.

Each section holds entries of one kind:

- ``pairs``: a keyword followed by its values, typed by the database; a keyword
  marked ``repeat`` may appear several times, and a last type ending in ``...``
  takes any number of values of that type; in the schema's raw section an unknown
  keyword is accepted with values read as an integer, a real or a word;
- ``infosets``: an information set, either one line (a tag and one value per
  variable) or, when the schema states the information-set layout, a title line
  (index, value count, tag) followed by value lines of a fixed width;
- ``boundaries``: a selector, a boundary code, the code's parameters and, when the
  layout says so, the family name of the grid;
- ``lines``: free lines kept as written.

When the schema names a comment prefix, comment and blank lines are kept as
:class:`Trivia`: inside a section as items, before a section as its ``lead`` and
after the last section as the model's ``tail``; they never take part in equality
or in :func:`diff`. A parsed entry also keeps its original text, so that
:func:`render` writes a parsed input back byte for byte; an entry built by
cfdbender, or one whose values no longer match its text, is written from its
values in the schema's number format.

:func:`render` is deterministic (the same model gives the same text on every
platform; the line ending is the schema's) and writes sections in schema order and
keyword pairs in keyword order, unknown keywords after the known ones in the order
given. :func:`parse` and :func:`render` are inverse on every input in that order.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field, replace

from cfdbender.solvers.cfdpp._schema import (
    InfoSetTag,
    Keyword,
    Layout,
    Param,
    Scalar,
    Schema,
    Section,
)

_INT = re.compile(r"[+-]?\d+")
_VARIADIC = "..."


@dataclass(frozen=True)
class Pair:
    """A keyword and its values; ``text`` is the line it was read from, if any."""

    name: str
    values: tuple[Scalar, ...]
    text: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class InfoSet:
    """An information set: its tag and one value per variable of the tag."""

    tag: str
    values: tuple[Scalar, ...]
    text: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Boundary:
    """A boundary block: the selector of the grid, the code, its parameters, the family."""

    selector: int
    code: int
    values: tuple[Scalar, ...]
    family: str | None = None
    text: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Trivia:
    """A comment or blank line kept for a faithful rewrite; it carries no meaning."""

    text: str


Item = Pair | InfoSet | Boundary | Trivia | str


@dataclass(frozen=True)
class Block:
    """One section of an input and its entries, in order; ``lead`` is the trivia before it."""

    section: str
    items: tuple[Item, ...]
    lead: tuple[str, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class InputModel:
    """An input as ordered sections of entries; ``tail`` is the trivia after the last."""

    blocks: tuple[Block, ...]
    tail: tuple[str, ...] = field(default=(), compare=False)
    source: str | None = field(default=None, compare=False, repr=False)

    def block(self, section: str) -> Block | None:
        """Return the block of ``section``, or None when the input does not hold it."""
        return next((block for block in self.blocks if block.section == section), None)

    def pair(self, name: str) -> Pair | None:
        """Return the first keyword pair called ``name`` in any section, or None."""
        for block in self.blocks:
            for item in block.items:
                if isinstance(item, Pair) and item.name == name:
                    return item
        return None


@dataclass(frozen=True)
class Change:
    """One difference between two models: ``added``, ``removed`` or ``changed``."""

    kind: str
    section: str
    key: str
    old: str | None
    new: str | None


def parse(text: str, schema: Schema, *, free: Collection[str] = ()) -> InputModel:
    """Read ``text`` into a model, refusing any token the schema does not hold.

    ``free`` names keywords accepted untyped in any keyword-pair section, as in the
    raw section: the raw keywords a setup file places with a section hint.
    """
    schema.require_grammar()
    if schema.layout is not None:
        endings = re.findall(r"\r\n|\r|\n", text)
        if "\r" in endings or len(set(endings)) > 1:
            raise ValueError(f"{_where(schema)}: mixed or bare-CR line endings are not supported")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [line[:-1] if line.endswith("\r") else line for line in lines]
    reader = _Reader(schema, lines, free)
    model = reader.read()
    return replace(model, source=text) if schema.layout is not None else model


def render(model: InputModel, schema: Schema, *, free: Collection[str] = ()) -> str:
    """Write ``model`` as text, refusing an entry the schema does not hold.

    ``free`` names keywords written untyped in any keyword-pair section (see :func:`parse`).
    """
    number_format = schema.require_grammar()
    if model.source is not None and schema.layout is not None:
        try:
            original = parse(model.source, schema, free=free)
        except ValueError:
            pass  # A different schema may require rebuilding the text from the values.
        else:
            # Equality deliberately ignores lexical fields. Compare those too: editing
            # a comment, wrapper or stored entry text must not replay stale input.
            if asdict(original) == asdict(model):
                return model.source
    rank = {section.name: position for position, section in enumerate(schema.sections)}
    names = [block.section for block in model.blocks]
    for block in model.blocks:
        if block.section not in rank:
            schema.section(block.section)
        if names.count(block.section) > 1:
            raise ValueError(f"{_where(schema)}: the model holds section {block.section} twice")
    out: list[str] = []
    for block in sorted(model.blocks, key=lambda block: rank[block.section]):
        section = schema.section(block.section)
        out.extend(_trivia_lines(block.lead, schema))
        items = _ordered(block, section, schema)
        entries = [item for item in items if not isinstance(item, Trivia)]
        if section.head is not None:
            out.append(f"{section.head} {len(entries)}")
            if section.header is not None:
                out.append(section.header)
        elif section.open is not None:
            out.append(section.open)
            if section.counted:
                out.append(str(len(entries)))
        position = 0
        for item in items:
            if isinstance(item, Trivia):
                out.extend(_trivia_lines((item.text,), schema))
                continue
            position += 1
            out.extend(_render_item(item, section, schema, free, position))
        if section.head is None and section.close is not None:
            out.append(section.close)
    out.extend(_trivia_lines(model.tail, schema))
    return "".join(line + number_format.line_ending for line in out)


def diff(a: InputModel, b: InputModel) -> list[Change]:
    """List what changes from model ``a`` to model ``b``, section by section."""
    changes: list[Change] = []
    order = [block.section for block in a.blocks]
    order += [block.section for block in b.blocks if block.section not in order]
    for name in order:
        left, right = a.block(name), b.block(name)
        if left is None:
            changes.append(Change("added", name, "section", None, None))
        if right is None:
            changes.append(Change("removed", name, "section", None, None))
        old = _keyed(left.items if left else ())
        new = _keyed(right.items if right else ())
        for key, text in old.items():
            if key not in new:
                changes.append(Change("removed", name, key, text, None))
            elif new[key] != text:
                changes.append(Change("changed", name, key, text, new[key]))
        for key, text in new.items():
            if key not in old:
                changes.append(Change("added", name, key, None, text))
    return changes


def _where(schema: Schema) -> str:
    return f"{schema.solver} {schema.version} {schema.input_file}"


def _layout(schema: Schema) -> Layout:
    return schema.layout if schema.layout is not None else Layout()


def _is_trivia(line: str, schema: Schema) -> bool:
    comment = _layout(schema).comment
    return not line.strip() or (comment is not None and line.lstrip().startswith(comment))


def _trivia_lines(lines: Sequence[str], schema: Schema) -> list[str]:
    for line in lines:
        if "\n" in line or "\r" in line or not _is_trivia(line, schema):
            raise ValueError(f"{_where(schema)}: {line!r} is not a comment or blank line")
    return list(lines)


class _Reader:
    """One pass over the lines of an input, section by section."""

    def __init__(self, schema: Schema, lines: list[str], free: Collection[str]) -> None:
        self.schema = schema
        self.lines = lines
        self.free = free
        self.keep = _layout(schema).comment is not None
        self.opens = {section.open for section in schema.sections if section.open is not None}
        self.seen: set[str] = set()

    def read(self) -> InputModel:
        blocks: list[Block] = []
        pending: list[str] = []
        index = 0
        while index < len(self.lines):
            line = self.lines[index]
            if _is_trivia(line, self.schema):
                if self.keep:
                    pending.append(line)
                elif line.strip():
                    raise ValueError(
                        f"{_where(self.schema)} line {index + 1}: {line.strip()!r} is outside "
                        "every section"
                    )
                index += 1
                continue
            section = self._opening(line, index)
            if self.schema.layout is not None and blocks:
                order = [entry.name for entry in self.schema.sections]
                if order.index(section.name) < order.index(blocks[-1].section):
                    raise ValueError(
                        f"{_where(self.schema)} line {index + 1}: section {section.name} "
                        "is out of schema order"
                    )
            self.seen.add(section.name)
            if section.head is not None:
                block, index, after = self._headed(section, index)
            elif section.open is not None:
                block, index, after = self._enclosed(section, index)
            else:
                block, index, after = self._bare(section, index)
            blocks.append(Block(block.section, block.items, tuple(pending)))
            pending = after
        return InputModel(tuple(blocks), tuple(pending))

    def _opening(self, line: str, index: int) -> Section:
        schema = self.schema
        stripped = line.strip()
        enclosed = [s for s in schema.sections if s.open is not None and s.open == stripped]
        if enclosed:
            fresh = [s for s in enclosed if s.name not in self.seen]
            if not fresh:
                raise ValueError(
                    f"{_where(schema)} line {index + 1}: section {enclosed[0].name} is opened twice"
                )
            return fresh[0]
        first = stripped.split()[0]
        for section in schema.sections:
            if section.head is not None and section.head == first:
                if section.name in self.seen:
                    raise ValueError(
                        f"{_where(schema)} line {index + 1}: section {section.name} is opened twice"
                    )
                return section
        keyword = schema.keywords.get(first)
        if keyword is not None and keyword.section is not None:
            section = schema.section(keyword.section)
            if _bare(section):
                if section.name in self.seen:
                    raise ValueError(
                        f"{_where(schema)} line {index + 1}: section {section.name} is opened twice"
                    )
                return section
        expected = ", ".join(
            repr(s.open) if s.open is not None else repr(s.head or s.name) for s in schema.sections
        )
        raise ValueError(
            f"{_where(schema)} line {index + 1}: {stripped!r} is outside every section; a "
            f"section opens with one of: {expected}"
        )

    def _enclosed(self, section: Section, start: int) -> tuple[Block, int, list[str]]:
        schema, lines = self.schema, self.lines
        index = start + 1
        count: int | None = None
        if section.counted:
            while index < len(lines) and not lines[index].strip():
                index += 1
            token = lines[index].strip() if index < len(lines) else ""
            if not _INT.fullmatch(token):
                raise ValueError(
                    f"{_where(schema)} line {index + 1}: section {section.name} starts with "
                    f"the count of its entries; found {token!r}"
                )
            count = int(token)
            index += 1
        items: list[Item] = []
        names: set[str] = set()
        while True:
            if index >= len(lines) or lines[index].strip() in self.opens - {section.close}:
                raise ValueError(
                    f"{_where(schema)}: section {section.name} opened at line {start + 1} is "
                    f"not closed; expected {section.close!r} before line {index + 1}"
                )
            line = lines[index]
            if line.strip() == section.close:
                break
            if section.kind == "lines":
                items.append(line)
            elif _is_trivia(line, schema):
                if self.keep:
                    items.append(Trivia(line))
            elif section.kind in ("infosets",) and _layout(schema).infoset_header:
                raise ValueError(
                    f"{_where(schema)} line {index + 1}: a multi-line information set sits "
                    "in a headed section"
                )
            else:
                item = _parse_line(section, line, index + 1, schema, self.free)
                _once(item, names, section, schema, index + 1)
                items.append(item)
            index += 1
        entries = [item for item in items if not isinstance(item, Trivia)]
        if count is not None and count != len(entries):
            raise ValueError(
                f"{_where(schema)} line {start + 2}: section {section.name} states {count} "
                f"entries and holds {len(entries)}"
            )
        return Block(section.name, tuple(items)), index + 1, []

    def _headed(self, section: Section, start: int) -> tuple[Block, int, list[str]]:
        schema, lines = self.schema, self.lines
        tokens = lines[start].split()
        if len(tokens) != 2 or not _INT.fullmatch(tokens[1]):
            raise ValueError(
                f"{_where(schema)} line {start + 1}: section {section.name} opens with "
                f"{section.head!r} and the count of its entries; found {lines[start].strip()!r}"
            )
        count = int(tokens[1])
        if count < 0:
            raise ValueError(f"{_where(schema)}: section {section.name} count must be nonnegative")
        index = start + 1
        if section.header is not None:
            if index >= len(lines) or lines[index].split() != section.header.split():
                found = lines[index].strip() if index < len(lines) else "the end of the input"
                raise ValueError(
                    f"{_where(schema)} line {index + 1}: section {section.name} continues "
                    f"with {section.header!r}; found {found!r}"
                )
            index += 1
        items: list[Item] = []
        pending: list[str] = []
        names: set[str] = set()
        entries = 0
        while entries < count:
            if index >= len(lines):
                raise ValueError(
                    f"{_where(schema)}: section {section.name} states {count} entries and "
                    f"holds {entries}"
                )
            line = lines[index]
            if _is_trivia(line, schema):
                if self.keep:
                    pending.append(line)
                index += 1
                continue
            header = _layout(schema).infoset_header
            if section.kind == "infosets" and header and _match(header, line.split()) is None:
                raise ValueError(
                    f"{_where(schema)} line {index + 1}: section {section.name} states {count} "
                    f"entries and holds {entries}"
                )
            items.extend(Trivia(text) for text in pending)
            pending = []
            item: Item
            if section.kind == "infosets" and header:
                item, index = _read_infoset(lines, index, schema, entries + 1)
            else:
                item = _parse_line(section, line, index + 1, schema, self.free)
                _once(item, names, section, schema, index + 1)
                index += 1
            items.append(item)
            entries += 1
        return Block(section.name, tuple(items)), index, pending

    def _bare(self, section: Section, start: int) -> tuple[Block, int, list[str]]:
        schema, lines = self.schema, self.lines
        items: list[Item] = []
        pending: list[str] = []
        names: set[str] = set()
        index = start
        while index < len(lines):
            line = lines[index]
            if _is_trivia(line, schema):
                if self.keep:
                    pending.append(line)
                index += 1
                continue
            keyword = schema.keywords.get(line.split()[0])
            if keyword is None or keyword.section != section.name:
                break
            items.extend(Trivia(text) for text in pending)
            pending = []
            item = _parse_line(section, line, index + 1, schema, self.free)
            _once(item, names, section, schema, index + 1)
            items.append(item)
            index += 1
        return Block(section.name, tuple(items)), index, pending


def _bare(section: Section) -> bool:
    return section.open is None and section.close is None and section.head is None


def _once(item: Item, names: set[str], section: Section, schema: Schema, number: int) -> None:
    if not isinstance(item, Pair):
        return
    keyword = schema.keywords.get(item.name)
    if keyword is not None and keyword.repeat:
        return
    if item.name in names:
        raise ValueError(
            f"{_where(schema)} line {number}: keyword {item.name} is stated twice in section "
            f"{section.name}"
        )
    names.add(item.name)


def _parse_line(
    section: Section, line: str, number: int, schema: Schema, free: Collection[str] = ()
) -> Item:
    tokens = line.split()
    where = f"{_where(schema)} line {number}"
    if section.kind == "pairs":
        name, rest = tokens[0], tokens[1:]
        keyword = _typed_keyword(name, section, schema, where, free)
        if keyword is None:
            return Pair(name, tuple(_guess(token) for token in rest), line)
        assert keyword.types is not None
        kinds = _expand(keyword.types, len(rest), f"{where}: keyword {name}")
        values = tuple(
            _read_value(token, kind, keyword, f"{where}: keyword {name}")
            for token, kind in zip(rest, kinds, strict=True)
        )
        return Pair(name, values, line)
    if section.kind == "infosets":
        tag = tokens[0]
        known = _tag(tag, schema, where)
        _check_count(known, len(tokens) - 1, where)
        kinds = _set_types(known, len(tokens) - 1)
        values = tuple(
            _read_value(token, kind, None, f"{where}: set {tag}")
            for token, kind in zip(tokens[1:], kinds, strict=True)
        )
        return InfoSet(tag, values, line)
    if section.kind != "boundaries":
        raise ValueError(f"{where}: section {section.name} holds {section.kind}")
    if len(tokens) < 2 or not _INT.fullmatch(tokens[0]) or not _INT.fullmatch(tokens[1]):
        raise ValueError(f"{where}: a boundary block is a selector and a code; found {line!r}")
    selector, code = int(tokens[0]), int(tokens[1])
    params = _params(code, schema, where)
    named = _layout(schema).boundary_family
    extra = len(tokens) - 2 - len(params)
    if extra != (1 if named else 0):
        names = ", ".join(param.name for param in params) or "none"
        family = " and the family name" if named else ""
        raise ValueError(
            f"{where}: boundary code {code} takes {len(params)} parameters ({names}){family}, "
            f"found {len(tokens) - 2} values"
        )
    values = tuple(
        _read_value(token, param.type, None, f"{where}: code {code} {param.name}")
        for token, param in zip(tokens[2 : 2 + len(params)], params, strict=True)
    )
    return Boundary(selector, code, values, tokens[-1] if named else None, line)


def _read_infoset(
    lines: Sequence[str], start: int, schema: Schema, position: int
) -> tuple[InfoSet, int]:
    layout = _layout(schema)
    assert layout.infoset_header is not None and layout.infoset_values is not None
    where = f"{_where(schema)} line {start + 1}"
    fields = _match(layout.infoset_header, lines[start].split())
    if fields is None:
        raise ValueError(
            f"{where}: an information set opens with {layout.infoset_header!r}; "
            f"found {lines[start].strip()!r}"
        )
    for name in ("index", "count"):
        if not _INT.fullmatch(fields[name]):
            raise ValueError(f"{where}: the information-set {name} is an integer")
    if int(fields["index"]) != position:
        raise ValueError(
            f"{where}: information set {fields['index']} sits at position {position}; "
            "the sets are numbered in order"
        )
    count = int(fields["count"])
    known = _tag(fields["tag"], schema, where)
    _check_count(known, count, where)
    tokens: list[str] = []
    text = [lines[start]]
    index = start + 1
    while len(tokens) < count:
        line = lines[index] if index < len(lines) else ""
        parts = line.split()
        if not parts or parts[0] != layout.infoset_values:
            raise ValueError(
                f"{_where(schema)} line {index + 1}: information set {fields['tag']} states "
                f"{count} values and holds {len(tokens)}"
            )
        tokens += parts[1:]
        text.append(line)
        index += 1
    if len(tokens) != count:
        raise ValueError(
            f"{_where(schema)} line {index}: information set {fields['tag']} states {count} "
            f"values and holds {len(tokens)}"
        )
    kinds = _set_types(known, count)
    values = tuple(
        _read_value(token, kind, None, f"{where}: set {fields['tag']}")
        for token, kind in zip(tokens, kinds, strict=True)
    )
    return InfoSet(fields["tag"], values, "\n".join(text)), index


def _match(pattern: str, tokens: list[str]) -> dict[str, str] | None:
    words = pattern.split()
    if len(words) != len(tokens):
        return None
    fields: dict[str, str] = {}
    for word, token in zip(words, tokens, strict=True):
        if word.startswith("{") and word.endswith("}"):
            fields[word[1:-1]] = token
        elif word != token:
            return None
    return fields


def _tag(tag: str, schema: Schema, where: str) -> InfoSetTag:
    known = schema.infosets.get(tag)
    if known is None or known.variables is None:
        tags = ", ".join(sorted(schema.infosets))
        raise ValueError(
            f"{where}: {tag!r} is not an information-set tag with a known variable order; "
            f"the tags are: {tags}"
        )
    return known


def _check_count(tag: InfoSetTag, count: int, where: str) -> None:
    assert tag.variables is not None
    variables = len(tag.variables)
    if count == variables or (tag.turbulence and count > variables):
        return
    more = " and the turbulence values" if tag.turbulence else ""
    raise ValueError(
        f"{where}: information set {tag.tag} takes {variables} values "
        f"({', '.join(tag.variables)}){more}, found {count}"
    )


def _set_types(tag: InfoSetTag, count: int) -> tuple[str, ...]:
    base = tag.types or ("real",) * len(tag.variables or ())
    return tuple(base) + ("real",) * (count - len(base))


def _expand(types: tuple[str, ...], count: int, label: str) -> tuple[str, ...]:
    if types and types[-1].endswith(_VARIADIC):
        fixed = types[:-1]
        if count < len(fixed):
            raise ValueError(f"{label} takes at least {len(fixed)} values, found {count}")
        return fixed + (types[-1][: -len(_VARIADIC)],) * (count - len(fixed))
    if count != len(types):
        raise ValueError(f"{label} takes {len(types)} values, found {count}")
    return types


def _typed_keyword(
    name: str, section: Section, schema: Schema, where: str, free: Collection[str] = ()
) -> Keyword | None:
    """Return the keyword's typed entry, None for a free keyword of the raw section."""
    keyword = schema.keywords.get(name)
    if name in free and (
        keyword is None or keyword.types is None or keyword.section != section.name
    ):
        return None
    if keyword is not None and keyword.section is not None and keyword.section != section.name:
        raise ValueError(
            f"{where}: keyword {name} belongs to section {keyword.section}, "
            f"found in section {section.name}"
        )
    if keyword is not None and keyword.section == section.name and keyword.types is not None:
        return keyword
    if section.name == schema.raw_section:
        return None
    raise ValueError(
        f"{where}: {name!r} is not a keyword of section {section.name} in the interface database"
    )


def _params(code: int, schema: Schema, where: str) -> tuple[Param, ...]:
    known = schema.boundary_codes.get(code)
    if known is None or known.params is None:
        codes = ", ".join(str(code) for code in sorted(schema.boundary_codes))
        raise ValueError(
            f"{where}: boundary code {code} is not in the interface database with its "
            f"parameters; the codes are: {codes}"
        )
    return known.params


def _read_value(token: str, kind: str, keyword: Keyword | None, label: str) -> Scalar:
    value: Scalar
    if kind == "int":
        if not _INT.fullmatch(token):
            raise ValueError(f"{label} takes an int; found {token!r}")
        value = int(token)
    elif kind == "real":
        value = _real(token, label)
    elif kind == "word":
        value = token
    else:
        assert keyword is not None and keyword.choices is not None
        for choice in keyword.choices:
            if token == str(choice) or (
                isinstance(choice, float) and _is_number(token) and float(token) == choice
            ):
                return choice
        allowed = ", ".join(str(choice) for choice in keyword.choices)
        raise ValueError(f"{label} takes one of {allowed}; found {token!r}")
    if keyword is not None:
        _check_range(value, keyword, label)
    return value


def _real(token: str, label: str) -> float:
    if not _is_number(token):
        raise ValueError(f"{label} takes a finite real; found {token!r}")
    return float(token)


def _is_number(token: str) -> bool:
    try:
        return math.isfinite(float(token))
    except ValueError:
        return False


def _guess(token: str) -> Scalar:
    if _INT.fullmatch(token):
        return int(token)
    if _is_number(token):
        return float(token)
    return token


def _check_range(value: Scalar, keyword: Keyword, label: str) -> None:
    if keyword.range is None or isinstance(value, str):
        return
    low, high = keyword.range
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{label}: {value} is outside its range [{_bound(low)}, {_bound(high)}]")


def _bound(value: float | None) -> str:
    return "open" if value is None else str(value)


def _ordered(block: Block, section: Section, schema: Schema) -> list[Item]:
    if section.kind != "pairs":
        return list(block.items)
    names = [item.name for item in block.items if isinstance(item, Pair)]
    repeated = sorted(
        {
            name
            for name in names
            if names.count(name) > 1
            and not (schema.keywords.get(name) is not None and schema.keywords[name].repeat)
        }
    )
    if repeated:
        raise ValueError(
            f"{_where(schema)}: keyword {', '.join(repeated)} is stated twice in section "
            f"{section.name}"
        )

    def rank(item: Item) -> tuple[int, int]:
        keyword = schema.keywords.get(item.name) if isinstance(item, Pair) else None
        if keyword is not None and keyword.section == section.name and keyword.order:
            return 0, keyword.order
        return 1, 0

    pairs = [item for item in block.items if not isinstance(item, Trivia)]
    ranks = [rank(item) for item in pairs]
    if ranks == sorted(ranks):
        return list(block.items)
    return sorted(pairs, key=rank)


def _render_item(
    item: Item, section: Section, schema: Schema, free: Collection[str], position: int
) -> list[str]:
    where = f"{_where(schema)} section {section.name}"
    expected = {"pairs": Pair, "infosets": InfoSet, "boundaries": Boundary, "lines": str}
    if not isinstance(item, expected[section.kind]):
        raise ValueError(f"{where} holds {section.kind}; found {item!r}")
    if isinstance(item, Pair):
        if not item.name or any(character.isspace() for character in item.name):
            raise ValueError(f"{where}: keyword name {item.name!r} must be one non-empty word")
        comment = _layout(schema).comment
        if comment is not None and item.name.startswith(comment):
            raise ValueError(f"{where}: keyword name {item.name!r} starts a comment")
    if isinstance(item, str):
        closing = {section.close} - {None}
        if "\n" in item or "\r" in item or item.strip() in closing:
            raise ValueError(f"{where}: the line {item!r} cannot be written in this section")
        return [item]
    if isinstance(item, Pair | InfoSet | Boundary) and item.text is not None:
        if _faithful(item, section, schema, free, position):
            return item.text.split("\n")
    if isinstance(item, Pair):
        return [_render_pair(item, section, schema, free, where)]
    if isinstance(item, InfoSet):
        return _render_set(item, schema, where, position)
    assert isinstance(item, Boundary)
    return [_render_boundary(item, schema, where)]


def _faithful(
    item: Pair | InfoSet | Boundary,
    section: Section,
    schema: Schema,
    free: Collection[str],
    position: int,
) -> bool:
    """Say whether the item's original text still reads back as the item itself."""
    assert item.text is not None
    try:
        if isinstance(item, InfoSet) and _layout(schema).infoset_header:
            lines = item.text.split("\n")
            again, end = _read_infoset(lines, 0, schema, position)
            return end == len(lines) and again == item
        if "\n" in item.text:
            return False
        return _parse_line(section, item.text, 0, schema, free) == item
    except ValueError:
        return False


def _render_pair(
    item: Pair, section: Section, schema: Schema, free: Collection[str], where: str
) -> str:
    keyword = _typed_keyword(item.name, section, schema, where, free)
    if keyword is None:
        return _join(item.name, [_free(value, item.name, where, schema) for value in item.values])
    assert keyword.types is not None
    kinds = _expand(keyword.types, len(item.values), f"{where}: keyword {item.name}")
    return _join(
        item.name,
        [
            _typed(value, kind, keyword, f"{where}: keyword {item.name}", schema)
            for value, kind in zip(item.values, kinds, strict=True)
        ],
    )


def _render_set(item: InfoSet, schema: Schema, where: str, position: int) -> list[str]:
    known = _tag(item.tag, schema, where)
    _check_count(known, len(item.values), f"{where}: set {item.tag}")
    kinds = _set_types(known, len(item.values))
    tokens = [
        _typed(value, kind, None, f"{where}: set {item.tag}", schema)
        for value, kind in zip(item.values, kinds, strict=True)
    ]
    layout = _layout(schema)
    if layout.infoset_header is None or layout.infoset_values is None:
        return [_join(item.tag, tokens)]
    header = layout.infoset_header.format(index=position, count=len(tokens), tag=item.tag)
    width = layout.infoset_per_line or len(tokens) or 1
    tail = " " if layout.infoset_trailing_space else ""
    rows = [
        _join(layout.infoset_values, tokens[start : start + width]) + tail
        for start in range(0, len(tokens), width)
    ]
    return [header, *rows]


def _render_boundary(item: Boundary, schema: Schema, where: str) -> str:
    params = _params(item.code, schema, where)
    if len(item.values) != len(params):
        raise ValueError(
            f"{where}: boundary code {item.code} takes {len(params)} parameters, "
            f"found {len(item.values)}"
        )
    columns = [
        _typed(item.selector, "int", None, where, schema),
        str(item.code),
        *(
            _typed(value, param.type, None, f"{where}: code {item.code} {param.name}", schema)
            for value, param in zip(item.values, params, strict=True)
        ),
    ]
    layout = _layout(schema)
    if layout.boundary_width:
        columns = [column.rjust(layout.boundary_width) for column in columns]
    if layout.boundary_family:
        if item.family is None:
            raise ValueError(f"{where}: selector {item.selector} needs its family name")
        columns.append(_word(item.family, f"{where}: family of selector {item.selector}"))
    elif item.family is not None:
        raise ValueError(f"{where}: this input carries no family name on a boundary block")
    return " ".join(columns)


def _join(head: str, values: Sequence[str]) -> str:
    return " ".join([head, *values])


def _typed(value: Scalar, kind: str, keyword: Keyword | None, label: str, schema: Schema) -> str:
    if kind == "choice":
        assert keyword is not None and keyword.choices is not None
        if isinstance(value, bool) or value not in keyword.choices:
            allowed = ", ".join(str(choice) for choice in keyword.choices)
            raise ValueError(f"{label} takes one of {allowed}; found {value!r}")
        return _free(value, label, label, schema)
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} takes an int; found {value!r}")
        if keyword is not None:
            _check_range(value, keyword, label)
        return str(value)
    if kind == "real":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{label} takes a finite real; found {value!r}")
        if keyword is not None:
            _check_range(value, keyword, label)
        return _number(float(value), label, schema)
    if not isinstance(value, str):
        raise ValueError(f"{label} takes a word; found {value!r}")
    return _word(value, label)


def _free(value: Scalar, name: str, label: str, schema: Schema) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{label}: {name} takes integers, reals and words; found {value!r}")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _number(value, label, schema)
    return _word(value, label)


def _number(value: float, label: str, schema: Schema) -> str:
    if not math.isfinite(value):
        raise ValueError(f"{label} takes a finite real; found {value!r}")
    spec = schema.require_grammar().real
    return repr(value) if spec == "shortest" else format(value, spec)


def _word(value: str, label: str) -> str:
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{label} takes a word without spaces; found {value!r}")
    return value


def _keyed(items: Sequence[Item]) -> dict[str, str]:
    keyed: dict[str, str] = {}
    position = 0
    for item in items:
        if isinstance(item, Trivia):
            continue
        position += 1
        if isinstance(item, Pair):
            key = item.name
            occurrence = 1
            while key in keyed:
                occurrence += 1
                key = f"{item.name} #{occurrence}"
            keyed[key] = " ".join(_show(value) for value in item.values)
        elif isinstance(item, InfoSet):
            keyed[f"set {position}"] = " ".join([item.tag, *(_show(v) for v in item.values)])
        elif isinstance(item, Boundary):
            shown = [str(item.code), *(_show(value) for value in item.values)]
            if item.family is not None:
                shown.append(item.family)
            keyed[f"selector {item.selector}"] = " ".join(shown)
        else:
            keyed[f"line {position}"] = item
    return keyed


def _show(value: Scalar) -> str:
    return repr(value) if isinstance(value, float) else str(value)
