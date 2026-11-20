"""The read side of ``cfd-matrix status``: one row per polar, or per datapoint, as pyfs-matrix.

A polar is the points of one simulation (POL) of one matrix. A row says how many of its points
the records hold out of those the plan receipt names, and the status of each. A point the
receipt (``post/<matrix>/plan.json``) names and no record carries reads ``planned``, in lower
case because no recorded status is written so. Nothing is written: no record, log or lock.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cfdbender.core.naming import POINT_NAME_FIELDS
from cfdbender.core.registry import RegistryError, RunRecord, RunRegistry
from cfdbender.core.settings import WorkspaceError
from cfdbender.matrix._console import status_word
from cfdbender.matrix._journal import MARKS, read_journal

PLANNED = "planned"
STATUS_SCHEMA = "cfdbender.status/1"

#: The keys of one polar row, in the order the table, the CSV and the JSON print them.
STATUS_COLUMNS = ("sim", "polar", "sweep", "points", "matrix", "status")
#: The keys of one datapoint row (``status --points``), in print order.
POINT_COLUMNS = (
    "sim",
    "point",
    "matrix",
    "status",
    "was",
    "iterations",
    "residual",
    "wall_s",
    "build",
    "where",
)
HEADINGS = {"iterations": "ITER", "wall_s": "WALL"}

#: What the footer says of a plan, by the state :func:`freshness` gives.
PLAN_STATES = {
    "current": "made from the matrix on disk",
    "stale": "made from another revision of the matrix; plan it again (cfd-matrix plan {stem})",
    "unpinned": "does not say which matrix it measured; plan it again (cfd-matrix plan {stem})",
    "no matrix": "the matrix {stem} is in neither matrix home of the workspace",
    "unreadable": "the plan cannot be read",
    "absent": "no plan",
}

SWEEP_NAME_VALUE = "+sweep"
_CODE_KEYS = {field.code: key for key, field in POINT_NAME_FIELDS.items()}
_TOKEN = re.compile(r"([A-Z]+)([+-]?\d+(?:\.\d+)?)")
#: Words a ``--status`` takes besides any ``FAILED_*``: the recorded statuses, the derived word
#: and the verdicts a completed point carries.
_KNOWN = (
    "RUNNING",
    "SUBMITTED",
    "COMPLETED",
    "CONVERGED",
    "ITERATION_LIMIT",
    "FAILED",
    "PLANNED",
    PLANNED,
)
#: The order of the status words of a mixed polar: concluded, failed, in flight, planned.
_GROUPS = ("COMPLETED", "FAILED", "SUBMITTED", "RUNNING", "PLANNED")


def check_status_words(words: Sequence[str]) -> None:
    """Refuse a ``--status`` word that no recorded or derived status takes."""
    for word in words:
        if word.endswith("*"):
            stem = word[:-1]
            known = (
                any(item.startswith(stem) for item in _KNOWN)
                or "FAILED".startswith(stem)
                or stem.startswith("FAILED")
            )
        else:
            known = word in _KNOWN or word.startswith("FAILED")
        if not known:
            raise WorkspaceError(
                f"status (CLI: --status) names {word!r}, which no status takes; the words are "
                f"{', '.join(_KNOWN)}, any FAILED_ word, and a trailing * matches every word it "
                "begins"
            )


def status_text(counts: Mapping[str, int]) -> str:
    """Return a polar's status: the one word, or the count of each word, concluded first."""
    present = {word: count for word, count in counts.items() if count}
    if len(present) == 1:
        return next(iter(present))
    return ", ".join(f"{word} {present[word]}" for word in sorted(present, key=_word_order))


def _word_order(word: str) -> tuple[int, str]:
    for position, group in enumerate(_GROUPS):
        if word.upper().startswith(group) and (word == word.upper() or word == PLANNED):
            return position, word
    return len(_GROUPS), word


@dataclass(frozen=True)
class _Entry:
    matrix: str
    sim: str
    name: str
    record: RunRecord | None
    word: str


@dataclass
class _Plan:
    points: dict[str, list[str]] = field(default_factory=dict)
    sha256: str | None = None
    matrix_path: str | None = None
    error: str | None = None


def matrix_stem(matrix: str) -> str:
    """Return the stem of a recorded matrix path or a matrix name."""
    return Path(matrix).name.removesuffix(".fs")


def _sim_order(sim: str) -> tuple[int, int, str]:
    """Order simulation ids as numbers, an id that is not a number after them."""
    return (0, int(sim), sim) if sim.isdigit() else (1, 0, sim)


def _split(point_id: str) -> tuple[str, str]:
    sim, _, name = point_id.partition("/")
    return sim, name.removeprefix("DP-")


def _read_plans(root: Path) -> dict[str, _Plan]:
    found: dict[str, _Plan] = {}
    post = root / "post"
    if not post.is_dir():
        return found
    for folder in sorted(p for p in post.iterdir() if p.is_dir()):
        path = folder / "plan.json"
        if not path.is_file():
            continue
        plan = found.setdefault(folder.name, _Plan())
        try:
            data = json.loads(path.read_bytes())
            for entry in data["points"]:
                sim, name = _split(str(entry["point_id"]))
                plan.points.setdefault(sim, []).append(f"{sim}/DP-{name}")
            digest = data.get("matrix_sha256")
            plan.sha256 = digest if isinstance(digest, str) else None
            plan.matrix_path = str(data.get("matrix") or "")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            plan.points.clear()
            plan.error = str(error)
    return found


def _tokens(name: str) -> list[tuple[str, str]] | None:
    found = _TOKEN.findall(name)
    return found if found and "".join(a + b for a, b in found) == name else None


def _axis(code: str) -> str:
    key = _CODE_KEYS.get(code, code)
    return key.lower() if key.isupper() else key


def polar_of(names: Sequence[str]) -> tuple[str, str]:
    """Return the polar's name and what it sweeps, from the names of its points.

    The fields of the point names that differ are the swept ones: the polar name writes each as
    its code and ``+sweep``, as pyflightstream does. A polar of one point, or whose names cannot
    be read field by field, sweeps nothing.
    """
    first = names[0]
    split = [_tokens(name) for name in names]
    if any(item is None for item in split) or len({len(item or ()) for item in split}) != 1:
        return first, "single point" if len(set(names)) == 1 else "unknown"
    rows = [item for item in split if item is not None]
    swept = [i for i in range(len(rows[0])) if len({row[i] for row in rows}) > 1]
    if not swept:
        return first, "single point"
    polar = "".join(
        code + SWEEP_NAME_VALUE if i in swept else code + number
        for i, (code, number) in enumerate(rows[0])
    )
    return polar, ", ".join(_axis(rows[0][i][0]) for i in swept)


def _where(record: RunRecord) -> str:
    submission = record.submission
    if not submission:
        return "local"
    job = submission.get("job")
    if isinstance(job, Mapping) and job.get("label"):
        return str(job["label"])
    return "cluster"


def _wall(record: RunRecord) -> float | None:
    try:
        began = datetime.fromisoformat(record.started_at or "")
        ended = datetime.fromisoformat(record.completed_at or "")
    except ValueError:
        return None
    return max((ended - began).total_seconds(), 0.0)


def _collected(root: Path, stem: str) -> dict[str, Mapping[str, object]]:
    """Return the last iteration and its residual of each collected point of a matrix."""
    path = root / "post" / stem / "campaign_summary.json"
    try:
        rows = json.loads(path.read_bytes())["points"]
        return {str(row["point_id"]): row for row in rows}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


class Status:
    """One snapshot of the records, the plan receipts and the posts of a workspace."""

    def __init__(self, root: Path, runs: str = "runs.json") -> None:
        self.root = root
        self.read_at = datetime.now(UTC).isoformat(timespec="seconds")
        registry = RunRegistry(root, runs)
        self.records: list[RunRecord] = list(registry.read())
        self.plans = _read_plans(root)
        self.marks = self._marks()
        self.entries = self._entries()
        self.sources = {
            path.name: datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(
                timespec="seconds"
            )
            for path in (registry.path,)
            if path.is_file()
        }
        self._summaries: dict[str, dict[str, Mapping[str, object]]] = {}

    def _marks(self) -> dict[str, Mapping[str, object]]:
        found: dict[str, Mapping[str, object]] = {}
        try:
            journal = read_journal(self.root, MARKS)
        except WorkspaceError:
            return found
        current = {r.point_id: r for r in self.records}
        for entry in journal:
            record = current.get(str(entry.get("point_id")))
            if record is not None and entry.get("created_at") == record.created_at:
                found[record.point_id] = entry
        return found

    def _entries(self) -> list[_Entry]:
        entries: list[_Entry] = []
        recorded = {r.point_id for r in self.records}
        for record in self.records:
            sim, name = _split(record.point_id)
            word = status_word(record.status, record.classification, record.error)
            entries.append(_Entry(matrix_stem(record.matrix), sim, name, record, word))
        for stem, plan in self.plans.items():
            for sim, ids in plan.points.items():
                entries.extend(
                    _Entry(stem, sim, _split(point_id)[1], None, PLANNED)
                    for point_id in ids
                    if point_id not in recorded
                )
        return entries

    def _groups(
        self, sims: Iterable[str] | None, matrix: str | None
    ) -> dict[tuple[str, str], list[_Entry]]:
        wanted = None if sims is None else set(sims)
        stem = None if matrix is None else matrix_stem(matrix)
        groups: dict[tuple[str, str], list[_Entry]] = {}
        for entry in self.entries:
            if (wanted is None or entry.sim in wanted) and (stem is None or entry.matrix == stem):
                groups.setdefault((entry.matrix, entry.sim), []).append(entry)
        return dict(sorted(groups.items(), key=lambda item: (item[0][0], _sim_order(item[0][1]))))

    @staticmethod
    def _wanted(entry: _Entry, statuses: Sequence[str]) -> bool:
        words = {entry.word}
        if entry.record is not None:
            words |= {entry.record.status}
            if entry.record.classification:
                words.add(entry.record.classification.upper())
        return any(fnmatch.fnmatchcase(word, pattern) for word in words for pattern in statuses)

    def _keep(self, entries: Sequence[_Entry], statuses: Sequence[str], failed: bool) -> bool:
        if failed and not any(e.word.startswith("FAILED") for e in entries):
            return False
        return not statuses or any(self._wanted(e, statuses) for e in entries)

    def status(
        self,
        *,
        sims: Iterable[str] | None = None,
        matrix: str | None = None,
        statuses: Sequence[str] = (),
        failed: bool = False,
    ) -> list[dict[str, object]]:
        """Return one row per polar."""
        rows: list[dict[str, object]] = []
        for (stem, sim), entries in self._groups(sims, matrix).items():
            if not self._keep(entries, statuses, failed):
                continue
            counts = Counter(e.word for e in entries)
            recorded = sum(1 for e in entries if e.record is not None)
            plan = self.plans.get(stem)
            planned = len(plan.points[sim]) if plan is not None and sim in plan.points else None
            polar, swept = polar_of([e.name for e in entries])
            rows.append(
                {
                    "sim": sim,
                    "polar": polar,
                    "sweep": swept,
                    "points": str(recorded) if planned is None else f"{recorded}/{planned}",
                    "matrix": stem,
                    "status": status_text(counts),
                    "recorded": recorded,
                    "planned": planned,
                    "counts": dict(sorted(counts.items(), key=lambda item: _word_order(item[0]))),
                }
            )
        return rows

    def points(
        self,
        *,
        sims: Iterable[str] | None = None,
        matrix: str | None = None,
        statuses: Sequence[str] = (),
        failed: bool = False,
    ) -> list[dict[str, object]]:
        """Return one row per datapoint: the recorded ones, then those only planned."""
        rows: list[dict[str, object]] = []
        for (stem, sim), entries in self._groups(sims, matrix).items():
            for entry in entries:
                if failed and not entry.word.startswith("FAILED"):
                    continue
                if statuses and not self._wanted(entry, statuses):
                    continue
                rows.append(self._point_row(stem, sim, entry))
        return rows

    def _point_row(self, stem: str, sim: str, entry: _Entry) -> dict[str, object]:
        row: dict[str, object] = dict.fromkeys(POINT_COLUMNS)
        row.update(sim=sim, point=entry.name, matrix=stem, status=entry.word)
        record = entry.record
        if record is None:
            return row
        mark = self.marks.get(record.point_id)
        if mark is not None and entry.word.startswith("FAILED_MARKED"):
            row["was"] = " ".join(
                str(part)
                for part in (mark.get("from_status"), mark.get("from_classification"))
                if part
            )
        summary = self._summaries.setdefault(stem, _collected(self.root, stem))
        result = summary.get(record.point_id, {}).get("result")
        history = result.get("history") if isinstance(result, Mapping) else None
        if isinstance(history, list) and history and isinstance(history[-1], Mapping):
            row["iterations"] = history[-1].get("iteration")
            row["residual"] = history[-1].get("residual")
        row["wall_s"] = _wall(record)
        row["build"] = record.solver_version or record.point.solver_build
        row["where"] = _where(record)
        row["error"] = record.error
        return row

    def unmatched(self, sims: Iterable[str]) -> list[str]:
        """Return each id that no record and no planned point carries."""
        held = {e.sim for e in self.entries}
        return [sim for sim in dict.fromkeys(sims) if sim not in held]

    def freshness(self, stems: Iterable[str]) -> list[dict[str, object]]:
        """Return, per matrix, whether its plan and its post are current."""
        return [self._freshness_of(stem) for stem in sorted(set(stems) - {""})]

    def _freshness_of(self, stem: str) -> dict[str, object]:
        recorded = [r.point_id for r in self.records if matrix_stem(r.matrix) == stem]
        folder = self.root / "post" / stem
        products = folder / "products.json"
        post, posted_at, missing = "absent", None, recorded
        if products.is_file():
            try:
                data = json.loads(products.read_bytes())
                post = "complete" if data.get("complete") is True else "incomplete"
                posted_at = datetime.fromtimestamp(products.stat().st_mtime, UTC).isoformat(
                    timespec="seconds"
                )
                posted = set(_collected(self.root, stem))
                missing = [point for point in recorded if point not in posted]
            except (OSError, ValueError, AttributeError):
                post = "incomplete"
        return {
            "matrix": stem,
            "plan": self._plan_state(stem),
            "post": post,
            "posted_at": posted_at,
            "not_posted": missing,
        }

    def _plan_state(self, stem: str) -> str:
        plan = self.plans.get(stem)
        if plan is None:
            return "absent"
        if plan.error is not None:
            return "unreadable"
        if plan.sha256 is None:
            return "unpinned"
        candidates = [self.root / "inputs" / "matrices" / f"{stem}.fs", self.root / f"{stem}.fs"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            return "no matrix"
        return "current" if hashlib.sha256(path.read_bytes()).hexdigest() == plan.sha256 else "stale"

    def document(
        self, rows: Sequence[Mapping[str, object]], freshness: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        """Return the ``--json`` document: schema, workspace, read time, sources and rows."""
        return {
            "schema": STATUS_SCHEMA,
            "workspace": str(self.root),
            "read_at": self.read_at,
            "sources": dict(self.sources),
            "rows": [dict(row) for row in rows],
            "freshness": [dict(item) for item in freshness],
        }


def listed_sims(values: Sequence[str] | None) -> list[str]:
    """Return the ids of ``--sims`` given with or without brackets, comma or space separated."""
    if not values:
        return []
    text = " ".join(values)
    listed = re.sub(r"[,\s\[\]]+", ",", text.strip())
    return [item for item in listed.split(",") if item]


def freshness_lines(item: Mapping[str, object]) -> list[str]:
    """Return the two footer lines of a matrix: its plan, then its post."""
    stem = str(item["matrix"])
    plan = PLAN_STATES[str(item["plan"])].format(stem=stem)
    if item["post"] == "absent":
        post = "no post yet"
    else:
        state = "complete" if item["post"] == "complete" else "NOT complete"
        post = f"written {item['posted_at']}, {state}"
    missing = item["not_posted"]
    if isinstance(missing, list) and missing and item["post"] != "absent":
        post += f", {len(missing)} recorded run(s) not yet posted (cfd-matrix post {stem})"
    return [f"plan/{stem}: {plan}", f"post/{stem}: {post}"]


__all__ = [
    "POINT_COLUMNS",
    "STATUS_COLUMNS",
    "RegistryError",
    "Status",
    "check_status_words",
    "freshness_lines",
    "listed_sims",
    "polar_of",
    "status_text",
]
