"""The human-readable log of a geometry run, written beside the stage records.

Pipeline role: :mod:`cfdbender.geometry.pipeline` calls the writer once per stage as the
run proceeds, so a crash leaves the stages done so far. Every line is derived from the stage
record or the report the runner already holds, never measured again; the stage record stays the
source. The file is plain text a reader opens without the package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["LOG_NAME", "RunLog"]

#: The file name of the log, inside the run folder or the set folder.
LOG_NAME = "geom-run.log"

_MAX_VALUE = 160


def _short(value: Any) -> str:
    text = json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= _MAX_VALUE else text[: _MAX_VALUE - 3] + "..."


def _verdict(check: Mapping[str, Any]) -> str:
    return str(check["status"]).upper()


def _check_line(check: Mapping[str, Any]) -> str:
    if check["status"] == "skipped":
        return f"  check {check['id']}: SKIPPED ({check['note']})"
    sign = "<=" if check.get("bound", "max") == "max" else ">="
    unit = f" {check['unit']}" if check.get("unit") else ""
    where = f" where {_short(check['where'])}" if check.get("where") else ""
    note = f" ({check['note']})" if check.get("note") else ""
    return (
        f"  check {check['id']}: measured {check['measured']}{unit}, "
        f"threshold {sign} {check['limit']}{unit}: {_verdict(check)}{where}{note}"
    )


class RunLog:
    """Appends the sections of one run to ``<folder>/geom-run.log``, flushed at each call."""

    def __init__(self, folder: Path) -> None:
        self.path = Path(folder) / LOG_NAME

    def _write(self, lines: Sequence[str], *, mode: str = "a") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open(mode, encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()

    def start(
        self,
        *,
        config: str,
        package_version: str,
        configuration: str | None,
        variant: str,
        until: str,
        started: str,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """Begin the file: the configuration file, the package version and the run's parameters."""
        self._write(
            [
                "cfdbender geometry run log",
                f"configuration file: {config}",
                f"package version: {package_version}",
                f"started: {started}",
                f"configuration: {configuration or '(all components)'}  variant: {variant}",
                f"until: {until}",
                f"overrides: {_short(dict(overrides or {}))}",
                "",
            ],
            mode="w",
        )

    def note(self, line: str) -> None:
        """Append one line, such as the parameter id once the configuration is resolved."""
        self._write([line, ""])

    def stage(self, record: Mapping[str, Any]) -> None:
        """Append one stage: inputs, checks, wall time, warnings and findings."""
        outputs = dict(record["outputs"])
        inputs = outputs.pop("inputs", None)
        lines = [
            f"sub-step {record['stage']} {record['name']}: {str(record['status']).upper()}"
            f" (wall {record['wall_s']:.3f} s)"
        ]
        if isinstance(inputs, Mapping):  # every input of each piece, whole: never shortened
            lines.extend(
                f"  inputs of {piece}: {json.dumps(value, default=str, sort_keys=True)}"
                for piece, value in inputs.items()
            )
        elif inputs is not None:
            lines.append(f"  inputs: {json.dumps(inputs, default=str, sort_keys=True)}")
        for key, value in outputs.items():
            lines.append(f"  file or output {key}: {_short(value)}")
        lines.extend(_check_line(c) for c in record["checks"])
        lines.extend(f"  finding: {item}" for item in record["findings"])
        lines.extend(f"  warning: {item}" for item in record["warnings"])
        lines.append("")
        self._write(lines)

    def finish(self, report: Mapping[str, Any], not_reached: Sequence[str], wall_s: float) -> None:
        """Append the ending: completed, or where the run stopped and why; stages not run."""
        status = report["status"]
        lines = [f"result: {status.upper()} (total wall {wall_s:.3f} s)"]
        if status in ("failed", "refused"):
            lines.append(f"stopped at sub-step {report.get('stage')}, check {report.get('check')}")
            if report.get("measured") is not None:
                lines.append(f"  measured {report['measured']}, threshold {report.get('limit')}")
            lines.append(f"  why: {report.get('message') or '(no message)'}")
        elif status == "accepted":
            lines.append("completed: every sub-step ran and every check was measured")
        else:
            lines.append("completed with checks not measured (incomplete)")
        if not_reached:
            lines.append("sub-steps not reached: " + ", ".join(not_reached))
        not_run = list(report.get("not_run") or [])
        if not_run:
            lines.append("checks not measured (incomplete):")
            lines.extend(f"  {item}" for item in not_run)
        self._write(lines)
