"""Stage records and the run report of the geometry pipeline.

Pipeline role: pure helper of the geometry package. A stage record is the file a paper cites:
it names the package version, the identity of the inputs, and for each check its id, its
limit and the value measured. A report is what a design loop reads after a parameter set
that failed: it names the stage and the check, and it is returned, never raised.

``started`` and ``wall_s`` change from run to run and stay out of every identity; use
:func:`identity_view` to compare two records.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from cfdbender._errors import CheckFailedError as CheckFailedError
from cfdbender._errors import InputValueError
from cfdbender._textio import write_text

__all__ = [
    "REPORT_FORMAT",
    "STAGE_FORMAT",
    "CheckFailedError",
    "check_entry",
    "exit_code",
    "geometry_report",
    "identity_view",
    "stage_record",
    "validate_stage_record",
    "write_record",
]

STAGE_FORMAT = "cfdbender-geom-stage/1"
REPORT_FORMAT = "cfdbender-geom-report/1"
_STATUSES = ("passed", "failed", "skipped")
#: The status of a run and its exit code. ``incomplete`` is a run that reached its last stage
#: with a check of the stage table not measured (a stage not built, or a check its stage did
#: not return): no check failed, and it is not ``accepted``. 3 is the package's code for a
#: result with parts outstanding.
_OUTCOMES = {"accepted": 0, "failed": 1, "refused": 2, "incomplete": 3}
_VOLATILE = ("started", "wall_s")
_BOUNDS = ("max", "min")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _agrees(check_id, status, measured, limit, bound):
    """Refuse a status its own two numbers contradict; ``bound`` says which side is good."""
    if bound not in _BOUNDS:
        raise InputValueError(f"check {check_id}: bound expected one of {_BOUNDS}")
    within = measured <= limit if bound == "max" else measured >= limit
    if within != (status == "passed"):
        raise InputValueError(
            f"check {check_id}: status {status} contradicts measured {measured} "
            f"against the {bound} {limit}"
        )


def check_entry(
    check_id, status, measured=None, limit=None, *, unit="", where=None, note="", bound="max"
):
    """Return one check of a stage record.

    A check that ran (``passed`` or ``failed``) carries a finite measured value and a finite
    limit, so a paper can cite both, and its status must follow from them: ``bound`` says whether
    the limit is a maximum, the default, or a minimum. A ``skipped`` check carries a note that says
    why.

    Parameters
    ----------
    check_id : str
        The id of the check in the stage table, such as ``"input.valid"``.
    status : str
        ``"passed"``, ``"failed"`` or ``"skipped"``.
    measured : float, optional
        The measured value; required when the check ran.
    limit : float, optional
        The limit it is held to; required when the check ran.
    unit : str, optional
        The unit of ``measured`` and ``limit``.
    where : Mapping, optional
        Where the worst value sits, such as a component or a station.
    note : str, optional
        Why a skipped check was skipped; required for ``"skipped"``.
    bound : str, optional
        ``"max"`` (default) if ``measured`` may not exceed ``limit``, ``"min"`` if it may not
        fall below it.

    Returns
    -------
    dict
        The check as plain values.

    Raises
    ------
    InputValueError
        If the status is unknown, a number is missing or not finite, a skipped check has no
        note, or the status contradicts the numbers.

    Examples
    --------
    >>> from cfdbender.geometry.stages import check_entry
    >>> check_entry("input.valid", "passed", 0, 0, unit="refused keys")["status"]
    'passed'
    """
    if status not in _STATUSES:
        raise InputValueError(f"check {check_id}: status expected one of {_STATUSES}")
    if status == "skipped":
        if not note:
            raise InputValueError(f"check {check_id}: a skipped check names its reason in note")
    elif not (_finite(measured) and _finite(limit)):
        raise InputValueError(f"check {check_id}: measured and limit are required numbers")
    else:
        _agrees(check_id, status, measured, limit, bound)
    return {
        "id": check_id,
        "status": status,
        "measured": measured,
        "limit": limit,
        "bound": bound,
        "unit": unit,
        "where": dict(where or {}),
        "note": note,
    }


def stage_record(
    stage,
    name,
    *,
    package_version,
    inputs_id,
    parameter_id,
    checks=(),
    findings=(),
    warnings=(),
    outputs=None,
    kernel=None,
    started="",
    wall_s=0.0,
):
    """Return the record of one stage; its status follows from its checks.

    ``failed`` when any check failed, ``skipped`` when none ran, else ``passed``. The record
    names ``package_version`` and ``inputs_id`` (the identity of what the stage read). A check
    :func:`check_entry` would refuse is refused here too, whoever built it.

    Parameters
    ----------
    stage : str
        The stage id, such as ``"S0"``.
    name : str
        The name of the stage.
    package_version : str
        The version of this package that ran the stage.
    inputs_id : str
        The identity of what the stage read.
    parameter_id : str
        The identity of the parameters of the run.
    checks : sequence of Mapping, optional
        The checks of the stage, each as :func:`check_entry` returns it.
    findings : sequence of str, optional
        What the stage found that is not a check.
    warnings : sequence of str, optional
        The warnings of the stage.
    outputs : Mapping, optional
        What the stage produced, such as file names and counts.
    kernel : str, optional
        The geometry kernel it used, or None.
    started : str, optional
        The start time, ISO 8601.
    wall_s : float, optional
        The wall time in seconds.

    Returns
    -------
    dict
        The record as plain values.

    Examples
    --------
    >>> from cfdbender.geometry.stages import check_entry, stage_record
    >>> check = check_entry("input.valid", "passed", 0, 0)
    >>> stage_record("S0", "Definition", package_version="0", inputs_id="a",
    ...              parameter_id="b", checks=[check])["status"]
    'passed'
    """
    checks = [_rechecked(c) for c in checks]
    status = _status_of(checks)
    return {
        "format": STAGE_FORMAT,
        "stage": stage,
        "name": name,
        "status": status,
        "package_version": package_version,
        "inputs_id": inputs_id,
        "parameter_id": parameter_id,
        "started": started,
        "wall_s": wall_s,
        "kernel": kernel,
        "checks": checks,
        "findings": list(findings),
        "warnings": list(warnings),
        "outputs": dict(outputs or {}),
    }


def _rechecked(check: Mapping) -> dict:
    """Return ``check`` through :func:`check_entry`, so no route around it exists."""
    return check_entry(
        check.get("id"),
        check.get("status"),
        check.get("measured"),
        check.get("limit"),
        unit=check.get("unit", ""),
        where=check.get("where"),
        note=check.get("note", ""),
        bound=check.get("bound", "max"),
    )


def _status_of(checks) -> str:
    """Return the status of a stage from its checks."""
    if any(c["status"] == "failed" for c in checks):
        return "failed"
    return "passed" if any(c["status"] != "skipped" for c in checks) else "skipped"


def validate_stage_record(record: Mapping) -> None:
    """Refuse a stage record a paper could not cite.

    A missing or empty field, a check without a value or whose status contradicts its numbers, or
    a stage status its checks do not give.
    """
    required = ("format", "stage", "name", "status", "package_version", "inputs_id", "checks")
    for key in required:
        if key not in record or record[key] in ("", None):
            raise InputValueError(f"stage record: {key} is required")
    if record["format"] != STAGE_FORMAT:
        raise InputValueError(f"stage record: format expected {STAGE_FORMAT}")
    checks = [_rechecked(check) for check in record["checks"]]
    if record["status"] != _status_of(checks):
        raise InputValueError(
            f"stage record: status {record['status']} is not what its checks give, "
            f"{_status_of(checks)}"
        )


def identity_view(record: Mapping) -> dict:
    """Return a record without ``started`` and ``wall_s``, the form two runs are compared in."""
    return {k: v for k, v in record.items() if k not in _VOLATILE}


def geometry_report(
    status,
    parameter_id,
    *,
    surface_id=None,
    build_id=None,
    stage=None,
    check=None,
    message="",
    measured=None,
    limit=None,
    where=None,
    stages=(),
    outputs=None,
    warnings=(),
    not_run=(),
):
    """Return the report of a run: ``accepted``, ``incomplete``, ``failed`` or ``refused``.

    ``failed`` names a stage and a check. ``not_run`` lists, as ``"<stage> <check>"``, every
    check of the stage table up to the last stage run that was not measured; a run with one is
    ``incomplete``, never ``accepted``.

    Parameters
    ----------
    status : str
        ``"accepted"``, ``"incomplete"``, ``"failed"`` or ``"refused"``.
    parameter_id : str
        The identity of the parameters of the run.
    surface_id : str, optional
        The identity of the surface that was built.
    build_id : str, optional
        The identity of the build (parameters, package version and kernel).
    stage : str, optional
        The stage that failed or refused; required for ``"failed"``.
    check : str, optional
        The check that failed; required for ``"failed"``.
    message : str, optional
        The reason, in words.
    measured : float, optional
        The measured value of the failed check.
    limit : float, optional
        The limit of the failed check.
    where : Mapping, optional
        Where the failed check sits.
    stages : sequence of Mapping, optional
        The stage records of the run.
    outputs : Mapping, optional
        The files the run wrote.
    warnings : sequence of str, optional
        The warnings of the run.
    not_run : sequence of str, optional
        The checks not measured, as ``"<stage> <check>"``.

    Returns
    -------
    dict
        The report as plain values.

    Raises
    ------
    InputValueError
        If the status is unknown, a failed run names no stage or check, or an accepted run
        has checks not measured.

    Examples
    --------
    >>> from cfdbender.geometry.stages import geometry_report
    >>> geometry_report("accepted", "p")["status"]
    'accepted'
    """
    if status not in _OUTCOMES:
        raise InputValueError(f"report: status expected one of {sorted(_OUTCOMES)}")
    if status == "failed" and not (stage and check):
        raise InputValueError("report: a failed run names its stage and its check")
    if status == "accepted" and not_run:
        raise InputValueError("report: a run with checks not measured is not accepted")
    return {
        "format": REPORT_FORMAT,
        "status": status,
        "parameter_id": parameter_id,
        "surface_id": surface_id,
        "build_id": build_id,
        "stage": stage,
        "check": check,
        "message": message,
        "measured": measured,
        "limit": limit,
        "where": dict(where or {}),
        "stages": list(stages),
        "outputs": dict(outputs or {}),
        "warnings": list(warnings),
        "not_run": list(not_run),
    }


def exit_code(report: Mapping) -> int:
    """Return the exit code of a run: 0 accepted, 1 failed, 2 refused, 3 incomplete.

    Parameters
    ----------
    report : Mapping
        A report as :func:`geometry_report` returns it.

    Returns
    -------
    int
        The process exit code of the report's status.

    Examples
    --------
    >>> from cfdbender.geometry.stages import exit_code
    >>> exit_code({"status": "failed"})
    1
    """
    return _OUTCOMES[report["status"]]


def write_record(path, record: Mapping) -> None:
    """Write a record or report as sorted, indented JSON; a stage record is validated first.

    Parameters
    ----------
    path : str or Path
        The file to write.
    record : Mapping
        A stage record or a report.

    Returns
    -------
    None
        The file is written.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from cfdbender.geometry.stages import geometry_report, write_record
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> write_record(folder / "report.json", geometry_report("accepted", "p"))
    >>> (folder / "report.json").exists()
    True
    """
    if record.get("format") == STAGE_FORMAT:
        validate_stage_record(record)
    write_text(Path(path), json.dumps(record, indent=2, sort_keys=True) + "\n")
