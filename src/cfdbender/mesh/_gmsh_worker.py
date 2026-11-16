"""Run a geometry-kernel conversion in a disposable worker process.

The OpenCASCADE kernel behind gmsh can fault natively on a malformed surface (an access
violation), and a Python ``except`` cannot establish that the kernel is usable afterwards. So
each conversion that needs the kernel runs in a worker process of its own: the parent never
initializes gmsh for it, a fault ends only the worker, and the parent reports the failure with
its cause and goes on. A failed conversion is fatal for that conversion: no kernel state is
reused, no retry is made in the same process, and nothing is published: an operation that
names an output (``out``) writes it, and everything beside it, to a staging folder, and the
parent moves the files to their place only after the worker has answered without an error,
so a worker that is killed, times out or refuses part way leaves nothing behind.

The request and the answer travel as JSON files named on the worker's command line, never
through the standard streams, which the kernel writes to.
"""

from __future__ import annotations

import builtins
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cfdbender._errors import (
    GeometryError,
    GeometryWarning,
    InputArtifactError,
    InputValueError,
    warn_by_name,
)
from cfdbender._textio import write_text

__all__ = ["run_isolated"]

#: The environment variable that, set to ``0``, runs the kernel in the calling process instead
#: of a worker. It exists for tests that patch the kernel in their own process.
ISOLATE_ENV = "CFDBENDER_GMSH_ISOLATE"

#: The environment variable that sets the seconds a worker may run before it is ended and
#: reported. A conversion of an assembly takes seconds to minutes, and a hung kernel must not
#: hold the caller for ever; the default is an hour.
TIMEOUT_ENV = "CFDBENDER_GMSH_TIMEOUT"
_DEFAULT_TIMEOUT_S = 3600.0

#: The Windows process-creation flag that opens no console window for the worker.
_NO_WINDOW = 0x08000000
#: The last lines of the worker's error stream carried into the refusal, at most.
_TAIL_LINES = 12


def _describe(code: int) -> str:
    if code < 0 and os.name != "nt":
        return f"signal {-code}"
    return f"exit code {code} ({code & 0xFFFFFFFF:#x})"


def _timeout() -> float:
    """Return the worker's time limit in seconds, refusing a value that is not a positive number."""
    text = os.environ.get(TIMEOUT_ENV)
    if text is None:
        return _DEFAULT_TIMEOUT_S
    try:
        limit = float(text)
    except ValueError:
        limit = math.nan
    if not (math.isfinite(limit) and limit > 0):
        raise InputValueError(
            f"{TIMEOUT_ENV}={text!r} must be a positive number of seconds (default "
            f"{_DEFAULT_TIMEOUT_S:g}). Nothing was written."
        )
    return limit


def run_isolated(
    operation: str, arguments: dict[str, Any], direct: Callable[..., dict[str, Any]]
) -> dict[str, Any]:
    """Return the record of ``operation`` run in a worker with ``arguments``.

    ``operation`` is ``"<module>:<function>"``, a function of this package that takes
    ``arguments`` as keywords and returns a JSON-serializable record. ``direct`` is the same
    function, called here when :data:`ISOLATE` is off. When ``arguments`` holds ``out``, the
    worker writes there under a staging folder and the files are moved beside ``out`` after a
    successful answer.

    Raises
    ------
    InputArtifactError
        The worker ended without an answer (a kernel fault, a timeout): the message gives the
        cause and says nothing was written.
    """
    if os.environ.get(ISOLATE_ENV) == "0":
        return direct(**arguments)
    with tempfile.TemporaryDirectory(prefix="cfdb-gmsh-") as folder:
        request, answer = Path(folder) / "request.json", Path(folder) / "answer.json"
        requested = Path(arguments["out"]) if "out" in arguments else None
        stage = Path(folder) / "publish"
        sent = arguments if requested is None else {**arguments, "out": str(stage / requested.name)}
        write_text(request, json.dumps({"operation": operation, "arguments": sent}))
        limit = _timeout()
        flags = _NO_WINDOW if os.name == "nt" else 0
        env = {**os.environ, _ENV: "1"}
        try:
            done = subprocess.run(
                [sys.executable, "-m", __name__, str(request), str(answer)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limit,
                creationflags=flags,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise _failed(f"it ran longer than {limit:g} s and was ended", "") from error
        if not answer.is_file():
            tail = "\n".join((done.stderr or "").strip().splitlines()[-_TAIL_LINES:])
            raise _failed(f"its worker process ended with {_describe(done.returncode)}", tail)
        reply = json.loads(answer.read_text(encoding="utf-8"))
        if requested is not None:
            reply = json.loads(
                json.dumps(reply).replace(
                    json.dumps(str(stage))[1:-1], json.dumps(str(requested.parent))[1:-1]
                )
            )
            if "error" not in reply:
                publish_stage(stage, requested.parent)
    for category, message in reply.get("warnings", []):
        warn_by_name(message, category, 2)
    if "error" in reply:
        raise _rebuilt(reply["error"])
    result: dict[str, Any] = reply["result"]
    return result


def publish_stage(stage: Path, folder: Path) -> None:
    """Move the files of a finished worker from its staging folder to ``folder``.

    An earlier record of these outputs goes first and the new one last, so a folder that holds
    a record holds its whole set, and each file arrives whole or not at all.
    """
    files = sorted(stage.iterdir()) if stage.is_dir() else []
    folder.mkdir(parents=True, exist_ok=True)
    records = [path for path in files if path.name.endswith(".json")]
    for path in records:
        (folder / path.name).unlink(missing_ok=True)
    for path in [*(path for path in files if path not in records), *records]:
        _arrive_whole(path, folder / path.name)


def _arrive_whole(path: Path, destination: Path) -> None:
    """Move one file; across file systems, copy beside the destination and rename it there."""
    try:
        os.replace(path, destination)
    except OSError:
        partial = destination.with_name(destination.name + ".partial")
        try:
            shutil.copy2(path, partial)
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)


def _failed(cause: str, tail: str) -> InputArtifactError:
    detail = f" The worker said:\n{tail}\n" if tail else " "
    return InputArtifactError(
        f"the geometry kernel failed on this input: {cause}.{detail}This conversion cannot "
        "continue: the kernel is not reused after a fault. Nothing was written. Check the "
        "IGES file (a malformed surface can fault the kernel), write it again from its CAD "
        "system and run again; the calling process is unaffected.",
        kind="geometry",
    )


def _rebuilt(error: dict[str, Any]) -> Exception:
    """Return the exception the worker raised, as the parent's own type and message.

    ``MissingExtraError`` and ``InputValueError`` keep their type, and any other error of this
    package returns as ``InputArtifactError``. A bare ``Exception`` is how the kernel's binding
    reports a failure, so it returns as the ``InputArtifactError`` the public functions document,
    saying the kernel failed. Anything else is a defect, not a refusal of the input: it returns
    as the same builtin type where there is one, else as a ``RuntimeError`` naming the type.
    Both carry the worker's traceback as a note.
    """
    name, message = str(error["type"]), str(error["message"])
    if name == "MissingExtraError":
        return _errors_missing(message, error.get("attrs") or {})
    if name == "KernelUnavailableError":
        from cfdbender.mesh._extras import KernelUnavailableError

        attrs = error.get("attrs") or {}
        return KernelUnavailableError(
            purpose=attrs.get("purpose", "the geometry kernel"),
            reason=attrs.get("reason", message),
        )
    if name == "InputValueError":
        return InputValueError(message)
    if error.get("package"):
        return InputArtifactError(message, kind=error.get("kind") or "geometry")
    found = getattr(builtins, name, None)
    rebuilt: Exception
    try:
        if found is Exception:
            rebuilt = InputArtifactError(
                f"the geometry kernel failed: {message}. Nothing was written.", kind="geometry"
            )
        elif isinstance(found, type) and issubclass(found, Exception):
            rebuilt = found(message)
        else:
            rebuilt = RuntimeError(f"the geometry worker raised {name}: {message}")
    except Exception:
        rebuilt = RuntimeError(f"the geometry worker raised {name}: {message}")
    trace = str(error.get("traceback") or "").strip()
    if trace:
        rebuilt.add_note(f"worker traceback:\n{trace}")
    return rebuilt


def _errors_missing(message: str, attrs: dict[str, str]) -> Exception:
    from cfdbender.mesh._extras import MissingExtraError

    error = MissingExtraError.__new__(MissingExtraError)
    Exception.__init__(error, message)
    for key, value in attrs.items():
        setattr(error, key, value)
    return error


_ENV = "CFDBENDER_GMSH_WORKER"


def _warning_text(caught: warnings.WarningMessage) -> str:
    """Return a worker warning's text, naming its category where the replay cannot keep it.

    The parent raises every warning as a category of this package, so a builtin one such as
    numpy's RuntimeWarning arrives as GeometryWarning with its own category in the text.
    """
    text = str(caught.message)
    if issubclass(caught.category, GeometryWarning):
        return text
    return f"{caught.category.__name__}: {text}"


def _main(request: Path, answer: Path) -> int:
    """Run the requested operation and write its answer; a native fault writes none."""
    body = json.loads(request.read_text(encoding="utf-8"))
    module, _, function = str(body["operation"]).partition(":")
    reply: dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            call = getattr(importlib.import_module(module), function)
            reply["result"] = call(**body["arguments"])
        except Exception as error:
            reply["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "kind": getattr(error, "kind", None),
                "package": isinstance(error, GeometryError),
                "attrs": {
                    key: str(getattr(error, key))
                    for key in ("extra", "package", "purpose", "remedy", "reason")
                    if hasattr(error, key)
                },
                "traceback": traceback.format_exc(),
            }
    reply["warnings"] = [(w.category.__name__, _warning_text(w)) for w in caught]
    write_text(answer, json.dumps(reply))
    return 0


if __name__ == "__main__":
    sys.exit(_main(Path(sys.argv[1]), Path(sys.argv[2])))
