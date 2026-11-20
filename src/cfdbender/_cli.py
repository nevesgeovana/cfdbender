"""Console outcome reporting shared by every cfdbender command, below the domain layers.

Decorating a console ``main`` preserves its return value and its exceptions, standard
output and the exit code. Only the outermost entry point reports, argparse's early exits
included. Every invocation ends with the package's signature on standard error: a box drawn
from :mod:`cfdbender._signature` for an outcome, or one short line for ``--help`` and
``--version`` so their output stays compact. The signature always appears: no flag, stream
kind or error path hides it, and a stream that cannot take it never changes the command's own
result or exit code.

The outcome is ``success`` for a return of ``None`` or ``0``, ``failed`` for any other return
and for an invalid command line, ``cancelled`` for a keyboard interrupt, and ``help`` or
``version`` for a clean exit that asked for one. A success that ran a post (the ``post`` verb,
or the post a ``collect`` runs) is a ``post`` outcome.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import TextIO

import cfdbender._signature as _signature

_ACTIVE: ContextVar[bool] = ContextVar("cfdbender_cli_active", default=False)
#: Whether this invocation ran a post; a success that did is signed with the koala.
_POSTED: ContextVar[list[bool] | None] = ContextVar("cfdbender_cli_posted", default=None)
#: Whether this invocation was handed to pyflightstream, whose own entry point signs it.
_FORWARDED: ContextVar[list[bool] | None] = ContextVar("cfdbender_cli_forwarded", default=None)

_MESSAGES = {
    "help": ("Help delivered; the map is yours.", "Help ready; pick your next turn."),
    "version": ("Version reported; coordinates checked.", "Version ready; now you know the build."),
}
#: The chooser of drawings and phrases; a test replaces it with a seeded one.
_RNG = random.Random()


def note_post_ran() -> None:
    """Record that this invocation ran a post, so its success is signed as one."""
    posted = _POSTED.get()
    if posted is not None:
        posted[0] = True


def note_forwarded() -> None:
    """Record that pyflightstream's console entry point runs this invocation and signs it."""
    forwarded = _FORWARDED.get()
    if forwarded is not None:
        forwarded[0] = True


def _signature_text(outcome: str) -> str:
    """Return what the invocation prints on stderr for ``outcome``."""
    if outcome in _MESSAGES:
        return f"{_RNG.choice(_MESSAGES[outcome])} {_signature.SEES_YOU}"
    name, phrase = _signature.pick(outcome, _RNG)
    return f"\n{_signature.box(name, phrase)}\n"


def _write_replaced(text: str, stream: TextIO) -> None:
    """Write ``text`` with the stream's own encoding, a replacement mark for what it lacks."""
    encoding = getattr(stream, "encoding", None) or "ascii"
    data = text.encode(encoding, errors="replace")
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        stream.flush()
        buffer.write(data)
        buffer.flush()
    else:
        stream.write(data.decode(encoding, errors="replace"))


def _sign(outcome: str) -> None:
    """Print the signature of ``outcome`` on stderr; it always appears.

    A stream that cannot encode the drawing gets the same text with replacement marks, and a
    drawing that cannot be built falls back to the bare line. Only a closed or broken stream
    prints nothing, and it never replaces the command's own result or exit code.
    """
    try:
        text = _signature_text(outcome)
    except Exception:
        text = _signature.SEES_YOU
    stream = sys.stderr
    if stream is None:
        return
    try:
        try:
            print(text, file=stream)
        except UnicodeEncodeError:
            _write_replaced(f"{text}\n", stream)
    except (OSError, ValueError, AttributeError):
        pass


def cli_entrypoint[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Report the actual CLI outcome once, without changing CLI semantics.

    Parameters
    ----------
    function : Callable[P, R]
        Console entry point whose return value and exceptions are preserved.

    Returns
    -------
    Callable[P, R]
        The entry point, ending its call with the signature on stderr.

    Examples
    --------
    >>> def main(argv=None):
    ...     return 0
    >>> callable(cli_entrypoint(main))
    True
    """

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        if _ACTIVE.get():
            return function(*args, **kwargs)
        token = _ACTIVE.set(True)
        supplied = args[0] if args else kwargs.get("argv")
        argv = supplied if isinstance(supplied, (list, tuple)) else sys.argv[1:]
        posted = [False]
        posted_token = _POSTED.set(posted)
        forwarded = [False]
        forwarded_token = _FORWARDED.set(forwarded)
        outcome = "failed"
        try:
            result = function(*args, **kwargs)
            outcome = "success" if result is None or result == 0 else "failed"
            return result
        except SystemExit as error:
            if error.code is None or error.code == 0:
                outcome = (
                    "help"
                    if "--help" in argv or "-h" in argv
                    else "version"
                    if "--version" in argv
                    else "success"
                )
            raise
        except KeyboardInterrupt:
            outcome = "cancelled"
            raise
        finally:
            _POSTED.reset(posted_token)
            _FORWARDED.reset(forwarded_token)
            _ACTIVE.reset(token)
            if outcome == "success" and posted[0]:
                outcome = "post"
            if not forwarded[0]:
                _sign(outcome)

    return wrapped
