"""Text and CSV writing with LF line ends on every platform, replacing a file atomically.

A file is written to a temporary sibling and moved over the target, so a reader never sees a
half-written file and a failed write leaves the previous file in place.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO, Any

__all__ = ["csv_writer", "write_bytes", "write_csv", "write_json", "write_lines", "write_text"]


def write_text(
    path: str | Path, text: str, *, encoding: str = "utf-8", errors: str | None = None
) -> int:
    r"""Write ``text`` to ``path`` with LF line ends, replacing the file atomically.

    Parameters
    ----------
    path : str or Path
        The file to create or replace; its folder must exist.
    text : str
        The text, whose own line ends are written unchanged.
    encoding : str, optional
        The encoding, UTF-8 by default.
    errors : str, optional
        The ``errors`` argument of :func:`open`.

    Returns
    -------
    int
        The number of characters written.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> target = Path(tempfile.mkdtemp()) / "lines.txt"
    >>> write_text(target, "a\nb\n")
    4
    >>> target.read_bytes()
    b'a\nb\n'
    """
    target = Path(path)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding=encoding, errors=errors, newline="\n") as stream:
            written = stream.write(text)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return written


def csv_writer(handle: IO[str], **fmt: Any) -> Any:
    """Return a :func:`csv.writer` that ends each row with LF.

    Parameters
    ----------
    handle : file
        A text stream opened with ``newline=""``.
    **fmt
        Further format parameters of :func:`csv.writer`.

    Returns
    -------
    csv writer
        The writer.
    """
    return csv.writer(handle, lineterminator="\n", **fmt)


def write_bytes(path: str | Path, data: bytes) -> int:
    """Replace ``path`` with ``data`` atomically and return the number of bytes written.

    Parameters
    ----------
    path : str or Path
        The file to create or replace; its folder must exist.
    data : bytes
        The complete payload, written unchanged.

    Returns
    -------
    int
        The number of bytes written.
    """
    return write_text(path, data.decode("latin-1"), encoding="latin-1")


def write_lines(path: str | Path, lines: Iterable[str], *, encoding: str = "utf-8") -> int:
    """Write ``lines`` to ``path``, each ended by LF, and return how many were written.

    Parameters
    ----------
    path : str or Path
        The file to create or replace; its folder must exist.
    lines : iterable of str
        The lines, without their line end.
    encoding : str, optional
        The encoding, UTF-8 by default.

    Returns
    -------
    int
        The number of lines written.
    """
    written = list(lines)
    write_text(path, "".join(line + "\n" for line in written), encoding=encoding)
    return len(written)


def write_csv(
    path: str | Path, rows: Iterable[Iterable[object]], *, header: Sequence[str] | None = None
) -> int:
    """Write ``rows`` (and an optional ``header``) as a CSV file with LF record ends.

    Parameters
    ----------
    path : str or Path
        The file to create or replace; its folder must exist.
    rows : iterable of iterables
        The data rows.
    header : sequence of str, optional
        The header row.

    Returns
    -------
    int
        The number of data rows written.
    """
    buffer = io.StringIO(newline="")
    writer = csv_writer(buffer)
    if header is not None:
        writer.writerow(header)
    count = 0
    for row in rows:
        writer.writerow(row)
        count += 1
    write_text(path, buffer.getvalue())
    return count


def write_json(path: str | Path, payload: object, *, indent: int | None = 2, **dump: Any) -> int:
    """Write ``payload`` as JSON ending with one LF; non-finite floats are refused.

    Parameters
    ----------
    path : str or Path
        The file to create or replace; its folder must exist.
    payload : object
        The JSON-serializable value.
    indent : int, optional
        The indent of :func:`json.dumps`, 2 by default.
    **dump
        Further arguments of :func:`json.dumps`; ``allow_nan`` is always False.

    Returns
    -------
    int
        The number of characters written.
    """
    dump["allow_nan"] = False
    return write_text(path, json.dumps(payload, indent=indent, **dump) + "\n")
