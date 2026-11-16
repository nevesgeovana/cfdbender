"""SHA-256 digests of files and texts, for pins and input identities.

A file is hashed as bytes, so its line endings are part of the digest; a text is hashed as
UTF-8, so the same text gives the same digest on every platform.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["file_sha256", "optional_file_sha256", "text_sha256"]

#: Bytes read per block; any size gives the same digest, this one bounds the memory used.
_BLOCK = 1 << 20


def file_sha256(path: str | Path) -> str:
    """Return the lowercase hexadecimal SHA-256 of a file's bytes.

    Parameters
    ----------
    path : str or Path
        The file to hash.

    Returns
    -------
    str
        The digest.

    Raises
    ------
    OSError
        If the file cannot be read.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> target = Path(tempfile.mkdtemp()) / "a.txt"
    >>> _ = target.write_bytes(b"a")
    >>> file_sha256(target)[:12]
    'ca978112ca1b'
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_file_sha256(path: str | Path) -> str | None:
    """Return the SHA-256 of a file's bytes, or None when it cannot be read.

    Parameters
    ----------
    path : str or Path
        The file to hash.

    Returns
    -------
    str or None
        The digest, or None for an unreadable file, so a record can say the digest is absent.
    """
    try:
        return file_sha256(path)
    except OSError:
        return None


def text_sha256(text: str) -> str:
    """Return the lowercase hexadecimal SHA-256 of a string's UTF-8 encoding.

    Parameters
    ----------
    text : str
        The text to hash.

    Returns
    -------
    str
        The digest.

    Examples
    --------
    >>> text_sha256("a")[:12]
    'ca978112ca1b'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
