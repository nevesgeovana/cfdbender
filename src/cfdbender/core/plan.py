"""Expand matrix rows into a read-only plan through an injected solver hook.

Adapters implement validate_row(row, inputs) -> RowRequirements. They inspect
inputs.setup, inputs.reference and inputs.post, and read any additional files
through inputs.read_bytes/read_toml so every byte read is included in provenance.
The hook validates and describes actions; it must not run a solver or write files.
"""

from __future__ import annotations

import contextvars
import difflib
import hashlib
import json
import os
import re
import tempfile
import time
import tomllib
import warnings
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Protocol, cast

from cfdbender.core.flight import ATTITUDE_KEYS, STATE_KEYS
from cfdbender.core.matrix import HPC_OVERRIDE_PREFIX, RETIRED_COLUMNS, MatrixRow, parse_matrix
from cfdbender.core.mesh import MESH_FOLDER, mesh_candidates, resolve_setup_paths
from cfdbender.core.naming import PointNameError, datapoint_dir_name, point_name
from cfdbender.core.refusals import warn_or_refuse
from cfdbender.core.settings import WorkspaceError, _validate, setup_defaults


def _empty() -> Mapping[str, object]:
    return MappingProxyType({})


class PlanError(WorkspaceError):
    """A workspace input, solver validation or output claim prevents planning."""


def workspace_path(root: Path, name: str) -> Path:
    """Resolve a workspace-relative path, refusing escapes and ambiguous separators.

    Parameters
    ----------
    root : Path
        Resolved workspace root.
    name : str
        Relative POSIX path.

    Returns
    -------
    Path
        Contained absolute path, including symlink resolution.
    """
    path = checked_relative(name)
    target = (root / path).resolve()
    if not target.is_relative_to(root) or target == root:
        raise PlanError(f"{name!r}: path outside workspace")
    return target


def checked_relative(name: str) -> Path:
    """Refuse a workspace path that escapes or is ambiguous on Windows, without resolving it.

    Parameters
    ----------
    name : str
        Relative POSIX path.

    Returns
    -------
    Path
        The unresolved relative path.
    """
    path = Path(name)
    if not name or chr(92) in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise PlanError(f"{name!r}: path outside workspace or not a relative POSIX path")
    for part in path.parts:
        stem = part.split(".", 1)[0].upper()
        if (
            part.endswith((".", " "))
            or re.search(r"~[0-9]+(?:[.]|$)", part, re.IGNORECASE)
            or stem in {"CON", "PRN", "AUX", "NUL"}
            or re.fullmatch(r"(?:COM|LPT)[1-9]", stem)
            or any(ord(c) < 32 or c in '<>"|?*' for c in part)
        ):
            raise PlanError(f"{name!r}: ambiguous or invalid Windows path component")
    return path


def _checked_file_name(name: str) -> str:
    """Require one portable filename before composing an output path."""
    if len(checked_relative(name).parts) != 1 or "/" in name:
        raise PlanError(f"{name!r}: output name must be one filename")
    return name


def owned_path(root: Path, name: str) -> Path:
    """Return a management path, refusing links or junctions in any component."""
    relative = checked_relative(name)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink() or os.path.isjunction(path):
            raise PlanError(f"{name!r}: linked management path; refusing to follow it")
    workspace_path(root, name)
    return path


def is_matrix_file(name: str) -> bool:
    """Return whether an input names a matrix in one of the supported matrix homes.

    Every point records the matrix's digest, but a point is as stale as the files it read
    (its s, r and p files, its geometry), not as the matrix, which a new row changes: the
    digest is kept in the record and never judged as a point's input.
    """
    return re.fullmatch(r"(?:inputs/matrices/)?[A-Za-z0-9][A-Za-z0-9_-]*[.]fs", name) is not None


#: Inputs up to this size are read whole and held, so their digest and their parse come from one
#: byte snapshot; a larger input is streamed through the digest and never held by planning.
LARGE_INPUT_BYTES = 8 << 20
#: Bytes read per block when a large input is streamed.
_BLOCK = 1 << 20


def _open_input(path: Path) -> BinaryIO:
    """Open an input for reading; every read of a tracked input goes through here."""
    return open(path, "rb")


#: Blocks streamed between two requests to drop the bytes already read from the page cache.
_DROP_EVERY = 16


def _drop_cached(stream: BinaryIO, upto: int) -> None:
    """Ask the kernel to drop the first ``upto`` bytes of a streamed file from its page cache.

    A large input is read once, so keeping it cached only fills the memory of a shared host
    (on Linux a cgroup counts page cache against the session). Where ``os.posix_fadvise`` is
    missing (Windows) or the stream has no descriptor, nothing happens; nothing here fails.
    """
    advise = getattr(os, "posix_fadvise", None)
    flag = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or flag is None or upto <= 0:
        return
    try:
        advise(stream.fileno(), 0, upto, flag)
    except (AttributeError, OSError, ValueError):
        pass


def _stream_blocks(stream: BinaryIO) -> Iterator[bytes]:
    """Yield a large input block by block, dropping what was read from the page cache."""
    done = 0
    blocks = 0
    while chunk := stream.read(_BLOCK):
        done += len(chunk)
        blocks += 1
        yield chunk
        if blocks % _DROP_EVERY == 0:
            _drop_cached(stream, done)
    _drop_cached(stream, done)


def _hasher() -> hashlib._Hash:
    """Return a new SHA-256 state; every digest of a tracked input is fed through one."""
    return hashlib.sha256()


class ByteScanner(Protocol):
    """Collect a result from the bytes of an input while planning digests them."""

    def feed(self, chunk: bytes) -> None:
        """Take the next block of bytes, in file order."""
        ...

    def result(self) -> object:
        """Return what the scan found, after the last block."""
        ...


#: A file whose newest timestamp is younger than this at the time it was hashed is not cached:
#: coarse network-file-system clocks could hide a later rewrite behind the same timestamps.
CACHE_SETTLE_NS = 2_000_000_000
_REHASH: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cfdbender_rehash_inputs", default=False
)


_NEW_POST: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cfdbender_new_post_file", default=False
)


@contextmanager
def new_post_file(enabled: bool = True) -> Iterator[None]:
    """Let a point be loaded with a PPROC file that differs from the planned one, in the block.

    ``collect`` and ``post`` read saved outputs only, so a new grouping of the products needs no
    new plan and no new run; a command that runs never sets it.
    """
    token = _NEW_POST.set(enabled)
    try:
        yield
    finally:
        _NEW_POST.reset(token)


def new_post_file_allowed() -> bool:
    """Return whether the running command accepts a PPROC file that differs from the plan."""
    return _NEW_POST.get()


@contextmanager
def rehash_inputs(enabled: bool = True) -> Iterator[None]:
    """Ignore the persistent digest cache inside the block, reading every input again.

    Parameters
    ----------
    enabled : bool, optional
        False leaves the cache in use, so a caller can pass a command flag straight through.
    """
    token = _REHASH.set(enabled)
    try:
        yield
    finally:
        _REHASH.reset(token)


#: Values of the init.toml setting ``large_input_check``: ``digest`` reads and hashes a large
#: input (the default), ``identity`` never opens it and records its size and modification time.
LARGE_INPUT_CHECKS = ("digest", "identity")
#: Environment variable that turns the large-input check off, as ``--no-read-large-inputs`` does.
NO_READ_ENV = "CFDBENDER_NO_READ_LARGE_INPUTS"
_IDENTITY_TEXT = re.compile(r"identity:size=([0-9]{1,19});mtime_ns=([0-9]{1,28})")
_CHECK: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cfdbender_large_input_check", default=None
)
_WARNED: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "cfdbender_large_input_warned", default=None
)


@contextmanager
def large_input_check(mode: str | None) -> Iterator[None]:
    """Set the large-input check of the commands inside the block.

    Parameters
    ----------
    mode : str or None
        ``"digest"`` or ``"identity"``; None leaves the workspace setting and the environment
        variable in charge. Each warning about an unread file is given once inside the block.
    """
    if mode is not None and mode not in LARGE_INPUT_CHECKS:
        raise ValueError(f"large_input_check must be one of {', '.join(LARGE_INPUT_CHECKS)}")
    token = _CHECK.set(mode)
    warned = _WARNED.set(set())
    try:
        yield
    finally:
        _WARNED.reset(warned)
        _CHECK.reset(token)


def workspace_large_input_check(root: str | Path) -> str:
    """Return the ``large_input_check`` setting of a workspace's init.toml (``digest`` unset).

    Raises
    ------
    WorkspaceError
        The setting is present and is not one of ``LARGE_INPUT_CHECKS``.
    """
    source = Path(root) / "init.toml"
    try:
        values = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return "digest"
    value = values.get("large_input_check", "digest")
    if value not in LARGE_INPUT_CHECKS:
        raise WorkspaceError(f'{source}: large_input_check must be "digest" or "identity"')
    return cast(str, value)


def reads_large_inputs(root: str | Path) -> bool:
    """Return whether this command reads and hashes inputs above ``LARGE_INPUT_BYTES``.

    ``--rehash-inputs`` always reads. Otherwise the command's option, then the environment
    variable ``CFDBENDER_NO_READ_LARGE_INPUTS`` (any value but empty or ``0``), then the
    workspace setting decide.
    """
    if _REHASH.get():
        return True
    mode = _CHECK.get()
    if mode is not None:
        return mode == "digest"
    if os.environ.get(NO_READ_ENV, "") not in ("", "0"):
        return False
    return workspace_large_input_check(root) == "digest"


def identity_value(stat: os.stat_result) -> str:
    """Return the recorded value of a large input that was not read: its size and mtime.

    It is never 64 hexadecimal characters, so it cannot be taken for a SHA-256.
    """
    return f"identity:size={stat.st_size};mtime_ns={stat.st_mtime_ns}"


def is_identity_value(value: object) -> bool:
    """Return whether a recorded input value is a file identity rather than a SHA-256."""
    match = _IDENTITY_TEXT.fullmatch(value) if isinstance(value, str) else None
    # Bound conversion before calling int, including on untrusted registry/receipt data.
    return (
        match is not None
        and int(match[1]) <= (1 << 63) - 1
        and int(match[2]) <= ((1 << 63) - 1) * 10**9
    )


def recorded_value_valid(value: object) -> bool:
    """Return whether a recorded input or output value is a SHA-256 or a file identity."""
    return isinstance(value, str) and (
        re.fullmatch("[0-9a-f]{64}", value) is not None or is_identity_value(value)
    )


def shown_digest(value: object) -> object:
    """Return a recorded value for display: ``not hashed`` for a file identity."""
    return "not hashed" if is_identity_value(value) else value


def warn_once(key: str, message: str, warned: set[str] | None = None) -> None:
    """Warn once per command about something the large-input check did not do.

    It never refuses, also under ``warning_refusals``: the check is off by choice.
    """
    seen = _WARNED.get()
    seen = seen if seen is not None else warned
    if seen is not None:
        if key in seen:
            return
        seen.add(key)
    warnings.warn(message, UserWarning, stacklevel=3)


def skipped_content(name: str, what: str) -> str:
    """Return the warning text of a content check skipped because a file was not read."""
    return (
        f"{name}: {what} not checked; the large-input check is off, so the file is not read "
        "(use --rehash-inputs without --no-read-large-inputs to check it)"
    )


def recorded_values_agree(root: Path, name: str, recorded: object, current: str) -> bool:
    """Return whether two recorded values of one input describe the file on disk now.

    Equal values agree. A file identity on either side agrees when it is the identity of the
    file now; the other side, a SHA-256, was then taken of this same file or not at all.
    """
    if not recorded_value_valid(recorded) or not recorded_value_valid(current):
        return False
    if recorded == current:
        return True
    if not isinstance(recorded, str):
        return False
    if not (is_identity_value(recorded) or is_identity_value(current)):
        return False
    try:
        now = identity_value(workspace_path(root, name).stat())
    except OSError:
        return False
    return all(now == value for value in (recorded, current) if is_identity_value(value))


def _clock_ns() -> int:
    """Return the wall clock in nanoseconds; tests replace it to fake a recent write."""
    return time.time_ns()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _identity(stat: os.stat_result | None) -> tuple[int, ...] | None:
    """Return a usable followed-file identity, or disable reuse on this filesystem."""
    fields = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_dev", "st_ino")
    values = tuple(getattr(stat, name, None) for name in fields)
    if not all(_is_int(value) for value in values) or not values[-1]:
        return None
    return cast(tuple[int, ...], values)


def _persistent_identity(stat: os.stat_result | None) -> tuple[int, ...] | None:
    """Return the identity valid across the nodes of a shared file system, or None.

    The device number is left out: a network mount can report a different one on each client.
    """
    values = _identity(stat)
    return None if values is None else values[:3] + values[4:]


def _valid_entry(entry: object) -> bool:
    """Return whether a cache entry has the documented shape."""
    if not isinstance(entry, dict):
        return False
    digest = entry.get("sha256")
    scans = entry.get("scans")
    return (
        all(_is_int(entry.get(key)) for key in ("size", "mtime_ns", "ctime_ns", "ino"))
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and isinstance(scans, dict)
        and all(
            isinstance(tag, str)
            and isinstance(names, list)
            and all(isinstance(name, str) for name in names)
            for tag, names in scans.items()
        )
    )


class DigestCache:
    """Remember the digest of each large input between commands, keyed by its identity on disk.

    The cache is one small JSON file in the workspace root, ``.input-digests.cache``, next to the
    run records. It is never a planned input: it does not enter ``inputs_sha256`` or any record.
    An entry holds a file's size, modification time and change time in nanoseconds, its SHA-256
    and the results of scans of its bytes (for example the marker names of an SU2 mesh). An entry
    is served only when size, timestamps and inode are unchanged. The host name, the workspace
    path and the device number are not checked, so that a workspace on a shared file system can
    be used from several nodes, and a copied workspace (new inodes) is hashed again.
    A missing, unreadable, corrupt or wrong-version file is an empty cache.
    Failed writes are ignored.

    Parameters
    ----------
    root : Path
        Workspace root, where the cache file lives.
    """

    NAME = ".input-digests.cache"
    FORMAT = 1
    MAX_BYTES = 8 << 20

    def __init__(self, root: Path) -> None:
        self.path = root / self.NAME
        self._entries: dict[str, dict[str, object]] | None = None

    def _read(self) -> dict[str, dict[str, object]]:
        try:
            with self.path.open("rb") as stream:
                data = stream.read(self.MAX_BYTES + 1)
            if len(data) > self.MAX_BYTES:
                return {}
            document = json.loads(data)
        except (OSError, ValueError, RecursionError):
            return {}
        if not isinstance(document, dict) or document.get("format") != self.FORMAT:
            return {}
        entries = document.get("entries")
        if not isinstance(entries, dict):
            return {}
        return {k: v for k, v in entries.items() if isinstance(k, str) and _valid_entry(v)}

    def _all(self) -> dict[str, dict[str, object]]:
        if self._entries is None:
            self._entries = self._read()
        return self._entries

    def get(self, key: str, stat: os.stat_result) -> dict[str, object] | None:
        """Return the entry for a file whose identity still matches, or None.

        Parameters
        ----------
        key : str
            Workspace-relative POSIX name.
        stat : os.stat_result
            Current ``stat`` of the file.
        """
        if _REHASH.get():
            return None
        entry = self._all().get(key)
        identity = _persistent_identity(stat)
        if (
            identity is not None
            and entry is not None
            and tuple(entry[k] for k in ("size", "mtime_ns", "ctime_ns", "ino")) == identity
        ):
            return entry
        return None

    def put(
        self,
        key: str,
        stat: os.stat_result,
        digest: str,
        scans: Mapping[str, tuple[str, ...]],
    ) -> None:
        """Store an entry, unless the file is too recent to trust its timestamps.

        Parameters
        ----------
        key : str
            Workspace-relative POSIX name.
        stat : os.stat_result
            ``stat`` taken before the file was read.
        digest : str
            SHA-256 of the bytes read.
        scans : Mapping
            Tag to tuple of names produced by scans made in the same read.
        """
        entries = self._all()
        if _persistent_identity(stat) is None:
            return
        newest = max(stat.st_mtime_ns, stat.st_ctime_ns)
        if _clock_ns() - newest < CACHE_SETTLE_NS:
            if entries.pop(key, None) is not None:
                self._flush()
            return
        entries[key] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "ino": stat.st_ino,
            "sha256": digest,
            "scans": {tag: list(names) for tag, names in scans.items()},
        }
        self._flush()

    def _flush(self) -> None:
        """Replace the file atomically; a concurrent writer may win, a reader never sees half."""
        text = json.dumps({"format": self.FORMAT, "entries": self._all()}, sort_keys=True)
        temp = ""
        try:
            handle, temp = tempfile.mkstemp(
                dir=self.path.parent, prefix=".input-digests-", suffix=".tmp"
            )
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            os.replace(temp, self.path)
        except OSError:
            if temp:
                try:
                    os.unlink(temp)
                except OSError:
                    pass


@dataclass
class _Snapshot:
    """What one plan knows of one file: its identity on disk, digest, and cached derivations."""

    path: Path
    size: int
    mtime_ns: int
    digest: str
    data: bytes | None = None
    derived: dict[str, object] = field(default_factory=dict)
    ctime_ns: int = 0
    stat: os.stat_result | None = None
    unread: bool = False


class InputSession:
    """The reads of one command, shared by every point it touches.

    A command that loads many points builds one session and passes it to ``load_point``, so a
    large input is read and hashed once for the command rather than once per point. A file
    whose size or timestamps differ when it is touched again is read again, never trusted.
    Small inputs are always read again, as they are without a session.

    Parameters
    ----------
    root : str or Path
        Workspace root; it is resolved, and the digest cache is kept there.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.snapshots: dict[str, _Snapshot] = {}
        self.cache = DigestCache(self.root)
        self.warned: set[str] = set()

    def files(self) -> InputFiles:
        """Return a fresh tracking view over the shared reads."""
        files = InputFiles(self.root, self.snapshots, cache=self.cache, refresh=True)
        files._warned = self.warned
        return files


class InputFiles:
    """Read byte snapshots once, tracking workspace-relative SHA-256 keys.

    Every file is read and hashed at most once per plan: the snapshot is cached by resolved path,
    size and modification time in nanoseconds, and shared by the row-level objects of one plan. A
    file up to ``LARGE_INPUT_BYTES`` is held whole; a larger one is streamed through its digest
    (and through a scanner or ``derive`` callback when an adapter needs something from it) and is
    held only when an adapter asks for its bytes. A file that changes after it was snapshotted
    is refused rather than recorded with a digest that does not match what was parsed.

    A larger file is also remembered between commands by the workspace's ``DigestCache`` when its
    size, modification time and change time all match. With ``refresh`` (an ``InputSession``),
    a changed file is read again instead of refused, so the caller compares the new digest.
    """

    def __init__(
        self,
        root: Path,
        snapshots: dict[str, _Snapshot] | None = None,
        *,
        cache: DigestCache | None = None,
        refresh: bool = False,
    ) -> None:
        self.root = root
        self._snapshots: dict[str, _Snapshot] = {} if snapshots is None else snapshots
        self._tracked: set[str] = set()
        self._cache = DigestCache(root) if cache is None else cache
        self._refresh = refresh
        self._reads: bool | None = None
        self._warned: set[str] = set()

    def reads_large(self) -> bool:
        """Return whether this reader opens inputs above ``LARGE_INPUT_BYTES`` (decided once)."""
        if self._reads is None or _REHASH.get():
            self._reads = reads_large_inputs(self.root)
        return self._reads

    def warn(self, key: str, message: str) -> None:
        """Warn once per command (or per reader without a command) under ``key``."""
        warn_once(key, message, self._warned)

    def unread(self, name: str, tag: str | None = None) -> bool:
        """Return whether ``name`` is a large input this command does not open.

        True when the large-input check is off and the file is larger than
        ``LARGE_INPUT_BYTES``, unless the result of the scan ``tag`` is already known from the
        command's reads or the digest cache. A caller that needs the contents then skips its
        check with a warning (see ``skip_content``).

        Parameters
        ----------
        name : str
            Workspace-relative input path.
        tag : str, optional
            Scan whose cached result would make reading unnecessary.
        """
        key, path = self._locate(name)
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size <= LARGE_INPUT_BYTES or self.reads_large():
            return False
        snapshot = self._current(name, key, path)
        if snapshot is None or snapshot.unread:
            cached = self._cached(key, path)
            if cached is not None:
                if snapshot is not None:
                    cached.derived.update(snapshot.derived)
                self._snapshots[key] = snapshot = cached
        return not (snapshot is not None and tag is not None and tag in snapshot.derived)

    def skip_content(self, name: str, what: str) -> None:
        """Track an unread large input and warn once that ``what`` was not checked."""
        self.track(name)
        key, _ = self._locate(name)
        self.warn(f"{key}|{what}", skipped_content(name, what))

    def _locate(self, name: str) -> tuple[str, Path]:
        path = workspace_path(self.root, name)
        return path.relative_to(self.root).as_posix(), path

    def _current(self, name: str, key: str, path: Path) -> _Snapshot | None:
        """Return the cached snapshot after checking the file still is the one it describes.

        With ``refresh``, a small file is always read again and a changed large file is
        dropped so it is read again; otherwise a changed file is refused.
        """
        snapshot = self._snapshots.get(key)
        if snapshot is None:
            return None
        try:
            stat = path.stat()
        except OSError as error:
            raise PlanError(f"{name}: cannot read: {error}") from error
        if self._refresh and snapshot.size <= LARGE_INPUT_BYTES:
            del self._snapshots[key]
            return None
        if _identity(stat) is None:
            del self._snapshots[key]
            return None
        if _identity(stat) != _identity(snapshot.stat):
            if self._refresh:
                del self._snapshots[key]
                return None
            raise PlanError(f"{name}: input changed while planning; plan again")
        return snapshot

    def _cached(self, key: str, path: Path) -> _Snapshot | None:
        """Return a snapshot built from the persistent cache for a large file, or None."""
        try:
            stat = path.stat()
        except OSError:
            return None
        if stat.st_size <= LARGE_INPUT_BYTES:
            return None
        entry = self._cache.get(key, stat)
        if entry is None:
            return None
        snapshot = _Snapshot(
            path,
            stat.st_size,
            stat.st_mtime_ns,
            str(entry["sha256"]),
            ctime_ns=stat.st_ctime_ns,
            stat=stat,
        )
        scans = cast(Mapping[str, list[str]], entry["scans"])
        for tag, names in scans.items():
            snapshot.derived[tag] = tuple(names)
        return snapshot

    def _remember(self, key: str, snapshot: _Snapshot) -> None:
        """Store a large file's digest and its name-list scans in the persistent cache."""
        if snapshot.stat is None or snapshot.size <= LARGE_INPUT_BYTES or snapshot.unread:
            return
        scans = {
            tag: value
            for tag, value in snapshot.derived.items()
            if isinstance(value, tuple) and all(isinstance(name, str) for name in value)
        }
        self._cache.put(key, snapshot.stat, snapshot.digest, scans)

    def _load(
        self,
        name: str,
        key: str,
        path: Path,
        *,
        hold: bool,
        scanner: ByteScanner | None = None,
        expect: str | None = None,
    ) -> _Snapshot:
        """Read a file once, hashing it (and feeding ``scanner``) in the same pass."""
        try:
            before = path.stat()
            state = _hasher()
            data: bytes | None = None
            count = 0
            with _open_input(path) as stream:
                if hold or before.st_size <= LARGE_INPUT_BYTES:
                    data = stream.read()
                    count = len(data)
                    state.update(data)
                    if scanner is not None:
                        scanner.feed(data)
                else:
                    for chunk in _stream_blocks(stream):
                        count += len(chunk)
                        state.update(chunk)
                        if scanner is not None:
                            scanner.feed(chunk)
            after = path.stat()
        except OSError as error:
            raise PlanError(f"{name}: cannot read: {error}") from error
        digest = state.hexdigest()
        unchanged = (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) and (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
        if not unchanged or count != before.st_size or (expect is not None and digest != expect):
            raise PlanError(f"{name}: input changed while planning; plan again")
        return _Snapshot(
            path,
            before.st_size,
            before.st_mtime_ns,
            digest,
            data,
            ctime_ns=before.st_ctime_ns,
            stat=before,
        )

    def _unread_snapshot(self, path: Path) -> _Snapshot | None:
        """Return a snapshot of a large file from its identity alone, never opening it."""
        try:
            stat = path.stat()
        except OSError:
            return None
        if stat.st_size <= LARGE_INPUT_BYTES:
            return None
        return _Snapshot(
            path,
            stat.st_size,
            stat.st_mtime_ns,
            identity_value(stat),
            ctime_ns=stat.st_ctime_ns,
            stat=stat,
            unread=True,
        )

    def _snapshot(self, name: str, *, hold: bool) -> tuple[str, _Snapshot]:
        key, path = self._locate(name)
        snapshot = self._current(name, key, path)
        derived: dict[str, object] = {}
        if snapshot is not None and snapshot.unread and (hold or self.reads_large()):
            derived, snapshot = snapshot.derived, None  # the bytes are wanted now: read them
        if snapshot is None:
            if not hold:
                snapshot = self._cached(key, path)
            if snapshot is None and not hold and not self.reads_large():
                snapshot = self._unread_snapshot(path)
            if snapshot is None:
                snapshot = self._load(name, key, path, hold=hold)
                self._remember(key, snapshot)
            snapshot.derived = {**derived, **snapshot.derived}
            self._snapshots[key] = snapshot
        elif hold and snapshot.data is None:
            loaded = self._load(name, key, path, hold=True, expect=snapshot.digest)
            snapshot.data = loaded.data
        self._tracked.add(key)
        return key, snapshot

    def acknowledge_link(self, name: str, before: os.stat_result, after: os.stat_result) -> None:
        """Refresh metadata only for an observed hard-link count increment.

        Parameters
        ----------
        name : str
            Input linked by the library.
        before, after : os.stat_result
            Followed source stats immediately around the library's hard-link call.
        """
        key, _ = self._locate(name)
        snapshot = self._snapshots.get(key)
        identity = _identity(before)
        current = _identity(after)
        if (
            snapshot is None
            or identity is None
            or current is None
            or _identity(snapshot.stat) != identity
            or identity[:2] + identity[3:] != current[:2] + current[3:]
            or getattr(before, "st_nlink", 0) < 1
            or getattr(after, "st_nlink", 0) != before.st_nlink + 1
        ):
            return
        snapshot.stat = after
        snapshot.ctime_ns = after.st_ctime_ns
        self._remember(key, snapshot)

    def digest_output(self, path: Path, candidates: Iterable[str]) -> str:
        """Return the SHA-256 of a point output, streaming it and never holding it whole.

        An output that is the very file of a planned input (the mesh link of a point folder,
        a symbolic or hard link to it) is the same bytes, so it takes that input's digest from
        this reader (the command's own read, or the cache) instead of reading the file again.

        Parameters
        ----------
        path : Path
            The output file.
        candidates : iterable of str
            Workspace-relative names of the inputs the output may be a link to.
        """
        if path.stat().st_size > LARGE_INPUT_BYTES:
            for name in candidates:
                if is_matrix_file(name):
                    continue
                try:
                    same = os.path.samefile(path, workspace_path(self.root, name))
                except OSError:
                    continue
                if same:
                    return self.digest(name)
        state = _hasher()
        with _open_input(path) as stream:
            for chunk in _stream_blocks(stream):
                state.update(chunk)
        return state.hexdigest()

    def digest_via(self, name: str, path: Path) -> str:
        """Return the SHA-256 of the file at ``path``, a link to the workspace input ``name``.

        A large file whose size and timestamps (followed through the link) match the command's
        own read of ``name``, or the digest cache, is not read again; any other file is streamed.

        Parameters
        ----------
        name : str
            Workspace-relative input path the link points at.
        path : Path
            The link, for example in a point folder.
        """
        key, source = self._locate(name)
        stat = path.stat()
        try:
            same = stat.st_size > LARGE_INPUT_BYTES and os.path.samefile(source, path)
        except OSError:
            same = False
        if same:
            snapshot = self._snapshots.get(key)
            if (
                snapshot is not None
                and _identity(stat) is not None
                and (_identity(snapshot.stat) == _identity(stat))
            ):
                return snapshot.digest
            entry = self._cache.get(key, stat)
            if entry is not None:
                return str(entry["sha256"])
        if stat.st_size > LARGE_INPUT_BYTES and not self.reads_large():
            return identity_value(stat)
        state = _hasher()
        with _open_input(path) as stream:
            for chunk in _stream_blocks(stream):
                state.update(chunk)
        return state.hexdigest()

    def digest(self, name: str) -> str:
        """Return the SHA-256 of an input, streaming a large file and trusting the cache.

        Parameters
        ----------
        name : str
            Workspace-relative input path.

        Raises
        ------
        OSError
            The file cannot be read (the original error, not a plan refusal).
        """
        try:
            return self._snapshot(name, hold=False)[1].digest
        except PlanError as error:
            if isinstance(error.__cause__, OSError):
                raise error.__cause__ from None
            raise

    def agrees(self, name: str, planned: str, *, via: Path | None = None) -> bool:
        """Return whether an input still is the file its planned value describes.

        A planned file identity (``identity:size=...;mtime_ns=...``) is compared with the size
        and modification time of the file now, without reading it; a command that reads large
        inputs says once that it could only check that much. A planned SHA-256 is compared with
        the digest now, unless this command does not read the large file and the digest cache
        has no entry for it: the file is then accepted unverified, with one warning.

        Parameters
        ----------
        name : str
            Workspace-relative input path.
        planned : str
            The value recorded when the point was planned.
        via : Path, optional
            A link to the input (for example in a point folder), checked instead of the input.
        """
        if not recorded_value_valid(planned):
            return False
        key, path = self._locate(name)
        if is_identity_value(planned):
            target = path if via is None else via
            try:
                stat = target.stat()
            except OSError as error:
                raise PlanError(f"{name}: cannot read: {error}") from error
            if identity_value(stat) != planned:
                return False
            if via is None:
                known = self._snapshots.get(key)
                if known is None or _identity(known.stat) != _identity(stat):
                    self._snapshots[key] = _Snapshot(
                        path,
                        stat.st_size,
                        stat.st_mtime_ns,
                        planned,
                        ctime_ns=stat.st_ctime_ns,
                        stat=stat,
                        unread=True,
                    )
                self._tracked.add(key)
            if self.reads_large():
                self.warn(
                    f"{key}|identity",
                    f"{name}: checked by size and modification time only; it was planned "
                    "with the large-input check off, so no SHA-256 was taken (plan again "
                    "with --rehash-inputs to hash it)",
                )
            return True
        current = self.digest(name) if via is None else self.digest_via(name, via)
        if is_identity_value(current):
            self.warn(
                f"{key}|unverified",
                f"{name}: not verified against its planned SHA-256; the large-input check is "
                "off, so the file is not read (use --rehash-inputs without "
                "--no-read-large-inputs to verify it)",
            )
            return True
        return current == planned

    def output_agrees(self, path: Path, candidates: Iterable[str], expected: str) -> bool:
        """Return whether a recorded output still matches its recorded value.

        Parameters
        ----------
        path : Path
            The output file.
        candidates : iterable of str
            Workspace-relative names of the inputs the output may be a link to.
        expected : str
            The recorded SHA-256 or file identity.
        """
        if not recorded_value_valid(expected):
            return False
        if is_identity_value(expected):
            return identity_value(path.stat()) == expected
        current = self.digest_output(path, candidates)
        if is_identity_value(current):
            self.warn(
                f"{path}|unverified",
                f"{path.name}: output not verified against its recorded SHA-256; the "
                "large-input check is off, so the file is not read (use --rehash-inputs "
                "without --no-read-large-inputs to verify it)",
            )
            return True
        return current == expected

    def read_bytes(self, name: str) -> bytes:
        """Read and memoize an input, refusing a path outside this workspace.

        Parameters
        ----------
        name : str
            Workspace-relative input path, using forward slashes.

        Returns
        -------
        bytes
            Exact first-read bytes, also used for the recorded digest.
        """
        data = self._snapshot(name, hold=True)[1].data
        assert data is not None
        return data

    def track(self, name: str) -> None:
        """Record an input in the provenance without handing its bytes to the caller.

        A large file is streamed through its digest and not held.
        """
        self._snapshot(name, hold=False)

    def scan(self, name: str, tag: str, factory: Callable[[], ByteScanner]) -> object:
        """Return what a scanner finds in an input, reading and hashing the file only once.

        The scan runs in the same pass that digests the file; the result is cached for the plan
        under ``tag``, so a file shared by many rows and points is scanned once.

        Parameters
        ----------
        name : str
            Workspace-relative input path.
        tag : str
            Identity of the scan, for example the kind of marker listing.
        factory : callable
            Builds a fresh scanner.
        """
        key, path = self._locate(name)
        snapshot = self._current(name, key, path)
        if snapshot is None:
            snapshot = self._cached(key, path)
            if snapshot is not None:
                self._snapshots[key] = snapshot
        if snapshot is not None and tag in snapshot.derived:
            validate = getattr(factory(), "valid_result", None)
            if validate is not None and not validate(snapshot.derived[tag]):
                del snapshot.derived[tag]
        if snapshot is None:
            scanner = factory()
            snapshot = self._load(name, key, path, hold=False, scanner=scanner)
            snapshot.derived[tag] = scanner.result()
            self._snapshots[key] = snapshot
            self._remember(key, snapshot)
        elif tag not in snapshot.derived:
            scanner = factory()
            if snapshot.data is not None:
                scanner.feed(snapshot.data)
            else:
                self._load(name, key, path, hold=False, scanner=scanner, expect=snapshot.digest)
            snapshot.derived[tag] = scanner.result()
            self._remember(key, snapshot)
        self._tracked.add(key)
        return snapshot.derived[tag]

    def derive(self, name: str, tag: str, compute: Callable[[Path], object]) -> object:
        """Return ``compute(path)`` for a tracked input, computed once per plan.

        The file is digested first (streamed when large); ``compute`` may open it by path, for a
        format that is read by seeking rather than in full. The file is checked unchanged after.
        """
        key, snapshot = self._snapshot(name, hold=False)
        if tag not in snapshot.derived:
            snapshot.derived[tag] = compute(snapshot.path)
            self._current(name, key, snapshot.path)
            self._remember(key, snapshot)
        return snapshot.derived[tag]

    def read_toml(self, name: str) -> Mapping[str, object]:
        """Read TOML from the same byte snapshot used for its digest.

        Parameters
        ----------
        name : str
            Workspace-relative TOML input path.

        Returns
        -------
        Mapping
            Parsed TOML fields. Solver-specific schema validation belongs to the hook.
        """
        data = self.read_bytes(name)
        try:
            return tomllib.loads(data.decode("utf-8"))
        except (UnicodeError, tomllib.TOMLDecodeError) as error:
            raise PlanError(f"{name}: invalid TOML: {error}") from error

    def digests(self) -> Mapping[str, str]:
        """Return a sorted, immutable digest snapshot of every file read."""
        return MappingProxyType({key: self._snapshots[key].digest for key in sorted(self._tracked)})


def apply_variables(
    setup: Mapping[str, object], variables: Mapping[str, object]
) -> Mapping[str, object]:
    """Return the setup with a row's VAR_NAMES_VALUES overrides applied.

    Parameters
    ----------
    setup : Mapping
        Parsed setup file.
    variables : Mapping
        Row overrides; a dotted key such as ``solver.threads`` sets a key of a table.

    Returns
    -------
    Mapping
        A new mapping; the parsed setup is not modified.
    """
    result: dict[str, object] = dict(setup)
    for key, value in variables.items():
        if key.casefold() in {"cluster", "hpc"} or key.startswith(HPC_OVERRIDE_PREFIX):
            continue
        *tables, name = key.split(".")
        target = result
        for part in tables:
            inner = target.get(part, {})
            if not isinstance(inner, Mapping):
                raise ValueError(f"VAR_NAMES_VALUES {key}: setup key {part} is not a table")
            copied = dict(inner)
            target[part] = copied
            target = copied
        target[name] = value
    return result


def resolve_geometry(root: Path, cell: str | None, *, folders: bool = False) -> str | None:
    """Resolve a MESH cell to a workspace-relative file, without reading it.

    Parameters
    ----------
    root : Path
        Resolved workspace root.
    cell : str or None
        A file name in inputs/mesh/ (legacy inputs/geometry/, inputs/geometries/ or
        inputs/meshes/ still read), a stem naming exactly one file there, or a
        workspace-relative path containing ``/``.
    folders : bool, optional
        Also accept a folder of inputs/mesh/ named exactly, or a workspace-relative
        folder: the adapter's validator declares ``geometry_folders = True`` when its
        mesh is a folder of files (a CFD++ native grid). Other adapters keep files only.

    Returns
    -------
    str or None
        The workspace-relative POSIX path, or None when the row states none.
    """
    if cell is None:
        return None
    if "/" in cell:
        path = workspace_path(root, cell)
        found = [cell] if path.is_file() or (folders and path.is_dir()) else []
    elif folders and (root / MESH_FOLDER / cell).is_dir():
        found = [f"{MESH_FOLDER}/{cell}"]
    else:
        found = mesh_candidates(root, cell)
    if len(found) != 1:
        raise ValueError(
            f"MESH {cell!r} names "
            + (f"several files: {', '.join(found)}" if found else "no file")
            + "; name one file of inputs/mesh/ (with its extension when stems repeat)"
        )
    return found[0]


NUMERICS_HOME = "inputs/numerics"
#: Top-level keys a numerics file may not state: execution and geometry, not numerics.
NUMERICS_REFUSED = ("solver", "mesh", "MESH_FILENAME")


def numerics_refusal(name: str, numerics: Mapping[str, object]) -> str | None:
    """Return the refusal text for a numerics file that states execution or the mesh, or None.

    Parameters
    ----------
    name : str
        Workspace-relative path of the numerics file, used in the message.
    numerics : Mapping
        Its parsed top-level keys.
    """
    for key in numerics:
        if key in NUMERICS_REFUSED:
            what = "the [solver] table" if key == "solver" else f"the mesh key {key}"
            kind = "execution" if key == "solver" else "geometry"
            return (
                f"{name}: states {what}, which is {kind}, not numerics; state it in the setup file"
            )
    return None


def _leaves(table: Mapping[str, object], prefix: str = "") -> set[str]:
    found: set[str] = set()
    for key, value in table.items():
        if isinstance(value, Mapping):
            found |= _leaves(value, f"{prefix}{key}.")
        else:
            found.add(f"{prefix}{key}")
    return found


def numerics_conflicts(
    numerics: Mapping[str, object], setup: Mapping[str, object]
) -> tuple[str, ...]:
    """Return the sorted keys (dotted for tables) that both files state; the setup wins."""
    return tuple(sorted(_leaves(numerics) & _leaves(setup)))


def resolve_setup(
    defaults: Mapping[str, object],
    numerics: Mapping[str, object],
    setup: Mapping[str, object],
    variables: Mapping[str, object],
    *,
    omitted: Collection[str] = (),
) -> Mapping[str, object]:
    """Merge a point's setup: defaults, then the numerics file, then the setup, then the row.

    Parameters
    ----------
    defaults : Mapping
        The init.toml defaults table.
    numerics : Mapping
        The parsed numerics file, empty when the row names none.
    setup : Mapping
        The parsed setup file; it wins over the numerics file for a key both state.
    variables : Mapping
        The row's VAR_NAMES_VALUES overrides, applied last.
    omitted : collection of str, optional
        Default keys that explicit row columns replace.

    Returns
    -------
    Mapping
        The setup the adapter reads. Without a numerics file it equals the earlier merge.
    """
    base = setup_defaults(defaults, numerics, omitted=omitted) if numerics else defaults
    merged = setup_defaults(base, setup, omitted=() if numerics else omitted)
    return apply_variables(merged, variables)


def setup_origins(
    defaults: Mapping[str, object],
    numerics: Mapping[str, object],
    setup: Mapping[str, object],
    variables: Mapping[str, object],
    *,
    omitted: Collection[str] = (),
) -> Mapping[str, str]:
    """Return, for each top-level key of the resolved setup, where its value came from.

    The origin is ``defaults``, ``numerics``, ``setup`` or ``row``; the last writer wins.
    """
    origins: dict[str, str] = {key: "defaults" for key in defaults if key not in omitted}
    for origin, table in (("numerics", numerics), ("setup", setup)):
        for key in table:
            origins[key] = origin
    for key in variables:
        if key.casefold() not in {"cluster", "hpc"} and not key.startswith(HPC_OVERRIDE_PREFIX):
            origins[key.split(".")[0]] = "row"
    return origins


class RowInputs(InputFiles):
    """One row's s/r/p tables plus tracked access to further workspace inputs.

    ``setup`` carries the row's VAR_NAMES_VALUES overrides. ``geometry`` is the
    resolved MESH file, or None; an adapter reads it through read_bytes.
    """

    def __init__(
        self,
        root: Path,
        row: MatrixRow,
        common: InputFiles,
        default_overrides: Mapping[str, Collection[str]] | None = None,
        *,
        accepts_numerics: bool = True,
        geometry_folders: bool = False,
    ) -> None:
        super().__init__(root, common._snapshots, cache=common._cache)
        self._warned = common._warned
        self._tracked.update(common._tracked)
        defaults = cast(Mapping[str, object], self.read_toml("init.toml").get("defaults", {}))
        overridden = {
            name
            for key, names in (default_overrides or {}).items()
            if key in row.condition or (key == "GEOMETRY" and row.geometry is not None)
            for name in names
        }
        self.defaults_omitted = tuple(sorted(set(defaults) & overridden))
        raw_setup = self._named("SETUP", f"inputs/setups/{row.setup_id}.toml", row.setup_id)
        raw_numerics: Mapping[str, object] = {}
        if row.numerics_id and accepts_numerics:
            where = f"inputs/numerics/{row.numerics_id}.toml"
            raw_numerics = self._named("NUMERICS", where, row.numerics_id)
            refusal = numerics_refusal(where, raw_numerics)
            if refusal:
                raise PlanError(refusal)
            for key in numerics_conflicts(raw_numerics, raw_setup):
                common.warn(
                    f"numerics|{where}|{row.setup_id}|{key}",
                    f"{key!r} is stated in both {where} and inputs/setups/{row.setup_id}.toml; "
                    "the setup file wins",
                )
        self.numerics_id = row.numerics_id if accepts_numerics else ""
        self.numerics = raw_numerics
        self.origins = setup_origins(
            defaults, raw_numerics, raw_setup, row.variables, omitted=self.defaults_omitted
        )
        self.setup = resolve_setup_paths(
            root,
            resolve_setup(
                defaults, raw_numerics, raw_setup, row.variables, omitted=self.defaults_omitted
            ),
        )
        self.reference = self._named(
            "REF", f"inputs/references/{row.reference_id}.toml", row.reference_id
        )
        self.post = self._named("PPROC", f"inputs/pproc/{row.post_id}.toml", row.post_id)
        self.geometry = resolve_geometry(root, row.geometry, folders=geometry_folders)
        self._row = row

    def _named(self, column: str, name: str, ident: str) -> Mapping[str, object]:
        """Read the TOML a row's SET, REF or PPROC cell names, refusing a missing file by cell."""
        if not workspace_path(self.root, name).is_file():
            raise PlanError(f"{column} {ident} names {name}, which does not exist")
        return self.read_toml(name)

    def file_names(self) -> Mapping[str, str]:
        """Return the words an adapter refusal uses for the row's files and the files they are."""
        row = self._row
        numerics = (
            {"numerics file": f"inputs/numerics/{row.numerics_id}.toml"} if row.numerics_id else {}
        )
        return {
            **numerics,
            "setup file": f"inputs/setups/{row.setup_id}.toml",
            "REF file": f"inputs/references/{row.reference_id}.toml",
            "PPROC file": f"inputs/pproc/{row.post_id}.toml",
        }

    def warn_unread(
        self, solver: str, *, reference: Collection[str] = (), post: Collection[str] = ()
    ) -> None:
        """Warn once per entry that the REF or PPROC file holds and ``solver`` never reads.

        An entry that does not apply is skipped, not refused: planning goes on
        without it, and the warning names the row, the file, the entry and why.

        Parameters
        ----------
        solver : str
            Solver label used in the message.
        reference : collection of str, optional
            REF-file keys the solver reads; any other key is unread.
        post : collection of str, optional
            PPROC-file keys the solver reads; any other key is unread.
        """
        row = self._row
        for kind, table, read, ident in (
            ("references", self.reference, reference, row.reference_id),
            ("pproc", self.post, post, row.post_id),
        ):
            for key in table:
                if key not in read:
                    where = f"inputs/{kind}/{ident}.toml"
                    warn_or_refuse(
                        f"POL {row.pol}: {key!r} of {where} is skipped; {solver} does not read "
                        f"it ({'reads only ' + ', '.join(sorted(read)) if read else 'reads no key'}"
                        f" of this file)",
                        UserWarning,
                        stacklevel=2,
                    )


@dataclass(frozen=True)
class RowRequirements:
    """Solver hook response: actions and workspace-relative output templates.

    Templates may use {pol} and {point}. The point token is a stable DP name.
    Example: sims/sim_{pol}/datapoints/{point}/history.csv. No file is created.
    """

    actions: tuple[str, ...]
    outputs: tuple[str, ...]
    #: What the point's record states about how it was set up (its run type, the identity of
    #: the run type's declaration set, the numerics id, the turbulence model and its origin).
    provenance: Mapping[str, object] = field(default_factory=dict)
    #: The origin of the solver keys the run type carried, by key, for ``plan --verbose``.
    origins: Mapping[str, str] = field(default_factory=dict)


class RowValidator(Protocol):
    """The solver-independent callable interface supplied by a composing layer."""

    def __call__(self, row: MatrixRow, inputs: RowInputs) -> RowRequirements:
        """Validate one row, read tracked inputs and describe the proposed work.

        Parameters
        ----------
        row : MatrixRow
            Matrix data and its input ids.
        inputs : RowInputs
            Parsed s/r/p and tracked readers for further inputs.

        Returns
        -------
        RowRequirements
            Actions and output paths. Raise ValueError to refuse the row.
        """
        ...


@dataclass(frozen=True)
class Point:
    """One stable point with all inputs, proposed actions and claimed output paths."""

    point_id: str
    pol: str
    setup_id: str
    reference_id: str
    post_id: str
    condition: Mapping[str, float]
    walltime: str | None
    directory: str
    actions: tuple[str, ...]
    outputs: tuple[str, ...]
    inputs_sha256: Mapping[str, str]
    geometry: str | None = None
    variables: Mapping[str, object] = field(default_factory=_empty)
    ncpus: int | None = None
    solver_build: str | None = None
    description: str = ""
    aircraft: str = ""
    configuration: str = ""
    hidden: bool = False
    restart_from: str | None = None
    restart_directory: str | None = None
    restart_sha256: Mapping[str, str] = field(default_factory=dict)
    #: The source of RESTART runs earlier in the same batch job: the solution is staged by the
    #: job script after the source ends, so no digest is recorded at planning.
    restart_in_job: bool = False
    #: The row's RESTART cell is ``MRF``: a steady frozen-rotor stage runs before the point.
    precursor: bool = False
    defaults_omitted: tuple[str, ...] = ()
    numerics_id: str = ""
    #: The run type the row selects in its WORKFLOW cell; a steady record omits it.
    run_type: str = "steady"
    #: Where each top-level setup key came from (planning only; never recorded).
    origins: Mapping[str, str] = field(default_factory=dict, compare=False, repr=False)
    #: How the adapter set the point up, as the record states it; empty when it says nothing.
    provenance: Mapping[str, object] = field(default_factory=_empty)
    #: A time-accurate point continued in place (``run --continue``): the solver iteration it
    #: restarts at, its segments and where the earlier run's evidence was kept; empty otherwise.
    continuation: Mapping[str, object] = field(default_factory=_empty)


#: Point fields beyond the original record, with the value an older record implies.
POINT_EXTRAS: Mapping[str, object] = MappingProxyType(
    {
        "geometry": None,
        "variables": {},
        "ncpus": None,
        "solver_build": None,
        "description": "",
        "aircraft": "",
        "configuration": "",
        "hidden": False,
        "restart_from": None,
        "restart_directory": None,
        "restart_sha256": {},
        "restart_in_job": False,
        "precursor": False,
        "defaults_omitted": (),
        "numerics_id": "",
        "run_type": "steady",
        "provenance": {},
        "continuation": {},
    }
)


@dataclass(frozen=True)
class RunPlan:
    """An in-memory, solver-named plan; creating it writes no workspace files."""

    root: Path
    matrix: str
    solver: str
    points: tuple[Point, ...]


def _points(root: Path, row: MatrixRow, inputs: RowInputs, needs: RowRequirements) -> list[Point]:
    if not needs.actions or any(not isinstance(a, str) or not a.strip() for a in needs.actions):
        raise PlanError("solver validation must describe at least one non-empty action")
    if not needs.outputs:
        raise PlanError("solver validation must declare output paths")
    points: list[Point] = []
    names: dict[str, float | None] = {}
    for value in row.sweep_values if row.sweep_key else (None,):
        condition = dict(row.condition)
        if row.sweep_key is not None and value is not None:
            condition[row.sweep_key] = value
        try:
            name = point_name(row.condition_order, condition)
        except PointNameError as error:
            raise PlanError(f"POL {row.pol}: {error}") from error
        if name in names:
            raise PlanError(
                f"POL {row.pol}: sweep values {names[name]!r} and {value!r} of "
                f"{row.sweep_key} round to the same point name {name}; a datapoint folder "
                "is named from its flight condition, so give the values distinct names "
                "(for example further apart)"
            )
        names[name] = value
        tag = datapoint_dir_name(name)
        directory = f"sims/sim_{row.pol}/datapoints/{tag}"
        outputs: list[str] = []
        for template in needs.outputs:
            try:
                output = template.format(pol=row.pol, point=tag)
            except (KeyError, IndexError, ValueError) as error:
                raise PlanError(f"invalid output template {template!r}: {error}") from error
            # A staged mesh output may already exist as a link to inputs/geometry: judge the
            # output by its folder and name, never by what the link points at.
            checked_relative(output)
            parent, _, leaf = output.rpartition("/")
            folder = workspace_path(root, parent) if parent else root
            normalized = f"{folder.relative_to(root).as_posix()}/{leaf}" if parent else leaf
            path = folder / leaf
            if not parent or not normalized.startswith(("sims/", "post/")):
                raise PlanError(f"{output}: output must be under sims/ or post/")
            if path.is_dir():
                raise PlanError(f"{output}: output is a directory")
            outputs.append(normalized)
        points.append(
            Point(
                point_id=f"{row.pol}/{tag}",
                pol=row.pol,
                setup_id=row.setup_id,
                reference_id=row.reference_id,
                post_id=row.post_id,
                condition=MappingProxyType(condition),
                walltime=row.walltime,
                directory=directory,
                actions=tuple(needs.actions),
                outputs=tuple(outputs),
                inputs_sha256=inputs.digests(),
                geometry=inputs.geometry,
                variables=row.variables,
                ncpus=row.ncpus,
                solver_build=row.solver_build,
                description=row.description,
                aircraft=row.aircraft,
                configuration=row.configuration,
                hidden=row.hidden,
                restart_from=row.restart_from,
                precursor=row.precursor,
                defaults_omitted=inputs.defaults_omitted,
                numerics_id=inputs.numerics_id,
                run_type=row.workflow,
                origins={**inputs.origins, **needs.origins},
                provenance=MappingProxyType(dict(needs.provenance)),
            )
        )
    return points


def check_claims(points: tuple[Point, ...]) -> None:
    """Refuse duplicate point ids, overlapping directories and output path claims.

    Parameters
    ----------
    points : tuple of Point
        All points sharing a workspace, including any existing run records.
    """
    ids: set[str] = set()
    claims: list[tuple[str, str, bool]] = []
    for point in points:
        if point.point_id.casefold() in ids:
            raise PlanError(f"point {point.point_id!r} is recorded or planned twice")
        ids.add(point.point_id.casefold())
        for name, is_directory in [(point.directory, True), *((p, False) for p in point.outputs)]:
            folded = name.casefold().rstrip("/")
            for previous, owner, previous_dir in claims:
                equal = folded == previous
                overlaps = folded.startswith(previous + "/") or previous.startswith(folded + "/")
                own_file_in_directory = owner == point.point_id and (
                    (previous_dir and folded.startswith(previous + "/"))
                    or (is_directory and previous.startswith(folded + "/"))
                )
                if (equal or overlaps) and not own_file_in_directory:
                    raise PlanError(
                        f"{name}: point {point.point_id!r} and point {owner!r} "
                        "would write the same path; choose distinct output paths"
                    )
            claims.append((folded, point.point_id, is_directory))


MATRIX_HOME = "inputs/matrices"


def matrix_homes(name: str) -> tuple[str, str]:
    """Return the two workspace-relative paths a matrix named ``name`` may live at.

    The first is the canonical ``inputs/matrices/<stem>.fs``, the second the
    workspace root. Raises PlanError for a name that is not a plain matrix stem.
    """
    stem = name.removesuffix(".fs")
    if re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]*", stem) is None:
        raise PlanError(f"{name!r}: name a .fs matrix in inputs/matrices/ or the workspace root")
    return f"{MATRIX_HOME}/{stem}.fs", f"{stem}.fs"


def resolve_matrix(root: str | Path, name: str) -> str:
    """Return the workspace-relative path of the one matrix file ``name`` names.

    Parameters
    ----------
    root : str or Path
        Workspace root.
    name : str
        Matrix filename or stem.

    Raises
    ------
    PlanError
        The name is malformed, no file exists in either home, or both do (ambiguous).
    """
    target = Path(root).resolve()
    homes = matrix_homes(name)
    found = [home for home in homes if (target / home).is_file()]
    if len(found) == 2:
        raise PlanError(
            f"matrix {name!r} exists in both {homes[0]} and {homes[1]}; keep one, so a run "
            "never reads the wrong file"
        )
    if not found:
        stem = homes[1].removesuffix(".fs")
        have = sorted(p.stem for p in (target / MATRIX_HOME).glob("*.fs"))
        raise PlanError(
            f"no matrix {stem} in inputs/matrices/ (found: {', '.join(have) or 'none'}) and no "
            f"{homes[1]} in the workspace root"
        )
    return found[0]


def _closed_condition_keys(row: MatrixRow, native: tuple[str, ...]) -> None:
    """Refuse a FLIGHT_CONDITION key outside the grammar, as pyflightstream does.

    The set is closed: the flight-condition and attitude keys of the shared grammar and the
    keys the solver's adapter declares as its own. A mistyped key costs a message rather
    than a campaign solved at a condition nobody asked for.
    """
    known = (*STATE_KEYS, *ATTITUDE_KEYS, *native)
    for key in row.condition:
        if key in known or key.upper() in {k.upper() for k in known}:
            continue
        close = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
        hint = f" (closest known key: {close[0]})" if close else ""
        raise ValueError(
            f"FLIGHT_CONDITION key {key!r}{hint} is not one this package knows; the accepted "
            f"keys are {', '.join(known)}. The set is closed so that a mistyped key costs a "
            "message rather than a campaign solved at a condition nobody asked for."
        )


def _warn_retired_header(common: InputFiles, root: Path, matrix: str, retired: str) -> None:
    """Warn once per matrix file and command that its header still says ``retired``."""
    text = _RETIRED.format(matrix=matrix, old=retired, new=RETIRED_COLUMNS[retired])
    common.warn(f"retired {retired}|{root.as_posix()}|{matrix}", text)


def _post_products(
    common: InputFiles, row: MatrixRow, inputs: RowInputs, hook: object, solver: str
) -> None:
    """Read the p file's product tables; skip, with one warning, those the adapter lacks.

    A malformed table is refused with pyflightstream's message. A skipped table is removed
    from the row's p file as the adapter sees it, so the adapter never warns about it again.
    """
    from cfdbender.core.pproc import DECLARING_TABLES, read_plots, read_post, support_key

    if solver == "su2":
        read_plots(inputs.post, where=f"inputs/pproc/{row.post_id}.toml")
    if not any(table in inputs.post for table in DECLARING_TABLES):
        return
    supported = frozenset(getattr(hook, "post_products", ()))
    where = f"inputs/pproc/{row.post_id}.toml"

    def warn(key: str, message: str) -> None:
        if key not in common._warned:
            common._warned.add(key)
            warn_or_refuse(f"POL {row.pol}: {message}", UserWarning, stacklevel=4)

    read_post(
        inputs.post, supported=supported, where=where, supplier=f"solver {solver!r}", warn=warn
    )
    inputs.post = {
        key: value
        for key, value in inputs.post.items()
        if key not in DECLARING_TABLES or support_key(key) in supported
    }


_RETIRED = (
    "{matrix}: column {old} is now {new}; run the migration script "
    "(python -m cfdbender.migrate_setup_numerics WORKSPACE --apply)"
)


def plan_matrix(
    root: str | Path,
    name: str,
    *,
    validators: Mapping[str, RowValidator],
    accepts_numerics: bool = False,
) -> RunPlan:
    """Plan a named matrix (inputs/matrices/<name>.fs or <name>.fs in the root) read-only.

    Parameters
    ----------
    root : str or Path
        Workspace root containing init.toml.
    name : str
        Matrix filename or stem, without directory components.
    validators : Mapping of str to RowValidator
        Solver name to hook, supplied by the application. Core imports no adapters.
    accepts_numerics : bool, optional
        Whether the solver declares the ``numerics_file`` capability. When it does not, a
        NUMERICS cell is ignored with one warning per matrix.

    Returns
    -------
    RunPlan
        Stable points, actions, path claims and SHA-256 of exact input bytes.

    Raises
    ------
    PlanError
        Missing input, malformed TOML, hook refusal or colliding output claims.
    """
    target = Path(root).resolve()
    common = InputFiles(target)
    try:
        config = _validate(common.read_toml("init.toml"), target / "init.toml", validators.keys())
    except WorkspaceError as error:
        raise PlanError(str(error)) from error
    matrix = resolve_matrix(target, name)
    rows = parse_matrix(
        common.read_bytes(matrix),
        Path(matrix),
        on_retired_header=lambda old: _warn_retired_header(common, target, matrix, old),
    )
    points: list[Point] = []
    if not accepts_numerics and any(row.numerics_id for row in rows):
        warn_or_refuse(
            f"{matrix}: NUMERICS is ignored for solver {config.solver}; "
            "its setup file carries the numerics"
        )
    for row in rows:
        names: Mapping[str, str] = {}
        try:
            hook = validators[config.solver]
            inputs = RowInputs(
                target,
                row,
                common,
                getattr(hook, "default_setup_overrides", {}),
                accepts_numerics=accepts_numerics,
                geometry_folders=bool(getattr(hook, "geometry_folders", False)),
            )
            names = inputs.file_names()
            _closed_condition_keys(row, tuple(getattr(hook, "native_condition_keys", ())))
            for column in ("SYMMETRY", "SYMMETRY_LOADS"):
                if column in row.variables and not getattr(hook, "reads_symmetry", False):
                    raise ValueError(
                        f"{column} {row.variables[column]!r} is not read by solver "
                        f"{config.solver!r}, which takes symmetry from the mesh and the "
                        "setup; write - in the cell"
                    )
            _post_products(common, row, inputs, hook, config.solver)
            needs = hook(row, inputs)
            read = inputs.digests()
            if (
                inputs.geometry is not None
                and inputs.geometry not in read
                # A grid folder is read file by file.
                and not any(name.startswith(inputs.geometry + "/") for name in read)
            ):
                warn_or_refuse(
                    f"POL {row.pol}: MESH {row.geometry!r} is skipped; solver "
                    f"{config.solver!r} does not read the MESH column (name the geometry "
                    "where its setup documents it, and write - in the cell to silence this)",
                    UserWarning,
                    stacklevel=2,
                )
                inputs.geometry = None
            points.extend(_points(target, row, inputs, needs))
        except (ValueError, OSError) as error:
            text = str(error)
            for word, name in names.items():
                text = text.replace(word, name)
            raise PlanError(
                f"{matrix}, data row {row.row_number} (POL {row.pol}): {text}"
            ) from error
    result = tuple(points)
    check_claims(result)
    return RunPlan(target, matrix, config.solver, result)
