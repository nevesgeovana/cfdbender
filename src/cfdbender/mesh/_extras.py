"""The refusals for a library a geometry or mesh command needs and cannot load.

gmsh is a base dependency of cfdbender, so these refusals name an installation that is broken
(a missing shared library on Linux, a removed package) rather than an extra left out.
"""

from __future__ import annotations

from typing import Any

from cfdbender._errors import GeometryError

__all__ = ["KernelUnavailableError", "MissingExtraError"]


class MissingExtraError(GeometryError, ImportError):
    """A code path needs a library that is not installed in this environment.

    Parameters
    ----------
    extra : str
        The name of the library as ``pip`` installs it.
    package : str
        The distribution or module that was actually missing.
    purpose : str
        What the caller was doing, as a sentence subject.
    """

    def __init__(self, *, extra: str, package: str, purpose: str) -> None:
        self.extra = extra
        self.package = package
        self.purpose = purpose
        self.remedy = f'pip install "{extra}"'
        super().__init__(
            f"{purpose} needs {package}, which is not installed in this environment. "
            f"Install it with: {self.remedy}"
        )

    def __reduce__(self) -> Any:
        """Pickle by the keyword fields, since ``__init__`` takes no positional message."""
        return (_rebuild_missing_extra, (self.extra, self.package, self.purpose))


def _rebuild_missing_extra(extra: str, package: str, purpose: str) -> MissingExtraError:
    return MissingExtraError(extra=extra, package=package, purpose=purpose)


class KernelUnavailableError(MissingExtraError):
    """The CAD kernel library gmsh is not installed or cannot load in this environment.

    Raised for an ``ImportError`` and for an ``OSError`` (a shared library gmsh needs failed to
    load). The message names the library that failed, the Linux package line, and that the
    stages that need no kernel (S0 to S10 without their kernel sub-steps) still run.

    Parameters
    ----------
    purpose : str
        What the caller was doing, as a sentence subject.
    reason : str
        The text of the original import error.
    """

    #: The Linux packages the kernel library links against.
    LINUX_PACKAGES = "libglu1-mesa libxcursor1 libxft2 libxinerama1"

    def __init__(self, *, purpose: str, reason: str) -> None:
        self.extra = "gmsh"
        self.package = "gmsh"
        self.purpose = purpose
        self.reason = reason
        self.remedy = 'pip install "gmsh>=4.15.1,<5"'
        Exception.__init__(
            self,
            f"{purpose} needs the CAD kernel library gmsh, which could not be loaded: {reason}. "
            f"gmsh is a dependency of cfdbender; install it with: {self.remedy}. "
            f"On Linux the library also needs {self.LINUX_PACKAGES}. The stages that do not "
            "use the kernel still run without it.",
        )

    def __reduce__(self) -> Any:
        """Pickle by the keyword fields, since ``__init__`` takes no positional message."""
        return (_rebuild_kernel_unavailable, (self.purpose, self.reason))


def _rebuild_kernel_unavailable(purpose: str, reason: str) -> KernelUnavailableError:
    return KernelUnavailableError(purpose=purpose, reason=reason)
