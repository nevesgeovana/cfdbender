"""Compatibility alias for the shared host RAM guard.

Keep the module object shared so existing measurement injection still applies
to every adapter, without any adapter importing the SU2 implementation.
"""

import sys

from cfdbender.solvers._host import _ram
from cfdbender.solvers._host._ram import free_ram_gib as free_ram_gib
from cfdbender.solvers._host._ram import require_ram as require_ram

sys.modules[__name__] = _ram
