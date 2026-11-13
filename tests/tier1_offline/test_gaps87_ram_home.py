"""Gap 19 (part): the 0.5 GiB RAM floor has one home, read by core and by every adapter."""

import pytest

from cfdbender.core import run
from cfdbender.core.registry import RegistryError
from cfdbender.solvers._host import _ram


def test_the_host_guard_reads_the_one_measurement_of_core():
    assert _ram.free_ram_gib is run.free_ram_gib
    assert _ram.FLOOR_GIB == run.FLOOR_GIB == 0.5


@pytest.mark.parametrize("free,refused", [(0.49, True), (0.5, False), (4.0, False)])
def test_core_check_and_host_guard_share_the_floor(monkeypatch, free, refused):
    monkeypatch.setattr(run, "free_ram_gib", lambda: free)
    if refused:
        with pytest.raises(RegistryError, match="0.5 GiB floor"):
            run.check_available_ram()
    else:
        run.check_available_ram()
    # The host guard with a zero estimate decides the same way for the same reading.
    if refused:
        with pytest.raises(ValueError, match="0.5 GiB floor"):
            _ram.require_ram(0, measure=lambda: free)
    else:
        _ram.require_ram(0, measure=lambda: free)


def test_an_unmeasurable_host_refuses_the_core_check(monkeypatch):
    def broken():
        raise OSError("no meminfo")

    monkeypatch.setattr(run, "free_ram_gib", broken)
    with pytest.raises(RegistryError, match="cannot measure free RAM"):
        run.check_available_ram()
