"""--cpu-mem host-bank primitives: file-backed (spill) HostBanks and the per-layer
residency plans threading through alloc/pin helpers.

File-backed banks are Linux-only (shared mmap + MADV_DONTNEED re-fault), so every spill
test self-skips elsewhere; the anon-bank behavior tests run anywhere torch runs."""

from __future__ import annotations

import platform

import pytest
import torch

from freetoken.moe.host_banks import FILE_BACKING, HostBank, alloc_layer_banks, pin_banks

_LINUX = platform.system() == "Linux"


def test_host_bank_anon_roundtrip():
    bank = HostBank((4, 8), torch.float32)
    assert bank.backing == "anon"
    bank.tensor.copy_(torch.arange(32, dtype=torch.float32).view(4, 8))
    assert torch.equal(bank.tensor, torch.arange(32, dtype=torch.float32).view(4, 8))


@pytest.mark.skipif(not _LINUX, reason="file-backed spill banks are Linux-only")
def test_host_bank_file_backed_survives_drop_pages():
    """The --cpu-mem promise: after fill + drop_pages the DATA survives (it re-faults
    from the spill file) even though the resident pages were returned to the OS."""
    bank = HostBank((4, 8), torch.float32, backing=FILE_BACKING)
    assert bank.backing == FILE_BACKING
    expected = torch.arange(32, dtype=torch.float32).view(4, 8)
    bank.tensor.copy_(expected)
    bank.drop_pages()
    assert torch.equal(bank.tensor, expected)


@pytest.mark.skipif(not _LINUX, reason="file-backed spill banks are Linux-only")
def test_host_bank_file_backed_never_pins():
    bank = HostBank((4, 8), torch.float32, backing=FILE_BACKING)
    with pytest.raises(AssertionError, match="must never be registered"):
        bank.pin()


@pytest.mark.skipif(not _LINUX, reason="file-backed spill banks are Linux-only")
def test_alloc_layer_banks_honors_per_layer_backing():
    specs = {"gate_up": ((4, 8), torch.float32), "down": ((8, 2), torch.float32)}
    hb = alloc_layer_banks(specs, 3, backing=["pinned", "pageable", "pinned"])
    for name in specs:
        assert [b.backing for b in hb[name]] == ["anon", "file", "anon"]
    # a constant string applies to every layer
    hb_all = alloc_layer_banks(specs, 2, backing="pageable")
    for name in specs:
        assert all(b.backing == FILE_BACKING for b in hb_all[name])


@pytest.mark.skipif(not _LINUX, reason="file-backed spill banks are Linux-only")
def test_pin_banks_pins_anon_and_drops_file(monkeypatch):
    """pin_banks is the loaders' after-fill step: anon banks get cudaHostRegister'd,
    file-backed banks get their pages dropped instead -- never registered."""
    registered: list[int] = []
    import freetoken.kernel.pinned as pinned_mod

    monkeypatch.setattr(pinned_mod, "host_register", lambda addr, n: registered.append(n))

    anon = HostBank((2, 4), torch.float32)
    file_backed = HostBank((2, 4), torch.float32, backing=FILE_BACKING)
    expected = torch.arange(8, dtype=torch.float32).view(2, 4)
    anon.tensor.copy_(expected)
    file_backed.tensor.copy_(expected)

    pin_banks({"a": anon, "f": file_backed})

    assert registered == [len(anon._buf)]  # exactly the anon bank was registered
    assert anon._pinned is True
    assert file_backed._pinned is False
    assert torch.equal(file_backed.tensor, expected)  # dropped pages re-fault intact


@pytest.mark.skipif(not _LINUX, reason="file-backed spill banks are Linux-only")
def test_pin_pipeline_drops_file_backed_banks_without_blocking():
    """The PinPipeline's after-fill step treats a file-backed bank as an unpin (drop),
    not a registration; a clean exit still drains the queue."""
    from freetoken.moe.host_banks import PinPipeline

    bank = HostBank((2, 4), torch.float32, backing=FILE_BACKING)
    expected = torch.arange(8, dtype=torch.float32).view(2, 4)
    bank.tensor.copy_(expected)
    with PinPipeline() as pins:
        pins(0, {"gate_up": bank})
    assert bank._pinned is False
    assert torch.equal(bank.tensor, expected)
