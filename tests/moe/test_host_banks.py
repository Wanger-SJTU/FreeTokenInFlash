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


# --------------------------------------------------------------------------------------
# --cpu-mem-mode expert: the split (anonymous prefix + file suffix) bank layout.
# --------------------------------------------------------------------------------------


def test_parse_layer_backing_values():
    from freetoken.moe.host_banks import parse_layer_backing

    assert parse_layer_backing("pinned") == ("anon", None)
    assert parse_layer_backing("pageable") == ("file", None)
    assert parse_layer_backing("split") == ("split", None)
    assert parse_layer_backing("split:37") == ("split", 37)
    # raw backing constants pass through untouched
    assert parse_layer_backing("anon") == ("anon", None)
    assert parse_layer_backing("file") == ("file", None)


@pytest.mark.skipif(not _LINUX, reason="split banks are Linux-only (MAP_FIXED)")
def test_host_bank_split_survives_pin_and_drop():
    """The expert-mode promise: the tensor is ONE contiguous [E, ...] buffer straddling
    two mappings; pin() registers only the anonymous prefix, drop_pages() returns only
    the file-backed suffix -- and all the data survives both."""
    from freetoken.moe.host_banks import SPLIT_BACKING, HostResidency

    E, R, RB = 16, 4, 128
    bank = HostBank((E, R, RB), torch.uint8, backing=SPLIT_BACKING, split_experts=6)
    assert bank.backing == SPLIT_BACKING
    assert bank.split_experts == 6
    assert bank.residency == HostResidency.SPLIT
    assert 0 < bank.anon_bytes < bank.nbytes  # genuinely straddles

    expected = torch.arange(E * R * RB, dtype=torch.uint8).reshape(E, R, RB)
    bank.tensor.copy_(expected)
    assert torch.equal(bank.tensor, expected)  # contiguous across both mappings

    if torch.cuda.is_available():
        bank.pin()  # registers ONLY [addr, addr + anon_bytes)
    bank.drop_pages()  # drops ONLY the file suffix
    assert torch.equal(bank.tensor[:6], expected[:6])  # prefix resident, verbatim
    assert torch.equal(bank.tensor, expected)  # suffix re-faulted from the spill file


@pytest.mark.skipif(not _LINUX, reason="split banks are Linux-only (MAP_FIXED)")
def test_alloc_layer_banks_parses_split_plan_entries():
    """"split:<k>" plan values (the --cpu-mem expert plan rides layer_backing) allocate
    split banks with the parsed k; mixed plans stay per-layer."""
    from freetoken.moe.host_banks import ANON_BACKING, FILE_BACKING, SPLIT_BACKING

    specs = {"gate_up": ((8, 16), torch.float32)}
    hb = alloc_layer_banks(specs, 3, backing=["split:4", "pinned", "pageable"])
    gate_up = hb["gate_up"]
    assert gate_up[0].backing == SPLIT_BACKING and gate_up[0].split_experts == 4
    assert gate_up[1].backing == ANON_BACKING
    assert gate_up[2].backing == FILE_BACKING

    with pytest.raises(AssertionError, match="split"):
        # a bare "split" entry needs the count from somewhere
        alloc_layer_banks(specs, 1, backing=["split"])


@pytest.mark.skipif(not _LINUX, reason="split banks are Linux-only (MAP_FIXED)")
def test_pin_banks_finishes_split_prefix_and_suffix():
    """pin_banks (the loaders' after-fill step) pins the split prefix AND drops the
    suffix, in one pass."""
    from freetoken.moe.host_banks import SPLIT_BACKING

    specs = {"gate_up": ((8, 16), torch.float32)}
    hb = alloc_layer_banks(specs, 1, backing=["split:4"])
    bank = hb["gate_up"][0]
    bank.tensor.copy_(torch.randn(8, 16))
    expected = bank.tensor.clone()
    pin_banks(hb)
    assert bank.backing == SPLIT_BACKING
    assert torch.equal(bank.tensor, expected)
    if torch.cuda.is_available():
        assert bank._pinned
