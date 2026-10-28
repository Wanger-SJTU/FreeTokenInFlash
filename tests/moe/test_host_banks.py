"""HostBank file backing (--cpu-mem spill): allocate over an unlinked tmpfile, drop the
filled pages, and read the same bytes back by faulting them in again."""

from __future__ import annotations

import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="spill banks are Linux-only")


def test_file_backed_bank_survives_drop_pages():
    from freetoken.moe.host_banks import HostBank

    bank = HostBank((64, 1024), torch.uint8, backing="file")
    expected = torch.arange(64 * 1024, dtype=torch.int64).to(torch.uint8).reshape(64, 1024)
    bank.tensor.copy_(expected)
    bank.drop_pages()
    # force the pages back through the page table: a fresh read must see the same bytes
    assert torch.equal(bank.tensor.clone(), expected)
    assert bank.residency.value == "pageable"


def test_file_backed_bank_never_pins():
    from freetoken.moe.host_banks import HostBank

    bank = HostBank((8, 256), torch.uint8, backing="file")
    with pytest.raises(AssertionError, match="never pinned"):
        bank.pin()


def test_drop_pages_is_a_noop_for_anonymous_banks():
    from freetoken.moe.host_banks import HostBank

    bank = HostBank((8, 256), torch.uint8)
    bank.tensor.fill_(7)
    bank.drop_pages()
    assert bank.tensor.eq(7).all()  # anon pages have no backing store: dropping would lose data


def test_residency_plan_makes_pageable_layers_file_backed(tmp_path):
    from freetoken.moe.host_banks import HostBank, requested_residency

    labels = ["pinned", "pageable"]
    with requested_residency(labels, spill_dir=str(tmp_path)):
        bank_pinned = HostBank((4, 64), torch.uint8, layer_id=0)
        bank_spilled = HostBank((4, 64), torch.uint8, layer_id=1)
        # no layer id -> the plan cannot pick a backing; stay anonymous
        bank_unknown = HostBank((4, 64), torch.uint8)
    assert not bank_pinned._file_backed
    assert bank_spilled._file_backed
    assert not bank_unknown._file_backed


def test_settle_drops_a_pageable_file_bank_and_keeps_the_data(tmp_path):
    from freetoken.moe.host_banks import HostBank, _settle, requested_residency

    expected = torch.randint(0, 255, (4, 256), dtype=torch.uint8)
    with requested_residency(["pageable"], spill_dir=str(tmp_path)):
        bank = HostBank((4, 256), torch.uint8, layer_id=0)
        bank.tensor.copy_(expected)
        _settle(bank, "pageable")
    assert torch.equal(bank.tensor.clone(), expected)
