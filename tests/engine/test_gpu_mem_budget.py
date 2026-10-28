"""--gpu-mem (absolute GPU budget) policy: pure arithmetic + the startup KV solve.

The absolute mode replaces ``memory_ratio x baseline`` with ``budget - consumed`` in both
budget consumers (the MoE-vs-KV split and the startup KV solve), and rejects an exhausted
budget with a readable error instead of OOMing in a later CUDA allocation.
"""

from __future__ import annotations

import pytest

from freetoken.engine.cache_budget import (
    absolute_pool_budget_bytes,
    net_cache_budget_bytes,
    resolve_moe_cache_auto,
)
from freetoken.engine.engine import _startup_kv_budget


def test_absolute_mode_ignores_ratio_and_baseline():
    # weights + fixed come out of the absolute cap; ratio/baseline are not consulted.
    assert net_cache_budget_bytes(0.9, 10_000, 2_000, 500, gpu_budget=8_000) == 5_500
    assert net_cache_budget_bytes(0.9, 10_000, 2_000, 500) == 10_000 * 0.9 - 2_500


def test_absolute_pool_budget_bytes_is_cap_minus_consumed():
    assert absolute_pool_budget_bytes(16 << 30, 3 << 30) == 13 << 30


def test_resolve_moe_cache_auto_absolute_budget_splits_moe_first():
    # 8 GiB cap, 1 GiB weights+fixed -> 7 GiB for 1 MiB slots / 64 KiB pages.
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=1 << 40,
        weights_bytes=1 << 30,
        memory_ratio=0.9,
        cache_per_page=64 << 10,
        fixed_cache_size=0,
        per_expert_bytes=1 << 20,
        num_experts=8,
        total_experts=64,
        prefill_overlap=True,
        kv_reserve_tokens=64,
        page_size=1,
        max_slots=None,
        gpu_budget=8 << 30,
    )
    budget = (8 << 30) - (1 << 30)
    assert size * (1 << 20) + pages * (64 << 10) <= budget
    assert size >= 2 * 8  # prefill overlap floor
    assert overlap


def test_resolve_moe_cache_auto_rejects_exhausted_absolute_budget():
    with pytest.raises(ValueError, match="exhausted by the resident weights"):
        resolve_moe_cache_auto(
            baseline_free=1 << 40,
            weights_bytes=4 << 30,
            memory_ratio=0.9,
            cache_per_page=64 << 10,
            fixed_cache_size=1 << 30,
            per_expert_bytes=1 << 20,
            num_experts=8,
            total_experts=64,
            prefill_overlap=False,
            kv_reserve_tokens=64,
            page_size=1,
            gpu_budget=4 << 30,
        )


def test_startup_kv_budget_absolute_mode():
    # 1000 pre-load free, 600 consumed by the model -> 400 left of a 700 cap.
    assert _startup_kv_budget(0.9, 1000, 400, gpu_budget=700) == 100
    # no budget -> the ratio path is untouched
    assert _startup_kv_budget(0.9, 1000, 400) == int(0.9 * 1000) - 600


def test_startup_kv_budget_rejects_exhausted_budget():
    with pytest.raises(ValueError, match="exhausted before the KV pool"):
        _startup_kv_budget(0.9, 1000, 400, gpu_budget=500)
