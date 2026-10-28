"""--cpu-mem (host RAM budget for the pinned expert banks): the greedy per-layer residency
plan and the flags' decode-target/validation consequences."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine.cache_budget import clamp_host_budget, plan_host_residency


def test_plan_host_residency_pins_in_id_order_until_the_budget_runs_out():
    assert plan_host_residency(4, 100, 250) == ["pinned", "pinned", "pageable", "pageable"]
    assert plan_host_residency(4, 100, 1000) == ["pinned"] * 4
    assert plan_host_residency(4, 100, 0) == ["pageable"] * 4


def test_plan_host_residency_unset_or_unmeasurable_pins_everything():
    assert plan_host_residency(3, 100, None) == ["pinned"] * 3
    assert plan_host_residency(3, 0, 10) == ["pinned"] * 3


def test_clamp_host_budget_keeps_a_reserve():
    # MemAvailable 12 GiB, 2 GiB reserve -> at most 10 GiB may be pinned
    assert clamp_host_budget(7 << 30, 12 << 30, 2 << 30) == 7 << 30
    assert clamp_host_budget(11 << 30, 12 << 30, 2 << 30) == 10 << 30
    assert clamp_host_budget(1 << 30, 1 << 30, 2 << 30) == 0
    assert clamp_host_budget(7 << 30, None, 2 << 30) == 7 << 30  # unreadable -> unchanged


def test_decode_target_follows_cpu_mem():
    from freetoken.engine.engine import _decode_target

    def cfg(**kw):
        base = dict(moe_strategy="offload", moe_cpu_layers=None, cpu_mem_budget=None)
        base.update(kw)
        return SimpleNamespace(**base)

    assert _decode_target(cfg()) == "gpu"
    assert _decode_target(cfg(cpu_mem_budget=8 << 30)) == "cpu"
    assert _decode_target(cfg(moe_strategy="hybrid", cpu_mem_budget=8 << 30)) == "hybrid"
    assert _decode_target(cfg(moe_strategy="cpu", cpu_mem_budget=8 << 30)) == "cpu"


def test_cpu_mem_requires_an_offload_strategy():
    """The flag sizes pinned host banks: dense models and fused experts must be rejected."""
    from freetoken.engine.engine import _check_cpu_mem_config

    def cfg(strategy, budget=8 << 30):
        return SimpleNamespace(moe_strategy=strategy, cpu_mem_budget=budget)

    _check_cpu_mem_config(cfg("offload"), is_moe=True)  # no raise
    _check_cpu_mem_config(cfg("hybrid"), is_moe=True)
    _check_cpu_mem_config(cfg("cpu"), is_moe=True)
    _check_cpu_mem_config(cfg("offload", budget=None), is_moe=True)  # unset: nothing to check
    with pytest.raises(ValueError, match="needs a MoE model"):
        _check_cpu_mem_config(cfg("fused"), is_moe=True)
    with pytest.raises(ValueError, match="for a dense model"):
        _check_cpu_mem_config(cfg("fused"), is_moe=False)
