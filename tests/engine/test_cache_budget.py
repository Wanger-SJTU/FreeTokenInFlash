from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.cache_budget import expert_bytes_per_slot, plan_cache_budget, resolve_moe_cache_auto


def test_moe_priority_fills_experts_up_to_total():
    # budget large enough to cache every expert; KV gets the remainder.
    # per_expert=100, cache_per_page=10, total=8 experts (L*E), E=4.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_pages=5, max_slots=8,
    )
    assert size == 8  # capped at full residency
    assert pages == (2000 - 8 * 100) // 10  # == 120, remainder to KV
    assert overlap is True


def test_offload_case_experts_take_most_kv_gets_reserve_floor():
    # budget too small for full residency: experts take what they can, KV keeps its floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=1000, per_expert_bytes=100, cache_per_page=10,
        num_experts=2, total_experts=50, prefill_overlap=True,
        kv_reserve_pages=10, max_slots=50,
    )
    # raw = (1000 - 10*10) // 100 = 9 ; clamped to [4, 50] -> 9
    assert size == 9
    assert pages == max((1000 - 9 * 100) // 10, 10)  # remainder 10 pages, == floor
    assert overlap is True


def test_marlin_cap_clamps_count_and_rolls_bytes_to_kv():
    # budget would fund 1500 experts, but marlin caps at 992; freed bytes become KV pages.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=200_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=128, total_experts=4000, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=992,
    )
    assert size == 992
    assert pages == (200_000 - 992 * 100) // 10


def test_small_cache_disables_prefill_overlap():
    # cap below 2*num_experts -> overlap impossible, falls back to num_experts floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=10_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=8, total_experts=12, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=12,
    )
    assert overlap is False
    # raw = 10000//100 = 100, clamped to hi = min(12, 12) = 12.
    assert size == 12


def test_insufficient_kv_memory_raises():
    with pytest.raises(AssertionError, match="not enough memory"):
        plan_cache_budget(
            budget_bytes=410, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=0, max_slots=4,
        )  # experts eat 400, KV gets 1 page -> not > 1


def test_budget_too_small_for_min_moe_plus_reserve_raises():
    # Budget cannot fund even the minimum MoE slots + the KV reserve, so the floored plan
    # would exceed budget_bytes. Reject in arithmetic rather than OOM in a later CUDA alloc.
    with pytest.raises(AssertionError, match="budget too small"):
        plan_cache_budget(
            budget_bytes=300, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=10, max_slots=4,
        )  # min moe = 4 slots (400 B) + reserve (10 pages = 100 B) = 500 B > 300 B budget


def test_prefill_overlap_false_is_honored():
    # Even when the cache could fit 2*num_experts, an explicit False stays False.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=False,
        kv_reserve_pages=0, max_slots=8,
    )
    assert size == 8
    assert overlap is False
    assert pages == (2000 - 8 * 100) // 10


def test_expert_bytes_per_slot_sums_row_bytes_over_banks():
    sources = {
        "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)],  # row = 32*8*2 = 512
        "down": [torch.zeros(4, 8, 16, dtype=torch.float16)],     # row = 8*16*2 = 256
    }
    assert expert_bytes_per_slot(sources) == 512 + 256


def test_resolve_auto_applies_ratio_once_and_marlin_cap():
    # baseline 1000, weights 100, ratio 0.9 -> budget = 900 - 100 - 0(fixed) = 800
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=1000, weights_bytes=100, memory_ratio=0.9,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=50,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=1, quant_format="bf16",
    )
    # budget 800: experts cap at 8 -> 400 bytes; KV = 400//10 = 40 pages
    assert size == 8 and pages == 40 and overlap is True


def test_resolve_auto_marlin_caps_slots():
    size, _, _ = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=0, memory_ratio=1.0,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=100,
        num_experts=128, total_experts=4000, prefill_overlap=False,
        kv_reserve_tokens=0, page_size=1, quant_format="nvfp4_marlin",
    )
    assert size == 992


def _dsv4_adjust_cfg(**over):
    # A DSV4 _adjust_config stub mirroring the real checkpoint (ds_fp4 experts, dsv4_sparse
    # attention, offload MoE backend).
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, dsv4_args=SimpleNamespace(window_size=128), is_moe=True,
        expert_quant="ds_fp4", has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = True
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "offload"
        max_running_req = 1
        cuda_graph_max_bs = 1
        cuda_graph_bs = [1]
        max_seq_len = 1024
        max_extend_tokens = 4096
        page_size = 1
        attention_backend = "dsv4_sparse"
        moe_cpu_layers = None
        nvfp4_backend = "auto"
        num_page_override = None
        num_token_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return cfg


def test_adjust_config_allows_auto_for_dsv4():
    # DSV4 now supports --moe-cache-auto via the affine KV cost bridge (dsv4_auto_cost_model);
    # _adjust_config must NOT reject it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg()
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_backend == "offload"
    assert cfg.moe_cache_auto is True  # resolved later at engine init, not here
    assert cfg.page_size == 128  # DSV4's KV page is the P-token window page


def test_adjust_config_resolves_num_tokens_for_dsv4():
    # --num-tokens resolves AFTER every page_size override, so DSV4's P=128 page divides it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072)
    _adjust_config(cfg)
    assert cfg.page_size == 128
    assert cfg.num_page_override == 1024


def test_adjust_config_rejects_num_tokens_not_multiple_of_page():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131000)  # not a multiple of 128
    with pytest.raises(ValueError, match="not a multiple"):
        _adjust_config(cfg)


def test_adjust_config_rejects_num_tokens_with_num_pages():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072, num_page_override=1024)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _adjust_config(cfg)


def test_adjust_config_resolves_num_tokens_generic():
    # Generic model keeps its page_size (1 here): tokens map 1:1 onto pages.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "fi"
        nvfp4_backend = "auto"
        num_page_override = None
        num_token_override = 5000

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    _adjust_config(cfg)
    assert cfg.num_page_override == 5000


def test_mha_kv_cost_simple_full_attention():
    import torch

    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.utils import div_even

    class StubModelConfig:
        has_swa_attention = False

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=tuple(range(3)),
                num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.bfloat16
        page_size = 16
        max_running_req = 4
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    per_token = 2 * 64 * div_even(8, 1, allow_replicate=True) * 2 * 3
    assert cache_per_page == per_token * 16
    assert fixed == 0


def test_engine_resolve_auto_moe_cache_size_maps_kwargs():
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 4
        num_moe_layers = 2  # total_experts = 8

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        gpu_mem_budget = None
        moe_prefill_overlap = True
        kv_reserve_tokens = 0
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    class StubBanks:
        quant_format = "bf16"
        # 2 layers (num_moe_layers above) x 4 experts each -- per-layer host bank contract.
        sources = {
            "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)] * 2,  # row = 32*8*2 = 512
            "down": [torch.zeros(4, 8, 16, dtype=torch.float16)] * 2,     # row = 8*16*2 = 256
        }

    from freetoken.kvcache.mha_pool import MHAKVCache

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache  # __init__ skipped -> install the generic pool family

    size, pages, overlap = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks())

    # cross-check against the same pure functions, proving the kwarg mapping is faithful
    from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto
    from freetoken.kvcache.mha_pool import MHAKVCache

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    expected = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=1_000_000, memory_ratio=0.9,
        cache_per_page=cache_per_page, fixed_cache_size=fixed,
        per_expert_bytes=expert_bytes_per_slot(StubBanks.sources),
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=16, quant_format="bf16",
    )
    assert (size, pages, overlap) == expected


# ---------------------------------------------------------------------------
# offload-cache sizing guard + auto-resolution (_require_offload_cache_size / _adjust_config),
# the floor rule compute_cache_floors documents above.
# ---------------------------------------------------------------------------


def _offload_engine_config(**overrides):
    """A frozen EngineConfig for a quantized-experts MoE checkpoint in the bare-invocation state
    (moe_backend="auto") unless overridden — the shared fixture for the _adjust_config tests."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        attention_backend="fi",
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="nvfp4",  # quantized experts -> must resolve to an offload backend
            moe_backend="auto",
        ),
    )
    return config


def test_guard_passes_when_size_covers_one_expert_per_layer():
    from freetoken.engine.engine import _require_offload_cache_size

    _require_offload_cache_size(cache_size=128, num_experts=128)  # no raise


def test_guard_raises_actionable_error_when_too_small():
    from freetoken.engine.engine import _require_offload_cache_size

    with pytest.raises(ValueError) as exc:
        _require_offload_cache_size(cache_size=0, num_experts=128)
    msg = str(exc.value)
    assert "128" in msg and "moe-cache" in msg


def test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend():
    """Bare `ft serve <FTW MoE checkpoint>`: no --moe-backend, no --moe-cache-* flags at all.

    args.py's parse-time default only fires when the backend is *already*
    offload-family at parse time -- but a bare invocation leaves moe_backend="auto" at parse
    time, and the "auto" -> offload/cpu/hybrid resolution only happens later, in _adjust_config,
    once the model_config (and its expert_quant) is known. This proves the engine-level
    resolution: a quantized-experts model auto-resolving to an offload-family backend also gets
    moe_cache_auto=True, so _init_offload_moe_cache's _require_offload_cache_size guard is never
    reached with moe_cache_size still 0.
    """
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_backend

    config = _offload_engine_config()
    _adjust_config(config)

    # Which member of the family gets picked is not this test's claim, and is not ours to
    # decide: a bare "auto" consults ~/.cache/freetoken/benchbw.json, so a box that has run
    # `ft bench bw` resolves nvfp4 experts to hybrid instead. Assert the family, not the member.
    assert is_offload_moe_backend(config.moe_backend)
    assert config.moe_cache_auto is True
    assert config.moe_cache_size == 0  # still unresolved -- the scheduler sizes it from VRAM


def test_page_table_width_covers_whole_trailing_pages():
    # _write_page_table writes WHOLE trailing pages, so the width must reach the last
    # page's end, not just the next multiple of 32 (DSV4's P=128 exposed the gap).
    from freetoken.engine.engine import _page_table_width

    assert _page_table_width(4001, 128) == 4096   # align32 alone gave 4032 -> OOB
    assert _page_table_width(4096, 128) == 4096   # page-aligned length unchanged
    assert _page_table_width(100, 1) == 128       # page_size 1 degenerates to align32
    assert _page_table_width(33, 128) == 128
    for max_seq_len in (1, 31, 33, 4001, 4095, 4096):
        for page_size in (1, 32, 64, 128):
            w = _page_table_width(max_seq_len, page_size)
            last_col = -(-max_seq_len // page_size) * page_size - 1
            assert w > last_col and w % 32 == 0


def _generic_rotary_cfg(max_position, override):
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
        rotary_config=SimpleNamespace(max_position=max_position),
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton"
        nvfp4_backend = "auto"
        num_page_override = None
        num_token_override = None
        max_seq_len_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    object.__setattr__(cfg, "max_seq_len_override", override)
    return cfg


def test_adjust_config_rejects_override_past_rope_table():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="rope table"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=2048))


def test_adjust_config_allows_override_at_rope_table_boundary():
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=1024))  # must not raise


def test_adjust_config_rope_gate_exempts_dsv4():
    # DSV4 sizes its own rope table from the resolved max_seq_len (_adjust_dsv4_config),
    # so the generic gate must not fire even when the override dwarfs max_position.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(max_seq_len_override=10_000_000)
    cfg.model_config.rotary_config = SimpleNamespace(max_position=1024)
    _adjust_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# --gpu-mem / --cpu-mem budget planning (absolute GPU budget + host residency plan)
# ---------------------------------------------------------------------------


def test_plan_host_residency_pins_greedy_then_spills():
    from freetoken.engine.cache_budget import plan_host_residency

    assert plan_host_residency(4, 100, 250) == ["pinned", "pinned", "pageable", "pageable"]
    assert plan_host_residency(4, 100, 400) == ["pinned"] * 4
    assert plan_host_residency(4, 100, 0) == ["pageable"] * 4


def test_plan_host_residency_no_budget_or_unknown_geometry_pins_all():
    from freetoken.engine.cache_budget import plan_host_residency

    # None budget = the historical all-pinned behavior; an unmeasurable layer size
    # (0 bytes) must NOT silently spill everything.
    assert plan_host_residency(3, 100, None) == ["pinned"] * 3
    assert plan_host_residency(3, 0, 0) == ["pinned"] * 3


def test_absolute_pool_budget_bytes_subtracts_consumed():
    from freetoken.engine.cache_budget import absolute_pool_budget_bytes

    assert absolute_pool_budget_bytes(160, 100) == 60
    assert absolute_pool_budget_bytes(100, 160) == -60  # caller rejects negatives


def test_resolve_auto_honors_absolute_gpu_budget():
    # --gpu-mem 500 with weights 100 + fixed 0 -> budget 400, identical split to the
    # ratio path with an equivalent net budget.
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=10**9, weights_bytes=100, memory_ratio=0.9,  # ratio path must be ignored
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=25,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=1, quant_format="bf16",
        gpu_budget=500,
    )
    assert size == 8  # cap at full residency (8 x 25 = 200 of the 400 budget)
    assert pages == (400 - 8 * 25) // 10
    assert overlap is True


def test_resolve_auto_rejects_gpu_budget_exhausted_by_weights():
    with pytest.raises(ValueError, match="gpu-mem budget .* exhausted"):
        resolve_moe_cache_auto(
            baseline_free=10**9, weights_bytes=600, memory_ratio=0.9,
            cache_per_page=10, fixed_cache_size=50, per_expert_bytes=50,
            num_experts=4, total_experts=8, prefill_overlap=True,
            kv_reserve_tokens=0, page_size=1, quant_format="bf16",
            gpu_budget=500,
        )


def test_startup_kv_budget_absolute_subtracts_consumed():
    from freetoken.engine.engine import _startup_kv_budget_absolute

    # --gpu-mem cap 800; 650 consumed since the pre-load baseline -> 150 left for KV.
    assert _startup_kv_budget_absolute(800, init_free_memory=1000, new_free_memory=350) == 150


# Per-format host-bank size estimates must match the real bank row shapes
# (expert_bytes_per_slot over dummy sources built with the loaders' formulas) -- the
# drift guard for --cpu-mem planning, which runs BEFORE any bank exists.
_H, _I, _E = 128, 64, 4


def _dummy_sources(quant: str, nvfp4_native: bool):
    import torch

    def bank(shape, dtype=torch.uint8):
        return [torch.zeros(shape, dtype=dtype)]

    if quant == "none":
        return {
            "gate_up": bank((_E, 2 * _I, _H), torch.bfloat16),
            "down": bank((_E, _H, _I), torch.bfloat16),
        }
    if quant == "fp8_block":
        B = 128
        return {
            "gate_up": bank((_E, 2 * _I, _H)),
            "gate_up_scale": bank((_E, 2 * _I // B, _H // B), torch.bfloat16),
            "down": bank((_E, _H, _I)),
            "down_scale": bank((_E, _H // B, _I // B), torch.bfloat16),
        }
    if quant == "q4_0":
        return {
            "gate_up": bank((_E, 2 * _I, _H // 32 * 18)),
            "down": bank((_E, _H, _I // 32 * 18)),
        }
    if quant == "nvfp4":
        if nvfp4_native:
            return {
                "gate_up_packed": bank((_E, 2 * _I, _H // 2)),
                "gate_up_scale": bank((_E, 2 * _I, _H // 16)),
                "gate_up_global": bank((_E, 2 * _I), torch.float16),
                "down_packed": bank((_E, _H, _I // 2)),
                "down_scale": bank((_E, _H, _I // 16)),
                "down_global": bank((_E, _H), torch.float16),
            }
        return {  # marlin/b12x: globals folded into GPU-resident alphas, not host banks
            "gate_up_packed": bank((_E, 2 * _I, _H // 2)),
            "gate_up_scale": bank((_E, 2 * _I, _H // 16)),
            "down_packed": bank((_E, _H, _I // 2)),
            "down_scale": bank((_E, _H, _I // 16)),
        }
    if quant == "mxfp4_triton":
        return {
            "gate_up_blocks": bank((_E, _H // 2, 2 * _I)),
            "gate_up_scales": bank((_E, _H // 32, 2 * _I)),
            "gate_up_bias": bank((_E, 2 * _I), torch.bfloat16),
            "down_blocks": bank((_E, _I // 2, _H)),
            "down_scales": bank((_E, _I // 32, _H)),
            "down_bias": bank((_E, _H), torch.bfloat16),
        }
    if quant == "ds_fp4":
        return {
            "gate_up_packed": bank((_E, 2 * _I, _H // 2)),
            "gate_up_scale": bank((_E, 2 * _I, _H // 32)),
            "down_packed": bank((_E, _H, _I // 2)),
            "down_scale": bank((_E, _H, _I // 32)),
        }
    raise AssertionError(quant)


@pytest.mark.parametrize("quant,native", [
    ("none", False),
    ("fp8_block", False),
    ("q4_0", False),
    ("nvfp4", True),
    ("nvfp4", False),
    ("mxfp4_triton", False),
    ("ds_fp4", False),
])
def test_estimate_layer_bank_bytes_matches_real_bank_shapes(quant, native):
    from freetoken.engine.cache_budget import estimate_layer_bank_bytes, expert_bytes_per_slot

    sources = _dummy_sources(quant, native)
    expected = expert_bytes_per_slot(sources) * _E
    got = estimate_layer_bank_bytes(
        hidden_size=_H, moe_intermediate_size=_I, num_experts=_E,
        expert_quant=quant, tp_size=1, nvfp4_native=native,
    )
    assert got == expected


def test_estimate_layer_bank_bytes_applies_tp_to_intermediate():
    from freetoken.engine.cache_budget import estimate_layer_bank_bytes

    one = estimate_layer_bank_bytes(
        hidden_size=_H, moe_intermediate_size=_I, num_experts=_E,
        expert_quant="none", tp_size=1,
    )
    two = estimate_layer_bank_bytes(
        hidden_size=_H, moe_intermediate_size=_I, num_experts=_E,
        expert_quant="none", tp_size=2,
    )
    assert two == one // 2  # bf16 rows scale linearly with the sharded intermediate


def test_estimate_layer_bank_bytes_unknown_quant_raises():
    from freetoken.engine.cache_budget import estimate_layer_bank_bytes

    with pytest.raises(ValueError, match="no host-bank size estimate"):
        estimate_layer_bank_bytes(
            hidden_size=_H, moe_intermediate_size=_I, num_experts=_E,
            expert_quant="mystery", tp_size=1,
        )


def test_estimate_routes_gpt_oss_mxfp4_via_weight_format():
    # gpt-oss keeps expert_quant="none" with moe_weight_format="mxfp4"; the estimate
    # must use the mxfp4 row math (the real geometry), not the bf16 fallback -- else
    # --cpu-mem over-reserves 3x and spills layers that would have fit.
    from freetoken.engine.cache_budget import estimate_layer_bank_bytes

    bf16 = estimate_layer_bank_bytes(
        hidden_size=_H, moe_intermediate_size=_I, num_experts=_E, expert_quant="none",
    )
    mxfp4 = estimate_layer_bank_bytes(
        hidden_size=_H, moe_intermediate_size=_I, num_experts=_E, expert_quant="none",
        moe_weight_format="mxfp4",
    )
    direct = estimate_layer_bank_bytes(
        hidden_size=_H, moe_intermediate_size=_I, num_experts=_E,
        expert_quant="mxfp4_triton",
    )
    assert mxfp4 == direct
    assert mxfp4 < bf16 // 2


def test_net_cache_budget_absolute_mode_ignores_ratio_and_baseline():
    from freetoken.engine.cache_budget import net_cache_budget_bytes

    # --gpu-mem: the ratio/baseline terms are inert; budget = cap - weights - fixed.
    assert net_cache_budget_bytes(0.9, 10**10, 100, 50, gpu_budget=1000) == 850
    # without a budget the historical ratio math is untouched
    assert net_cache_budget_bytes(0.9, 1000, 100, 50) == 750


def test_validate_rebuild_fit_check_honors_gpu_budget():
    """A rebuild target that fits the RATIO budget but exceeds the --gpu-mem cap must
    reject: the fit-check's budget is the absolute cap, not ratio x baseline."""
    from freetoken.kvcache.base import CacheRebuildRejected
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 1
        max_running_req = 4
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        memory_ratio = 1.0  # ratio budget would be the whole (huge) baseline
        gpu_mem_budget = 10_000_000  # ... but the absolute cap is small
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    cache_per_page, _, _, _ = MHAKVCache.kv_cost(StubConfig())
    target_pages = 12_000_000 // cache_per_page  # KV bytes ~12MB > the 10MB absolute cap

    with pytest.raises(CacheRebuildRejected, match="old cache kept"):
        MHAKVCache.validate_rebuild(
            MHAKVCache.__new__(MHAKVCache), StubConfig(),
            num_pages=target_pages, target_moe=0, per_expert_bytes=0,
            baseline_free=10**9, weights_bytes=0, current_num_pages=1,
        )
