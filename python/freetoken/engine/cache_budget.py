"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4_paged_pool.py.
    return sum(t[0][0].numel() * t[0].element_size() for t in sources.values())


def net_cache_budget_bytes(
    memory_ratio: float,
    baseline_free: int,
    weights_bytes: int,
    fixed_cache_size: int,
    gpu_budget: int | None = None,
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check.

    ``gpu_budget`` (--gpu-mem absolute mode) replaces the ratio term: the budget is the
    absolute cap minus weights + fixed, ignoring baseline/ratio (graphs/activations sit
    on top of the cap)."""
    if gpu_budget is not None:
        return int(gpu_budget) - int(weights_bytes) - int(fixed_cache_size)
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def absolute_pool_budget_bytes(gpu_budget: int, consumed_bytes: int) -> int:
    """Net bytes left of an ABSOLUTE GPU budget (--gpu-mem) after everything already
    consumed. ``consumed_bytes`` is measured the same way the ratio path measures
    ``ratio * baseline_free``: for the MoE/KV split it is weights + fixed cache; for the
    startup KV solve it is everything allocated since the pre-load baseline. The
    (negative -> invalid) result is the caller's to reject with a readable message."""
    return int(gpu_budget) - int(consumed_bytes)


def plan_host_residency(
    num_layers: int, layer_bytes: int, budget_bytes: int | None
) -> list[str]:
    """Greedy per-layer host residency plan for --cpu-mem: pin layers in id order until
    the budget is exhausted, then spill the rest to file-backed (page-cache-paged) banks.

    Returns ``HostResidency``-compatible values ("pinned"/"pageable"); ``budget_bytes``
    None (unset) pins everything -- today's behavior. ``layer_bytes <= 0`` (unknown
    geometry) also pins everything: an unmeasurable budget must not silently spill.
    """
    if budget_bytes is None or layer_bytes <= 0:
        return ["pinned"] * num_layers
    plan = []
    remaining = int(budget_bytes)
    for _ in range(num_layers):
        if remaining >= layer_bytes:
            plan.append("pinned")
            remaining -= layer_bytes
        else:
            plan.append("pageable")
    return plan


def plan_host_pool_slots(
    num_layers: int, layer_bytes: int, num_experts: int, budget_bytes: int | None
) -> int:
    """Global host-pool slot count for --cpu-mem-mode expert: the budget buys N
    pinned expert slots shared across ALL layers (one global pool, mirroring the
    unified GPU slot cache), clamped to ``[0, num_layers * num_experts]``.

    ``budget_bytes`` None pins everything (``N == L * E``); ``layer_bytes <= 0``
    (unknown geometry) also pins everything, mirroring :func:`plan_host_residency`:
    an unmeasurable budget must not silently spill."""
    total = num_layers * num_experts
    if budget_bytes is None or layer_bytes <= 0:
        return total
    per_expert = layer_bytes // num_experts
    if per_expert <= 0:
        return 0
    return max(0, min(int(budget_bytes) // per_expert, total))


def host_available_bytes() -> int | None:
    """/proc/meminfo MemAvailable (the OOM-relevant figure, reclaimable cache
    included), or None when unreadable (non-Linux / no /proc)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, StopIteration):
        pass
    return None


def clamp_pool_slots_to_headroom(
    n_slots: int, per_expert: int, available: int | None, headroom: int
) -> int:
    """Shrink the planned pool until it fits ``available - headroom`` bytes.

    The pool is unreclaimable shared-anon memory stacked on top of the serving
    processes' fixed torch overhead, while the file-backed banks keep faulting
    pages in at decode time -- without a reserve the box OOMs or thrashes reclaim.
    ``available=None`` (unreadable) keeps the plan as-is."""
    if available is None or per_expert <= 0 or n_slots <= 0:
        return n_slots
    fitting = max(0, available - headroom) // per_expert
    return min(n_slots, fitting)


def estimate_layer_bank_bytes(
    *,
    hidden_size: int,
    moe_intermediate_size: int,
    num_experts: int,
    expert_quant: str,
    tp_size: int = 1,
    nvfp4_native: bool = False,
    moe_weight_format: str | None = None,
) -> int:
    """Bytes ONE MoE layer's host banks occupy (all experts, all banks of the format),
    for --cpu-mem planning BEFORE the banks are loaded.

    Mirrors the per-format row shapes of ``_BANK_SCHEMAS`` (offload_cache) / the weight
    loaders; ``tests/engine/test_cache_budget.py`` pins every format against
    ``expert_bytes_per_slot`` on dummy banks so the two cannot drift. TP shards the
    intermediate dim (``div_even``, the loaders' convention); gpt-oss's mxfp4 uses its
    own TP range split, so its estimate is approximate by up to one block per rank.
    ``nvfp4_native`` selects the 6-bank native layout the loaders produce for
    decode_target=="cpu" (the layout a --cpu-mem split forces); the marlin/b12x
    alternatives fold their globals into GPU-resident alphas and are smaller -- plan
    against the native layout whenever a budget is active (conservative).

    gpt-oss keeps its quant on ``moe_weight_format`` (``expert_quant`` stays "none"),
    so callers pass both; "none" + mxfp4 resolves to the mxfp4 row math, not bf16.
    """
    from freetoken.utils import div_even

    if expert_quant == "none" and (moe_weight_format or None) == "mxfp4":
        expert_quant = "mxfp4_triton"
    H, I = hidden_size, div_even(moe_intermediate_size, tp_size)
    E = num_experts
    if expert_quant == "none":
        per_expert = (2 * I * H + H * I) * 2  # bf16 gate_up [2I, H] + down [H, I]
    elif expert_quant == "fp8_block":
        B = 128
        per_expert = (2 * I * H + H * I) * 1 + (
            (2 * I // B) * (H // B) + (H // B) * (I // B)
        ) * 2
    elif expert_quant == "q4_0":
        # packed 4-bit blocks: 18 bytes per 32 elements, per output row
        per_expert = 2 * I * (H // 32 * 18) + H * (I // 32 * 18)
    elif expert_quant == "gguf_kq":
        # Native GGUF K-quant banks: Q4_K gate_up + Q5_K down (the dominant real
        # layout; UD stragglers requant onto Q5_K at load, and other mixes shift
        # the estimate by well under a slot-count rounding step). K-quant rows
        # occupy whole super-blocks, so round dims up to the 256-element block.
        from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q5_K, row_bytes

        def _rb(n: int, ty: int) -> int:
            return row_bytes(((n + 255) // 256) * 256, ty)

        per_expert = 2 * I * _rb(H, GGML_Q4_K) + H * _rb(I, GGML_Q5_K)
    elif expert_quant == "nvfp4":
        globals_bytes = (2 * I + H) * 2 if nvfp4_native else 0
        per_expert = 2 * I * (H // 2 + H // 16) + H * (I // 2 + I // 16) + globals_bytes
    elif expert_quant == "mxfp4_triton":
        # bias banks are bf16 (2 B/row-element); packed blocks and e8m0 scales are 1 B
        per_expert = (
            2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2)
        )
    elif expert_quant == "ds_fp4":
        per_expert = 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32)
    else:
        raise ValueError(f"no host-bank size estimate for expert_quant={expert_quant!r}")
    return per_expert * E


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    quant_format: str,
    gpu_budget: int | None = None,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    Applies memory_ratio to the persisted pre-model baseline exactly once (or, with
    ``gpu_budget`` set -- the --gpu-mem absolute budget -- takes weights + fixed out of
    the absolute cap instead), then defers the MoE-vs-KV split to plan_cache_budget. The
    (1-memory_ratio) remainder is the CUDA-graph/activation headroom (not subtracted
    here); under an absolute budget graphs/activations sit on top of the cap.
    """
    if gpu_budget is not None:
        budget_bytes = absolute_pool_budget_bytes(gpu_budget, weights_bytes + fixed_cache_size)
        if budget_bytes <= 0:
            raise ValueError(
                f"--gpu-mem budget ({gpu_budget} B) is exhausted by the resident weights "
                f"({weights_bytes} B) + fixed cache ({fixed_cache_size} B); raise the "
                "budget or free GPU memory"
            )
    else:
        budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = 992 if quant_format == "nvfp4_marlin" else total_experts
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )
