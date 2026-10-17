"""CPU MoE executor -- native GGUF K-quant experts (``gguf_kq`` banks, the qwen35moe
GGUF checkpoint family; Q4_K gate_up + Q5_K down is the dominant unsloth UD mix).

The CPU W4A8 GEMV (``q4_k_dot_i8`` / ``q5_k_dot_i8`` in csrc/cpu_moe/cpu_moe_ext.cpp)
reads the *same* packed banks the GPU offload path streams and dequantizes the
super-blocks inside the K-loop, pairing them with Q8_0 activations like the Q4_0 path
(the Q8_0 block granularity coincides with the K-quant 32-elem sub-block). Checked
against the reference dequant (models/gguf/dequant.py) + the bf16 GPU decode kernel,
and against the GPU ggml MMVQ kernel on byte-identical banks (both W4A8 there, so the
spread is just activation-quant noise). Part 3 covers CUDA-graph capture/replay.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_SB_BYTES = {"q4_k": 144, "q5_k": 176}
_GGML = {"q4_k": 12, "q5_k": 13}  # GGML_Q4_K / GGML_Q5_K


def _rand_kq_rows(S: int, OUT: int, K: int, sb_bytes: int, seed: int = 0) -> torch.Tensor:
    """Random-but-valid packed K-quant rows ``[S, OUT, K//256*sb_bytes]``: every bit
    pattern decodes, so only the fp16 (d, dmin) header needs sanitizing (random bytes
    could encode inf/nan scales). Exercises the six-bit scale packing adversarially."""
    g = torch.Generator().manual_seed(seed)
    n_sb = K // 256
    row = torch.randint(0, 256, (S, OUT, n_sb * sb_bytes), dtype=torch.uint8, generator=g)
    row = row.reshape(S, OUT, n_sb, sb_bytes)

    def _f16_bytes(t):
        return t.to(torch.float16).unsqueeze(-1).view(torch.uint8).squeeze(-1)

    row[..., 0:2] = _f16_bytes(0.02 + 0.04 * torch.rand(S, OUT, n_sb, generator=g))
    row[..., 2:4] = _f16_bytes(0.002 + 0.004 * torch.rand(S, OUT, n_sb, generator=g))
    return row.reshape(S, OUT, n_sb * sb_bytes).contiguous()


def _make_gguf_kq_cache(L, E, H, I, gu="q4_k", dn="q5_k", seed=0):
    """Random gguf_kq banks for the cpu backend (pinned host tensors)."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    S = L * E

    def rows(OUT, K, qtype, tag):
        packed = _rand_kq_rows(S, OUT, K, _SB_BYTES[qtype], seed=seed + tag)
        pinned = alloc_pinned_tensor(*packed.shape, dtype=torch.uint8)
        pinned.copy_(packed)
        return pinned

    return SimpleNamespace(
        quant_format="gguf_kq",
        bank_sources={
            "gate_up": list(rows(2 * I, H, gu, 1).split(E)),
            "down": list(rows(H, I, dn, 2).split(E)),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


def _dequant_rows(packed: torch.Tensor, K: int, qtype: str, dev) -> torch.Tensor:
    from freetoken.models.gguf.dequant import dequantize

    flat = dequantize(packed.reshape(-1), _GGML[qtype], torch.bfloat16)
    return flat.reshape(*packed.shape[:2], K).to(dev)


@pytest.mark.parametrize("gu,dn", [("q4_k", "q5_k"), ("q5_k", "q5_k")])
@pytest.mark.parametrize("bs", [1, 4])
def test_cpu_decode_gguf_kq_matches_dequant_then_gpu(gu, dn, bs):
    """CPU inline-dequant K-quant GEMV vs. canonical dequant + bf16 GPU decode."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(400 + bs)
    L, E, H, I, top_k = 2, 8, 1024, 256, 8  # H, I % 256 == 0 (super-block tiling)
    layer = 1
    dev = torch.device("cuda")
    cache = _make_gguf_kq_cache(L, E, H, I, gu=gu, dn=dn)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )
    assert "+kq-w4a8" in ex.isa

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gate_up_layer = _dequant_rows(b["gate_up"][layer], H, gu, dev)  # [E, 2I, H]
    down_layer = _dequant_rows(b["down"][layer], I, dn, dev)        # [E, H, I]
    gpu_out = fused_experts_decode_impl(
        hidden, gate_up_layer, down_layer, w, ids.clone(), "silu", False
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 3e-2, f"gguf_kq ({gu}/{dn}) bs={bs} rel err {rel.item()}"


@pytest.mark.slow
def test_cpu_decode_gguf_kq_matches_ggml_mmvq():
    """Sanity: the CPU W4A8 K-quant GEMV lands close to the GPU ggml MMVQ kernel the
    offload path uses -- both quantize activations to int8, so only the independent
    Q8_0 roundings differ."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_kq

    torch.manual_seed(77)
    L, E, H, I, top_k = 2, 8, 1024, 256, 8
    layer, bs = 1, 4
    dev = torch.device("cuda")
    cache = _make_gguf_kq_cache(L, E, H, I, gu="q4_k", dn="q5_k")

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu",
        apply_router_weight_on_input=False, num_threads=0, max_tokens=bs, device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gpu_out = fused_experts_gguf_kq(
        hidden, b["gate_up"][layer].to(dev), b["down"][layer].to(dev), w, ids.clone(),
        "silu", hidden_size=H, intermediate_size=I,
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 6e-2, f"gguf_kq vs ggml-mmvq rel err {rel.item()}"


def test_cpu_moe_decode_gguf_kq_cuda_graph_replay():
    """K-quant CPU path under capture/replay: the host nodes must recompute the GEMV
    from the freshly written pinned routing on each replay."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(9)
    L, E, H, I, top_k = 2, 8, 1024, 256, 8
    layer, bs = 1, 4
    cache = _make_gguf_kq_cache(L, E, H, I, gu="q4_k", dn="q5_k")

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu",
        apply_router_weight_on_input=False, num_threads=8, max_tokens=bs, device=dev,
    )
    b = cache.bank_sources
    gate_up_layer = _dequant_rows(b["gate_up"][layer], H, "q4_k", dev)
    down_layer = _dequant_rows(b["down"][layer], I, "q5_k", dev)

    def reference(hidden, ids, w):
        return fused_experts_decode_impl(
            hidden, gate_up_layer, down_layer, w, ids.clone(), "silu", False
        ).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    ex.decode(layer, hidden, w, ids)  # eager warmup: materialize buffers + task
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    for it in range(3):
        torch.manual_seed(600 + it)
        hidden.copy_(torch.randn(bs, H, dtype=torch.bfloat16) * 0.5)
        ids.copy_(torch.randint(0, E, (bs, top_k), dtype=torch.int32))
        w.copy_(torch.rand(bs, top_k, dtype=torch.float32))
        g.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, ids, w)
        rel = (out_cap.float() - ref).abs().max() / (ref.abs().max() + 1e-6)
        assert rel < 3e-2, f"gguf_kq replay {it} rel err {rel.item()}"


if __name__ == "__main__":
    for gu, dn in (("q4_k", "q5_k"), ("q5_k", "q5_k")):
        for bs in (1, 4):
            test_cpu_decode_gguf_kq_matches_dequant_then_gpu(gu, dn, bs)
            print(f"gguf_kq ({gu}/{dn}) bs={bs} OK")
    test_cpu_decode_gguf_kq_matches_ggml_mmvq()
    print("gguf_kq vs ggml-mmvq OK")
    test_cpu_moe_decode_gguf_kq_cuda_graph_replay()
