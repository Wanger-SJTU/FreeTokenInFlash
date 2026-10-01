"""qwen35moe GGUF adapter: end-to-end over a synthesized mini checkpoint.

Builds a tiny llama.cpp-style ``qwen35moe`` GGUF (8 layers, 6 GDN + 2 full, 8
experts) with the unsloth-UD shape of quant types -- Q8_0 dense projections, a
Q6_K lm_head, Q4_K gate/up experts and a Q5_K/Q6_K down mix -- and checks the
config parse, the weight mapping (fused qkv / GDN in_proj split / norm
passthrough), the strict model load, the K-quant expert banks (including the
Q6_K straggler requant onto Q5_K) and the fused MoE kernel against a torch
reference. The test-local packers are only required to round-trip through
freetoken's (gguf-py/CUDA-verified) dequantizers, not to match ggml's optimal
quantizers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

H = 256          # hidden (K-quant super-block aligned)
HEAD = 64        # full-attention head dim (rope kernel supports {64,128,256,512})
N_Q, N_KV = 4, 2
K_HEADS, V_HEADS = 2, 4          # GDN
KEY_DIM, VALUE_DIM = K_HEADS * HEAD, V_HEADS * HEAD  # 64, 128
CONV_DIM = 2 * KEY_DIM + VALUE_DIM                    # 256
E, TOPK, I = 8, 2, 256
VOCAB, L, INTERVAL = 512, 8, 4
DOWNS_Q6 = {6}  # layer ids whose down experts are Q6_K stragglers (rest Q5_K)


# --------------------------------------------------------------------------- test quantizers

def _pack_q8_0(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, 32).float()
    d = (x.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    q = (x / d).round().clamp(-127, 127).to(torch.int8)
    return torch.cat([d.half().view(torch.uint8), q.view(torch.uint8)], dim=1).reshape(-1)


def _pack_kq_4bit(x: torch.Tensor) -> torch.Tensor:
    """Q4_K blocks: w = d*sc*q - dmin*m, 4-bit q (joint-scale scheme, like quantize_q5_k).

    Byte (g, b) holds element g*64+b in its low nibble and g*64+32+b in its high nibble.
    """
    from freetoken.models.gguf.dequant import _pack_k_scales

    x = x.reshape(-1, 256).float()
    n = x.shape[0]
    sb = x.reshape(n, 8, 32)
    lo = sb.min(dim=2).values.clamp(max=0.0)
    hi = sb.max(dim=2).values.clamp(min=0.0)
    span = (hi - lo) / 15.0
    off = -lo
    d = (span.max(dim=1).values / 63.0).clamp(min=1e-6).to(torch.float16)
    dmin = (off.max(dim=1).values / 63.0).clamp(min=1e-6).to(torch.float16)
    sc = (span / d.float().unsqueeze(1)).ceil().clamp(0, 63)
    m = (off / dmin.float().unsqueeze(1)).ceil().clamp(0, 63)
    step = (d.float().unsqueeze(1) * sc.float()).clamp(min=1e-30)
    base = dmin.float().unsqueeze(1) * m.float()
    q = ((sb + base.unsqueeze(2)) / step.unsqueeze(2)).round().clamp(0, 15).to(torch.int64)
    return _finish_kq_4bit(x, n, q, d, dmin, sc, m, _pack_k_scales, 144)


def _finish_kq_4bit(x, n, q, d, dmin, sc, m, pack_scales, block_bytes):
    out = torch.empty(n, block_bytes, dtype=torch.uint8)
    out[:, 0:2] = d.view(torch.uint8).reshape(n, 2)
    out[:, 2:4] = dmin.view(torch.uint8).reshape(n, 2)
    out[:, 4:16] = pack_scales(sc.to(torch.int64), m.to(torch.int64))
    qf = q.reshape(n, 256)
    f = torch.arange(256)
    byte = (f // 64) * 32 + (f % 32)
    low = torch.where((f % 64) < 32, qf & 0xF, torch.zeros_like(qf))
    high = torch.where((f % 64) >= 32, qf & 0xF, torch.zeros_like(qf))
    qs = torch.zeros(n, 128, dtype=torch.int64)
    qs.scatter_add_(1, byte.unsqueeze(0).expand(n, -1), low)
    qs.scatter_add_(1, byte.unsqueeze(0).expand(n, -1), high << 4)
    out[:, 16:block_bytes] = qs.to(torch.uint8)
    return out.reshape(-1)


def _pack_q6_k(x: torch.Tensor) -> torch.Tensor:
    """Q6_K blocks: w = sc_j * (q - 32) with fp16 d = 1.0 and 16 int8 sub-scales.

    Element e (h = e//128, l = e%32, sub = (e%128)//32): low nibble in ql[h*64 +
    (sub%2)*32 + l] when sub < 2, high nibble when sub >= 2; code bits 4-5 in
    qh[h*32 + l] at bit pair 2*sub.
    """
    x = x.reshape(-1, 256).float()
    n = x.shape[0]
    sb = x.reshape(n, 16, 16)
    sc = (sb.abs().amax(dim=2) / 31.0).round().clamp(1, 127)
    q = (sb / sc.unsqueeze(-1)).round().clamp(-32, 31).to(torch.int64) + 32  # [n,16,16]

    qf = q.reshape(n, 256)
    f = torch.arange(256)
    h, l, sub = f // 128, f % 32, (f % 128) // 32
    ql = torch.zeros(n, 128, dtype=torch.int64)
    qh = torch.zeros(n, 64, dtype=torch.int64)  # two 32-byte halves (h = e // 128)
    nib = qf & 0xF
    ql_pos = (h * 64 + (sub % 2) * 32 + l).unsqueeze(0).expand(n, -1)
    shifted = torch.where(sub >= 2, nib << 4, nib)  # [n, 256]
    ql.scatter_add_(1, ql_pos, shifted)
    qh_bits = ((qf >> 4) & 3) << (2 * sub).unsqueeze(0)
    qh.scatter_add_(1, (h * 32 + l).unsqueeze(0).expand(n, -1), qh_bits)

    out = torch.empty(n, 210, dtype=torch.uint8)
    out[:, 0:128] = ql.to(torch.uint8)
    out[:, 128:192] = qh.to(torch.uint8)
    out[:, 192:208] = sc.to(torch.int8).view(torch.uint8).reshape(n, 16)
    out[:, 208:210] = torch.ones(n, 1, dtype=torch.float16).view(torch.uint8).reshape(n, 2)
    return out.reshape(-1)


# --------------------------------------------------------------------------- fixture build

def _build_gguf(path: str):
    import gguf

    w = gguf.GGUFWriter(path, "qwen35moe")
    w.add_uint32("qwen35moe.block_count", L)
    w.add_uint32("qwen35moe.embedding_length", H)
    w.add_uint32("qwen35moe.attention.head_count", N_Q)
    w.add_uint32("qwen35moe.attention.head_count_kv", N_KV)
    w.add_uint32("qwen35moe.attention.key_length", HEAD)
    w.add_uint32("qwen35moe.attention.value_length", HEAD)
    w.add_float32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32("qwen35moe.context_length", 512)
    w.add_uint32("qwen35moe.expert_count", E)
    w.add_uint32("qwen35moe.expert_used_count", TOPK)
    w.add_uint32("qwen35moe.expert_feed_forward_length", I)
    w.add_uint32("qwen35moe.expert_shared_feed_forward_length", I)
    w.add_uint32("qwen35moe.full_attention_interval", INTERVAL)
    w.add_uint32("qwen35moe.rope.dimension_count", 16)
    w.add_float32("qwen35moe.rope.freq_base", 1_000_000.0)
    w.add_uint32("qwen35moe.ssm.conv_kernel", 4)
    w.add_uint32("qwen35moe.ssm.group_count", K_HEADS)
    w.add_uint32("qwen35moe.ssm.inner_size", VALUE_DIM)
    w.add_uint32("qwen35moe.ssm.state_size", HEAD)
    w.add_uint32("qwen35moe.ssm.time_step_rank", V_HEADS)

    torch.manual_seed(7)
    fp32: dict[str, torch.Tensor] = {}

    def put(name: str, tensor: torch.Tensor, raw: torch.Tensor, qtype):
        # the packed shape mirrors llama.cpp: [out, rb] (2D) or [E, rows, rb] (3D experts)
        rb = raw.numel() // (tensor.numel() // tensor.shape[-1])
        w.add_tensor(name, raw.numpy().reshape(*tensor.shape[:-1], rb), raw_dtype=qtype)
        fp32[name] = tensor

    def q8(name, out, inn):
        t = torch.randn(out, inn) * 0.1
        put(name, t, _pack_q8_0(t), gguf.GGMLQuantizationType.Q8_0)

    def f32(name, t: torch.Tensor):
        w.add_tensor(name, t.float().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
        fp32[name] = t.float()

    q8("token_embd.weight", VOCAB, H)
    t = torch.randn(VOCAB, H) * 0.1
    put("output.weight", t, _pack_q6_k(t), gguf.GGMLQuantizationType.Q6_K)
    f32("output_norm.weight", torch.randn(H) + 1.0)

    from freetoken.models.gguf.dequant import quantize_q5_k

    for layer in range(L):
        f32(f"blk.{layer}.attn_norm.weight", torch.randn(H) + 1.0)
        f32(f"blk.{layer}.post_attention_norm.weight", torch.randn(H) + 1.0)
        f32(f"blk.{layer}.ffn_gate_inp.weight", torch.randn(E, H))
        f32(f"blk.{layer}.ffn_gate_inp_shexp.weight", torch.randn(H))
        q8(f"blk.{layer}.ffn_gate_shexp.weight", I, H)
        q8(f"blk.{layer}.ffn_up_shexp.weight", I, H)
        q8(f"blk.{layer}.ffn_down_shexp.weight", H, I)
        for part in ("gate", "up"):  # Q4_K routed experts
            t = torch.randn(E, I, H) * 0.1
            put(f"blk.{layer}.ffn_{part}_exps.weight", t,
                _pack_kq_4bit(t), gguf.GGMLQuantizationType.Q4_K)
        t = torch.randn(E, H, I) * 0.1
        if layer in DOWNS_Q6:
            put(f"blk.{layer}.ffn_down_exps.weight", t,
                _pack_q6_k(t), gguf.GGMLQuantizationType.Q6_K)
        else:
            put(f"blk.{layer}.ffn_down_exps.weight", t,
                quantize_q5_k(t.reshape(-1)), gguf.GGMLQuantizationType.Q5_K)
        if (layer + 1) % INTERVAL == 0:  # full attention
            q8(f"blk.{layer}.attn_q.weight", N_Q * HEAD * 2, H)
            q8(f"blk.{layer}.attn_k.weight", N_KV * HEAD, H)
            q8(f"blk.{layer}.attn_v.weight", N_KV * HEAD, H)
            q8(f"blk.{layer}.attn_output.weight", H, N_Q * HEAD)
            f32(f"blk.{layer}.attn_q_norm.weight", torch.randn(HEAD) + 1.0)
            f32(f"blk.{layer}.attn_k_norm.weight", torch.randn(HEAD) + 1.0)
        else:  # GDN
            q8(f"blk.{layer}.attn_qkv.weight", CONV_DIM, H)
            q8(f"blk.{layer}.attn_gate.weight", VALUE_DIM, H)
            q8(f"blk.{layer}.ssm_out.weight", H, VALUE_DIM)
            f32(f"blk.{layer}.ssm_alpha.weight", torch.randn(V_HEADS, H))
            f32(f"blk.{layer}.ssm_beta.weight", torch.randn(V_HEADS, H))
            f32(f"blk.{layer}.ssm_conv1d.weight", torch.randn(CONV_DIM, 4))
            f32(f"blk.{layer}.ssm_dt.bias", torch.randn(V_HEADS))
            a = -torch.rand(V_HEADS).exp()  # A = -exp(A_log), strictly negative
            f32(f"blk.{layer}.ssm_a", a)
            fp32[f"blk.{layer}.A_log"] = torch.log(-a)
            f32(f"blk.{layer}.ssm_norm.weight", torch.randn(HEAD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return fp32


@pytest.fixture(scope="module")
def mini_gguf(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("qwen35moe") / "mini.gguf")
    return path, _build_gguf(path)


@pytest.fixture(scope="module")
def parsed(mini_gguf):
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config

    path, _ = mini_gguf
    return parse_gguf_config(build_gguf_shim(path))


@pytest.fixture(autouse=True)
def _tp1():
    from freetoken.distributed import get_tp_info, set_tp_info

    try:
        info = get_tp_info()
        if info.rank == 0 and info.size == 1:
            return
    except RuntimeError:
        pass
    set_tp_info(rank=0, size=1)


def test_config_geometry(parsed):
    assert parsed.num_layers == L and parsed.hidden_size == H
    assert (parsed.num_qo_heads, parsed.num_kv_heads, parsed.head_dim) == (N_Q, N_KV, HEAD)
    assert parsed.rotary_config.rotary_dim == 16 and parsed.rotary_config.scaling is None
    groups = {g.kind: g for g in parsed.attention_groups}
    assert groups["full"].layer_ids == tuple(i for i in range(L) if (i + 1) % INTERVAL == 0)
    lin = groups["linear_gated_delta"]
    assert (lin.num_key_heads, lin.num_value_heads) == (K_HEADS, V_HEADS)
    assert (lin.key_head_dim, lin.value_head_dim, lin.conv_kernel_dim) == (HEAD, HEAD, 4)
    assert (parsed.num_experts, parsed.num_experts_per_tok, parsed.moe_intermediate_size) == (E, TOPK, I)
    assert parsed.expert_quant == "gguf_kq" and not parsed.tie_word_embeddings
    assert dict(parsed.gguf_quant_types)["lm_head"] == 14  # Q6_K
    assert dict(parsed.gguf_quant_types)["attn_qkv"] == 8  # Q8_0


def _rows_raw(path, name):
    from freetoken.models.gguf.reader import _reader

    for t in _reader(path).tensors:
        if t.name == name:
            return torch.from_numpy(np.ascontiguousarray(t.data).reshape(-1).view(np.uint8))
    raise KeyError(name)


def test_weight_mapping_and_load(mini_gguf, parsed):
    import dataclasses

    torch.set_default_dtype(torch.bfloat16)
    path, fp32 = mini_gguf
    from freetoken.models.gguf.dequant import dequantize
    from freetoken.models.register import get_model_class
    from freetoken.models.weight import load_weight

    cfg = dataclasses.replace(parsed, moe_backend="offload")
    model = get_model_class(cfg.architectures[0], cfg)
    ms = model.state_dict()
    sd = {k: w.to(dtype=ms[k].dtype) for k, w in load_weight(path, torch.device("cpu"),
                                                             include_moe_experts=False)}
    assert set(sd) == set(ms)
    model.load_state_dict(sd)  # strict

    # Fused GDN in_proj_qkvz computes dequantized [qkv | z] rows.
    layer = model.model.layers.op_list[0]
    want = torch.cat([
        dequantize(_rows_raw(path, "blk.0.attn_qkv.weight"), 8, torch.float32).view(CONV_DIM, H),
        dequantize(_rows_raw(path, "blk.0.attn_gate.weight"), 8, torch.float32).view(VALUE_DIM, H),
    ])
    x = torch.randn(3, H, dtype=torch.bfloat16)
    if torch.cuda.is_available():  # packed ggml kernel path (the serving path)
        from freetoken.layers.gguf import fused_mul_mat_gguf

        got = fused_mul_mat_gguf(
            x.cuda(), layer.linear_attn.in_proj_qkvz.qweight.cuda(), 8
        ).float().cpu()
    else:  # dequantize fallback path of the same layer
        from freetoken.models.gguf.dequant import dequant_q8_0

        q = layer.linear_attn.in_proj_qkvz.qweight
        dense = dequant_q8_0(q, torch.float32).reshape(-1, H)
        got = x.float() @ dense.T
    torch.testing.assert_close(got, x.float() @ want.T, rtol=0.05, atol=0.05)
    # ssm_a -> A_log mapping
    torch.testing.assert_close(layer.linear_attn.A_log, fp32["blk.0.A_log"], rtol=1e-5, atol=1e-6)
    # llama.cpp baked the (1+w) Gemma norms; the loader passes them through verbatim
    torch.testing.assert_close(
        layer.input_layernorm.weight.float(), fp32["blk.0.attn_norm.weight"], rtol=1e-2, atol=1e-2)


def test_expert_banks_and_requant(mini_gguf, parsed):
    path, fp32 = mini_gguf
    from freetoken.models.gguf.dequant import dequant_q5_k
    from freetoken.models.qwen3_5_moe.gguf import load_gguf_kq_expert_sources

    banks = load_gguf_kq_expert_sources(path, parsed)
    gu, dn = banks["gate_up"], banks["down"]
    assert len(gu) == len(dn) == L
    assert tuple(gu[0].shape) == (E, 2 * I, 144)    # Q4_K row bytes over H=256
    assert tuple(dn[0].shape) == (E, H, 176)        # Q5_K row bytes over I=256
    # gate|up concatenation is byte-exact (same type)
    gate = _rows_raw(path, "blk.0.ffn_gate_exps.weight").reshape(E, I, 144)
    up = _rows_raw(path, "blk.0.ffn_up_exps.weight").reshape(E, I, 144)
    assert torch.equal(gu[0][:, :I], gate) and torch.equal(gu[0][:, I:], up)
    # straggler Q6_K down layer requantized onto Q5_K: within Q5_K quantization error
    ref = fp32["blk.6.ffn_down_exps.weight"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    back = dequant_q5_k(dn[6][3].to(dev), torch.float32).cpu().reshape(H, I)
    span = (ref[3].max(dim=1).values - ref[3].min(dim=1).values)
    assert bool(((back - ref[3]).abs() <= span.unsqueeze(1) / 31 + 1e-2).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_moe_kernel_vs_torch(mini_gguf, parsed):
    path, fp32 = mini_gguf
    from freetoken.models.gguf.dequant import dequant_q4_k, dequant_q5_k
    from freetoken.models.qwen3_5_moe.gguf import load_gguf_kq_expert_sources
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_kq

    banks = load_gguf_kq_expert_sources(path, parsed)
    gate_up = banks["gate_up"][0][:4].cuda()  # [4, 2I, rb]
    down = banks["down"][0][:4].cuda()

    torch.manual_seed(3)
    x = torch.randn(4, H, dtype=torch.bfloat16, device="cuda")
    ids = torch.arange(4, device="cuda", dtype=torch.int32).view(4, 1)
    weights = torch.ones(4, 1, device="cuda", dtype=torch.bfloat16)
    out = fused_experts_gguf_kq(x, gate_up, down, weights, ids, "silu",
                                hidden_size=H, intermediate_size=I)

    ref = torch.zeros(4, H, device="cuda")
    for e in range(4):
        gu = dequant_q4_k(gate_up[e], torch.float32).reshape(2 * I, H)
        dn_ = dequant_q5_k(down[e], torch.float32).reshape(H, I)
        gate = x[e].float() @ gu[:I].T
        up = x[e].float() @ gu[I:].T
        inter = torch.nn.functional.silu(gate) * up
        ref[e] = inter @ dn_.T
    torch.testing.assert_close(out.float(), ref, rtol=0.05, atol=0.05)
