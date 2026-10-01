"""Qwen3.5/3.6 MoE (hybrid GDN + full attention) GGUF adapter.

Reads a llama.cpp ``qwen35moe`` GGUF (e.g. unsloth's dynamically-quantized
``Qwen3.6-35B-A3B-UD-Q4_K_M``) natively: dense projections (attention q/k/v/o,
GDN qkv/z/out, shared expert) stay in their packed block layout and compute
inside the borrowed ggml kernels; the routed experts stream to the offload
cache as packed K-quant banks (``expert_quant="gguf_kq"``).

Two conventions matter vs the HF loader:

* Norms: llama.cpp's conversion BAKES the Gemma-style ``(1 + w)`` into the stored
  norm weights (values center on 1.0, not 0), so -- unlike the HF path -- nothing
  is added at load. The GDN gated norm is plain ``w * x`` either way.
* ``ssm_a`` carries FLA's ``A = -exp(A_log)`` (strictly negative); the model keeps
  ``A_log``, so ``A_log = log(-a)``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import (
    GGML_NAME,
    GGML_Q4_K,
    GGML_Q5_K,
    dequant_q4_k,
    dequant_q5_k,
    dequant_q6_k,
    dequantize,
    quantize_q5_k,
    row_bytes,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

# Routed-expert GGUF tensors (loaded into the offload banks, not iter_gguf_weights).
_EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
_LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# Dense-projection roles: (gguf suffix, role) for the per-role quant-type scan. Every
# layer must agree on a role's type (llama.cpp quantizes per tensor-role uniformly;
# per-layer variation only happens for the routed experts).
_ROLE_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("attn_q.weight", "attn_qkv"), ("attn_k.weight", "attn_qkv"),
    ("attn_v.weight", "attn_qkv"),
    ("attn_output.weight", "attn_o"),
    ("attn_qkv.weight", "gdn_qkvz"), ("attn_gate.weight", "gdn_qkvz"),
    ("ssm_out.weight", "gdn_out"),
    ("ffn_gate_shexp.weight", "shexp_gate_up"), ("ffn_up_shexp.weight", "shexp_gate_up"),
    ("ffn_down_shexp.weight", "shexp_down"),
)
_ROOT_ROLES = (("token_embd.weight", "embed"), ("output.weight", "lm_head"))
# Packed ops exist for the block-quants (+Q4_0); F32/F16 projections would need the
# dense path, which no qwen35moe GGUF on the ground uses for these tensors.
_PACKED_TYPES = (2, 8, 12, 13, 14)  # Q4_0, Q8_0, Q4_K, Q5_K, Q6_K


def _scan_quant_types(shim: "GgufConfigShim") -> tuple[tuple[str, int], ...]:
    """Per-role packed ggml types from the GGUF tensor table (uniform across layers)."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    roles: dict[str, int] = {}
    for t in iter_gguf_tensors(shim.model_path):
        role = None
        for root, r in _ROOT_ROLES:
            if t.name == root:
                role = r
        if role is None:
            lm = _LAYER_RE.match(t.name)
            if lm:
                suffix = lm.group(2)
                for suf, r in _ROLE_SUFFIXES:
                    if suffix == suf:
                        role = r
        if role is None:
            continue
        prev = roles.setdefault(role, t.ggml_type)
        if prev != t.ggml_type:
            raise ValueError(
                f"GGUF quant type for {role!r} varies across tensors "
                f"({GGML_NAME.get(prev, prev)} vs {GGML_NAME.get(t.ggml_type, t.ggml_type)}); "
                "FreeToken requires per-role uniformity for the dense projections"
            )
    missing = {"embed", "lm_head", "attn_qkv", "attn_o", "gdn_qkvz", "gdn_out",
               "shexp_gate_up", "shexp_down"} - set(roles)
    if missing:
        raise ValueError(f"GGUF is missing dense-projection tensors for roles: {sorted(missing)}")
    for role, ty in roles.items():
        if ty not in _PACKED_TYPES:
            raise ValueError(
                f"GGUF stores {role!r} as {GGML_NAME.get(ty, ty)}; only packed "
                "Q4_0/Q8_0/Q4_K/Q5_K/Q6_K dense projections are supported"
            )
    return tuple(sorted(roles.items()))


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str):
        val = m.get(f"qwen35moe.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key qwen35moe.{key}")
        return val

    num_layers = int(g("block_count"))
    num_q = int(g("attention.head_count"))
    num_kv = int(g("attention.head_count_kv"))
    head_dim = int(g("attention.key_length"))
    assert int(g("attention.value_length")) == head_dim, "Qwen3.5 requires k/v head dims equal"

    # Every full_attention_interval-th layer (1-indexed) is full attention; the rest GDN.
    interval = int(g("full_attention_interval"))
    full_ids = tuple(i for i in range(num_layers) if (i + 1) % interval == 0)
    linear_ids = tuple(i for i in range(num_layers) if (i + 1) % interval != 0)

    # Partial rope: llama.cpp writes rope.dimension_count for the rotated width. The
    # mrope dimension_sections reduce to plain partial rope for text-only serving
    # (same as the HF path, which drops the scaling dict for the default rope type).
    rotary_dim = int(g("rope.dimension_count"))
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(g("context_length")),
        base=float(g("rope.freq_base")),
        scaling=None,
    )

    # GDN geometry from the ssm.* metadata (llama.cpp semantics: state_size = head
    # dim, inner_size = total value dim, group_count = key heads; k/v head dims must
    # be equal for the fla kernels). The load asserts these against tensor shapes.
    def s(key: str) -> float:
        val = m.get(f"qwen35moe.ssm.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key qwen35moe.ssm.{key}")
        return float(val)

    head_k = head_v = int(s("state_size"))
    num_v = int(s("inner_size")) // head_v
    num_k = int(s("group_count"))
    conv = int(s("conv_kernel"))

    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=num_k,
        num_value_heads=num_v,
        key_head_dim=head_k,
        value_head_dim=head_v,
        conv_kernel_dim=conv,
        output_gate=True,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        rotary_config=rotary,
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=int(g("embedding_length")),
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(g("expert_shared_feed_forward_length")),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        norm_topk_prob=True,
        model_type="qwen3_5_moe",
        architectures=list(shim.architectures),
        moe_enabled=True,
        attention_groups=tuple(
            sorted(
                (full_group, linear_group),
                key=lambda x: x.layer_ids[0] if x.layer_ids else 1 << 30,
            )
        ),
        # Routed experts: packed GGUF K-quant banks served by the offload cache; dense
        # projections / shared expert / lm_head: native packed GGUF in GGUFLinear ops.
        expert_quant="gguf_kq",
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
        moe_weight_format="gguf_kq",
        use_qk_norm=True,
        gguf_quant_types=_scan_quant_types(shim),
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken qwen3_5_moe state-dict keys.
# --------------------------------------------------------------------------------------


def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q*) to a dense bf16 tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _require_tp1(what: str) -> None:
    """GGUF quant layers / expert banks are not sharded; reject TP>1 with a clear error
    (mirrors the gemma4 GGUF and the HF qwen3_5_moe loaders' TP=1 restriction)."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"qwen3_5_moe GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert qwen3_5_moe param.

    Quantized projections (attention q/k/v/o, GDN qkv/z/out, shared-expert MLP) and
    the embedding / untied lm_head stay in their native packed block layout (yielded
    as ``.qweight`` uint8); norms / router / GDN gate params dequantize to bf16 or
    stay fp32 (``A_log`` / ``dt_bias``). Fusions concatenate packed rows along the
    output dim (shared input dim -> shared ``row_bytes``): q|k|v -> ``qkv_proj``,
    GDN qkv|z -> ``in_proj_qkvz``, b|a -> ``in_proj_ba`` (F32 in the GGUF, bf16
    GEMM), shared gate|up -> ``gate_up_proj``. Routed experts are served from the
    offload cache and skipped here.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    assert not include_moe_experts, (
        "qwen3_5_moe GGUF stores experts as packed K-quants and only supports the "
        "offload backend; experts are loaded via load_gguf_kq_expert_sources()."
    )
    assert include_non_moe
    _require_tp1("weight loading")

    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    qkvz_buf: dict[int, dict[str, torch.Tensor]] = {}
    shexp_buf: dict[int, dict[str, torch.Tensor]] = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output.weight":
            yield "lm_head.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if _EXPERT_RE.match(name):
            continue  # routed experts -> offload banks
        layer_match = _LAYER_RE.match(name)
        assert layer_match is not None, f"unmapped qwen35moe GGUF tensor: {name}"
        layer, suffix = int(layer_match.group(1)), layer_match.group(2)
        base = f"model.layers.{layer}"

        if suffix == "attn_norm.weight":
            yield f"{base}.input_layernorm.weight", _to_bf16(t)
        elif suffix == "post_attention_norm.weight":
            yield f"{base}.post_attention_layernorm.weight", _to_bf16(t)
        # ---- full-attention layers ----
        elif suffix == "attn_q_norm.weight":
            yield f"{base}.self_attn.q_norm.weight", _to_bf16(t)
        elif suffix == "attn_k_norm.weight":
            yield f"{base}.self_attn.k_norm.weight", _to_bf16(t)
        elif suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
            slots = qkv_buf.setdefault(layer, {})
            slots[suffix.removeprefix("attn_").removesuffix(".weight")] = t.packed()
            if "q" in slots and "k" in slots and "v" in slots:
                yield f"{base}.self_attn.qkv_proj.qweight", torch.cat(
                    [slots["q"], slots["k"], slots["v"]], dim=0
                )
                del qkv_buf[layer]
        elif suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.qweight", t.packed()
        # ---- GDN layers ----
        elif suffix in ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"):
            slots = qkvz_buf.setdefault(layer, {})
            slots[{"attn_qkv.weight": "qkv", "attn_gate.weight": "z",
                   "ssm_beta.weight": "b", "ssm_alpha.weight": "a"}[suffix]] = (
                t.packed() if "attn" in suffix else _to_bf16(t)
            )
            if all(k in slots for k in ("qkv", "z", "b", "a")):
                yield f"{base}.linear_attn.in_proj_qkvz.qweight", torch.cat(
                    [slots["qkv"], slots["z"]], dim=0
                )
                yield f"{base}.linear_attn.in_proj_ba.weight", torch.cat(
                    [slots["b"], slots["a"]], dim=0
                )
                del qkvz_buf[layer]
        elif suffix == "ssm_out.weight":
            yield f"{base}.linear_attn.out_proj.qweight", t.packed()
        elif suffix == "ssm_conv1d.weight":
            conv = t.packed().view(torch.float32).reshape(*t.shape)
            yield f"{base}.linear_attn.conv1d.weight", conv.unsqueeze(1)  # [conv_dim, 1, kernel]
        elif suffix == "ssm_dt.bias":
            yield f"{base}.linear_attn.dt_bias", t.packed().view(torch.float32).reshape(-1)
        elif suffix == "ssm_a":  # A = -exp(A_log) -> the model keeps A_log
            a = t.packed().view(torch.float32).reshape(-1)
            assert bool((a < 0).all()), f"{name}: ssm_a must be negative (A = -exp(A_log))"
            yield f"{base}.linear_attn.A_log", torch.log(-a)
        elif suffix == "ssm_norm.weight":
            yield f"{base}.linear_attn.norm.weight", _to_bf16(t)
        # ---- MoE (every layer here is MoE) ----
        elif suffix == "ffn_gate_inp.weight":
            yield f"{base}.mlp.gate.weight", _to_bf16(t)
        elif suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
        elif suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            slots = shexp_buf.setdefault(layer, {})
            slots["gate" if suffix.startswith("ffn_gate") else "up"] = t.packed()
            if "gate" in slots and "up" in slots:
                yield f"{base}.mlp.shared_expert.gate_up_proj.qweight", torch.cat(
                    [slots["gate"], slots["up"]], dim=0
                )
                del shexp_buf[layer]
        elif suffix == "ffn_down_shexp.weight":
            yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed()
        else:
            raise ValueError(f"unmapped qwen35moe GGUF tensor: {name}")

    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"
    assert not qkvz_buf, f"incomplete GDN in_proj groups: {sorted(qkvz_buf)}"
    assert not shexp_buf, f"incomplete shared-expert merges: {sorted(shexp_buf)}"


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a qwen35moe GGUF checkpoint."""
    return getattr(config, "moe_weight_format", None) == "gguf_kq"


# --------------------------------------------------------------------------------------
# Model layer swap: dense bf16 projections -> native GGUF-quant ops.
# --------------------------------------------------------------------------------------


def convert_qwen35moe_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace the dense projections with native GGUF ops.

    Quantized in the checkpoint -> swapped: attention qkv/o, GDN qkvz/out, the
    shared-expert MLP, the token embedding and the untied lm_head. The fused GDN
    ``in_proj`` [qkv|z|b|a] splits into a packed ``in_proj_qkvz`` GEMM + a bf16
    ``in_proj_ba`` GEMM (b/a are F32 in the GGUF and cannot ride a packed GEMM;
    the same split the fp8 path uses). Left dense: the router, the shared-expert
    gate, all RMSNorms, conv1d and the GDN gate params. Routed experts are served
    from the offload cache.
    """
    from freetoken.layers import LinearColParallelMerged
    from freetoken.layers.gguf import GGUFLMHead, GGUFEmbedding, GGUFLinear

    types = dict(config.gguf_quant_types)
    inner = model.model

    inner.embed_tokens = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=types["embed"],
    )

    for layer in inner.layers.op_list:
        if layer._is_linear:
            gdn = layer.linear_attn
            del gdn.in_proj  # fused [qkv|z|b|a] -> split modules (fp8-shaped path)
            gdn.in_proj_qkvz = GGUFLinear(
                config.hidden_size, gdn.conv_dim + gdn.value_dim, types["gdn_qkvz"]
            )
            gdn.in_proj_ba = LinearColParallelMerged(
                config.hidden_size, [gdn.num_v_heads, gdn.num_v_heads], has_bias=False
            )
            gdn.out_proj = GGUFLinear(gdn.value_dim, config.hidden_size, types["gdn_out"])
            gdn._split_proj = True
        else:
            attn = layer.self_attn
            attn.qkv_proj = GGUFLinear(
                config.hidden_size, sum(attn._qkv_split), types["attn_qkv"]
            )
            attn.o_proj = GGUFLinear(attn.qo_attn_dim, config.hidden_size, types["attn_o"])
        shared = layer.mlp.shared_expert
        shared.gate_up_proj = GGUFLinear(
            config.hidden_size, 2 * config.shared_expert_intermediate_size,
            types["shexp_gate_up"],
        )
        shared.down_proj = GGUFLinear(
            config.shared_expert_intermediate_size, config.hidden_size, types["shexp_down"]
        )

    assert not config.tie_word_embeddings, "qwen35moe GGUF carries an untied output head"
    model.lm_head = GGUFLMHead(config.vocab_size, config.hidden_size, types["lm_head"])


# --------------------------------------------------------------------------------------
# Routed-expert host banks (packed K-quants) for the offload cache.
# --------------------------------------------------------------------------------------

# Straggler requant: source ggml type -> dequantizer. Layers whose down projection
# was quantized differently (Unsloth "UD" mixes, e.g. Q6_K among Q5_K) are requantized
# onto the majority type so every bank shares one shape -- the offload cache's
# uniform-layout contract. Requant is implemented onto Q5_K only.
_DEQUANT_KQ = {12: dequant_q4_k, 13: dequant_q5_k, 14: dequant_q6_k}
_REQUANT_TARGET = GGML_Q5_K
# Requant chunk: rows per iteration (bounds the fp32 intermediate to ~64 MiB).
_REQUANT_CHUNK_ROWS = 32_768


def _kq_bank_specs(config: ModelConfig, gu_type: int, down_type: int) -> dict:
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    return {
        "gate_up": ((E, 2 * I, row_bytes(H, gu_type)), torch.uint8),
        "down": ((E, H, row_bytes(I, down_type)), torch.uint8),
    }


def _majority_type(types: dict[int, int], what: str) -> tuple[int, set[int]]:
    counts: dict[int, int] = {}
    for ty in types.values():
        counts[ty] = counts.get(ty, 0) + 1
    best = max(counts, key=lambda k: counts[k])
    stragglers = {t for t in counts if t != best}
    for t in counts:
        assert t in _DEQUANT_KQ, (
            f"{what} experts use {GGML_NAME.get(t, t)}, which has no dequantizer; "
            "supported: Q4_K/Q5_K/Q6_K"
        )
    return best, stragglers


def _requant_to_q5_k(t, target_rows_bytes: int) -> torch.Tensor:
    """Requantize a whole packed expert tensor onto Q5_K rows (straggler alignment)."""
    flat = t.packed().reshape(-1)
    out = torch.empty(t.rows * target_rows_bytes, dtype=torch.uint8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for start in range(0, t.rows, _REQUANT_CHUNK_ROWS):
        end = min(start + _REQUANT_CHUNK_ROWS, t.rows)
        raw = flat[start * t.row_bytes : end * t.row_bytes].to(dev)
        vals = _DEQUANT_KQ[t.ggml_type](raw, torch.float32)
        blocks = quantize_q5_k(vals).reshape(-1).to(out.device)
        out[start * target_rows_bytes : end * target_rows_bytes] = blocks
    return out.reshape(t.rows, target_rows_bytes)


def load_gguf_kq_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None,
    layer_backing: list[str] | None = None, spill_dir: str | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' native K-quant block bytes.

    ``gate_up`` is one ``[E, 2I, row_bytes(H, qtype)]`` tensor per layer (gate|up
    concatenated along the output dim; both parts must share one quant type), ``down``
    one ``[E, H, row_bytes(I, qtype)]`` per layer. Down-projection layers quantized
    differently from the majority (Unsloth UD mixes) are requantized onto Q5_K so
    every bank shares one shape. gate|up + down = writes per layer, each layer's
    banks pinned after fully resident.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    _require_tp1("expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size

    gu_types: dict[int, int] = {}
    down_types: dict[int, int] = {}
    for t in iter_gguf_tensors(model_path):
        match = _EXPERT_RE.match(t.name)
        if match:
            (gu_types if match.group(2) in ("gate", "up") else down_types)[
                int(match.group(1))
            ] = t.ggml_type
    assert len(gu_types) == len(down_types) == L, "expert layers missing from the GGUF"
    gu_target, gu_strays = _majority_type(gu_types, "gate_up")
    down_target, down_strays = _majority_type(down_types, "down")
    assert not gu_strays, (
        f"gate_up experts mix quant types {sorted(GGML_NAME.get(t, t) for t in gu_strays | {gu_target})}; "
        "gate|up must share one type (no requant path)"
    )
    if down_strays and down_target != _REQUANT_TARGET:
        raise NotImplementedError(
            f"down experts mix types onto majority {GGML_NAME[down_target]}; "
            "requantizing stragglers is only implemented onto Q5_K"
        )
    # gate/up must ALSO share a type to concatenate rows; per-layer check.
    for layer, ty in gu_types.items():
        assert ty == gu_target

    # --cpu-mem: pageable layers allocate file-backed spill banks (never pinned).
    backing = "anon" if layer_backing is None else layer_backing
    hb = alloc_layer_banks(
        _kq_bank_specs(config, gu_target, down_target), L, backing=backing, spill_dir=spill_dir
    )
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    h_gu, h_dn = row_bytes(H, gu_target), row_bytes(I, down_target)
    seen: dict[str, set[int]] = {"gate_up": set(), "down": set()}

    def _load(sink) -> None:
        # gate|up + down = 2 notes/layer for the completion tracker (gate_up counts 1).
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        for t in iter_gguf_tensors(model_path):
            match = _EXPERT_RE.match(t.name)
            if match is None:
                continue
            layer, part = int(match.group(1)), match.group(2)
            if part in ("gate", "up"):
                packed = t.packed().reshape(E, I, h_gu)
                slot = slice(None, I) if part == "gate" else slice(I, None)
                banks["gate_up"][layer][:, slot].copy_(packed)
                if tracker is not None and part == "up":  # gate landed just before
                    tracker.note(layer)
                seen["gate_up"].add(layer)
            else:
                if t.ggml_type != down_target:
                    packed = _requant_to_q5_k(t, h_dn).reshape(E, H, h_dn)
                else:
                    packed = t.packed().reshape(E, H, h_dn)
                banks["down"][layer].copy_(packed)
                if tracker is not None:
                    tracker.note(layer)
                seen["down"].add(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned

    want = set(range(L))
    assert seen["gate_up"] == want and seen["down"] == want, (
        f"missing K-quant expert layers: gate_up {sorted(want - seen['gate_up'])}, "
        f"down {sorted(want - seen['down'])}"
    )
    return banks


def dummy_gguf_kq_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    """Random K-quant expert banks shaped like ``load_gguf_kq_expert_sources`` output
    (Q4_K gate_up / Q5_K down -- the dominant layout of the real checkpoints)."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L = config.num_layers
    hb = alloc_layer_banks(_kq_bank_specs(config, GGML_Q4_K, GGML_Q5_K), L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)  # match the other dummies: pin-after-fill (no-op mmap fill on CPU-only)
    return banks


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "is_gguf_model",
    "convert_qwen35moe_to_gguf",
    "load_gguf_kq_expert_sources",
    "dummy_gguf_kq_expert_sources",
]
