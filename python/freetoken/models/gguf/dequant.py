"""GGML block-quant dequantization in pure torch (the formats this repo's GGUF
checkpoints use: Q4_0, Q6_K, plus trivial F32/F16/BF16).

This is the *reference / CPU* path, NOT the engine's hot path: GGUF weights stay
packed and are dequantized inside the borrowed ggml CUDA kernels (see
``freetoken.kernel.gguf``). These routines are used only to (a) materialize the few
dense F32/F16 tensors at load (norms, scales, router) via :func:`dequantize`, and
(b) cross-check the CUDA kernels in tests. The ``BLOCK_SHAPE`` table and
:func:`row_bytes` are the type metadata the packed (kernel) path also relies on.

Each ``dequant_*`` takes the raw little-endian bytes as a ``uint8`` tensor whose
final axis spans whole blocks, and returns the values in *storage order* (ggml's
fastest axis first); the caller reshapes to the torch shape (``dims[::-1]``). The
math mirrors ``ggml-quants.c``.
"""

from __future__ import annotations

import torch

# ggml_type enum values (subset present in these checkpoints).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_BF16 = 30

# (block numel, bytes per block) per ggml type.
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q8_0: (32, 34),
    GGML_Q2_K: (256, 84),
    GGML_Q3_K: (256, 110),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_BF16: "BF16",
    GGML_Q4_0: "Q4_0",
    GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
}


def row_bytes(numel: int, ggml_type: int) -> int:
    """Packed byte length of one row of ``numel`` elements in ``ggml_type`` blocks.

    Single source of truth for the ``numel // block * type_size`` math shared by the
    packed-weight ops (``GGUFLinear``/``GGUFEmbedding``) and the expert bank loaders.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    assert numel % block == 0, (
        f"{numel} not a multiple of block {block} for {GGML_NAME.get(ggml_type, ggml_type)}"
    )
    return numel // block * type_size


def _f16_scales(raw: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Reinterpret bytes ``[lo:hi]`` (2 per block) of each block row as fp16 -> fp32 [N,1]."""
    return raw[:, lo:hi].contiguous().view(torch.float16).to(torch.float32)


def dequant_q4_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0: per 32-elem block = fp16 scale ``d`` + 16 packed nibbles; ``w = d*(q-8)``.

    Byte ``j`` of the 16 holds element ``j`` in its low nibble and ``j+16`` in its high
    nibble, so storage order within the block is ``[lo0..lo15, hi0..hi15]``.
    """
    raw = raw.reshape(-1, 18)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    qs = raw[:, 2:18]  # [N,16] uint8
    lo = (qs & 0x0F).to(torch.float32)
    hi = (qs >> 4).to(torch.float32)
    q = torch.cat([lo, hi], dim=1)  # [N,32]
    return ((q - 8.0) * d).reshape(-1).to(out_dtype)


def dequant_q6_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q6_K: 256-elem super-block = 128B low nibbles + 64B high 2-bits + 16 int8
    sub-scales + fp16 ``d``. Direct vectorization of ggml's two-half loop."""
    raw = raw.reshape(-1, 210)
    n = raw.shape[0]
    ql = raw[:, 0:128]  # [n,128]
    qh = raw[:, 128:192]  # [n,64]
    sc = raw[:, 192:208].view(torch.int8).to(torch.float32)  # [n,16]
    d = _f16_scales(raw, 208, 210)  # [n,1]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # l in 0..15 -> is=0; l in 16..31 -> is=1 (per ggml: is = l/16).
    is_idx = (torch.arange(32, device=raw.device) // 16)  # [32] in {0,1}
    for h in range(2):  # two 128-elem halves of the super-block
        qlh = ql[:, h * 64:(h + 1) * 64]  # [n,64]
        qhh = qh[:, h * 32:(h + 1) * 32]  # [n,32]
        sch = sc[:, h * 8:(h + 1) * 8]  # [n,8]
        a = qlh[:, 0:32].to(torch.int32)  # ql[l]
        b = qlh[:, 32:64].to(torch.int32)  # ql[l+32]
        hb = qhh.to(torch.int32)  # qh[l]
        q1 = ((a & 0x0F) | (((hb >> 0) & 3) << 4)) - 32
        q2 = ((b & 0x0F) | (((hb >> 2) & 3) << 4)) - 32
        q3 = ((a >> 4) | (((hb >> 4) & 3) << 4)) - 32
        q4 = ((b >> 4) | (((hb >> 6) & 3) << 4)) - 32
        s1 = sch.index_select(1, is_idx + 0).to(torch.float32)
        s2 = sch.index_select(1, is_idx + 2).to(torch.float32)
        s3 = sch.index_select(1, is_idx + 4).to(torch.float32)
        s4 = sch.index_select(1, is_idx + 6).to(torch.float32)
        base = h * 128
        y[:, base + 0:base + 32] = d * s1 * q1.to(torch.float32)
        y[:, base + 32:base + 64] = d * s2 * q2.to(torch.float32)
        y[:, base + 64:base + 96] = d * s3 * q3.to(torch.float32)
        y[:, base + 96:base + 128] = d * s4 * q4.to(torch.float32)
    return y.reshape(-1).to(out_dtype)


def _k_scales(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack Q4_K/Q5_K's 12 scale bytes per 256-elem super-block into 8 six-bit sub-block
    scales and 8 six-bit sub-block mins, each re-biased to 0..63 ints (the layout ggml
    packs; storage order is storage-major, fastest axis first)."""
    sc_bytes = raw[:, 0:12].to(torch.int64)  # [N,12]
    a = sc_bytes[:, 0:4]  # sc[0:4] low 6 bits + sc[4:8] bits 4..5 in the top 2 bits
    b = sc_bytes[:, 4:8]  # m[0:4] low 6 bits + m[4:8] bits 4..5 in the top 2 bits
    c = sc_bytes[:, 8:12]  # sc[4:8] bits 0..3 in low nibbles, m[4:8] in high nibbles
    sc_lo = a & 0x3F
    sc_hi = (c & 0x0F) | ((a >> 2) & 0x30)
    m_lo = b & 0x3F
    m_hi = (c >> 4) | ((b >> 2) & 0x30)
    return torch.cat([sc_lo, sc_hi], dim=1), torch.cat([m_lo, m_hi], dim=1)  # [N,8] each


def dequant_q4_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_K: 256-elem super-block = fp16 ``d`` + fp16 ``dmin`` + 12B six-bit scales/mins
    (8 sub-blocks of 16) + 128B packed nibbles. ``w = d*sc*q - dmin*m`` (per sub-block)."""
    raw = raw.reshape(-1, 144)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    dmin = _f16_scales(raw, 2, 4)  # [N,1]
    sc, m = _k_scales(raw[:, 4:16])  # [N,8]
    qs = raw[:, 16:144]  # [N,128]
    # Byte (g, b) holds element g*64+b (low nibble) and g*64+32+b (high nibble).
    lo = (qs & 0x0F).to(torch.float32).reshape(-1, 4, 32)
    hi = (qs >> 4).to(torch.float32).reshape(-1, 4, 32)
    q = torch.stack([lo, hi], dim=2).reshape(-1, 256).reshape(-1, 8, 32)  # sub-block-major
    w = d.unsqueeze(-1) * sc.unsqueeze(-1) * q - dmin.unsqueeze(-1) * m.unsqueeze(-1)
    return w.reshape(-1).to(out_dtype)


def dequant_q5_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q5_K: Q4_K's layout plus 32B of high bits: ``w = d*sc*q - dmin*m`` with 5-bit ``q``
    (high bit from ``qh`` bit ``f//32`` of byte ``f%32``, low nibble from ``qs`` byte
    ``(f//64)*32 + f%32`` at nibble position ``(f%64)//32``)."""
    raw = raw.reshape(-1, 176)
    d = _f16_scales(raw, 0, 2)
    dmin = _f16_scales(raw, 2, 4)
    sc, m = _k_scales(raw[:, 4:16])
    qh = raw[:, 16:48].to(torch.int64)  # [N,32] high-bit bytes
    qs = raw[:, 48:176]  # [N,128]
    n = raw.shape[0]
    f = torch.arange(256, device=raw.device)
    hi_bit = (qh[:, f % 32] >> (f // 32)) & 1  # [N,256] int64
    qs_byte = qs[:, (f // 64) * 32 + (f % 32)].to(torch.int64)  # [N,256]
    nibble = (f % 64) // 32  # 0 -> low nibble, 1 -> high nibble
    q = ((qs_byte >> (4 * nibble)) & 0x0F) | (hi_bit << 4)
    q = q.to(torch.float32)
    q = q.reshape(n, 8, 32)  # sub-block-major: sub-block f//32
    w = d.unsqueeze(-1) * sc.unsqueeze(-1) * q - dmin.unsqueeze(-1) * m.unsqueeze(-1)
    return w.reshape(-1).to(out_dtype)


# Six-bit scale packing inverse of _k_scales: 8 (sc, m) pairs in 0..63 -> 12 bytes.
def _pack_k_scales(sc: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    n = sc.shape[0]
    out = torch.empty(n, 12, dtype=torch.uint8, device=sc.device)
    out[:, 0:4] = ((sc[:, 0:4] & 0x3F) | ((sc[:, 4:8] & 0x30) << 2)).to(torch.uint8)
    out[:, 4:8] = ((m[:, 0:4] & 0x3F) | ((m[:, 4:8] & 0x30) << 2)).to(torch.uint8)
    out[:, 8:12] = ((sc[:, 4:8] & 0x0F) | ((m[:, 4:8] & 0x0F) << 4)).to(torch.uint8)
    return out


def quantize_q5_k(x: torch.Tensor) -> torch.Tensor:
    """Quantize flat fp32 ``x`` (multiple of 256) to packed Q5_K blocks ``[N, 176]``.

    FreeToken's own requantizer (used to align stragglers of a dynamically-quantized
    GGUF onto the majority down-projection format): per 32-elem sub-block, clamp the
    range to include zero and encode ``w ~= d*sc*q - dmin*m`` with the sub-block's
    5-bit ``q`` spanning ``[lo, hi]`` in 32 uniform steps. The super-block's fp16
    ``(d, dmin)`` normalize the six-bit ``(sc, m)`` to their largest sub-block needs.
    """
    x = x.reshape(-1, 256).to(torch.float32)
    n = x.shape[0]
    sb = x.reshape(n, 8, 32)  # 8 sub-blocks of 32 (the 12 scale bytes carry 8 (sc, m) pairs)
    lo = sb.min(dim=2).values.clamp(max=0.0)  # extend range to include 0
    hi = sb.max(dim=2).values.clamp(min=0.0)
    span = (hi - lo) / 31.0  # per-sub-block effective step d*sc
    off = -lo  # effective offset dmin*m
    # fp16-subnormal floor: keeps an all-zero block finite (d underflowing to exactly 0
    # would turn the sc/m divisions into NaN).
    d = (span.max(dim=1).values / 63.0).clamp(min=1e-6).to(torch.float16)  # [n]
    dmin = (off.max(dim=1).values / 63.0).clamp(min=1e-6).to(torch.float16)  # [n]
    # ceil (not round): the int (sc, m) reconstruct step/base = d*sc, dmin*m, which must
    # cover [lo, hi] or q clamps at 31 and distorts the range's top end.
    sc = (span / d.float().unsqueeze(1)).ceil().clamp(0, 63)  # [n,8]
    m = (off / dmin.float().unsqueeze(1)).ceil().clamp(0, 63)
    step = (d.float().unsqueeze(1) * sc.float()).clamp(min=1e-30)  # [n,8]
    base = dmin.float().unsqueeze(1) * m.float()  # [n,8]
    q = ((sb + base.unsqueeze(2)) / step.unsqueeze(2)).round().clamp(0, 31).to(torch.int64)

    out = torch.empty(n, 176, dtype=torch.uint8, device=x.device)
    out[:, 0:2] = d.view(torch.uint8).reshape(n, 2)
    out[:, 2:4] = dmin.view(torch.uint8).reshape(n, 2)
    out[:, 4:16] = _pack_k_scales(sc.to(torch.int64), m.to(torch.int64))

    # Element f's low nibble rides in qs byte (f//64)*32 + f%32 at position (f%64)//32;
    # its high bit rides in qh byte f%32 at bit f//32 (matching the dequant layout).
    qf = q.reshape(n, 256)
    f = torch.arange(256, device=x.device)
    qs = torch.zeros(n, 128, dtype=torch.int64, device=x.device)
    qs.scatter_add_(1, ((f // 64) * 32 + (f % 32)).unsqueeze(0).expand(n, -1),
                    (qf & 0x0F) << (4 * ((f % 64) // 32)).unsqueeze(0))
    qh = torch.zeros(n, 32, dtype=torch.int64, device=x.device)
    qh.scatter_add_(1, (f % 32).unsqueeze(0).expand(n, -1),
                    (qf >> 4) << (f // 32).unsqueeze(0))
    out[:, 16:48] = qh.to(torch.uint8)
    out[:, 48:176] = qs.to(torch.uint8)
    return out


def dequant_q8_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q8_0: per 32-elem block = fp16 scale ``d`` + 16 int8 codes; ``w = d * q``."""
    raw = raw.reshape(-1, 34)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    q = raw[:, 2:34].view(torch.int8).to(torch.float32)  # [N,16]
    return (d * q).reshape(-1).to(out_dtype)


_DEQUANT = {
    GGML_Q4_0: dequant_q4_0,
    GGML_Q8_0: dequant_q8_0,
    GGML_Q4_K: dequant_q4_k,
    GGML_Q5_K: dequant_q5_k,
    GGML_Q6_K: dequant_q6_k,
}


def dequantize(raw: torch.Tensor, ggml_type: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``raw`` (uint8) of any supported ggml type to flat ``out_dtype``."""
    if ggml_type == GGML_F32:
        return raw.view(torch.float32).to(out_dtype)
    if ggml_type == GGML_F16:
        return raw.view(torch.float16).to(out_dtype)
    if ggml_type == GGML_BF16:
        return raw.view(torch.bfloat16).to(out_dtype)
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"dequant for ggml type {GGML_NAME.get(ggml_type, ggml_type)} not implemented"
        )
    return fn(raw, out_dtype)


def ggml_type_for_row_bytes(numel: int, rb: int) -> int:
    """The (unique) ggml type whose packed row length over ``numel`` elements is ``rb``.

    Restricted to the expert-bank types {Q4_K, Q5_K, Q6_K, Q8_0}: their bytes-per-256
    ratios (144/176/210/272) are pairwise distinct, so a bank's quant type is recoverable
    from its row length. Q4_0 is excluded (same ratio as Q4_K).
    """
    for ty in (GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0):
        if row_bytes(numel, ty) == rb:
            return ty
    raise ValueError(f"no K-quant/Q8_0 type packs {numel} elements into {rb} bytes")


__all__ = [
    "GGML_F32",
    "GGML_F16",
    "GGML_BF16",
    "GGML_Q4_0",
    "GGML_Q8_0",
    "GGML_Q2_K",
    "GGML_Q3_K",
    "GGML_Q4_K",
    "GGML_Q5_K",
    "GGML_Q6_K",
    "GGML_NAME",
    "BLOCK_SHAPE",
    "row_bytes",
    "dequant_q4_0",
    "dequant_q8_0",
    "dequant_q4_k",
    "dequant_q5_k",
    "dequant_q6_k",
    "quantize_q5_k",
    "dequantize",
    "ggml_type_for_row_bytes",
]
