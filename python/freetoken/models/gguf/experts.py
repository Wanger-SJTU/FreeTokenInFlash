"""Routed-expert banks straight from a GGUF's ``blk.N.ffn_{gate,up,down}_exps`` tensors, in their packed ggml rows: pinned host banks for the offload cache, or device banks for resident experts.

Each layer keeps its own ggml types, so each layer gets its own bank shapes: ``gate_up`` [E, 2I, row_bytes] (expert e's gate rows, then its up rows) and ``down`` [E, H, row_bytes]; float experts keep their dtype ([E, 2I, H] and [E, H, I]).
A GGUF that stores ``ffn_gate_up_exps`` (gemma4) already holds the gate_up rows in that order.
When NVFP4 experts carry per-expert ``.scale`` tensors, every layer also gets ``global`` [E, 4] fp32: the (gate, up, down, 1) weight_scale_2, 1 where a tensor has none.

The bytes move with O_DIRECT (buffered fallback), NOT an mmap of the source: the expert set is
the bulk of a large GGUF, and mapping it would park gigabytes of page cache in front of the
pinned expert banks on exactly the RAM-tight hosts ``--cpu-mem`` exists for."""

from __future__ import annotations

import contextlib

import torch

from freetoken.layers.quantization import QuantKind
from freetoken.layers.quantization.ggml_types import float_dtype, has_expert_scales, mmq_tail_bytes, pad_rows, type_name
from freetoken.utils import init_logger

from .reader import GgufTensorInfo, byte_view, gguf_direct_reader, gguf_tensor_index
from .tensors import weight_scale

logger = init_logger(__name__)


def _expert_shape(info: GgufTensorInfo) -> tuple[int, int, int]:
    """``(experts, rows, row_bytes)`` of a packed expert tensor ``[k, rows, experts]``."""
    if len(info.ne) != 3:
        raise ValueError(f"{info.name}: expert tensor is {len(info.ne)}-D, expected [k, rows, experts]")
    k, rows, experts = (int(x) for x in info.ne)
    if experts <= 0 or rows <= 0 or info.nbytes % (experts * rows):
        raise ValueError(f"{info.name}: {info.nbytes} B is not {experts} x {rows} whole rows")
    return experts, rows, info.nbytes // (experts * rows)


def _bank_dtype(info: GgufTensorInfo) -> torch.dtype:
    return float_dtype(info.ggml_type) or torch.uint8


def _last_dim(info: GgufTensorInfo, dtype: torch.dtype) -> int:
    """Bank last dim: the packed row byte count for quantized rows, ``K`` for float rows."""
    return _expert_shape(info)[2] if dtype is torch.uint8 else int(info.ne[0])


def _fill_packed(bank: torch.Tensor, start_row: int, info: GgufTensorInfo) -> None:
    """Fill ``bank[:, start_row:start_row+rows]`` from ``info`` one expert at a time, O_DIRECT.

    A destination expert block (``rows`` whole rows of one leading dim) is contiguous, and
    the source's experts are contiguous too, so this stays a straight sequence of aligned
    reads with no extra staging copy at the layer level."""
    experts, rows, row_bytes = _expert_shape(info)
    per = rows * row_bytes
    reader = gguf_direct_reader(info.path)
    for e in range(experts):
        dst = bank[e, start_row : start_row + rows]
        reader.read_into(byte_view(dst), info.offset + e * per, per)


def _fill_gate_up(bank: torch.Tensor, parts: list[GgufTensorInfo], shapes: list[tuple[int, int, int]]) -> None:
    """Fill the fused ``gate_up`` bank: each part's rows go to their own column block."""
    start = 0
    for info, (_experts, rows, _row_bytes) in zip(parts, shapes):
        _fill_packed(bank, start, info)
        start += rows


def _device_bank(shape: tuple[int, ...], ggml_type: int, k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """An [E, rows, row] device bank; a zero row follows the last one when ggml's MMQ tiles read past it."""
    experts, rows, row = shape
    flat = torch.empty(experts * rows + pad_rows(ggml_type, k), row, dtype=dtype, device=device)
    flat[experts * rows :].zero_()
    return flat[: experts * rows].view(shape)


def _expert_scales(gate_up: list[GgufTensorInfo], down: GgufTensorInfo, experts: int) -> torch.Tensor:
    """[E, 4] fp32 (gate, up, down, 1) weight_scale_2 of one layer's experts."""
    gate, up = gate_up if len(gate_up) == 2 else (gate_up[0], gate_up[0])
    out = torch.ones(experts, 4, dtype=torch.float32)
    for col, info in enumerate((gate, up, down)):
        s = weight_scale(info)
        if s is None:
            continue
        if s.numel() != experts:
            raise NotImplementedError(f"{info.name}: {s.numel()} scales for {experts} experts")
        out[:, col] = s
    return out


def load_gguf_expert_banks(model_path: str, num_layers: int, *, first_layer: int = 0, dummy: bool = False, device: torch.device | None = None):
    """``ExpertBanks`` with one bank per (layer, role): pinned host banks for the offload cache, filled from the file and pinned while later layers load, or resident banks on ``device``.

    The packed rows are read with O_DIRECT straight into the bank (no mmap of the source), so
    the load leaves no page cache behind beyond the small ``.scale`` tensors."""
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.host_banks import HostBank, PinPipeline

    index = gguf_tensor_index(model_path)
    scaled = has_expert_scales({name: info.ggml_type for name, info in index.items()})
    roles = ("gate_up", "down", "global") if scaled else ("gate_up", "down")
    sources: dict[str, list[torch.Tensor]] = {role: [] for role in roles}
    tails: dict[str, list[int]] = {role: [] for role in roles}
    tags = []
    with PinPipeline() if device is None else contextlib.nullcontext() as pins:
        for layer in range(num_layers):
            blk = f"blk.{first_layer + layer}"
            down = index[f"{blk}.ffn_down_exps.weight"]
            if f"{blk}.ffn_gate_up_exps.weight" in index:
                gate = index[f"{blk}.ffn_gate_up_exps.weight"]
                parts = [gate]
            else:
                gate, up = index[f"{blk}.ffn_gate_exps.weight"], index[f"{blk}.ffn_up_exps.weight"]
                if gate.ggml_type != up.ggml_type:
                    raise NotImplementedError(f"{blk}: gate and up experts have different ggml types")
                parts = [gate, up]
            tags.append(("ggml", type_name(gate.ggml_type), type_name(down.ggml_type)))
            tails["gate_up"].append(mmq_tail_bytes(gate.ggml_type, gate.ne[0]))
            tails["down"].append(mmq_tail_bytes(down.ggml_type, down.ne[0]))
            if scaled:
                tails["global"].append(0)
            shapes = [_expert_shape(info) for info in parts]
            experts, rows_total = shapes[0][0], sum(rows for _, rows, _ in shapes)
            gate_dtype, down_dtype = _bank_dtype(parts[0]), _bank_dtype(down)
            gate_up_shape = (experts, rows_total, _last_dim(parts[0], gate_dtype))
            down_shape = _expert_shape(down)
            down_bank_shape = (down_shape[0], down_shape[1], _last_dim(down, down_dtype))
            if device is None:
                host = {
                    "gate_up": HostBank(gate_up_shape, gate_dtype, layer_id=layer),
                    "down": HostBank(down_bank_shape, down_dtype, layer_id=layer),
                }
                if scaled:
                    host["global"] = HostBank((experts, 4), torch.float32, layer_id=layer)
                banks = {role: bank.tensor for role, bank in host.items()}
            else:
                banks = {
                    "gate_up": _device_bank(gate_up_shape, gate.ggml_type, gate.ne[0], gate_dtype, device),
                    "down": _device_bank(down_bank_shape, down.ggml_type, down.ne[0], down_dtype, device),
                }
                if scaled:
                    banks["global"] = torch.empty(experts, 4, dtype=torch.float32, device=device)
            if scaled:
                banks["global"].copy_(_expert_scales(parts, down, experts))
            if dummy:
                banks["gate_up"].zero_()
                banks["down"].zero_()
            elif device is None:
                _fill_gate_up(banks["gate_up"], parts, shapes)
                _fill_packed(banks["down"], 0, down)
            else:
                # resident experts: stage on the host (O_DIRECT needs a CPU destination), then H2D
                staging = torch.empty(gate_up_shape, dtype=gate_dtype)
                _fill_gate_up(staging, parts, shapes)
                banks["gate_up"].copy_(staging.to(device))
                staging = torch.empty(down_bank_shape, dtype=down_dtype)
                _fill_packed(staging, 0, down)
                banks["down"].copy_(staging.to(device))
            if device is None:
                pins(layer, host)
            for role, bank in banks.items():
                sources[role].append(bank)
    total = sum(t.numel() * t.element_size() for ts in sources.values() for t in ts)
    logger.info_rank0(f"GGUF expert banks: {num_layers} layers, {total / 2**30:.1f} GiB {'on ' + str(device) if device is not None else 'in host RAM'}")
    return ExpertBanks("gguf_scaled" if scaled else "gguf", sources, kind=QuantKind.GGUF, kernel="ggml", layout=None, layer_tags=tags, row_tail_bytes=tails)


__all__ = ["load_gguf_expert_banks"]
