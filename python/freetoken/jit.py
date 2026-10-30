"""``ft jit``: compile the AOT kernels a checkpoint needs WITHOUT loading its weights.

Without a prebuilt kernel cache, the AOT modules are nvcc-built on first use: ggml-CUDA
for GGUF dense/experts, FlashInfer for attention. Doing that inside ``ft serve`` stacks
the build (several GiB of host RAM) on top of the pinned/spilled expert banks -- on a
RAM-tight host the first forward or request dies there.

This command builds exactly those modules from the checkpoint's *config* only (GGUF
metadata / HF ``config.json``), so it can run once ahead of time on a machine with spare
memory; ``ft serve`` then just loads the cached modules (0 s). Kernel selection mirrors
``Engine._precompile_kernels`` -- the same code paths, with errors surfaced instead of
swallowed.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from types import SimpleNamespace

_DTYPES = {
    "auto": None,
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Compile the AOT kernels this checkpoint's forward path needs (ggml-CUDA for "
            "GGUF weights, FlashInfer for attention). Model weights are NOT loaded: only "
            "the checkpoint config/GGUF metadata is read. Run once before `ft serve` on a "
            "machine with spare host memory."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="checkpoint path: a .gguf file, an HF safetensors dir, or an FTW dir",
    )
    parser.add_argument(
        "--attention-backend",
        default="auto",
        metavar="NAME",
        help="attention backend to compile for (auto = the same default `ft serve` picks)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=sorted(_DTYPES),
        help="weight/activation dtype for the kernel variants (auto = the checkpoint's)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="tensor-parallel size the kernels should match (default 1)",
    )
    return parser


def _resolve_dtype(hf_config, flag: str):
    import torch

    if flag != "auto":
        return getattr(torch, _DTYPES[flag])
    dtype = getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None)
    if isinstance(dtype, str):
        return getattr(torch, dtype.removeprefix("torch."), torch.bfloat16)
    return dtype if isinstance(dtype, torch.dtype) else torch.bfloat16


def _build_config(model_path: str, tp_size: int, backend: str, dtype_flag: str):
    """(config-like object for the engine's precompile helpers, resolved dtype, model_config)."""
    import torch

    from freetoken.distributed import DistributedInfo, set_tp_info
    from freetoken.engine.config import EngineConfig

    try:
        set_tp_info(rank=0, size=tp_size)  # raises if already set (e.g. embedded use)
    except RuntimeError:
        pass
    base = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=tp_size),
        dtype=torch.bfloat16,
    )
    dtype = _resolve_dtype(base.hf_config, dtype_flag)
    if dtype is not base.dtype:
        base = dataclasses.replace(base, dtype=dtype)
    # parsing the checkpoint config only: no weights are read above or below
    model_config = base.model_config
    config = SimpleNamespace(
        model_path=model_path,
        attention_backend=backend,
        dtype=base.dtype,
        model_config=model_config,
    )
    return config, base.dtype, model_config


def main(argv: list[str] | None = None, prog: str = "ft jit") -> int:
    args = _parser(prog).parse_args(sys.argv[1:] if argv is None else argv)
    if not os.path.exists(args.model):
        print(f"{prog}: no such checkpoint: {args.model}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    try:
        config, dtype, model_config = _build_config(
            args.model, args.tp_size, args.attention_backend, args.dtype
        )
        print(
            f"checkpoint: {args.model}\n"
            f"dtype: {dtype}, head_dim: {getattr(model_config, 'head_dim', 'n/a')}, "
            f"attention backend: {args.attention_backend}"
        )
        from freetoken.engine.engine import _precompile_kernels

        _precompile_kernels(config, raise_on_error=True)
    except Exception as exc:  # noqa: BLE001 -- the CLI must report the real reason
        import traceback

        traceback.print_exc()
        print(f"{prog}: kernel build failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    print(f"kernels ready in {time.perf_counter() - t0:.1f}s")
    return 0


__all__ = ["main"]
