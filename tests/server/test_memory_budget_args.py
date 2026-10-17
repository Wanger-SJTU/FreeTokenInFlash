"""--gpu-mem / --cpu-mem argument parsing: the memory-size type, the budget fields, and
the --memory-ratio mutual exclusion. The budgets' engine-side math lives in
tests/engine/test_cache_budget.py; this file owns the CLI surface."""

from __future__ import annotations

import pytest

from freetoken.server.args import parse_args

# Parsing resolves the checkpoint dtype off config.json, so point at a real-ish local
# dir and pin --dtype to skip the hub/config resolution entirely.
PATH_ARGS = ["--model-path", "/models/anon", "--dtype", "bfloat16"]


def _parse(*extra):
    args, _ = parse_args([*PATH_ARGS, *extra])
    return args


def test_mem_size_type_accepts_suffix_and_bare_bytes():
    assert _parse("--gpu-mem", "16G").gpu_mem_budget == 16 * 1024**3
    assert _parse("--gpu-mem", "512M").gpu_mem_budget == 512 * 1024**2
    assert _parse("--gpu-mem", str(1024)).gpu_mem_budget == 1024


def test_cpu_mem_budget_and_spill_dir_fields_map():
    args = _parse("--cpu-mem", "32G", "--cpu-mem-spill-dir", "/var/spill")
    assert args.cpu_mem_budget == 32 * 1024**3
    assert args.cpu_mem_spill_dir == "/var/spill"


def test_defaults_leave_both_budgets_unset():
    args = _parse()
    assert args.gpu_mem_budget is None
    assert args.cpu_mem_budget is None
    assert args.cpu_mem_spill_dir is None
    assert args.memory_ratio == 0.9  # ratio path untouched without --gpu-mem


def test_gpu_mem_excludes_memory_ratio():
    with pytest.raises(SystemExit):
        parse_args([*PATH_ARGS, "--gpu-mem", "16G", "--memory-ratio", "0.8"])


def test_bad_mem_size_rejected():
    with pytest.raises(SystemExit):
        parse_args([*PATH_ARGS, "--gpu-mem", "16X"])
    with pytest.raises(SystemExit):
        parse_args([*PATH_ARGS, "--cpu-mem", "0G"])


def test_cpu_mem_mode_default_and_choices():
    args = _parse("--cpu-mem", "9G")
    assert args.cpu_mem_mode == "expert"  # the default
    args = _parse("--cpu-mem", "9G", "--cpu-mem-mode", "layer")
    assert args.cpu_mem_mode == "layer"
    with pytest.raises(SystemExit):
        _parse("--cpu-mem", "9G", "--cpu-mem-mode", "bananas")
