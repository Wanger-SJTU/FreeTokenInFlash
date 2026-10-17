# 专家粒度宿主内存驻留（--cpu-mem expert 模式）

## 目标

`--cpu-mem` 预算不再按"整层 pinned / 整层 spill"贪心划分，改为**专家粒度均分**：预算平摊到所有 MoE 层，每层 pin 前 k 个 expert，其余 E−k 个 expert 每层落 file-backed spill bank。新增 `--cpu-mem-mode {expert,layer}`，默认 `expert`，`layer` 保留现行为。Prefill 维持现状（pageable staging）。

## 核心设计：连续双 mmap 的单张量 split bank

每层 bank 仍是一个逻辑连续的 `[E, rows, row_bytes]` 张量（`sources[name][layer]` 契约不变），底层两段 mmap 拼接：

- `[0, k_eff)` 专家 → 匿名段，fill 后 `cudaHostRegister` 只注册该前缀（pin-after-fill 子范围注册）
- `[k_eff, E)` 专家 → file-backed spill tmpfile 段，fill 后 MADV_DONTNEED，由页缓存管，缺页从盘上回
- `anon_bytes = round_up(k·slab, 4096)`，`k_eff = ceil(anon_bytes/slab)`（边界不切开专家）；张量基址即匿名段起点

这样 **CPU executor 完全不用改**（单基址 + `e·stride` 算术自动落到 file 半区，缺页透明），GPU 侧只要保证"被 fetch 的 expert 一定是 resident"。

## Decode 路径（复用 hybrid 机制）

- engine 在 `0 < k_eff < E` 时强制 `decode_target="hybrid"`（含 `--moe-backend offload`），`cpu_layer_ids` 保持空
- `_ensure_experts_hybrid_kernel` 增加 `residency_ptr` + `HAS_RESIDENCY` constexpr：候选集变为 `missing ∩ resident`（掩码 `[L,E]` bool，套 `expert_recency` 的模式）；spilled miss 自然留 -1 → CPU executor；`max_fetch`/带宽配比策略叠加在掩码之上（resident 但超 cap 的也转 CPU，合法——CPU 能读匿名内存）
- `_decode_hybrid` 的 `copy_missing` 改为 `copy_missing(masked=True)`：split 层用**匿名前缀基址**做 fused gather（掩码保证索引 resident，图捕获安全）
- split 层 `_layer_fused_ok=False` → prefill / 非 masked 调用自动走 `_copy_missing_pageable`（torch staging，与今天 spilled 层一致）
- k==E：无 split，纯 offload；k==0：全 file-backed，掩码全 False，一切走 CPU

## 逐文件改动

1. **moe/host_banks.py** — `HostBank` 新增 split 支持（PROT_NONE 保留 + MAP_FIXED 两段；`pin()` 只注册前缀；`drop_pages()` 只作用 file 段）；`alloc_layer_banks(..., split_k=)`；`pin_banks`/`PinPipeline` 对 split 先 pin 后 drop；暴露匿名段字节供 cache 用
2. **engine/cache_budget.py** — 新增纯函数 `plan_expert_residency(L, layer_bytes, E, budget) -> k`（budget//L//per_expert，clamp [0,E]，None→E）；`plan_host_residency` 保留给 layer 模式
3. **engine/config.py + server/args.py** — `cpu_mem_mode` 字段 + `--cpu-mem-mode` 参数（默认 expert），`--cpu-mem` help 更新
4. **engine/engine.py** — `_plan_host_residency` 按模式分派；split 时 decode_target="hybrid"、`_cpu_executor_supports_act` 检查提前到 `0<k<E`；`load_expert_banks(..., split_k=)`；`layer_residency` 每层记 `"split"`；构建 `expert_residency` 掩码；日志输出 k 值
5. **moe/expert_banks.py** — `ExpertBanks.split_k` 字段；`load_expert_banks` 透传 split_k 到各 provider（全部 provider 都经 `alloc_layer_banks`，改动是机械的）；`_stamp_layer_residency` 记 "split"
6. **moe/offload_kernels.py** — hybrid ensure kernel 加 residency 掩码（候选=missing∩resident），宿主包装加 `residency=None` 形参
7. **moe/offload_cache.py** — `set_bank_sources` 接受 "split"；存 `expert_residency` 掩码；`_build_copy_plan` 为 split 层另建匿名前缀基址的 `_split_src_ptrs`（`_layer_fused_ok` 置 False）；`copy_missing(masked=False)` 新分支
8. **layers/moe.py** — `_decode_hybrid` 调 `copy_missing(masked=True)`（非 split 层等价现状）
9. **moe/cpu_executor.py + C++** — **零改动**（设计锚点，测试锁定）
10. **checkpoint/ftw.py** — `_bank_for_layer` 支持按 `[0:k_eff)`/`[k_eff:E)` 两窗口分别读入两段；盘上格式不变
11. **上报/文档** — cache_status host meta 加 `mode` + `pinned_experts_per_layer`；cache_report/api_server 渲染；docs/cli.md、docs/model-loading.zh.md 更新
12. **测试**
    - test_cache_budget：plan_expert_residency（均分/clamp/None）
    - test_host_banks：split bank 生命周期（Linux-gated）：pin 只注册前缀、drop 只作用 file 段、file 段重缺页、匿名段数据保留
    - test_offload："split" residency、masked copy 分支、非 masked 绝不 fused
    - test_hybrid_fetch：掩码变体（resident miss 被取、spilled miss→-1、hit 不受影响）
    - 数值集成：dummy gguf_kq banks + split decode 输出 == 全 pinned 参考
    - test_memory_budget_args：--cpu-mem-mode 解析/默认/非法值

## 实施顺序

host_banks（原语+测试）→ cache_budget（纯函数+测试）→ expert_banks/engine 接线 → offload_kernels/offload_cache/layers（decode 路径+测试）→ FTW → 上报与文档 → 端到端验证

## 验证

单测全绿后，在你的机器上：`ft serve --model ~/models/unsloth/Qwen3.6-35B-A3B-GGUF --cpu-mem 9G` 确认日志的 k 值与 hybrid decode、`ft ctl stats` 看 tok/s，并与 `--cpu-mem-mode layer` A/B。

**预期管理**（已与你对齐）：expert 均分不减少期望磁盘流量（未驻留比例不变），收益是每层都能走 PCIe + GPU LRU、prefill 双缓冲覆盖全部层、消除"整层磁盘路径"；NVMe 的墙仍在。

## 风险与规避

- 绝不依赖整张量 `device_ptr` 成败判断 split 层 fused 资格（UVA 下可能恒成功）——显式 layer mode 路由
- cudaHostRegister 子范围：base/size 均页对齐（mmap + round_up 保证）
- marlin/b12x repack 假设整层单分配：split 强制 native layout（decode_target=hybrid 已如此），repack 不运行，无冲突
- split 同 file-backed 一样 Linux-only，沿用现有 skip 策略