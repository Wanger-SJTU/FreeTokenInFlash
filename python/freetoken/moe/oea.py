"""Simplified OEA (Opportunistic Expert Activation) for prefill expert loading.

From "Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode
Without Retraining" (arXiv:2511.02237). The paper's observation: what bounds MoE latency
is the number of *distinct* experts a batch activates, not the number of token-expert
pairs, because each activated expert's weights must be fetched. Their simplified routing
(Algorithm 1) keeps, per token, its top ``k0`` experts as a baseline and then lets it
"piggyback" on any further top-``k`` experts that the batch already activated -- the union
is fixed by the baselines, so no extra expert becomes resident.

The paper uses this for decode only (it says so explicitly: at prefill batch sizes the
union saturates anyway). The same selection is useful for **prefill expert loading** on an
offload cache: FreeToken's prefill normally materializes all ``num_experts`` of a layer
into GPU staging; with the union of activated experts we copy (and, for file-backed spill
banks, read from disk) only that union. ``k0 == k`` is the lossless variant: the union of
the model's own top-k sets, no rerouting at all.

Each token's routing slots must hold *distinct* expert ids: the grouped GEMM addresses
experts by id and its id-compaction helper assumes at most one occurrence per token (a
repeated id overflows the compact buffer). Short sets are therefore padded with distinct
resident experts from the union that the token did not pick, weight 0.
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def oea_select(
    scores: torch.Tensor,
    k: int,
    k0: int,
    *,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick per-token expert sets ``S_i`` (simplified OEA) and their weights.

    ``scores`` are the router's normalized scores over all experts, ``[tokens, num_experts]``.
    Returns ``(ids [tokens, K] int32, weights [tokens, K] float, union [T] int64)``:

    * ``k0`` (1..k) baseline experts per token -- their union is the batch's activated set;
    * ranks ``k0..k-1`` join a token's set only if that expert is already in the union;
    * ``K = max|S_i|`` slots per token (at most ``k``); a token with fewer kept experts is
      padded with distinct union members it did not pick, weight 0, so each row stays a
      set of unique ids;
    * weights keep the model's router scores and are renormalized over the kept set
      (paper Eq. 1) unless ``renormalize`` is false.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be [tokens, experts], got {tuple(scores.shape)}")
    num_tokens, num_experts = scores.shape
    k = min(int(k), num_experts)
    k0 = max(1, min(int(k0), k))
    if num_tokens == 0:
        empty = scores.new_empty((0, k))
        return (
            torch.empty((0, k), dtype=torch.int32, device=scores.device),
            empty,
            torch.empty((0,), dtype=torch.long, device=scores.device),
        )
    top = torch.topk(scores, k, dim=-1)
    ids = top.indices
    union_mask = torch.zeros(num_experts, dtype=torch.bool, device=scores.device)
    union_mask[ids[:, :k0].reshape(-1)] = True
    union = union_mask.nonzero(as_tuple=False).flatten()
    # base ids are in the union by construction; extra ids join only when already active
    keep = union_mask[ids]
    # compact the kept ids to the left (rank order preserved) so holes are at the right
    order = torch.argsort(keep.to(torch.int8), dim=-1, descending=True, stable=True)
    ids = ids.gather(1, order)
    keep = keep.gather(1, order)
    weights = torch.gather(scores, -1, ids) * keep
    n_keep = keep.sum(dim=-1)
    width = max(1, min(int(n_keep.max().item()), int(union.numel())))
    ids = ids[:, :width]
    keep = keep[:, :width]
    weights = weights[:, :width]
    # distinct resident fillers: union members the token did not pick, ascending
    union_pos = torch.full(
        (num_experts,), union.numel(), dtype=torch.long, device=scores.device
    )
    union_pos[union] = torch.arange(union.numel(), device=scores.device)
    # holes (experts outside the union) scatter into a throwaway last column so their
    # False cannot clobber a kept id that happens to share position 0
    used = torch.zeros((num_tokens, union.numel() + 1), dtype=torch.bool, device=scores.device)
    used.scatter_(1, union_pos[ids], keep)
    used = used[:, : union.numel()]
    candidates = union.unsqueeze(0).expand(num_tokens, -1).masked_fill(used, num_experts)
    candidates = candidates.sort(dim=-1).values
    fill = torch.arange(width, device=scores.device).unsqueeze(0) - n_keep.unsqueeze(1)
    fill = candidates.gather(1, fill.clamp_min(0).clamp_max(union.numel() - 1))
    is_fill = torch.arange(width, device=scores.device).unsqueeze(0) >= n_keep.unsqueeze(1)
    ids = torch.where(is_fill, fill, ids)
    weights = torch.where(is_fill, torch.zeros_like(weights), weights)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    return ids.to(torch.int32), weights, union


__all__ = ["oea_select"]
