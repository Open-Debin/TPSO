"""Losses from equations (5)-(7) of the TPSO paper."""

from __future__ import annotations

import torch


def semantic_loss(
    cosine_similarity: torch.Tensor,
    kappa: float,
    tolerance: float,
) -> torch.Tensor:
    """Return the source implementation's loss outside the target band."""

    lower = kappa - tolerance
    upper = kappa + tolerance
    zero = torch.zeros_like(cosine_similarity)
    return (
        torch.maximum(zero, lower - cosine_similarity)
        + torch.maximum(zero, cosine_similarity - upper)
    ).sum()


def diversity_loss(
    projected: torch.Tensor,
    num_variants: int,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Source-faithful average pairwise distance with active-row compensation."""

    if num_variants <= 1:
        return projected.new_zeros(())
    if projected.shape[0] % num_variants:
        raise ValueError("The batch size must be divisible by num_variants.")
    grouped = projected.reshape(-1, num_variants, projected.shape[-1])
    summed = grouped.sum(dim=1)
    sum_all = (summed * summed).sum(dim=1)
    average_pairwise_distance = 1.0 - (
        (sum_all - num_variants) / (num_variants * (num_variants - 1))
    )
    if active is None:
        active = torch.ones(
            projected.shape[0], dtype=torch.bool, device=projected.device
        )
    active_per_group = active.reshape(-1, num_variants).sum(dim=1)
    valid = active_per_group > 0
    if not valid.any():
        return projected.new_zeros(())
    scale = num_variants / active_per_group[valid].to(projected.dtype)
    return (-average_pairwise_distance[valid] * scale).mean()
