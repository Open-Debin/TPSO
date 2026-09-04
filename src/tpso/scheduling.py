"""Coarse-to-fine embedding schedule from equations (9)-(10)."""

from __future__ import annotations

import torch


def coarse_to_fine_alphas(
    num_steps: int,
    ratio: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return alpha values ordered in diffusion sampling order, from noisy to clean."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not -1.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [-1, 1].")
    resolved_dtype = dtype or torch.float32
    if ratio == 0.0:
        return torch.zeros(num_steps, device=device, dtype=resolved_dtype)

    reverse = ratio < 0.0
    ratio = abs(ratio)

    active_steps = int(num_steps * ratio)
    if active_steps == 0:
        return torch.zeros(num_steps, device=device, dtype=resolved_dtype)
    alphas = torch.cat(
        [
            torch.linspace(
                1.0,
                0.0,
                active_steps,
                device=device,
                dtype=resolved_dtype,
            ),
            torch.zeros(
                num_steps - active_steps, device=device, dtype=resolved_dtype
            ),
        ]
    )
    return alphas.flip(0) if reverse else alphas


def schedule_embeddings(
    optimized: torch.Tensor,
    original: torch.Tensor,
    num_steps: int,
    ratio: float,
) -> torch.Tensor:
    """Interpolate optimized embeddings back to the original condition."""

    if optimized.shape != original.shape:
        raise ValueError("optimized and original embeddings must have identical shapes.")
    alphas = coarse_to_fine_alphas(
        num_steps, ratio, device=optimized.device, dtype=optimized.dtype
    )
    shape = (1, num_steps) + (1,) * (optimized.ndim - 1)
    alphas = alphas.reshape(shape)
    return alphas * optimized.unsqueeze(1) + (1.0 - alphas) * original.unsqueeze(1)
