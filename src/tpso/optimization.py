"""Model-independent optimization of token-level prompt offsets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .config import TPSOConfig
from .losses import diversity_loss, semantic_loss


@dataclass(frozen=True)
class EncodedPrompt:
    """Prompt representations needed by optimization and diffusion inference."""

    hidden: torch.Tensor
    projected: torch.Tensor


@dataclass(frozen=True)
class OptimizationResult:
    """Optimized prompt representations and convergence diagnostics."""

    optimized: EncodedPrompt
    original: EncodedPrompt
    offsets: torch.Tensor
    cosine_similarity: torch.Tensor
    converged: torch.Tensor
    steps: int


EncodeOffsets = Callable[[torch.Tensor], EncodedPrompt]
EncodeOffsetSubset = Callable[[torch.Tensor, torch.Tensor], EncodedPrompt]


def initialize_offsets(
    shape: tuple[int, ...],
    config: TPSOConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Initialize offsets, including every distribution reported in Table IV."""

    if config.offset_init in {"approx_zero", "normal", "zero", "normal_1", "normal_2"}:
        scale = {
            "approx_zero": 1e-4,
            "normal": config.offset_std,
            "zero": 1e-4,
            "normal_1": 1.0,
            "normal_2": 2.0,
        }[config.offset_init]
        return (
            torch.randn(
                shape, device=device, dtype=dtype, generator=generator
            )
            * scale
        )

    if config.offset_init.startswith("uniform_"):
        low, high = {
            "uniform_0_1": (0.0, 1.0),
            "uniform_-1_1": (-1.0, 1.0),
            "uniform_-3_3": (-3.0, 3.0),
        }[config.offset_init]
        values = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        return values.mul_(high - low).add_(low)

    if config.offset_init.startswith("laplace_"):
        scale = 1.0 if config.offset_init == "laplace_1" else 1.5**0.5
        distribution = torch.distributions.Laplace(0.0, scale)
        if generator is None:
            return distribution.sample(shape).to(device=device, dtype=dtype)
        with torch.random.fork_rng():
            torch.manual_seed(generator.initial_seed())
            return distribution.sample(shape).to(device=device, dtype=dtype)

    concentration, rate = {
        "gamma_1_1": (1, 1.0),
        "gamma_4_4": (4, 4.0),
        "gamma_4_sqrt4": (4, 4.0**0.5),
        "gamma_7_sqrt7": (7, 7.0**0.5),
    }[config.offset_init]
    distribution = torch.distributions.Gamma(float(concentration), rate)
    if generator is None:
        return distribution.sample(shape).to(device=device, dtype=dtype)
    with torch.random.fork_rng():
        torch.manual_seed(generator.initial_seed())
        return distribution.sample(shape).to(device=device, dtype=dtype)


def optimize_prompt_offsets(
    encode_offsets: EncodeOffsets,
    original: EncodedPrompt,
    offset_shape: tuple[int, ...],
    num_variants: int,
    config: TPSOConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    trainable_mask: torch.Tensor | None = None,
    encode_subset: EncodeOffsetSubset | None = None,
    initial_offsets: torch.Tensor | None = None,
    active_subset: bool = True,
) -> OptimizationResult:
    """Run active-subset or full-batch offset optimization."""

    config.validate()
    if offset_shape[0] != original.projected.shape[0]:
        raise ValueError("offset_shape and original batch sizes do not match.")
    if trainable_mask is not None:
        if trainable_mask.shape[0] != offset_shape[0]:
            raise ValueError("trainable_mask and offset_shape batch sizes do not match.")
        trainable_rows = trainable_mask.reshape(trainable_mask.shape[0], -1).any(dim=1)
        if not trainable_rows.all():
            raise ValueError(
                "Every prompt must contain at least one trainable token; blank conditional "
                "prompts are not supported."
            )
    if initial_offsets is None:
        offsets = initialize_offsets(
            offset_shape,
            config,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        if tuple(initial_offsets.shape) != offset_shape:
            raise ValueError("initial_offsets and offset_shape do not match.")
        offsets = initial_offsets.to(device=device, dtype=dtype).clone()
    if trainable_mask is not None:
        mask = trainable_mask.to(device=device, dtype=dtype)
        try:
            offsets.mul_(mask)
        except RuntimeError as error:
            raise ValueError(
                "trainable_mask must be broadcastable to offset_shape."
            ) from error
    else:
        mask = None
    offsets = torch.nn.Parameter(offsets)
    optimizer_class = {
        "rmsprop": torch.optim.RMSprop,
        "adam": torch.optim.Adam,
    }[config.optimizer]
    optimizer = optimizer_class([offsets], lr=config.learning_rate)
    target = functional.normalize(original.projected.detach(), dim=-1)

    if not active_subset:
        cached_encoded: EncodedPrompt | None = None
        cached_cosine: torch.Tensor | None = None
        cached_offsets: torch.Tensor | None = None
        steps = 0
        for step_index in range(config.max_steps):
            steps = step_index + 1
            optimizer.zero_grad(set_to_none=True)
            encoded = encode_offsets(offsets)
            normalized = functional.normalize(encoded.projected, dim=-1)
            cosine = (normalized * target).sum(dim=-1)
            cached_encoded = EncodedPrompt(
                encoded.hidden.detach().clone(), encoded.projected.detach().clone()
            )
            cached_cosine = cosine.detach().clone()
            cached_offsets = offsets.detach().clone()
            outside_band = (
                (cached_cosine < config.kappa - config.tolerance)
                | (cached_cosine > config.kappa + config.tolerance)
            )
            # The source slow optimizer stops once every semantic target converges,
            # even when a diversity term is enabled.
            if not outside_band.any():
                break
            loss = semantic_loss(cosine, config.kappa, config.tolerance)
            if config.diversity_weight:
                loss = loss + config.diversity_weight * diversity_loss(
                    normalized, num_variants
                )
            loss.backward()
            if offsets.grad is None:
                raise RuntimeError("Offset optimization produced no gradients.")
            if mask is not None:
                offsets.grad.mul_(mask)
            optimizer.step()

        if cached_encoded is None or cached_cosine is None or cached_offsets is None:
            raise RuntimeError("Offset optimization did not produce a representation.")
        converged = (
            (cached_cosine >= config.kappa - config.tolerance)
            & (cached_cosine <= config.kappa + config.tolerance)
        )
        return OptimizationResult(
            optimized=cached_encoded,
            original=EncodedPrompt(original.hidden.detach(), original.projected.detach()),
            offsets=cached_offsets,
            cosine_similarity=cached_cosine,
            converged=converged,
            steps=steps,
        )

    steps = 0
    cached_hidden: torch.Tensor | None = None
    cached_projected: torch.Tensor | None = None
    cached_normalized: torch.Tensor | None = None
    cached_cosine: torch.Tensor | None = None
    cached_offsets: torch.Tensor | None = None
    for step_index in range(config.max_steps):
        steps = step_index + 1
        optimizer.zero_grad(set_to_none=True)

        if cached_cosine is None:
            encoded = encode_offsets(offsets)
            normalized = functional.normalize(encoded.projected, dim=-1)
            cosine = (normalized * target).sum(dim=-1)
            inactive = (
                (cosine.detach() >= config.kappa - config.tolerance)
                & (cosine.detach() <= config.kappa + config.tolerance)
            )
            active = ~inactive
            if not active.any():
                cached_hidden = encoded.hidden.detach().clone()
                cached_projected = encoded.projected.detach().clone()
                cached_normalized = normalized.detach().clone()
                cached_cosine = cosine.detach().clone()
                cached_offsets = offsets.detach().clone()
                break
            active_indices = active.nonzero(as_tuple=True)[0]
            active_encoded = EncodedPrompt(
                hidden=encoded.hidden.index_select(0, active_indices),
                projected=encoded.projected.index_select(0, active_indices),
            )
            active_normalized = normalized.index_select(0, active_indices)
            active_cosine = cosine.index_select(0, active_indices)
            mixed = normalized.detach().clone()
            mixed.index_copy_(0, active_indices, active_normalized)
        else:
            inactive = (
                (cached_cosine >= config.kappa - config.tolerance)
                & (cached_cosine <= config.kappa + config.tolerance)
            )
            active = ~inactive
            if not active.any():
                break
            active_indices = active.nonzero(as_tuple=True)[0]
            if encode_subset is None:
                encoded = encode_offsets(offsets)
                active_encoded = EncodedPrompt(
                    hidden=encoded.hidden.index_select(0, active_indices),
                    projected=encoded.projected.index_select(0, active_indices),
                )
            else:
                active_encoded = encode_subset(
                    offsets.index_select(0, active_indices), active_indices
                )
            active_normalized = functional.normalize(
                active_encoded.projected, dim=-1
            )
            active_cosine = (
                active_normalized * target.index_select(0, active_indices)
            ).sum(dim=-1)
            if cached_normalized is None:
                raise RuntimeError("Missing normalized representation cache.")
            mixed = cached_normalized.clone()
            mixed.index_copy_(0, active_indices, active_normalized)

        loss = semantic_loss(
            active_cosine, config.kappa, config.tolerance
        )
        if config.diversity_weight:
            loss = loss + config.diversity_weight * diversity_loss(
                mixed, num_variants, active
            )
        loss.backward()
        if offsets.grad is None:
            raise RuntimeError("Offset optimization produced no gradients.")
        offsets.grad[inactive] = 0
        if mask is not None:
            offsets.grad.mul_(mask)
        cached_offsets = offsets.detach().clone()
        optimizer.step()

        with torch.no_grad():
            if cached_hidden is None:
                cached_hidden = encoded.hidden.detach().clone()
                cached_projected = encoded.projected.detach().clone()
                cached_normalized = normalized.detach().clone()
                cached_cosine = cosine.detach().clone()
            else:
                cached_hidden.index_copy_(
                    0, active_indices, active_encoded.hidden.detach()
                )
                cached_projected.index_copy_(
                    0, active_indices, active_encoded.projected.detach()
                )
                cached_normalized.index_copy_(
                    0, active_indices, active_normalized.detach()
                )
                cached_cosine.index_copy_(
                    0, active_indices, active_cosine.detach()
                )

    if (
        cached_hidden is None
        or cached_projected is None
        or cached_normalized is None
        or cached_cosine is None
        or cached_offsets is None
    ):
        raise RuntimeError("Offset optimization did not produce a representation.")
    converged = (
        (cached_cosine >= config.kappa - config.tolerance)
        & (cached_cosine <= config.kappa + config.tolerance)
    )
    return OptimizationResult(
        optimized=EncodedPrompt(cached_hidden.detach(), cached_projected.detach()),
        original=EncodedPrompt(original.hidden.detach(), original.projected.detach()),
        offsets=cached_offsets.detach(),
        cosine_similarity=cached_cosine.detach(),
        converged=converged.detach(),
        steps=steps,
    )
