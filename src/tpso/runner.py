"""High-level generation and unconditional-context workflows."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .config import MODEL_SPECS, ModelSpec, TPSOConfig, default_config, load_config
from .contexts import (
    DEFAULT_CACHE_DIR,
    download_context,
    load_context,
    sample_context,
    save_context,
)

UNCONDITIONAL_TRAINABLE_TOKENS = 20
SOURCE_CONTEXT_PROTOCOL = {
    "sd15": {"count": 350, "group_size": 35},
    "sd21": {"count": 350, "group_size": 35},
    "sd35": {"count": 300, "group_size": 30},
}


def resolve_dtype(device: str, precision: str) -> torch.dtype:
    if precision == "fp32" or device == "cpu":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision != "auto":
        raise ValueError("precision must be auto, fp16, bf16, or fp32.")
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def resolve_generation_settings(
    spec: ModelSpec,
    *,
    num_steps: int | None,
    guidance_scale: float | None,
    height: int | None,
    width: int | None,
) -> dict[str, int | float]:
    """Resolve and validate model defaults plus generation-time overrides."""

    settings: dict[str, int | float] = {
        "num_steps": spec.inference_steps if num_steps is None else num_steps,
        "guidance_scale": spec.guidance_scale if guidance_scale is None else guidance_scale,
        "height": spec.image_size if height is None else height,
        "width": spec.image_size if width is None else width,
    }
    for name in ("num_steps", "height", "width"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
    guidance = settings["guidance_scale"]
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(guidance)
    ):
        raise ValueError("guidance_scale must be a finite number.")
    if settings["num_steps"] <= 0:
        raise ValueError("num_steps must be positive.")
    if settings["guidance_scale"] <= 1.0:
        raise ValueError(
            "TPSO generation requires guidance_scale > 1 for classifier-free guidance."
        )
    if settings["height"] <= 0 or settings["width"] <= 0:
        raise ValueError("height and width must be positive.")
    size_multiple = 16 if spec.name == "sd35" else 8
    if settings["height"] % size_multiple or settings["width"] % size_multiple:
        raise ValueError(
            f"height and width must be divisible by {size_multiple} for {spec.name}."
        )
    return settings


def load_adapter(
    spec: ModelSpec,
    *,
    device: str,
    dtype: torch.dtype,
    local_files_only: bool,
):
    if spec.name in {"sd15", "sd21"}:
        from .pipelines.stable_diffusion import StableDiffusionAdapter

        return StableDiffusionAdapter.load(
            spec, device=device, dtype=dtype, local_files_only=local_files_only
        )
    from .pipelines.stable_diffusion3 import StableDiffusion3Adapter

    return StableDiffusion3Adapter.load(
        spec, device=device, dtype=dtype, local_files_only=local_files_only
    )


def _context_metadata(
    spec: ModelSpec, config: TPSOConfig, count: int, group_size: int
) -> dict[str, object]:
    return {
        "config": asdict(config),
        "optimizer": config.optimizer,
        "count": count,
        "group_size": group_size,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "projection_model_id": spec.projection_model_id,
        "projection_revision": spec.projection_revision,
        "unconditional_prompt": " ",
        "trainable_token_count": UNCONDITIONAL_TRAINABLE_TOKENS,
    }


def unconditional_config(spec: ModelSpec, conditional: TPSOConfig) -> TPSOConfig:
    """Return the context settings used by the archived generation scripts."""

    base = default_config(spec.name)
    diversity_weight = 0.0 if spec.name == "sd35" else conditional.diversity_weight
    return replace(
        base,
        kappa=conditional.kappa,
        diversity_weight=diversity_weight,
        max_steps=(
            200
            if spec.name in {"sd15", "sd21"} and diversity_weight > 0.0
            else base.max_steps
        ),
    )


def _context_cache_path(spec: ModelSpec, config: TPSOConfig) -> Path:
    """Keep custom rebuilds separate from the paper-default artifact cache."""

    destination = DEFAULT_CACHE_DIR / spec.context_filename
    if config == unconditional_config(spec, default_config(spec.name)):
        return destination
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()[:12]
    return destination.with_name(f"{destination.stem}.{fingerprint}{destination.suffix}")


def rebuild_context(
    adapter: object,
    spec: ModelSpec,
    config: TPSOConfig,
    *,
    count: int,
    seed: int,
    output_path: str | Path,
    group_size: int = 10,
) -> Path:
    if count <= 0 or group_size <= 0:
        raise ValueError("count and group_size must be positive.")
    groups = math.ceil(count / group_size)
    # The paper code optimizes 20 interior positions for a blank prompt. Inferring
    # the mask from its EOS position would produce an all-zero mask and no update.
    optimization_options: dict[str, object] = {
        "trainable_token_count": UNCONDITIONAL_TRAINABLE_TOKENS,
        "active_subset": not (
            spec.name in {"sd15", "sd21"} and config.diversity_weight > 0.0
        ),
    }
    if spec.name == "sd35":
        optimization_options["optimization_autocast_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    values = adapter.optimize(
        [" "] * groups,
        group_size,
        config,
        seed=seed,
        **optimization_options,
    )
    actual_count = groups * group_size
    return save_context(
        output_path,
        model=spec.name,
        encoders=values,
        metadata=_context_metadata(spec, config, actual_count, group_size),
    )


def resolve_unconditional(
    adapter: object,
    spec: ModelSpec,
    config: TPSOConfig,
    *,
    count: int,
    seed: int,
    context_path: str | Path | None,
    rebuild_unconditional: bool,
    local_files_only: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    context_config = unconditional_config(spec, config)
    source_context_config = unconditional_config(spec, default_config(spec.name))
    is_published_config = context_config == source_context_config
    path: Path | None = None
    if context_path and not rebuild_unconditional:
        path = Path(context_path).expanduser()
    elif not rebuild_unconditional and is_published_config:
        cached_path = DEFAULT_CACHE_DIR / spec.context_filename
        if cached_path.is_file():
            path = cached_path
        else:
            try:
                path = download_context(spec.name, local_files_only=local_files_only)
            except (OSError, RuntimeError):
                custom_path = _context_cache_path(spec, context_config)
                if custom_path.is_file():
                    path = custom_path
                else:
                    warnings.warn(
                        "No published unconditional context is available; rebuilding it locally. "
                        "Pass --context-path to reuse an existing checkpoint.",
                        stacklevel=2,
                    )
    elif not rebuild_unconditional:
        custom_path = _context_cache_path(spec, context_config)
        if custom_path.is_file():
            path = custom_path
        else:
            warnings.warn(
                "No matching unconditional context is cached; rebuilding it locally.",
                stacklevel=2,
            )

    if path is None:
        destination = (
            Path(context_path).expanduser()
            if context_path
            else _context_cache_path(spec, context_config)
        )
        path = rebuild_context(
            adapter,
            spec,
            context_config,
            count=SOURCE_CONTEXT_PROTOCOL[spec.name]["count"],
            group_size=SOURCE_CONTEXT_PROTOCOL[spec.name]["group_size"],
            seed=seed,
            output_path=destination,
        )
    checkpoint = load_context(
        path,
        model=spec.name,
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        projection_model_id=spec.projection_model_id,
        projection_revision=spec.projection_revision,
        expected_group_size=SOURCE_CONTEXT_PROTOCOL[spec.name]["group_size"],
        device=adapter.device,
    )
    torch.manual_seed(seed)
    sampled = sample_context(checkpoint, count)
    return {
        name: {key: value.to(adapter.device) for key, value in values.items()}
        for name, values in sampled.items()
    }


def generate(
    *,
    model: str,
    prompts: str | Sequence[str],
    output_dir: str | Path,
    num_images: int = 4,
    seed: int = 2024,
    device: str = "cuda",
    precision: str = "auto",
    context_path: str | Path | None = None,
    rebuild_unconditional: bool = False,
    local_files_only: bool = False,
    config_path: str | Path | None = None,
    kappa: float | None = None,
    diversity_weight: float | None = None,
    scheduler_ratio: float | None = None,
    offset_init: str | None = None,
    offset_std: float | None = None,
    num_steps: int | None = None,
    guidance_scale: float | None = None,
    height: int | None = None,
    width: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Generate diverse images with one of the three paper backbones."""

    if model not in MODEL_SPECS:
        raise ValueError(f"Unsupported model {model!r}; choose from {tuple(MODEL_SPECS)}.")
    if isinstance(num_images, bool) or not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("num_images must be a positive integer.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompt_list:
        raise ValueError("At least one prompt is required.")
    if not all(isinstance(prompt, str) for prompt in prompt_list):
        raise TypeError("Every prompt must be a string.")
    if any(not prompt.strip() for prompt in prompt_list):
        raise ValueError("Prompts must contain at least one non-whitespace character.")
    from .pipelines.common import planned_image_paths

    planned_image_paths(output_dir, len(prompt_list) * num_images, overwrite=overwrite)

    if config_path:
        spec, config = load_config(config_path, model)
    else:
        spec, config = MODEL_SPECS[model], default_config(model)

    generation = resolve_generation_settings(
        spec,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )

    overrides = {
        key: value
        for key, value in {
            "kappa": kappa,
            "diversity_weight": diversity_weight,
            "scheduler_ratio": scheduler_ratio,
            "offset_init": offset_init,
            "offset_std": offset_std,
        }.items()
        if value is not None
    }
    config = replace(config, **overrides).validate()
    dtype = resolve_dtype(device, precision)
    adapter = load_adapter(
        spec, device=device, dtype=dtype, local_files_only=local_files_only
    )
    conditional = adapter.optimize(prompt_list, num_images, config, seed=seed)
    count = len(prompt_list) * num_images
    unconditional = resolve_unconditional(
        adapter,
        spec,
        config,
        count=count,
        seed=seed,
        context_path=context_path,
        rebuild_unconditional=rebuild_unconditional,
        local_files_only=local_files_only,
    )
    generation_args = {
        "conditional": conditional,
        "unconditional": unconditional,
        "num_variants": num_images,
        "config": config,
        **generation,
        "seed": seed,
        "output_dir": output_dir,
        "overwrite": overwrite,
    }
    if model == "sd35":
        generation_args.update(prompts=prompt_list)
    return adapter.generate(**generation_args)
