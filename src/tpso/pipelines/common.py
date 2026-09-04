"""Shared helpers for diffusion pipeline adapters."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from tpso.config import TPSOConfig
from tpso.optimization import OptimizationResult, optimize_prompt_offsets
from tpso.text_encoder import CLIPOffsetSession


def optimize_session(
    session: CLIPOffsetSession,
    *,
    num_variants: int,
    config: TPSOConfig,
    device: torch.device | str,
    seed: int | None,
    autocast_dtype: torch.dtype | None = None,
    initial_offsets: torch.Tensor | None = None,
    active_subset: bool = True,
) -> OptimizationResult:
    if seed is not None:
        torch.manual_seed(seed)
    device_type = torch.device(device).type
    enabled = autocast_dtype is not None and device_type == "cuda"
    with torch.autocast(
        device_type=device_type,
        dtype=autocast_dtype,
        enabled=enabled,
    ):
        return optimize_prompt_offsets(
            session.encode,
            session.original(),
            session.offset_shape,
            num_variants,
            config,
            device=device,
            generator=None,
            trainable_mask=session.token_mask,
            encode_subset=session.encode_subset,
            initial_offsets=initial_offsets,
            active_subset=active_subset,
        )


def result_tensors(result: OptimizationResult) -> dict[str, torch.Tensor]:
    return {
        "optimized_hidden": result.optimized.hidden,
        "original_hidden": result.original.hidden,
        "optimized_projected": result.optimized.projected,
        "original_projected": result.original.projected,
    }


def shared_latents(
    pipeline: Any,
    count: int,
    num_variants: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device | str,
    seed: int,
) -> torch.Tensor:
    if num_variants <= 0 or count % num_variants:
        raise ValueError("count must be divisible by a positive num_variants.")
    channels = getattr(pipeline, "unet", getattr(pipeline, "transformer", None)).config.in_channels
    prompt_count = count // num_variants
    latent = pipeline.prepare_latents(
        prompt_count,
        channels,
        height,
        width,
        dtype,
        device,
        None,
        None,
    )
    return latent.repeat_interleave(num_variants, dim=0)


def scheduler_step_count(
    pipeline: Any, requested_steps: int, device: torch.device | str
) -> int:
    """Return the actual denoising-step count, including scheduler-specific extras."""

    pipeline.scheduler.set_timesteps(requested_steps, device=device)
    count = len(pipeline.scheduler.timesteps)
    if count <= 0:
        raise RuntimeError("The diffusion scheduler produced no timesteps.")
    return count


def save_images(
    images: list[Any],
    output_dir: str | Path | None = None,
    *,
    output_paths: Sequence[str | Path] | None = None,
    output_size: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    if (output_dir is None) == (output_paths is None):
        raise ValueError("Provide exactly one of output_dir or output_paths.")
    if output_paths is None:
        paths = planned_image_paths(output_dir, len(images), overwrite=overwrite)
    else:
        paths = [Path(path).expanduser().resolve() for path in output_paths]
        if len(paths) != len(images):
            raise ValueError("output_paths must contain one path per image.")
        if len(paths) != len(set(paths)):
            raise ValueError("output_paths must be unique.")
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        conflicts = [path for path in paths if path.exists()]
        if conflicts and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {len(conflicts)} image(s); pass --overwrite."
            )
    temporary_paths = [
        path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp") for path in paths
    ]
    try:
        for image, temporary, path in zip(images, temporary_paths, paths, strict=True):
            if output_size is not None:
                if output_size <= 0:
                    raise ValueError("output_size must be positive.")
                from PIL import Image

                image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
            image_format = "JPEG" if path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
            image.save(temporary, format=image_format)
        for temporary, path in zip(temporary_paths, paths, strict=True):
            temporary.replace(path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return paths


def planned_image_paths(
    output_dir: str | Path, count: int, *, overwrite: bool = False
) -> list[Path]:
    """Resolve output paths and reject conflicts before expensive generation."""

    if count <= 0:
        raise ValueError("Image count must be positive.")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{index:02d}.png" for index in range(count)]
    conflicts = [path for path in paths if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(conflicts)} image(s) in {directory}; "
            "choose another output directory or pass --overwrite."
        )
    return paths
