"""TPSO integration for Stable Diffusion 1.5 and 2.1."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch

from tpso.config import ModelSpec, TPSOConfig
from tpso.scheduling import coarse_to_fine_alphas
from tpso.text_encoder import prepare_clip_session

from .common import (
    optimize_session,
    result_tensors,
    save_images,
    scheduler_step_count,
    shared_latents,
)


class StableDiffusionAdapter:
    def __init__(
        self,
        pipeline: object,
        projection: torch.nn.Module,
        spec: ModelSpec,
        device: str,
        text_model: torch.nn.Module | None = None,
    ):
        self.pipeline = pipeline
        self.projection = projection
        self.text_model = text_model
        self.spec = spec
        self.device = device

    @classmethod
    def load(
        cls,
        spec: ModelSpec,
        *,
        device: str,
        dtype: torch.dtype,
        local_files_only: bool = False,
    ):
        # Transformers 5 otherwise starts a background Hub conversion thread
        # for the projection-only CLIP checkpoints and prints a non-fatal traceback.
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
        from diffusers import StableDiffusionPipeline
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        pipeline = StableDiffusionPipeline.from_pretrained(
            spec.model_id,
            revision=spec.model_revision,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            local_files_only=local_files_only,
        ).to(device)
        projection_model = CLIPTextModelWithProjection.from_pretrained(
            spec.projection_model_id,
            revision=spec.projection_revision,
            torch_dtype=torch.float32,
            use_safetensors=False,
            local_files_only=local_files_only,
        )
        optimization_model = CLIPTextModel.from_pretrained(
            spec.model_id,
            subfolder="text_encoder",
            revision=spec.model_revision,
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        )
        projection = projection_model.text_projection
        del projection_model
        return cls(pipeline, projection, spec, device, optimization_model.text_model)

    def optimize(
        self,
        prompts: str | Sequence[str],
        num_variants: int,
        config: TPSOConfig,
        *,
        seed: int,
        trainable_token_count: int | None = None,
        active_subset: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        text_model = self.text_model or self.pipeline.text_encoder.text_model
        text_dtype = next(text_model.parameters()).dtype
        projection_dtype = next(self.projection.parameters()).dtype
        try:
            session = prepare_clip_session(
                self.pipeline.tokenizer,
                text_model,
                self.projection,
                prompts,
                num_variants,
                device=self.device,
                trainable_token_count=trainable_token_count,
                token_count_mode="whitespace",
            )
            result = optimize_session(
                session,
                num_variants=num_variants,
                config=config,
                device=self.device,
                seed=seed,
                active_subset=active_subset,
            )
        finally:
            text_model.to(dtype=text_dtype)
            self.projection.to(dtype=projection_dtype)
        return {"clip": result_tensors(result)}

    def generate(
        self,
        conditional: dict[str, dict[str, torch.Tensor]],
        unconditional: dict[str, dict[str, torch.Tensor]],
        *,
        num_variants: int,
        config: TPSOConfig,
        num_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        seed: int,
        output_dir: str | Path | None = None,
        output_paths: Sequence[str | Path] | None = None,
        output_size: int | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        cond = conditional["clip"]
        uncond = unconditional["clip"]
        schedule_steps = scheduler_step_count(self.pipeline, num_steps, self.device)
        alphas = coarse_to_fine_alphas(
            schedule_steps,
            config.scheduler_ratio,
            device=self.device,
            dtype=cond["optimized_hidden"].dtype,
        )

        def interpolate(values, step):
            alpha = alphas[step]
            return (
                alpha * values["optimized_hidden"]
                + (1.0 - alpha) * values["original_hidden"]
            ).to(self.pipeline.unet.dtype)

        cond_prompt = interpolate(cond, 0)
        uncond_prompt = interpolate(uncond, 0)
        count = cond_prompt.shape[0]

        def update_prompt(_pipeline, step, _timestep, callback_kwargs):
            next_step = min(step + 1, schedule_steps - 1)
            return {
                "prompt_embeds": torch.cat(
                    [interpolate(uncond, next_step), interpolate(cond, next_step)]
                )
            }

        latents = shared_latents(
            self.pipeline,
            count,
            num_variants,
            height,
            width,
            self.pipeline.unet.dtype,
            self.device,
            seed,
        )
        output = self.pipeline(
            prompt_embeds=cond_prompt,
            negative_prompt_embeds=uncond_prompt,
            latents=latents,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            callback_on_step_end=update_prompt,
            callback_on_step_end_tensor_inputs=["prompt_embeds"],
        )
        return save_images(
            output.images,
            output_dir,
            output_paths=output_paths,
            output_size=output_size,
            overwrite=overwrite,
        )
