"""TPSO integration for Stable Diffusion 3.5 Medium."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

from tpso.config import ModelSpec, TPSOConfig
from tpso.optimization import initialize_offsets
from tpso.scheduling import coarse_to_fine_alphas
from tpso.text_encoder import prepare_clip_session

from .common import (
    optimize_session,
    result_tensors,
    save_images,
    scheduler_step_count,
    shared_latents,
)

T5_MAX_SEQUENCE_LENGTH = 77


class StableDiffusion3Adapter:
    def __init__(
        self,
        pipeline: object,
        spec: ModelSpec,
        device: str,
        optimization_components: dict[str, tuple[object, torch.nn.Module, torch.nn.Module]]
        | None = None,
    ):
        self.pipeline = pipeline
        self.spec = spec
        self.device = device
        self.optimization_components = optimization_components

    @classmethod
    def load(
        cls,
        spec: ModelSpec,
        *,
        device: str,
        dtype: torch.dtype,
        local_files_only: bool = False,
    ):
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
        from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
        from transformers import CLIPTextModelWithProjection

        class StochasticFlowScheduler(FlowMatchEulerDiscreteScheduler):
            def __init__(self, *args, noise_scale: float = 0.03, **kwargs):
                super().__init__(*args, **kwargs)
                self._noise_scale = noise_scale

            def step(
                self,
                model_output,
                timestep,
                sample,
                generator=None,
                return_dict=True,
            ):
                output = super().step(
                    model_output,
                    timestep,
                    sample,
                    generator=generator,
                    return_dict=True,
                )
                output.prev_sample = (
                    output.prev_sample
                    + torch.randn_like(sample) * self._noise_scale
                )
                if not return_dict:
                    return (output.prev_sample,)
                return output

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            spec.model_id,
            revision=spec.model_revision,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=local_files_only,
        ).to(device)
        pipeline.scheduler = StochasticFlowScheduler.from_config(
            pipeline.scheduler.config, noise_scale=0.03
        )
        clip_l_encoder = CLIPTextModelWithProjection.from_pretrained(
            spec.model_id,
            subfolder="text_encoder",
            revision=spec.model_revision,
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        )
        clip_g_encoder = CLIPTextModelWithProjection.from_pretrained(
            spec.model_id,
            subfolder="text_encoder_2",
            revision=spec.model_revision,
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        )
        clip_l_projection = CLIPTextModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float32,
            use_safetensors=False,
            local_files_only=local_files_only,
        ).text_projection
        clip_g_projection = CLIPTextModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            torch_dtype=torch.float32,
            use_safetensors=False,
            local_files_only=local_files_only,
        ).text_projection
        components = {
            "clip_l": (
                pipeline.tokenizer,
                clip_l_encoder.text_model,
                clip_l_projection,
            ),
            "clip_g": (
                pipeline.tokenizer_2,
                clip_g_encoder.text_model,
                clip_g_projection,
            ),
        }
        return cls(pipeline, spec, device, components)

    def optimize(
        self,
        prompts: str | Sequence[str],
        num_variants: int,
        config: TPSOConfig,
        *,
        seed: int,
        trainable_token_count: int | None = None,
        active_subset: bool = True,
        optimization_autocast_dtype: torch.dtype = torch.float16,
    ) -> dict[str, dict[str, torch.Tensor]]:
        components = self.optimization_components or {
            "clip_l": (
                self.pipeline.tokenizer,
                self.pipeline.text_encoder.text_model,
                self.pipeline.text_encoder.text_projection,
            ),
            "clip_g": (
                self.pipeline.tokenizer_2,
                self.pipeline.text_encoder_2.text_model,
                self.pipeline.text_encoder_2.text_projection,
            ),
        }
        original_dtypes = {
            name: (next(text_model.parameters()).dtype, next(projection.parameters()).dtype)
            for name, (_, text_model, projection) in components.items()
        }
        try:
            sessions = {
                name: prepare_clip_session(
                    tokenizer,
                    text_model,
                    projection,
                    prompts,
                    num_variants,
                    device=self.device,
                    hidden_state_index=-2,
                    trainable_token_count=trainable_token_count,
                )
                for name, (tokenizer, text_model, projection) in components.items()
            }
            torch.manual_seed(seed)
            initial_offsets = {
                name: initialize_offsets(
                    session.offset_shape,
                    config,
                    device=self.device,
                    dtype=torch.float32,
                    generator=None,
                )
                for name, session in sessions.items()
            }
            results = {}
            for name, session in sessions.items():
                result = optimize_session(
                    session,
                    num_variants=num_variants,
                    config=config,
                    device=self.device,
                    seed=None,
                    autocast_dtype=optimization_autocast_dtype,
                    initial_offsets=initial_offsets[name],
                    active_subset=active_subset,
                )
                with torch.no_grad():
                    result = replace(result, optimized=session.encode(result.offsets))
                results[name] = result_tensors(result)
        finally:
            for name, (_, text_model, projection) in components.items():
                text_dtype, projection_dtype = original_dtypes[name]
                text_model.to(dtype=text_dtype)
                projection.to(dtype=projection_dtype)
        return results

    def _compose_at_step(
        self,
        values: dict[str, dict[str, torch.Tensor]],
        t5: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_l = values["clip_l"]
        clip_g = values["clip_g"]
        hidden_l = (
            alpha * clip_l["optimized_hidden"]
            + (1.0 - alpha) * clip_l["original_hidden"]
        )
        hidden_g = (
            alpha * clip_g["optimized_hidden"]
            + (1.0 - alpha) * clip_g["original_hidden"]
        )
        pooled_l = (
            alpha * clip_l["optimized_projected"]
            + (1.0 - alpha) * clip_l["original_projected"]
        )
        pooled_g = (
            alpha * clip_g["optimized_projected"]
            + (1.0 - alpha) * clip_g["original_projected"]
        )
        clip = torch.cat([hidden_l, hidden_g], dim=-1)
        clip = torch.nn.functional.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))
        dtype = self.pipeline.transformer.dtype
        return (
            torch.cat([clip.to(dtype), t5.to(dtype)], dim=-2),
            torch.cat([pooled_l, pooled_g], dim=-1).to(dtype),
        )

    def generate(
        self,
        conditional: dict[str, dict[str, torch.Tensor]],
        unconditional: dict[str, dict[str, torch.Tensor]],
        *,
        prompts: Sequence[str],
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
        max_sequence_length: int = T5_MAX_SEQUENCE_LENGTH,
        overwrite: bool = False,
    ) -> list[Path]:
        schedule_steps = scheduler_step_count(self.pipeline, num_steps, self.device)
        alphas = coarse_to_fine_alphas(
            schedule_steps,
            config.scheduler_ratio,
            device=self.device,
            dtype=conditional["clip_l"]["optimized_hidden"].dtype,
        )
        cond_t5 = self.pipeline._get_t5_prompt_embeds(
            prompt=list(prompts),
            num_images_per_prompt=num_variants,
            max_sequence_length=max_sequence_length,
            device=self.device,
        )
        uncond_t5 = self.pipeline._get_t5_prompt_embeds(
            prompt=[""] * len(prompts),
            num_images_per_prompt=num_variants,
            max_sequence_length=max_sequence_length,
            device=self.device,
        )
        count = cond_t5.shape[0]
        latents = shared_latents(
            self.pipeline,
            count,
            num_variants,
            height,
            width,
            self.pipeline.transformer.dtype,
            self.device,
            seed,
        )
        images = []
        # Run the complete optimized prompt batch together. The release target
        # assumes sufficient GPU memory and does not need the source script's
        # memory-saving two-way split.
        chunk_size = count
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            conditional_chunk = {
                name: {key: value[start:stop] for key, value in values.items()}
                for name, values in conditional.items()
            }
            unconditional_chunk = {
                name: {key: value[start:stop] for key, value in values.items()}
                for name, values in unconditional.items()
            }
            cond_t5_chunk = cond_t5[start:stop]
            uncond_t5_chunk = uncond_t5[start:stop]
            cond_prompt, cond_pooled = self._compose_at_step(
                conditional_chunk, cond_t5_chunk, alphas[0]
            )
            uncond_prompt, uncond_pooled = self._compose_at_step(
                unconditional_chunk, uncond_t5_chunk, alphas[0]
            )

            def update_prompt(
                _pipeline,
                step,
                _timestep,
                callback_kwargs,
                conditional_values=conditional_chunk,
                unconditional_values=unconditional_chunk,
                conditional_t5=cond_t5_chunk,
                unconditional_t5=uncond_t5_chunk,
            ):
                next_step = min(step + 1, schedule_steps - 1)
                next_cond_prompt, next_cond_pooled = self._compose_at_step(
                    conditional_values, conditional_t5, alphas[next_step]
                )
                next_uncond_prompt, next_uncond_pooled = self._compose_at_step(
                    unconditional_values, unconditional_t5, alphas[next_step]
                )
                return {
                    "prompt_embeds": torch.cat(
                        [next_uncond_prompt, next_cond_prompt]
                    ),
                    "pooled_prompt_embeds": torch.cat(
                        [next_uncond_pooled, next_cond_pooled]
                    ),
                }

            output = self.pipeline(
                prompt_embeds=cond_prompt,
                negative_prompt_embeds=uncond_prompt,
                pooled_prompt_embeds=cond_pooled,
                negative_pooled_prompt_embeds=uncond_pooled,
                latents=latents[start:stop],
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                max_sequence_length=max_sequence_length,
                callback_on_step_end=update_prompt,
                callback_on_step_end_tensor_inputs=[
                    "prompt_embeds",
                    "pooled_prompt_embeds",
                ],
            )
            images.extend(output.images)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return save_images(
            images,
            output_dir,
            output_paths=output_paths,
            output_size=output_size,
            overwrite=overwrite,
        )
