from types import SimpleNamespace

import torch
from PIL import Image

from tpso.config import MODEL_SPECS, TPSOConfig
from tpso.pipelines.stable_diffusion import StableDiffusionAdapter
from tpso.pipelines.stable_diffusion3 import (
    T5_MAX_SEQUENCE_LENGTH,
    StableDiffusion3Adapter,
)


class FakeDenoiser:
    dtype = torch.float32
    config = SimpleNamespace(in_channels=4)


class FakeScheduler:
    def __init__(self):
        self.timesteps = []

    def set_timesteps(self, count, device=None):
        self.timesteps = list(range(count + 1))


class FakePipeline:
    def __init__(self, family):
        self.family = family
        self.unet = FakeDenoiser()
        self.transformer = FakeDenoiser()
        self.scheduler = FakeScheduler()
        self.callback_verified = False
        self.call_count = 0
        self.t5_max_sequence_length = None
        self.call_max_sequence_length = None

    def prepare_latents(
        self, count, channels, height, width, dtype, device, generator, _latents
    ):
        return torch.zeros(count, channels, height // 8, width // 8, dtype=dtype, device=device)

    def _get_t5_prompt_embeds(
        self, *, prompt, num_images_per_prompt, max_sequence_length, device
    ):
        self.t5_max_sequence_length = max_sequence_length
        count = len(prompt) * num_images_per_prompt
        return torch.zeros(count, max_sequence_length, 8, device=device)

    def __call__(self, **kwargs):
        self.call_count += 1
        self.call_max_sequence_length = kwargs.get("max_sequence_length")
        prompt = torch.cat([kwargs["negative_prompt_embeds"], kwargs["prompt_embeds"]])
        callback_kwargs = {"prompt_embeds": prompt}
        if self.family == "sd3":
            pooled = torch.cat(
                [kwargs["negative_pooled_prompt_embeds"], kwargs["pooled_prompt_embeds"]]
            )
            callback_kwargs["pooled_prompt_embeds"] = pooled
        output = kwargs["callback_on_step_end"](self, 0, None, callback_kwargs)
        assert output["prompt_embeds"].shape[0] == prompt.shape[0]
        if self.family == "sd3":
            assert output["pooled_prompt_embeds"].shape == pooled.shape
            assert torch.allclose(
                output["pooled_prompt_embeds"],
                torch.full_like(output["pooled_prompt_embeds"], 0.5),
            )
        self.callback_verified = True
        count = kwargs["latents"].shape[0]
        return SimpleNamespace(images=[Image.new("RGB", (2, 2), "blue") for _ in range(count)])


def encoder_values(count, sequence, width, pooled_width):
    return {
        "optimized_hidden": torch.ones(count, sequence, width),
        "original_hidden": torch.zeros(count, sequence, width),
        "optimized_projected": torch.ones(count, pooled_width),
        "original_projected": torch.zeros(count, pooled_width),
    }


def test_stable_diffusion_stepwise_callback(tmp_path):
    pipeline = FakePipeline("sd")
    adapter = StableDiffusionAdapter(pipeline, torch.nn.Linear(4, 4), MODEL_SPECS["sd15"], "cpu")
    conditional = {"clip": encoder_values(2, 3, 4, 4)}
    unconditional = {"clip": encoder_values(2, 3, 4, 4)}
    paths = adapter.generate(
        conditional,
        unconditional,
        num_variants=2,
        config=TPSOConfig(scheduler_ratio=1.0),
        num_steps=2,
        guidance_scale=7.5,
        height=16,
        width=16,
        seed=1,
        output_dir=tmp_path,
    )
    assert pipeline.callback_verified
    assert pipeline.call_count == 1
    assert len(paths) == 2


def test_stable_diffusion3_updates_pooled_callback_tensor(tmp_path):
    pipeline = FakePipeline("sd3")
    adapter = StableDiffusion3Adapter(pipeline, MODEL_SPECS["sd35"], "cpu")
    conditional = {
        "clip_l": encoder_values(2, 3, 2, 2),
        "clip_g": encoder_values(2, 3, 3, 3),
    }
    unconditional = {
        "clip_l": encoder_values(2, 3, 2, 2),
        "clip_g": encoder_values(2, 3, 3, 3),
    }
    paths = adapter.generate(
        conditional,
        unconditional,
        prompts=["a red panda"],
        num_variants=2,
        config=TPSOConfig(scheduler_ratio=1.0),
        num_steps=2,
        guidance_scale=7.0,
        height=16,
        width=16,
        seed=1,
        output_dir=tmp_path,
    )
    assert pipeline.callback_verified
    assert pipeline.call_count == 1
    assert len(paths) == 2
    assert pipeline.t5_max_sequence_length == T5_MAX_SEQUENCE_LENGTH
    assert pipeline.call_max_sequence_length == T5_MAX_SEQUENCE_LENGTH
