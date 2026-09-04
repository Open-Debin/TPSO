from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tpso.pipelines.common import (
    planned_image_paths,
    save_images,
    scheduler_step_count,
    shared_latents,
)


class IndexedLatentPipeline:
    unet = SimpleNamespace(config=SimpleNamespace(in_channels=1))

    def prepare_latents(
        self, count, channels, height, width, dtype, device, _generator, _latents
    ):
        values = torch.arange(count, dtype=dtype, device=device)
        return values.reshape(count, 1, 1, 1).expand(count, channels, height // 8, width // 8)


class ExtraStepScheduler:
    timesteps = []

    def set_timesteps(self, count, device=None):
        self.timesteps = list(range(count + 1))


def test_save_images_refuses_implicit_overwrite(tmp_path):
    images = [Image.new("RGB", (2, 2), "red")]
    paths = save_images(images, tmp_path)
    assert paths[0].is_file()

    with pytest.raises(FileExistsError, match="--overwrite"):
        save_images(images, tmp_path)

    assert save_images(images, tmp_path, overwrite=True) == paths


def test_save_images_does_not_leave_partial_outputs_on_encoding_failure(tmp_path):
    class BrokenImage:
        def save(self, *_args, **_kwargs):
            raise OSError("encoding failed")

    with pytest.raises(OSError, match="encoding failed"):
        save_images([Image.new("RGB", (2, 2), "red"), BrokenImage()], tmp_path)

    assert not list(tmp_path.iterdir())


def test_save_images_supports_explicit_paper_jpeg_paths(tmp_path):
    paths = [tmp_path / "0_0.jpg", tmp_path / "0_1.jpg"]
    result = save_images(
        [Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")],
        output_paths=paths,
        output_size=4,
    )

    assert result == [path.resolve() for path in paths]
    assert all(Image.open(path).format == "JPEG" for path in paths)
    assert all(Image.open(path).size == (4, 4) for path in paths)


def test_planned_image_paths_reject_conflicts_before_generation(tmp_path):
    (tmp_path / "00.png").touch()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        planned_image_paths(tmp_path, 2)


def test_shared_latents_reuses_noise_only_within_each_prompt():
    latents = shared_latents(
        IndexedLatentPipeline(),
        count=6,
        num_variants=3,
        height=8,
        width=8,
        dtype=torch.float32,
        device="cpu",
        seed=7,
    )

    assert latents[:, 0, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_shared_latents_rejects_incomplete_prompt_groups():
    with pytest.raises(ValueError, match="divisible"):
        shared_latents(
            IndexedLatentPipeline(),
            count=5,
            num_variants=3,
            height=8,
            width=8,
            dtype=torch.float32,
            device="cpu",
            seed=7,
        )


def test_scheduler_step_count_includes_scheduler_specific_timesteps():
    pipeline = SimpleNamespace(scheduler=ExtraStepScheduler())

    assert scheduler_step_count(pipeline, requested_steps=50, device="cpu") == 51
