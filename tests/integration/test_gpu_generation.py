import os

import pytest
import torch

from tpso.runner import generate

RUN_GPU_TESTS = os.environ.get("TPSO_RUN_GPU_TESTS") == "1"


@pytest.mark.gpu
@pytest.mark.parametrize("model", ["sd15", "sd21", "sd35"])
@pytest.mark.skipif(
    not RUN_GPU_TESTS or not torch.cuda.is_available(),
    reason="Set TPSO_RUN_GPU_TESTS=1 on a CUDA host to run model smoke tests.",
)
def test_four_image_generation(model, tmp_path):
    paths = generate(
        model=model,
        prompts="A photograph of a red panda",
        output_dir=tmp_path / model,
        num_images=4,
        seed=2026,
        context_path=tmp_path / f"{model}-context.pt",
        rebuild_unconditional=True,
    )
    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert (tmp_path / f"{model}-context.pt").is_file()
