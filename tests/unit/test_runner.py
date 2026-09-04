from dataclasses import asdict, replace

import pytest
import torch

from tpso import runner
from tpso.config import MODEL_SPECS, default_config
from tpso.contexts import save_context


class CpuAdapter:
    device = "cpu"


class RecordingAdapter:
    device = "cpu"

    def __init__(self):
        self.optimized = None
        self.generated = None

    def optimize(self, prompts, num_variants, config, *, seed, **kwargs):
        self.optimized = (prompts, num_variants, config, seed, kwargs)
        return {"conditional": torch.tensor(1)}

    def generate(self, **kwargs):
        self.generated = kwargs
        return kwargs["output_paths"]


def test_default_cached_context_is_reused(tmp_path, monkeypatch):
    spec = MODEL_SPECS["sd15"]
    path = tmp_path / spec.context_filename
    values = {
        "clip": {
            "optimized_hidden": torch.ones(2, 77, 768),
            "original_hidden": torch.zeros(2, 77, 768),
            "optimized_projected": torch.ones(2, 768),
            "original_projected": torch.zeros(2, 768),
        }
    }
    save_context(
        path,
        model="sd15",
        encoders=values,
        metadata={
            "count": 2,
            "group_size": runner.SOURCE_CONTEXT_PROTOCOL["sd15"]["group_size"],
            "config": asdict(
                runner.unconditional_config(spec, default_config("sd15"))
            ),
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            "projection_model_id": spec.projection_model_id,
            "projection_revision": spec.projection_revision,
        },
    )
    monkeypatch.setattr(runner, "DEFAULT_CACHE_DIR", tmp_path)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("A cached context should avoid download and rebuild.")

    monkeypatch.setattr(runner, "download_context", unexpected)
    monkeypatch.setattr(runner, "rebuild_context", unexpected)
    sampled = runner.resolve_unconditional(
        CpuAdapter(),
        spec,
        default_config("sd15"),
        count=2,
        seed=7,
        context_path=None,
        rebuild_unconditional=False,
        local_files_only=True,
    )
    assert sampled["clip"]["optimized_hidden"].shape == (2, 77, 768)


def test_explicit_context_can_use_different_optimization_settings(tmp_path):
    spec = MODEL_SPECS["sd15"]
    path = tmp_path / "context.pt"
    values = {
        "clip": {
            "optimized_hidden": torch.ones(2, 77, 768),
            "original_hidden": torch.zeros(2, 77, 768),
            "optimized_projected": torch.ones(2, 768),
            "original_projected": torch.zeros(2, 768),
        }
    }
    legacy_config = asdict(runner.unconditional_config(spec, default_config("sd15")))
    legacy_config.update(kappa=0.7, offset_init="normal")
    save_context(
        path,
        model="sd15",
        encoders=values,
        metadata={
            "count": 2,
            "group_size": runner.SOURCE_CONTEXT_PROTOCOL["sd15"]["group_size"],
            "config": legacy_config,
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            "projection_model_id": spec.projection_model_id,
            "projection_revision": spec.projection_revision,
        },
    )

    sampled = runner.resolve_unconditional(
        CpuAdapter(),
        spec,
        replace(default_config("sd15"), kappa=0.9, diversity_weight=0.0),
        count=2,
        seed=7,
        context_path=path,
        rebuild_unconditional=False,
        local_files_only=True,
    )

    assert sampled["clip"]["optimized_hidden"].shape == (2, 77, 768)


def test_non_cfg_guidance_is_rejected_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="guidance_scale > 1"):
        runner.generate(
            model="sd15",
            prompts="a red panda",
            output_dir=tmp_path,
            device="cpu",
            guidance_scale=1.0,
        )


def test_sd35_size_is_validated_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="divisible by 16"):
        runner.generate(
            model="sd35",
            prompts="a red panda",
            output_dir=tmp_path,
            device="cpu",
            height=520,
            width=1024,
        )


@pytest.mark.parametrize("num_images", [True, 1.5, 0])
def test_invalid_num_images_is_rejected_before_model_loading(tmp_path, num_images):
    with pytest.raises(ValueError, match="positive integer"):
        runner.generate(
            model="sd15",
            prompts="a red panda",
            output_dir=tmp_path,
            num_images=num_images,
            device="cpu",
        )


def test_non_string_prompt_is_rejected_before_model_loading(tmp_path):
    with pytest.raises(TypeError, match="prompt must be a string"):
        runner.generate(
            model="sd15",
            prompts=["a red panda", 7],
            output_dir=tmp_path,
            device="cpu",
        )


def test_blank_prompt_is_rejected_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="non-whitespace"):
        runner.generate(
            model="sd15",
            prompts="   ",
            output_dir=tmp_path,
            device="cpu",
        )


def test_auto_precision_is_fp32_off_cuda():
    assert runner.resolve_dtype("cpu", "auto") == torch.float32
    assert runner.resolve_dtype("mps", "auto") == torch.float32


def test_auto_precision_matches_paper_experiment_fp16_on_cuda():
    assert runner.resolve_dtype("cuda", "auto") == torch.float16


def test_custom_context_rebuild_does_not_overwrite_paper_cache(tmp_path, monkeypatch):
    spec = MODEL_SPECS["sd15"]
    monkeypatch.setattr(runner, "DEFAULT_CACHE_DIR", tmp_path)

    default_context = runner.unconditional_config(spec, default_config("sd15"))
    default_path = runner._context_cache_path(spec, default_context)
    custom_path = runner._context_cache_path(
        spec,
        replace(default_context, kappa=0.7),
    )

    assert default_path == tmp_path / spec.context_filename
    assert custom_path.parent == tmp_path
    assert custom_path != default_path
    assert custom_path.suffix == ".pt"


def test_existing_output_is_rejected_before_model_loading(tmp_path, monkeypatch):
    (tmp_path / "0_0.jpg").touch()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Output conflicts must be detected before model loading.")

    monkeypatch.setattr(runner, "load_adapter", unexpected)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner.generate(
            model="sd15",
            prompts="a red panda",
            output_dir=tmp_path,
            num_images=1,
            device="cpu",
        )


def test_generate_orchestrates_multiple_prompts_and_variants(tmp_path, monkeypatch):
    adapter = RecordingAdapter()
    monkeypatch.setattr(runner, "load_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(
        runner,
        "resolve_unconditional",
        lambda *_args, **_kwargs: {"unconditional": torch.tensor(1)},
    )

    paths = runner.generate(
        model="sd15",
        prompts=["red panda", "wooden chair"],
        output_dir=tmp_path,
        num_images=3,
        seed=9,
        device="cpu",
    )

    assert adapter.optimized[0] == ["red panda", "wooden chair"]
    assert adapter.optimized[1] == 3
    assert adapter.generated["num_variants"] == 3
    assert adapter.generated["conditional"] == {"conditional": torch.tensor(1)}
    assert adapter.generated["unconditional"] == {"unconditional": torch.tensor(1)}
    assert paths == [
        tmp_path / "0_0.jpg",
        tmp_path / "0_1.jpg",
        tmp_path / "0_2.jpg",
        tmp_path / "1_0.jpg",
        tmp_path / "1_1.jpg",
        tmp_path / "1_2.jpg",
    ]
    assert adapter.generated["output_paths"] == paths


def test_rebuild_uses_archived_unconditional_token_mask(tmp_path, monkeypatch):
    adapter = RecordingAdapter()
    saved = {}

    def record_save(path, **kwargs):
        saved.update(kwargs)
        return path

    monkeypatch.setattr(runner, "save_context", record_save)
    output = tmp_path / "context.pt"
    result = runner.rebuild_context(
        adapter,
        MODEL_SPECS["sd15"],
        default_config("sd15"),
        count=12,
        group_size=10,
        seed=7,
        output_path=output,
    )

    assert result == output
    assert adapter.optimized[0] == [" ", " "]
    assert adapter.optimized[1] == 10
    assert adapter.optimized[4] == {
        "trainable_token_count": runner.UNCONDITIONAL_TRAINABLE_TOKENS,
        "active_subset": False,
    }
    assert saved["metadata"]["count"] == 20
    assert saved["metadata"]["group_size"] == 10


def test_sd35_unconditional_context_is_semantic_only():
    spec = MODEL_SPECS["sd35"]
    conditional = default_config("sd35")

    assert conditional.diversity_weight == 10.0
    context = runner.unconditional_config(spec, conditional)
    assert context.diversity_weight == 0.0
    assert context.max_steps == 200


@pytest.mark.parametrize("model", ["sd15", "sd21"])
def test_diverse_sd12_context_uses_slow_optimizer_budget(model):
    context = runner.unconditional_config(MODEL_SPECS[model], default_config(model))

    assert context.diversity_weight == 1.0
    assert context.max_steps == 200


@pytest.mark.parametrize(
    "model,count,group_size",
    [("sd15", 350, 35), ("sd21", 350, 35), ("sd35", 300, 30)],
)
def test_source_context_protocol(model, count, group_size):
    assert runner.SOURCE_CONTEXT_PROTOCOL[model] == {
        "count": count,
        "group_size": group_size,
    }
