import hashlib
import json

import pytest
from PIL import Image

from tpso import benchmark
from tpso.benchmark import PaperExperiment, PromptRecord


def test_official_table_registry_matches_paper_rows():
    names = ("table1", "table2", "table3", "table4", "table5")
    groups = {name: benchmark.select_experiments(name) for name in names}

    assert [(item.model, item.prompt_count) for item in groups["table1"]] == [
        ("sd15", 5_000),
        ("sd21", 5_000),
        ("sd35", 5_000),
    ]
    assert [item.overrides["kappa"] for item in groups["table2"]] == [0.7, 0.8, 0.9]
    assert [item.overrides["scheduler_ratio"] for item in groups["table3"]] == [
        0.4,
        0.6,
        -0.4,
        -0.6,
    ]
    assert len(groups["table4"]) == 8
    assert [item.overrides["diversity_weight"] for item in groups["table5"]] == [
        0.0,
        5.0,
        10.0,
    ]


def test_local_coco_csv_is_checksum_pinned(tmp_path, monkeypatch):
    path = tmp_path / "captions.csv"
    path.write_text("file_name,caption\na.jpg,A red panda\n", encoding="utf-8")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(benchmark, "COCO_CAPTIONS_SHA256", checksum)

    assert benchmark.resolve_coco_csv(path) == path.resolve()
    assert benchmark.load_coco_prompts(path, 1) == [PromptRecord("a.jpg", "A red panda")]


def test_coco_checksum_mismatch_is_rejected(tmp_path):
    path = tmp_path / "captions.csv"
    path.write_text("file_name,caption\na.jpg,A red panda\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.resolve_coco_csv(path)


def test_indexed_image_paths_follow_archived_naming(tmp_path):
    paths = benchmark.indexed_image_paths(tmp_path, range(2, 4), num_variants=2)
    assert [path.name for path in paths] == ["2_0.jpg", "2_1.jpg", "3_0.jpg", "3_1.jpg"]


def test_resume_rejects_truncated_or_wrong_size_images(tmp_path):
    truncated = tmp_path / "0_0.jpg"
    truncated.write_bytes(b"not an image")
    wrong_size = tmp_path / "0_1.jpg"
    Image.new("RGB", (2, 2), "red").save(wrong_size)
    valid = tmp_path / "0_2.jpg"
    Image.new("RGB", (512, 512), "red").save(valid)

    assert not benchmark.valid_paper_image(truncated)
    assert not benchmark.valid_paper_image(wrong_size)
    assert benchmark.valid_paper_image(valid)


def test_experiment_run_is_resumable_with_fixed_batches(tmp_path, monkeypatch):
    class Adapter:
        device = "cpu"

        def __init__(self):
            self.generate_calls = 0

        def optimize(self, prompts, num_variants, config, *, seed):
            return {"prompts": prompts, "seed": seed}

        def generate(self, **kwargs):
            self.generate_calls += 1
            for path in kwargs["output_paths"]:
                Image.new("RGB", (512, 512), "red").save(path)
            return kwargs["output_paths"]

    adapter = Adapter()
    monkeypatch.setattr(benchmark, "load_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(benchmark, "resolve_unconditional", lambda *_args, **_kwargs: {})
    experiment = PaperExperiment("table2", "test", "sd15", 2, {"diversity_weight": 0.0})
    records = [PromptRecord("a.jpg", "A"), PromptRecord("b.jpg", "B")]
    arguments = {
        "output_root": tmp_path,
        "batch_size": 1,
        "seed": 7,
        "device": "cpu",
        "precision": "fp32",
        "context_dir": None,
        "local_files_only": True,
    }

    output = benchmark.run_experiment(experiment, records, **arguments)
    benchmark.run_experiment(experiment, records, **arguments)

    assert adapter.generate_calls == 2
    assert len(list(output.glob("*.jpg"))) == 20
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["filename_template"] == "{prompt_id}_{variant_id}.jpg"
    assert manifest["expected_images"] == 20
