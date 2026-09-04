import json

import pytest
import torch

from tpso import __version__
from tpso.contexts import (
    ARTIFACT_VERSION,
    IMPLEMENTATION_ID,
    load_context,
    load_manifest,
    sample_context,
    save_context,
    sha256_file,
)


def test_context_round_trip_and_sampling(tmp_path):
    path = tmp_path / "context.pt"
    values = {
        "clip": {
            "optimized_hidden": torch.ones(3, 77, 768),
            "original_hidden": torch.zeros(3, 77, 768),
            "optimized_projected": torch.ones(3, 768),
            "original_projected": torch.zeros(3, 768),
        }
    }
    save_context(path, model="sd15", encoders=values, metadata={"count": 3})
    loaded = load_context(path, model="sd15", device="cpu")
    sampled = sample_context(loaded, 3, generator=torch.Generator().manual_seed(1))
    assert sampled["clip"]["optimized_hidden"].shape == (3, 77, 768)
    assert len(sha256_file(path)) == 64
    assert not list(tmp_path.glob(".*.tmp"))


def test_sample_context_rejects_implicit_replacement():
    rows = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1)
    checkpoint = {
        "encoders": {
            "clip": {
                "optimized_hidden": rows,
                "original_hidden": rows.clone(),
            }
        }
    }
    with pytest.raises(ValueError, match="stores only 3"):
        sample_context(checkpoint, 4)


def test_context_rejects_pre_compatibility_artifact(tmp_path):
    path = tmp_path / "old.pt"
    values = {
        "clip": {
            "optimized_hidden": torch.ones(1, 77, 768),
            "original_hidden": torch.zeros(1, 77, 768),
            "optimized_projected": torch.ones(1, 768),
            "original_projected": torch.zeros(1, 768),
        }
    }
    save_context(path, model="sd15", encoders=values, metadata={"count": 1})
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    del checkpoint["metadata"]["implementation"]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="incompatible TPSO implementation"):
        load_context(path, model="sd15", device="cpu")


def test_artifact_version_matches_package_release():
    assert ARTIFACT_VERSION == __version__


def test_sample_context_does_not_repeat_rows_when_enough_are_available():
    rows = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1)
    checkpoint = {
        "encoders": {
            "clip": {
                "optimized_hidden": rows,
                "original_hidden": rows.clone(),
            }
        }
    }

    sampled = sample_context(
        checkpoint,
        6,
        generator=torch.Generator().manual_seed(7),
    )

    selected = sampled["clip"]["optimized_hidden"].flatten()
    assert selected.unique().numel() == 6
    assert selected.sort().values.tolist() == list(range(6))


def test_manifest_schema(tmp_path):
    path = tmp_path / "contexts.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "artifact_version": "0.1.0",
                "artifacts": [
                    {
                        "model": "sd15",
                        "filename": "context.pt",
                        "repo_id": "owner/repo",
                        "revision": "main",
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_manifest(path)["sd15"].filename == "context.pt"


def test_manifest_rejects_invalid_checksum(tmp_path):
    path = tmp_path / "contexts.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "artifact_version": "0.1.0",
                "artifacts": [
                    {
                        "model": "sd15",
                        "filename": "context.pt",
                        "repo_id": "owner/repo",
                        "sha256": "not-a-checksum",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="64 hexadecimal"):
        load_manifest(path)


def test_context_rejects_mismatched_encoder_counts(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save(
        {
            "metadata": {
                "format_version": 1,
                "artifact_version": "0.1.0",
                "method": "TPSO",
                    "model": "sd35",
                    "implementation": IMPLEMENTATION_ID,
                "count": 2,
            },
            "encoders": {
                "clip_l": {
                    "optimized_hidden": torch.zeros(2, 77, 768),
                    "original_hidden": torch.zeros(2, 77, 768),
                    "optimized_projected": torch.zeros(2, 768),
                    "original_projected": torch.zeros(2, 768),
                },
                "clip_g": {
                    "optimized_hidden": torch.zeros(3, 77, 1280),
                    "original_hidden": torch.zeros(3, 77, 1280),
                    "optimized_projected": torch.zeros(3, 1280),
                    "original_projected": torch.zeros(3, 1280),
                },
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="same number"):
        load_context(path, model="sd35", device="cpu")
