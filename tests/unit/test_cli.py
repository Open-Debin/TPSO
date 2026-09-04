import json
from pathlib import Path

import pytest

from tpso.cli.benchmark import main as benchmark_main
from tpso.cli.generate import main
from tpso.cli.precompute import main as precompute_main


def test_generate_dry_run_does_not_load_models(capsys):
    result = main(["--model", "sd15", "--prompt", "a red panda", "--dry-run"])
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["model"]["name"] == "sd15"
    assert payload["tpso"]["kappa"] == 0.8


def test_generate_dry_run_loads_yaml_config(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  name: sd15\ntpso:\n  optimizer: adam\n  kappa: 0.9\n",
        encoding="utf-8",
    )
    result = main(
        ["--model", "sd15", "--prompt", "a red panda", "--config", str(path), "--dry-run"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["tpso"]["kappa"] == 0.9


def test_generate_dry_run_applies_cli_overrides(capsys):
    result = main(
        [
            "--model",
            "sd21",
            "--prompt",
            "a red panda",
            "--kappa",
            "0.7",
            "--num-steps",
            "20",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["tpso"]["kappa"] == 0.7
    assert payload["generation"]["num_steps"] == 20


def test_benchmark_dry_run_lists_official_rows_without_loading_data(capsys):
    result = benchmark_main(["--group", "table5", "--dry-run"])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert [item["name"] for item in payload["experiments"]] == [
        "lambda_0",
        "lambda_5",
        "lambda_10",
    ]


def test_generate_dry_run_rejects_invalid_generation_settings():
    with pytest.raises(ValueError, match="num_steps must be positive"):
        main(["--model", "sd15", "--prompt", "a red panda", "--num-steps", "0", "--dry-run"])


def test_generate_dry_run_rejects_blank_prompt():
    with pytest.raises(ValueError, match="non-whitespace"):
        main(["--model", "sd15", "--prompt", " ", "--dry-run"])


def test_precompute_refuses_to_overwrite_artifacts(tmp_path):
    artifact = tmp_path / "sd15_kappa0.8_lambda1.pt"
    artifact.touch()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        precompute_main(["--models", "sd15", "--output-dir", str(tmp_path)])


def test_precompute_rejects_unavailable_cuda(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
        precompute_main(["--models", "sd15", "--output-dir", str(tmp_path)])


def test_precompute_cuda_dry_run_does_not_require_cuda(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert (
        precompute_main(
            ["--models", "sd15", "--output-dir", str(tmp_path), "--dry-run"]
        )
        == 0
    )


def test_precompute_writes_versioned_checksum_manifest(tmp_path, monkeypatch):
    import tpso.runner

    monkeypatch.setattr(tpso.runner, "load_adapter", lambda *_args, **_kwargs: object())

    def fake_rebuild(_adapter, _spec, _config, *, output_path, **_kwargs):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"context")
        return path

    monkeypatch.setattr(tpso.runner, "rebuild_context", fake_rebuild)
    result = precompute_main(
        [
            "--models",
            "sd15",
            "sd21",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )

    manifest = json.loads((tmp_path / "contexts.json").read_text(encoding="utf-8"))
    assert result == 0
    assert manifest["format_version"] == 1
    assert manifest["artifact_version"] == "0.1.0"
    assert [item["model"] for item in manifest["artifacts"]] == ["sd15", "sd21"]
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"])
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".tpso-precompute-*"))


def test_precompute_failure_does_not_publish_partial_artifacts(tmp_path, monkeypatch):
    import tpso.runner

    monkeypatch.setattr(tpso.runner, "load_adapter", lambda *_args, **_kwargs: object())
    calls = 0

    def fail_second_rebuild(_adapter, _spec, _config, *, output_path, **_kwargs):
        nonlocal calls
        calls += 1
        path = Path(output_path)
        path.write_bytes(b"context")
        if calls == 2:
            raise RuntimeError("generation failed")
        return path

    monkeypatch.setattr(tpso.runner, "rebuild_context", fail_second_rebuild)
    with pytest.raises(RuntimeError, match="generation failed"):
        precompute_main(
            [
                "--models",
                "sd15",
                "sd21",
                "--output-dir",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )

    assert not list(tmp_path.iterdir())
