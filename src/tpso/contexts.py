"""Versioned unconditional-context loading, validation, and Hub download."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch

FORMAT_VERSION = 1
ARTIFACT_VERSION = "0.1.0"
IMPLEMENTATION_ID = "simple_cads-ijcnn-source-v1"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tpso" / "contexts"
EXPECTED_ENCODERS = {
    "sd15": {"clip"},
    "sd21": {"clip"},
    "sd35": {"clip_l", "clip_g"},
}
EXPECTED_ENCODER_SHAPES = {
    "sd15": {"clip": (77, 768, 768)},
    "sd21": {"clip": (77, 1024, 1024)},
    "sd35": {
        "clip_l": (77, 768, 768),
        "clip_g": (77, 1280, 1280),
    },
}


@dataclass(frozen=True)
class ContextArtifact:
    model: str
    filename: str
    repo_id: str
    revision: str
    sha256: str | None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path | None = None) -> dict[str, ContextArtifact]:
    manifest_path = Path(path) if path else Path(files("tpso.data").joinpath("contexts.json"))
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("The context manifest root must be a mapping.")
    if raw.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported context manifest version: {raw.get('format_version')!r}.")
    if raw.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported context artifact version: {raw.get('artifact_version')!r}."
        )
    items = raw.get("artifacts")
    if not isinstance(items, list):
        raise ValueError("The context manifest artifacts field must be a list.")

    artifacts: dict[str, ContextArtifact] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each context artifact must be a mapping.")
        missing = {"model", "filename", "repo_id"} - set(item)
        if missing:
            raise ValueError(f"Context artifact is missing fields: {sorted(missing)}")
        model = item["model"]
        if model not in EXPECTED_ENCODERS:
            raise ValueError(f"Unsupported context model: {model!r}.")
        if model in artifacts:
            raise ValueError(f"Duplicate context artifact for model {model!r}.")
        for field in ("filename", "repo_id"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Context artifact {field!r} must be a non-empty string.")
        revision = item.get("revision", "main")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("Context artifact 'revision' must be a non-empty string.")
        checksum = item.get("sha256")
        if checksum is not None and (
            not isinstance(checksum, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checksum) is None
        ):
            raise ValueError("Context artifact sha256 must be null or 64 hexadecimal characters.")
        artifacts[model] = ContextArtifact(
            model=item["model"],
            filename=item["filename"],
            repo_id=item["repo_id"],
            revision=revision,
            sha256=checksum.lower() if checksum else None,
        )
    return artifacts


def verify_artifact(path: str | Path, expected_sha256: str | None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Unconditional context not found: {resolved}")
    if expected_sha256:
        actual = sha256_file(resolved)
        if actual != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {resolved.name}: expected {expected_sha256}, got {actual}."
            )
    return resolved


def download_context(
    model: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    manifest_path: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Download a published context and verify its manifest checksum."""

    manifest = load_manifest(manifest_path)
    try:
        artifact = manifest[model]
    except KeyError as exc:
        raise ValueError(f"No unconditional context is registered for model {model!r}.") from exc
    if not artifact.sha256:
        raise RuntimeError(
            f"The {model} context has not been published yet; pass --context-path or "
            "--rebuild-unconditional."
        )
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=artifact.repo_id,
        filename=artifact.filename,
        revision=artifact.revision,
        cache_dir=str(Path(cache_dir).expanduser()),
        local_files_only=local_files_only,
    )
    return verify_artifact(path, artifact.sha256)


def load_context(
    path: str | Path,
    *,
    model: str,
    device: torch.device | str,
    model_id: str | None = None,
    model_revision: str | None = None,
    projection_model_id: str | None = None,
    projection_revision: str | None = None,
    expected_group_size: int | None = None,
) -> dict[str, Any]:
    """Load a TPSO context checkpoint and validate its public schema."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Context checkpoint must contain a mapping.")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Context checkpoint metadata must be a mapping.")
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported or missing context checkpoint format_version.")
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported or missing context checkpoint artifact_version.")
    if metadata.get("method") != "TPSO" or metadata.get("model") != model:
        raise ValueError(f"Context metadata does not match TPSO/{model}.")
    if metadata.get("implementation") != IMPLEMENTATION_ID:
        raise ValueError(
            "Context artifact was generated by an incompatible TPSO implementation; "
            "rebuild it with tpso-precompute --overwrite."
        )
    if (
        expected_group_size is not None
        and metadata.get("group_size") != expected_group_size
    ):
        raise ValueError(
            "Context group size does not match the archived generation protocol; "
            "rebuild it with tpso-precompute."
        )
    if model_id is not None and metadata.get("model_id") != model_id:
        raise ValueError(f"Context model_id does not match {model_id!r}.")
    if model_revision is not None and metadata.get("model_revision") != model_revision:
        raise ValueError(f"Context model_revision does not match {model_revision!r}.")
    if (
        projection_model_id is not None
        and metadata.get("projection_model_id") != projection_model_id
    ):
        raise ValueError(f"Context projection_model_id does not match {projection_model_id!r}.")
    if (
        projection_revision is not None
        and metadata.get("projection_revision") != projection_revision
    ):
        raise ValueError(
            f"Context projection_revision does not match {projection_revision!r}."
        )
    encoders = checkpoint.get("encoders")
    if not isinstance(encoders, dict) or not encoders:
        raise ValueError("Context checkpoint is missing the encoders mapping.")
    if set(encoders) != EXPECTED_ENCODERS.get(model):
        raise ValueError(
            f"Context encoders for {model!r} must be {sorted(EXPECTED_ENCODERS.get(model, set()))}."
        )
    expected_count: int | None = None
    for name, values in encoders.items():
        if not isinstance(name, str) or not isinstance(values, dict):
            raise ValueError("Each context encoder must be a named tensor mapping.")
        for key in ("optimized_hidden", "original_hidden"):
            value = values.get(key)
            if not torch.is_tensor(value) or value.ndim != 3:
                raise ValueError(f"Context encoder {name!r} is missing tensor {key!r}.")
        if values["optimized_hidden"].shape != values["original_hidden"].shape:
            raise ValueError(f"Context encoder {name!r} hidden tensor shapes do not match.")
        sequence, hidden_width, projected_width = EXPECTED_ENCODER_SHAPES[model][name]
        if tuple(values["optimized_hidden"].shape[1:]) != (sequence, hidden_width):
            raise ValueError(
                f"Context encoder {name!r} hidden shape must end in "
                f"({sequence}, {hidden_width})."
            )
        projected = ("optimized_projected" in values, "original_projected" in values)
        if projected != (True, True):
            raise ValueError(f"Context encoder {name!r} requires both projected tensors.")
        if not torch.is_tensor(values["optimized_projected"]) or not torch.is_tensor(
            values["original_projected"]
        ):
            raise ValueError(f"Context encoder {name!r} projected values must be tensors.")
        if values["optimized_projected"].shape != values["original_projected"].shape:
            raise ValueError(f"Context encoder {name!r} projected tensor shapes do not match.")
        count = int(values["optimized_hidden"].shape[0])
        if tuple(values["optimized_projected"].shape) != (count, projected_width):
            raise ValueError(
                f"Context encoder {name!r} projected shape must be ({count}, {projected_width})."
            )
        if expected_count is None:
            expected_count = count
        elif count != expected_count:
            raise ValueError("All context encoders must contain the same number of samples.")
    if metadata.get("count") != expected_count:
        raise ValueError("Context metadata count does not match the stored tensors.")
    return checkpoint


def sample_context(
    checkpoint: dict[str, Any],
    count: int,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Sample source-style context rows uniformly without replacement."""

    if count <= 0:
        raise ValueError("count must be positive.")
    encoders = checkpoint["encoders"]
    first = next(iter(encoders.values()))
    total = int(first["optimized_hidden"].shape[0])
    if total <= 0:
        raise ValueError("The context checkpoint contains no samples.")
    if count > total:
        raise ValueError(
            f"Requested {count} contexts, but the artifact stores only {total}; "
            "rebuild it with a larger --count."
        )
    reference_device = first["optimized_hidden"].device
    indices = torch.randperm(
        total, device=reference_device, generator=generator
    )[:count]
    sampled: dict[str, dict[str, torch.Tensor]] = {}
    for name, values in encoders.items():
        if int(values["optimized_hidden"].shape[0]) != total:
            raise ValueError("All context encoders must contain the same number of samples.")
        device_indices = indices.to(values["optimized_hidden"].device)
        sampled[name] = {
            key: value.index_select(0, device_indices)
            for key, value in values.items()
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == total
        }
    return sampled


def save_context(
    path: str | Path,
    *,
    model: str,
    encoders: dict[str, dict[str, torch.Tensor]],
    metadata: dict[str, Any],
) -> Path:
    """Write a portable CPU context checkpoint atomically."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            **metadata,
            "format_version": FORMAT_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "implementation": IMPLEMENTATION_ID,
            "method": "TPSO",
            "model": model,
        },
        "encoders": {
            name: {key: value.detach().cpu() for key, value in values.items()}
            for name, values in encoders.items()
        },
    }
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
