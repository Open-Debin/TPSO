"""Reproducible generation protocols for Tables I-V of the TPSO paper."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
import uuid
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PIL import Image

from .config import MODEL_SPECS, TPSOConfig, default_config
from .runner import load_adapter, resolve_dtype, resolve_generation_settings, resolve_unconditional

COCO_CAPTIONS_URL = (
    "https://huggingface.co/datasets/sayakpaul/sample-datasets/raw/main/"
    "coco_30k_randomly_sampled_2014_val.csv"
)
COCO_CAPTIONS_SHA256 = "4e34947cb2a5d77c9bbaa11e8032e1339a26ebc32d9535725ab4f8e36760b7d3"
COCO_CAPTIONS_FILENAME = "coco_30k_randomly_sampled_2014_val.csv"
DEFAULT_DATASET_CACHE = Path.home() / ".cache" / "tpso" / "datasets"
PAPER_VARIANTS = 10
PAPER_SAVED_IMAGE_SIZE = 512
ARCHIVED_SD35_T5_LENGTH = 77
SOURCE_BATCH_SIZE = {"sd15": 5, "sd21": 5, "sd35": 2}


@dataclass(frozen=True)
class PromptRecord:
    file_name: str
    caption: str


@dataclass(frozen=True)
class PaperExperiment:
    table: str
    name: str
    model: str
    prompt_count: int
    overrides: dict[str, float | str]

    @property
    def relative_output_dir(self) -> Path:
        return Path(self.table) / self.name


def _table_experiments() -> tuple[PaperExperiment, ...]:
    ablation = {"diversity_weight": 0.0}
    return (
        PaperExperiment("table1", "sd15", "sd15", 5_000, {}),
        PaperExperiment("table1", "sd21", "sd21", 5_000, {}),
        PaperExperiment("table1", "sd35", "sd35", 5_000, {}),
        PaperExperiment("table2", "kappa_0p7", "sd15", 1_000, {**ablation, "kappa": 0.7}),
        PaperExperiment("table2", "kappa_0p8", "sd15", 1_000, {**ablation, "kappa": 0.8}),
        PaperExperiment("table2", "kappa_0p9", "sd15", 1_000, {**ablation, "kappa": 0.9}),
        PaperExperiment(
            "table3", "coarse_to_fine_r0p4", "sd15", 1_000, {**ablation, "scheduler_ratio": 0.4}
        ),
        PaperExperiment(
            "table3", "coarse_to_fine_r0p6", "sd15", 1_000, {**ablation, "scheduler_ratio": 0.6}
        ),
        PaperExperiment(
            "table3", "fine_to_coarse_rm0p4", "sd15", 1_000, {**ablation, "scheduler_ratio": -0.4}
        ),
        PaperExperiment(
            "table3", "fine_to_coarse_rm0p6", "sd15", 1_000, {**ablation, "scheduler_ratio": -0.6}
        ),
        PaperExperiment(
            "table4",
            "normal_0_1em4",
            "sd15",
            1_000,
            {**ablation, "offset_init": "approx_zero", "offset_std": 1e-4},
        ),
        PaperExperiment(
            "table4",
            "normal_0_1",
            "sd15",
            1_000,
            {**ablation, "offset_init": "normal", "offset_std": 1.0},
        ),
        PaperExperiment(
            "table4",
            "laplace_0_1",
            "sd15",
            1_000,
            {**ablation, "offset_init": "laplace_1"},
        ),
        PaperExperiment(
            "table4",
            "laplace_0_sqrt1p5",
            "sd15",
            1_000,
            {**ablation, "offset_init": "laplace_sqrt1.5"},
        ),
        PaperExperiment(
            "table4",
            "gamma_1_1",
            "sd15",
            1_000,
            {**ablation, "offset_init": "gamma_1_1"},
        ),
        PaperExperiment(
            "table4",
            "gamma_4_sqrt4",
            "sd15",
            1_000,
            {**ablation, "offset_init": "gamma_4_sqrt4"},
        ),
        PaperExperiment(
            "table4",
            "uniform_m1_1",
            "sd15",
            1_000,
            {**ablation, "offset_init": "uniform_-1_1"},
        ),
        PaperExperiment(
            "table4",
            "uniform_m3_3",
            "sd15",
            1_000,
            {**ablation, "offset_init": "uniform_-3_3"},
        ),
        PaperExperiment("table5", "lambda_0", "sd15", 1_000, {"diversity_weight": 0.0}),
        PaperExperiment("table5", "lambda_5", "sd15", 1_000, {"diversity_weight": 5.0}),
        PaperExperiment("table5", "lambda_10", "sd15", 1_000, {"diversity_weight": 10.0}),
    )


PAPER_EXPERIMENTS = _table_experiments()


def select_experiments(group: str, names: list[str] | None = None) -> list[PaperExperiment]:
    if group not in {"all", "table1", "table2", "table3", "table4", "table5"}:
        raise ValueError("group must be all or table1 through table5.")
    selected = [item for item in PAPER_EXPERIMENTS if group == "all" or item.table == group]
    if names:
        wanted = set(names)
        selected = [item for item in selected if item.name in wanted]
        missing = wanted - {item.name for item in selected}
        if missing:
            raise ValueError(f"Unknown experiment names for {group}: {sorted(missing)}")
    return selected


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_coco_csv(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path = DEFAULT_DATASET_CACHE,
) -> Path:
    """Resolve the exact caption CSV used by the archived TPSO source code."""

    if path is not None:
        resolved = Path(path).expanduser().resolve()
    else:
        resolved = Path(cache_dir).expanduser().resolve() / COCO_CAPTIONS_FILENAME
        if not resolved.is_file():
            resolved.parent.mkdir(parents=True, exist_ok=True)
            temporary = resolved.with_name(f".{resolved.name}.{uuid.uuid4().hex}.tmp")
            try:
                urllib.request.urlretrieve(COCO_CAPTIONS_URL, temporary)
                actual = sha256_file(temporary)
                if actual != COCO_CAPTIONS_SHA256:
                    raise ValueError(
                        "Downloaded MS-COCO caption checksum mismatch: "
                        f"expected {COCO_CAPTIONS_SHA256}, got {actual}."
                    )
                temporary.replace(resolved)
            finally:
                temporary.unlink(missing_ok=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"MS-COCO caption CSV not found: {resolved}")
    actual = sha256_file(resolved)
    if actual != COCO_CAPTIONS_SHA256:
        raise ValueError(
            f"MS-COCO caption checksum mismatch: expected {COCO_CAPTIONS_SHA256}, got {actual}."
        )
    return resolved


def load_coco_prompts(path: str | Path, count: int) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"file_name", "caption"}.issubset(reader.fieldnames):
            raise ValueError("MS-COCO CSV must contain file_name and caption columns.")
        for row in reader:
            caption = row["caption"].strip()
            if not caption:
                raise ValueError(f"Empty caption encountered at row {len(records)}.")
            records.append(PromptRecord(row["file_name"], caption))
            if len(records) == count:
                break
    if len(records) != count:
        raise ValueError(f"Requested {count} prompts but the CSV contains only {len(records)}.")
    return records


def indexed_image_paths(
    output_dir: str | Path,
    prompt_indices: range,
    num_variants: int = PAPER_VARIANTS,
) -> list[Path]:
    root = Path(output_dir).expanduser().resolve()
    return [
        root / f"{prompt_index}_{variant_index}.jpg"
        for prompt_index in prompt_indices
        for variant_index in range(num_variants)
    ]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_prompt_index(path: Path, records: list[PromptRecord]) -> None:
    lines = ["prompt_id,file_name,caption\n"]
    for index, record in enumerate(records):
        escaped_name = record.file_name.replace('"', '""')
        escaped_caption = record.caption.replace('"', '""')
        lines.append(f'{index},"{escaped_name}","{escaped_caption}"\n')
    _atomic_write(path, "".join(lines))


def valid_paper_image(path: str | Path) -> bool:
    """Return whether an existing output is a complete paper-protocol JPEG."""

    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        with Image.open(candidate) as image:
            if image.format != "JPEG" or image.size != (
                PAPER_SAVED_IMAGE_SIZE,
                PAPER_SAVED_IMAGE_SIZE,
            ):
                return False
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


def experiment_manifest(
    experiment: PaperExperiment,
    config: TPSOConfig,
    *,
    prompt_count: int,
    seed: int,
    batch_size: int,
    records: list[PromptRecord],
    generation: dict[str, int | float],
    context: dict[str, str],
) -> dict[str, object]:
    spec = MODEL_SPECS[experiment.model]
    prompt_digest = hashlib.sha256(
        json.dumps([asdict(record) for record in records], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "protocol": "TPSO IJCNN 2026",
        "table": experiment.table,
        "experiment": experiment.name,
        "model": asdict(spec),
        "tpso": asdict(config),
        "dataset": {
            "source": COCO_CAPTIONS_URL,
            "source_sha256": COCO_CAPTIONS_SHA256,
            "selection": f"first {prompt_count} rows",
            "selected_prompts_sha256": prompt_digest,
        },
        "prompt_count": prompt_count,
        "variants_per_prompt": PAPER_VARIANTS,
        "expected_images": prompt_count * PAPER_VARIANTS,
        "filename_template": "{prompt_id}_{variant_id}.jpg",
        "generation": generation,
        "saved_image_size": PAPER_SAVED_IMAGE_SIZE,
        "unconditional_context": context,
        "seed": seed,
        "batch_size": batch_size,
        "shared_initial_latent_within_prompt": True,
    }


def describe_experiment(experiment: PaperExperiment) -> dict[str, object]:
    """Resolve the complete archived protocol for inspection and manifests."""

    spec = MODEL_SPECS[experiment.model]
    config = replace(default_config(experiment.model), **experiment.overrides).validate()
    generation = resolve_generation_settings(
        spec, num_steps=None, guidance_scale=None, height=None, width=None
    )
    if experiment.model == "sd35":
        generation.update(height=512, width=512)
    return {
        **asdict(experiment),
        "resolved_tpso": asdict(config),
        "resolved_generation": generation,
        "saved_image_size": PAPER_SAVED_IMAGE_SIZE,
        "sd35_t5_sequence_length": (
            ARCHIVED_SD35_T5_LENGTH if experiment.model == "sd35" else None
        ),
    }


def _prepare_case_files(
    output_dir: Path,
    manifest: dict[str, object],
    records: list[PromptRecord],
) -> None:
    manifest_path = output_dir / "manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(
            f"Existing manifest does not match the requested protocol: {manifest_path}"
        )
    if not manifest_path.exists():
        _atomic_write(manifest_path, serialized)
    prompt_path = output_dir / "prompts.csv"
    if not prompt_path.exists():
        _write_prompt_index(prompt_path, records)


def run_experiment(
    experiment: PaperExperiment,
    records: list[PromptRecord],
    *,
    output_root: str | Path,
    batch_size: int,
    seed: int,
    device: str,
    precision: str,
    context_dir: str | Path | None,
    local_files_only: bool,
) -> Path:
    """Run one paper row with deterministic fixed batches and resumable output."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    expected_batch_size = SOURCE_BATCH_SIZE[experiment.model]
    if batch_size != expected_batch_size:
        warnings.warn(
            f"Archived {experiment.model} runs used batch size {expected_batch_size}; "
            f"batch size {batch_size} changes the semantic/diversity gradient ratio.",
            stacklevel=2,
        )
    spec = MODEL_SPECS[experiment.model]
    description = describe_experiment(experiment)
    config = TPSOConfig(**description["resolved_tpso"])
    generation = description["resolved_generation"]
    output_dir = Path(output_root).expanduser().resolve() / experiment.relative_output_dir
    context_path = None
    if context_dir is not None:
        context_path = Path(context_dir).expanduser().resolve() / spec.context_filename
        if not context_path.is_file():
            raise FileNotFoundError(f"Context artifact not found: {context_path}")
        context_info = {
            "mode": "local",
            "filename": context_path.name,
            "sha256": sha256_file(context_path),
        }
    else:
        context_info = {"mode": "automatic", "filename": spec.context_filename}
    manifest = experiment_manifest(
        experiment,
        config,
        prompt_count=len(records),
        seed=seed,
        batch_size=batch_size,
        records=records,
        generation=generation,
        context=context_info,
    )
    _prepare_case_files(output_dir, manifest, records)

    dtype = resolve_dtype(device, precision)
    adapter = load_adapter(spec, device=device, dtype=dtype, local_files_only=local_files_only)
    total = len(records)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        paths = indexed_image_paths(output_dir, range(start, stop))
        valid = [valid_paper_image(path) for path in paths]
        if all(valid):
            print(f"[{experiment.table}/{experiment.name}] skip {start}:{stop} (complete)")
            continue

        batch_seed = seed + start
        prompts = [record.caption for record in records[start:stop]]
        conditional = adapter.optimize(prompts, PAPER_VARIANTS, config, seed=batch_seed)
        unconditional = resolve_unconditional(
            adapter,
            spec,
            config,
            count=len(prompts) * PAPER_VARIANTS,
            seed=batch_seed,
            context_path=context_path,
            rebuild_unconditional=False,
            local_files_only=local_files_only,
        )
        arguments = {
            "conditional": conditional,
            "unconditional": unconditional,
            "num_variants": PAPER_VARIANTS,
            "config": config,
            **generation,
            "seed": batch_seed,
            "output_paths": paths,
            "output_size": PAPER_SAVED_IMAGE_SIZE,
            # A partial fixed batch is regenerated with the same seed, preserving
            # deterministic results after interruption.
            "overwrite": any(path.exists() for path in paths),
        }
        if experiment.model == "sd35":
            arguments["prompts"] = prompts
            arguments["max_sequence_length"] = ARCHIVED_SD35_T5_LENGTH
        adapter.generate(**arguments)
        print(f"[{experiment.table}/{experiment.name}] generated {stop}/{total} prompts")
    return output_dir
