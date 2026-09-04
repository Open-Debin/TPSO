"""Paper-aligned model and optimization configuration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    """Static settings for one supported diffusion backbone."""

    name: str
    model_id: str
    model_revision: str
    projection_model_id: str | None
    projection_revision: str | None
    image_size: int
    inference_steps: int
    guidance_scale: float
    diversity_weight: float
    context_filename: str

    def validate(self) -> ModelSpec:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("model_revision must be a non-empty string.")
        if self.name in {"sd15", "sd21"} and (
            not isinstance(self.projection_model_id, str) or not self.projection_model_id.strip()
        ):
            raise ValueError(f"{self.name} requires a projection_model_id.")
        if self.name in {"sd15", "sd21"} and (
            not isinstance(self.projection_revision, str) or not self.projection_revision.strip()
        ):
            raise ValueError(f"{self.name} requires a projection_revision.")
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int):
            raise ValueError("image_size must be an integer.")
        if isinstance(self.inference_steps, bool) or not isinstance(self.inference_steps, int):
            raise ValueError("inference_steps must be an integer.")
        if not isinstance(self.guidance_scale, (int, float)) or not math.isfinite(
            self.guidance_scale
        ):
            raise ValueError("guidance_scale must be finite.")
        if not isinstance(self.diversity_weight, (int, float)) or not math.isfinite(
            self.diversity_weight
        ):
            raise ValueError("diversity_weight must be finite.")
        if self.image_size <= 0 or self.image_size % 8:
            raise ValueError("image_size must be positive and divisible by 8.")
        if self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive.")
        if self.guidance_scale <= 1.0:
            raise ValueError("guidance_scale must be greater than 1 for TPSO.")
        if self.diversity_weight < 0.0:
            raise ValueError("diversity_weight cannot be negative.")
        if (
            Path(self.context_filename).name != self.context_filename
            or not self.context_filename.endswith(".pt")
        ):
            raise ValueError("context_filename must be a local .pt filename.")
        return self


@dataclass(frozen=True)
class TPSOConfig:
    """Optimization defaults reported in the TPSO paper."""

    kappa: float = 0.8
    tolerance: float = 0.01
    diversity_weight: float = 1.0
    scheduler_ratio: float = 0.4
    learning_rate: float = 0.01
    optimizer: str = "rmsprop"
    max_steps: int = 50
    offset_init: str = "approx_zero"
    offset_std: float = 1e-4

    def validate(self) -> TPSOConfig:
        numeric = {
            "kappa": self.kappa,
            "tolerance": self.tolerance,
            "diversity_weight": self.diversity_weight,
            "scheduler_ratio": self.scheduler_ratio,
            "learning_rate": self.learning_rate,
            "offset_std": self.offset_std,
        }
        for name, value in numeric.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number.")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise ValueError("max_steps must be an integer.")
        if not 0.0 < self.kappa <= 1.0:
            raise ValueError("kappa must be in (0, 1].")
        if not 0.0 < self.tolerance < 1.0:
            raise ValueError("tolerance must be in (0, 1).")
        if self.diversity_weight < 0.0:
            raise ValueError("diversity_weight cannot be negative.")
        if not -1.0 <= self.scheduler_ratio <= 1.0:
            raise ValueError("scheduler_ratio must be in [-1, 1].")
        supported_initializers = {
            "approx_zero",
            "normal",
            "zero",
            "normal_1",
            "normal_2",
            "laplace_1",
            "laplace_1.5",
            "laplace_sqrt1.5",
            "gamma_1_1",
            "gamma_4_4",
            "gamma_4_sqrt4",
            "gamma_7_sqrt7",
            "uniform_0_1",
            "uniform_-1_1",
            "uniform_-3_3",
        }
        if self.offset_init not in supported_initializers:
            raise ValueError(
                f"offset_init must be one of {sorted(supported_initializers)}."
            )
        if self.learning_rate <= 0.0 or self.max_steps <= 0 or self.offset_std <= 0.0:
            raise ValueError("learning_rate, max_steps, and offset_std must be positive.")
        if self.optimizer not in {"rmsprop", "adam"}:
            raise ValueError("optimizer must be either 'rmsprop' or 'adam'.")
        return self


MODEL_SPECS: dict[str, ModelSpec] = {
    "sd15": ModelSpec(
        name="sd15",
        model_id="sd-legacy/stable-diffusion-v1-5",
        model_revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        projection_model_id="openai/clip-vit-large-patch14",
        projection_revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
        image_size=512,
        inference_steps=50,
        guidance_scale=7.5,
        diversity_weight=1.0,
        context_filename="sd15_kappa0.8_lambda1.pt",
    ),
    "sd21": ModelSpec(
        name="sd21",
        model_id="sd2-community/stable-diffusion-2-1",
        model_revision="bb2154823665391b4fb29b0b9cf82a198964ee05",
        projection_model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        projection_revision="1c2b8495b28150b8a4922ee1c8edee224c284c0c",
        image_size=768,
        inference_steps=50,
        guidance_scale=7.5,
        diversity_weight=1.0,
        context_filename="sd21_kappa0.8_lambda1.pt",
    ),
    "sd35": ModelSpec(
        name="sd35",
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_revision="b940f670f0eda2d07fbb75229e779da1ad11eb80",
        projection_model_id=None,
        projection_revision=None,
        image_size=512,
        inference_steps=35,
        guidance_scale=7.0,
        diversity_weight=10.0,
        context_filename="sd35_kappa0.8_lambda0.pt",
    ),
}


def default_config(model: str) -> TPSOConfig:
    """Return official-method defaults with the main-table model weight."""

    try:
        weight = MODEL_SPECS[model].diversity_weight
    except KeyError as exc:
        raise ValueError(f"Unsupported model {model!r}; choose from {tuple(MODEL_SPECS)}.") from exc
    return TPSOConfig(
        diversity_weight=weight,
        max_steps=200 if model == "sd35" else 50,
    )


def load_config(path: str | Path, model: str) -> tuple[ModelSpec, TPSOConfig]:
    """Load a public YAML config while retaining validated paper defaults."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("The config root must be a mapping.")
    unknown_sections = set(raw) - {"model", "tpso"}
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")
    model_values: dict[str, Any] = raw.get("model", {})
    tpso_values: dict[str, Any] = raw.get("tpso", {})
    if not isinstance(model_values, dict) or not isinstance(tpso_values, dict):
        raise ValueError("The model and tpso config sections must be mappings.")
    spec = MODEL_SPECS[model]
    allowed_model = set(asdict(spec))
    unknown_model = set(model_values) - allowed_model
    if unknown_model:
        raise ValueError(f"Unknown model config fields: {sorted(unknown_model)}")
    if model_values.get("name", model) != model:
        raise ValueError(f"Config model name does not match --model {model!r}.")
    spec = replace(spec, **{k: v for k, v in model_values.items() if k != "name"}).validate()

    unknown_tpso = set(tpso_values) - set(asdict(default_config(model)))
    if unknown_tpso:
        raise ValueError(f"Unknown TPSO config fields: {sorted(unknown_tpso)}")
    values = dict(tpso_values)
    if "optimizer" in values:
        values["optimizer"] = str(values["optimizer"]).lower()
    return spec, replace(default_config(model), **values).validate()
