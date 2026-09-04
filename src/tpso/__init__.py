"""TPSO: training-free semantic prompt embedding optimization."""

from .config import MODEL_SPECS, ModelSpec, TPSOConfig
from .optimization import OptimizationResult, optimize_prompt_offsets

__all__ = [
    "MODEL_SPECS",
    "ModelSpec",
    "OptimizationResult",
    "TPSOConfig",
    "optimize_prompt_offsets",
]

__version__ = "0.1.0"
