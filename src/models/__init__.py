"""Model construction via a registry-backed factory."""

from src.models.factory import (
    ModelBundle,
    available_models,
    build_model,
    build_model_from_config,
    freeze_backbone,
    make_classifier,
    unfreeze_all,
)

__all__ = [
    "ModelBundle",
    "available_models",
    "build_model",
    "build_model_from_config",
    "freeze_backbone",
    "make_classifier",
    "unfreeze_all",
]
