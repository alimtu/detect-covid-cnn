"""Lightweight configuration validation.

Fails fast with clear messages when a config would produce an invalid or
surprising experiment, rather than crashing deep inside training.
"""

from __future__ import annotations

from src.config.loader import Config

_MODELS = {"densenet121", "vgg16", "resnet50", "efficientnet_b0", "vit_b_16"}
_OPTIMIZERS = {"adam", "adamw", "sgd"}
_SCHEDULERS = {"reduce_on_plateau", "cosine", "step", "onecycle", "none"}
_LOSSES = {"cross_entropy", "weighted_cross_entropy", "focal"}
_MONITORS = {"val_loss", "val_acc"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid configuration: {message}")


def validate_config(config: Config) -> None:
    """Validate key configuration invariants.

    Args:
        config: The merged configuration.

    Raises:
        ValueError: If any invariant is violated.
    """
    split = config["dataset"]["split"]
    total = float(split["train"]) + float(split["val"]) + float(split["test"])
    _require(abs(total - 1.0) < 1e-6, f"dataset.split must sum to 1.0 (got {total}).")

    model_name = str(config["model"]["name"]).lower()
    _require(model_name in _MODELS, f"model.name '{model_name}' not in {sorted(_MODELS)}.")

    opt_name = str(config["optimizer"]["name"]).lower()
    _require(opt_name in _OPTIMIZERS, f"optimizer.name '{opt_name}' not in {sorted(_OPTIMIZERS)}.")

    sched_name = str(config["scheduler"]["name"]).lower()
    _require(sched_name in _SCHEDULERS, f"scheduler.name '{sched_name}' not in {sorted(_SCHEDULERS)}.")

    loss_name = str(config["loss"]["name"]).lower()
    _require(loss_name in _LOSSES, f"loss.name '{loss_name}' not in {sorted(_LOSSES)}.")

    monitor = str(config.get("training.early_stopping.monitor", "val_loss")).lower()
    _require(monitor in _MONITORS, f"early_stopping.monitor '{monitor}' not in {sorted(_MONITORS)}.")

    accum = int(config.get("training.gradient_accumulation_steps", 1))
    _require(accum >= 1, "training.gradient_accumulation_steps must be >= 1.")

    epochs = int(config["training"]["epochs"])
    _require(epochs >= 1, "training.epochs must be >= 1.")
