"""Build, validate, serialise and persist experiment configurations.

The Train page edits an in-memory ``dict`` that mirrors ``configs/base.yaml``.
This service is the single bridge between that dict and the backend:

* :func:`default_config` seeds a fresh config from ``base.yaml`` overlaid with the
  user's saved project defaults.
* :func:`get_in` / :func:`set_in` do dotted-path reads/writes on the dict.
* :func:`validate` runs the backend's own :func:`~src.config.validate_config`.
* :func:`to_yaml`, :func:`save_config`, :func:`load_config_file` handle I/O.

No configuration schema is duplicated here — ``base.yaml`` remains the source of
truth for defaults, and the backend remains the source of truth for validation.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from src.config import Config, load_config, validate_config

from frontend.services import paths
from frontend.services.settings_service import AppSettings, load_settings

# Option lists mirror the backend registries / validators. Kept as the display
# order for dropdowns; validation itself is delegated to the backend.
MODELS: list[str] = ["densenet121", "vgg16", "resnet50", "efficientnet_b0", "vit_b_16"]
OPTIMIZERS: list[str] = ["adam", "adamw", "sgd"]
SCHEDULERS: list[str] = ["reduce_on_plateau", "cosine", "step", "onecycle", "none"]
LOSSES: list[str] = ["cross_entropy", "weighted_cross_entropy", "focal"]
MONITORS: list[str] = ["val_loss", "val_acc"]

# Pretty labels for dropdowns.
MODEL_LABELS: dict[str, str] = {
    "densenet121": "DenseNet121",
    "vgg16": "VGG16",
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_b_16": "ViT-B/16",
}
OPTIMIZER_LABELS: dict[str, str] = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD"}
SCHEDULER_LABELS: dict[str, str] = {
    "reduce_on_plateau": "ReduceLROnPlateau",
    "cosine": "CosineAnnealing",
    "step": "StepLR",
    "onecycle": "OneCycleLR",
    "none": "None",
}
LOSS_LABELS: dict[str, str] = {
    "cross_entropy": "CrossEntropy",
    "weighted_cross_entropy": "Weighted CrossEntropy",
    "focal": "Focal Loss",
}


def base_config() -> dict[str, Any]:
    """Return the resolved ``base.yaml`` as a plain dict."""
    return load_config(paths.BASE_CONFIG_PATH).to_dict()


def default_config(settings: AppSettings | None = None) -> dict[str, Any]:
    """Return a fresh config seeded from ``base.yaml`` and the project defaults."""
    settings = settings or load_settings()
    config = base_config()
    # Overlay user defaults onto the base template.
    set_in(config, "dataset.path", settings.default_dataset_path)
    set_in(config, "dataset.image_size", int(settings.default_image_size))
    set_in(config, "experiment.output_root", settings.default_output_root)
    set_in(config, "model.name", settings.default_model)
    set_in(config, "optimizer.name", settings.default_optimizer)
    set_in(config, "scheduler.name", settings.default_scheduler)
    set_in(config, "training.epochs", int(settings.default_epochs))
    set_in(config, "training.batch_size", int(settings.default_batch_size))
    return config


def get_in(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``config`` at a dotted path (``"training.epochs"``)."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_in(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Write ``value`` into ``config`` at a dotted path, creating dicts as needed."""
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def clone(config: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a config dict (used when duplicating experiments)."""
    return copy.deepcopy(config)


def validate(config: dict[str, Any]) -> tuple[bool, str]:
    """Validate a config dict via the backend validator.

    Returns:
        ``(ok, message)`` — ``message`` is empty on success, else the error text.
    """
    try:
        validate_config(Config(config))
        return True, ""
    except ValueError as exc:
        return False, str(exc)


def to_yaml(config: dict[str, Any]) -> str:
    """Serialise a config dict to YAML (key order preserved)."""
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z._-]+", "_", name.strip())
    return slug.strip("_") or "config"


def save_config(config: dict[str, Any], name: str) -> Path:
    """Persist a config under ``configs/saved/<name>.yaml`` and return its path."""
    paths.SAVED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    filename = _slugify(name)
    if not filename.endswith((".yaml", ".yml")):
        filename += ".yaml"
    path = paths.SAVED_CONFIGS_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        handle.write(to_yaml(config))
    return path


def list_saved_configs() -> list[Path]:
    """Return saved configs plus the bundled experiment presets."""
    result: list[Path] = []
    if paths.EXPERIMENT_CONFIGS_DIR.is_dir():
        result += sorted(paths.EXPERIMENT_CONFIGS_DIR.glob("*.yaml"))
    if paths.SAVED_CONFIGS_DIR.is_dir():
        result += sorted(paths.SAVED_CONFIGS_DIR.glob("*.yaml"))
    return result


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a (possibly layered) config file into a plain dict."""
    return load_config(path).to_dict()


def load_config_bytes(data: bytes) -> dict[str, Any]:
    """Parse an uploaded YAML config (single file, no ``base:`` resolution)."""
    loaded = yaml.safe_load(data.decode("utf-8")) or {}
    # If the uploaded file is a bare override, merge it onto the base template so
    # the Train page always has a complete config to edit.
    if "base" in loaded:
        loaded.pop("base")
    merged = base_config()
    _deep_update(merged, loaded)
    return merged


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
