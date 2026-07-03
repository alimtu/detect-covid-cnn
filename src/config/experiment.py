"""Experiment directory management and tracking.

Each run gets an auto-incremented ``experiment_NNN`` folder holding everything
needed to understand and reproduce it: the fully resolved config, metrics,
per-epoch history, plots and checkpoints.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.config.loader import Config

_EXPERIMENT_PREFIX = "experiment_"
# Matches experiment_001 and experiment_001_<model>, capturing the index.
_EXPERIMENT_RE = re.compile(rf"^{_EXPERIMENT_PREFIX}(\d+)(?:_.*)?$")


def _slugify(value: str) -> str:
    """Make a filesystem-friendly slug (lowercase, no spaces/odd chars)."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    return slug.strip("_") or "model"


class ExperimentManager:
    """Owns the on-disk layout for a single experiment.

    Layout::

        outputs/experiment_001_densenet121/
            config.yaml            # fully resolved configuration
            metrics.json           # final evaluation metrics
            summary.json           # hyperparameters + final metrics
            training_history.csv   # per-epoch metrics
            plots/                 # accuracy/loss/confusion/roc/pr
            checkpoints/           # best/last/periodic weights
            tensorboard/           # optional TB event files
    """

    def __init__(self, root: Path, name: str) -> None:
        self.root = root
        self.name = name
        self.dir = root / name
        self.checkpoints_dir = self.dir / "checkpoints"
        self.plots_dir = self.dir / "plots"
        self.tensorboard_dir = self.dir / "tensorboard"
        self.config_path = self.dir / "config.yaml"
        self.metrics_path = self.dir / "metrics.json"
        self.summary_path = self.dir / "summary.json"
        self.history_path = self.dir / "training_history.csv"

    @classmethod
    def create(cls, config: Config) -> "ExperimentManager":
        """Create the next ``experiment_NNN_<model>`` folder and persist the config.

        Args:
            config: The fully resolved configuration for this run.

        Returns:
            The initialised :class:`ExperimentManager`.
        """
        root = Path(config.get("experiment.output_root", "outputs"))
        root.mkdir(parents=True, exist_ok=True)
        model_name = _slugify(config.get("model.name", "model"))
        name = cls._next_name(root, model_name)

        manager = cls(root, name)
        manager.dir.mkdir(parents=True, exist_ok=False)
        manager.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        manager.plots_dir.mkdir(parents=True, exist_ok=True)

        manager.save_config(config)
        return manager

    @classmethod
    def _next_name(cls, root: Path, model_name: str) -> str:
        """Return the next ``experiment_NNN_<model>`` name."""
        indices = [
            int(match.group(1))
            for child in root.iterdir()
            if child.is_dir() and (match := _EXPERIMENT_RE.match(child.name))
        ]
        next_index = (max(indices) + 1) if indices else 1
        return f"{_EXPERIMENT_PREFIX}{next_index:03d}_{model_name}"

    def save_config(self, config: Config) -> None:
        """Write the resolved config to ``config.yaml``."""
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """Write final evaluation metrics to ``metrics.json``."""
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    def save_summary(self, config: Config, metrics: dict[str, Any]) -> None:
        """Write a compact experiment summary for tracking/comparison."""
        cfg = config.to_dict()
        summary = {
            "experiment": self.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": cfg.get("model", {}),
            "dataset": cfg.get("dataset", {}),
            "augmentation": cfg.get("augmentation", {}),
            "optimizer": cfg.get("optimizer", {}),
            "scheduler": cfg.get("scheduler", {}),
            "loss": cfg.get("loss", {}),
            "training": cfg.get("training", {}),
            "reproducibility": cfg.get("reproducibility", {}),
            "final_metrics": {
                k: v for k, v in metrics.items() if not isinstance(v, list)
            },
        }
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
