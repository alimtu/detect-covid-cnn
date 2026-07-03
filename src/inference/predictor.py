"""Single-image inference from a saved experiment.

A :class:`Predictor` rebuilds the exact model architecture recorded in an
experiment's ``config.yaml`` (backbone, dropout, hidden head), loads a
checkpoint, and applies the same evaluation preprocessing used during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import load_config
from src.datasets import build_eval_transform, discover_class_names
from src.models import build_model_from_config


@dataclass(frozen=True)
class Prediction:
    """Result of running one model on one image."""

    model_name: str
    probabilities: dict[str, float]  # class -> probability, sorted high to low

    @property
    def top_class(self) -> str:
        return next(iter(self.probabilities))

    @property
    def top_confidence(self) -> float:
        return next(iter(self.probabilities.values()))


class Predictor:
    """Wraps a trained model for single-image inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable,
        class_names: Sequence[str],
        model_name: str,
        device: torch.device,
    ) -> None:
        self.model = model.eval().to(device)
        self.transform = transform
        self.class_names = list(class_names)
        self.model_name = model_name
        self.device = device

    @classmethod
    def from_experiment(
        cls,
        experiment_dir: str | Path,
        device: torch.device,
        checkpoint_filename: str = "best_model.pth",
    ) -> "Predictor":
        """Build a predictor from an experiment directory.

        Args:
            experiment_dir: Path to an ``experiment_NNN`` folder.
            device: Compute device.
            checkpoint_filename: Checkpoint under ``checkpoints/`` to load.

        Returns:
            A ready-to-use :class:`Predictor`.

        Raises:
            FileNotFoundError: If the config or checkpoint is missing.
        """
        experiment_dir = Path(experiment_dir)
        config_path = experiment_dir / "config.yaml"
        checkpoint_path = experiment_dir / "checkpoints" / checkpoint_filename
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        config = load_config(config_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        class_names = checkpoint.get("class_names") or config.get("dataset.class_names")
        if not class_names:
            class_names = discover_class_names(config["dataset"]["path"])

        bundle = build_model_from_config(config["model"], num_classes=len(class_names))
        bundle.model.load_state_dict(checkpoint["model_state_dict"])

        norm = config["augmentation"]["normalize"]
        transform = build_eval_transform(
            image_size=int(config["dataset"]["image_size"]),
            mean=norm["mean"],
            std=norm["std"],
        )
        return cls(
            model=bundle.model,
            transform=transform,
            class_names=class_names,
            model_name=str(config["model"]["name"]),
            device=device,
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        """Predict sorted class probabilities for a PIL image."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(tensor), dim=1).squeeze(0).cpu().tolist()
        paired = sorted(zip(self.class_names, probs), key=lambda kv: kv[1], reverse=True)
        return Prediction(
            model_name=self.model_name,
            probabilities={name: float(p) for name, p in paired},
        )
