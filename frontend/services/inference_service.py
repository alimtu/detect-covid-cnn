"""Inference orchestration: build a Predictor and run single-image predictions.

Thin wrapper over :class:`src.inference.Predictor`. UI-agnostic (the Streamlit
page adds caching); returns plain data structures for rendering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from src.inference import Prediction, Predictor

from frontend.services.device_service import resolve_device


@dataclass
class PredictionResult:
    """A prediction plus presentation metadata."""

    model_name: str
    top_class: str
    top_confidence: float
    probabilities: dict[str, float]
    inference_ms: float


def load_predictor(
    experiment_dir: str | Path,
    checkpoint_filename: str = "best_model.pth",
    device_preference: str = "auto",
) -> Predictor:
    """Build a :class:`Predictor` for an experiment/checkpoint on a device."""
    device = resolve_device(device_preference)
    return Predictor.from_experiment(
        experiment_dir=str(experiment_dir),
        device=device,
        checkpoint_filename=checkpoint_filename,
    )


def predict_image(predictor: Predictor, image: Image.Image) -> PredictionResult:
    """Run inference on a PIL image and time it."""
    start = time.perf_counter()
    prediction: Prediction = predictor.predict(image)
    inference_ms = (time.perf_counter() - start) * 1000.0
    return PredictionResult(
        model_name=prediction.model_name,
        top_class=prediction.top_class,
        top_confidence=prediction.top_confidence,
        probabilities=dict(prediction.probabilities),
        inference_ms=inference_ms,
    )


def model_info(experiment_dir: str | Path) -> dict[str, Optional[str]]:
    """Return a small model-info dict for the inference sidebar."""
    from frontend.services import experiment_service

    record = experiment_service.load_record(Path(experiment_dir))
    return {
        "experiment": record.experiment_id,
        "model": record.model,
        "accuracy": None if record.accuracy is None else f"{record.accuracy:.4f}",
        "f1_macro": None if record.f1_macro is None else f"{record.f1_macro:.4f}",
        "device": record.device,
    }
