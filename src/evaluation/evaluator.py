"""Config-driven test-set evaluation: metrics, curves and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.utils.visualization import (
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
)


def _collect(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference; return ``(y_true, y_pred, y_score)``."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[list[float]] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1).cpu().numpy()
            y_score.extend(probs.tolist())
            y_pred.extend(probs.argmax(axis=1).tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_score)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    config,
    plots_dir: str | Path,
) -> dict:
    """Evaluate ``model`` and persist enabled metrics/plots.

    Which metrics and plots are produced is controlled by
    ``config.evaluation.metrics``.

    Args:
        model: Trained model.
        loader: Dataloader (typically the test set).
        device: Compute device.
        class_names: Ordered class labels.
        config: The full experiment configuration.
        plots_dir: Directory where plots and the classification report are saved.

    Returns:
        A dictionary of computed metrics (also suitable for JSON serialisation).
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_cfg = config["evaluation"]["metrics"]
    labels_idx = list(range(len(class_names)))

    y_true, y_pred, y_score = _collect(model, loader, device)
    results: dict = {"num_samples": int(len(y_true)), "class_names": list(class_names)}

    if bool(metrics_cfg.get("accuracy", True)):
        results["accuracy"] = float(accuracy_score(y_true, y_pred))

    if any(bool(metrics_cfg.get(k, True)) for k in ("precision", "recall", "f1")):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
        )
        if bool(metrics_cfg.get("precision", True)):
            results["precision_macro"] = float(precision)
        if bool(metrics_cfg.get("recall", True)):
            results["recall_macro"] = float(recall)
        if bool(metrics_cfg.get("f1", True)):
            results["f1_macro"] = float(f1)

    if bool(metrics_cfg.get("confusion_matrix", True)):
        matrix = confusion_matrix(y_true, y_pred, labels=labels_idx)
        results["confusion_matrix"] = matrix.tolist()
        plot_confusion_matrix(matrix, class_names, plots_dir / "confusion_matrix.png")

    if bool(metrics_cfg.get("classification_report", True)):
        report = classification_report(
            y_true, y_pred, labels=labels_idx, target_names=list(class_names), zero_division=0
        )
        (plots_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    if bool(metrics_cfg.get("roc_curve", True)):
        results["roc_auc"] = plot_roc_curves(y_true, y_score, class_names, plots_dir / "roc_curve.png")

    if bool(metrics_cfg.get("pr_curve", True)):
        results["average_precision"] = plot_pr_curves(y_true, y_score, class_names, plots_dir / "pr_curve.png")

    return results
