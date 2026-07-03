"""Compare multiple experiments side by side.

Reads the same ``summary.json`` / ``metrics.json`` / ``training_history.csv``
artifacts as the ``compare.py`` CLI and assembles tidy pandas structures for the
Compare page (metrics table, per-class curves, learning curves, exports).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from frontend.services import experiment_service
from frontend.services.experiment_service import ExperimentRecord

# Scalar metrics compared in the headline table (higher is better for all).
SCALAR_METRICS: list[tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("precision_macro", "Precision"),
    ("recall_macro", "Recall"),
    ("f1_macro", "Macro F1"),
    ("roc_auc_macro", "ROC AUC"),
    ("ap_macro", "Avg Precision"),
]


def records_for(experiment_ids: list[str], output_root: Path | None = None) -> list[ExperimentRecord]:
    """Load records for the given experiment ids (skipping missing ones)."""
    root = output_root or experiment_service.paths.OUTPUTS_ROOT
    records = []
    for exp_id in experiment_ids:
        path = root / exp_id
        if experiment_service.is_experiment_dir(path):
            records.append(experiment_service.load_record(path))
    return records


def metrics_table(records: list[ExperimentRecord]) -> pd.DataFrame:
    """Return a DataFrame indexed by experiment with one column per metric."""
    data: dict[str, dict[str, Any]] = {}
    for record in records:
        data[record.experiment_id] = {
            label: getattr(record, key) for key, label in SCALAR_METRICS
        }
        data[record.experiment_id]["Model"] = record.model
        data[record.experiment_id]["Train time (s)"] = record.training_time_seconds
    frame = pd.DataFrame(data).T
    # Order columns: Model first, then metrics, then time.
    ordered = ["Model"] + [label for _, label in SCALAR_METRICS] + ["Train time (s)"]
    return frame[[c for c in ordered if c in frame.columns]]


def best_experiment_per_metric(records: list[ExperimentRecord]) -> dict[str, str]:
    """Map each metric label to the experiment id that maximises it."""
    winners: dict[str, str] = {}
    for key, label in SCALAR_METRICS:
        scored = [(r.experiment_id, getattr(r, key)) for r in records if getattr(r, key) is not None]
        if scored:
            winners[label] = max(scored, key=lambda kv: kv[1])[0]
    return winners


def per_class_metric(records: list[ExperimentRecord], metric: str) -> pd.DataFrame:
    """Return a per-class table for ``metric`` (``roc_auc`` or ``average_precision``).

    Rows are classes, columns are experiments.
    """
    columns: dict[str, dict[str, float]] = {}
    for record in records:
        metrics = experiment_service.load_metrics(record.path)
        values = metrics.get(metric) or {}
        columns[record.experiment_id] = {
            cls: float(val)
            for cls, val in values.items()
            if cls != "macro" and isinstance(val, (int, float))
        }
    return pd.DataFrame(columns)


def learning_curves(records: list[ExperimentRecord], column: str) -> pd.DataFrame:
    """Return a long-format DataFrame of ``column`` over epochs across experiments.

    Columns: ``epoch``, ``value``, ``experiment``.
    """
    frames = []
    for record in records:
        history = experiment_service.load_history(record.path)
        if history.empty or column not in history.columns:
            continue
        epochs = history["epoch"] if "epoch" in history.columns else range(1, len(history) + 1)
        frames.append(
            pd.DataFrame(
                {
                    "epoch": list(epochs),
                    "value": history[column].tolist(),
                    "experiment": record.experiment_id,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def confusion_matrices(records: list[ExperimentRecord]) -> dict[str, dict[str, Any]]:
    """Return ``{experiment_id: {"matrix": [[...]], "classes": [...]}}``."""
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        metrics = experiment_service.load_metrics(record.path)
        matrix = metrics.get("confusion_matrix")
        classes = metrics.get("class_names")
        if matrix and classes:
            result[record.experiment_id] = {"matrix": matrix, "classes": classes}
    return result


def export_csv(records: list[ExperimentRecord]) -> bytes:
    """Return the metrics table encoded as CSV bytes."""
    buffer = io.StringIO()
    metrics_table(records).to_csv(buffer)
    return buffer.getvalue().encode("utf-8")


def export_json(records: list[ExperimentRecord]) -> bytes:
    """Return the comparison (metrics + winners) encoded as JSON bytes."""
    payload = {
        "experiments": metrics_table(records).to_dict(orient="index"),
        "best_per_metric": best_experiment_per_metric(records),
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")
