"""Derive live training progress from a job record + its history CSV.

Per-epoch metrics are read straight from the ``training_history.csv`` that the
backend's ``CSVLogger`` appends each epoch, so the monitor needs no cooperation
from the training loop (zero backend changes). Estimated time remaining is
extrapolated from elapsed wall-clock and epochs completed so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from frontend.services.experiment_service import load_history
from frontend.services.job_models import JobRecord, JobState


@dataclass
class LiveProgress:
    """A snapshot of an in-flight (or finished) training job for the monitor."""

    state: str
    epochs_completed: int
    total_epochs: int
    latest: dict[str, float] = field(default_factory=dict)
    best_val_acc: Optional[float] = None
    best_val_loss: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    history: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def fraction(self) -> float:
        """Completion fraction in ``[0, 1]`` (epoch-granular)."""
        if self.total_epochs <= 0:
            return 0.0
        return max(0.0, min(1.0, self.epochs_completed / self.total_epochs))

    @property
    def is_active(self) -> bool:
        return self.state in JobState.ACTIVE


def compute_progress(record: JobRecord) -> LiveProgress:
    """Build a :class:`LiveProgress` snapshot for ``record``."""
    history = pd.DataFrame()
    if record.experiment_dir:
        history = load_history(Path(record.experiment_dir))

    epochs_completed = int(len(history))
    latest: dict[str, float] = {}
    best_val_acc: Optional[float] = None
    best_val_loss: Optional[float] = None

    if epochs_completed:
        row = history.iloc[-1]
        for key in ("train_loss", "train_acc", "val_loss", "val_acc", "lr"):
            if key in history.columns:
                latest[key] = _as_float(row.get(key))
        if "val_acc" in history.columns:
            best_val_acc = _as_float(history["val_acc"].max())
        if "val_loss" in history.columns:
            best_val_loss = _as_float(history["val_loss"].min())

    elapsed = _elapsed_seconds(record)
    eta = _estimate_eta(elapsed, epochs_completed, record.total_epochs, record.state)

    return LiveProgress(
        state=record.state,
        epochs_completed=epochs_completed,
        total_epochs=record.total_epochs,
        latest=latest,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        elapsed_seconds=elapsed,
        eta_seconds=eta,
        history=history,
    )


def _elapsed_seconds(record: JobRecord) -> Optional[float]:
    if not record.started_at:
        return None
    try:
        start = datetime.fromisoformat(record.started_at)
    except ValueError:
        return None
    end = datetime.now()
    if record.ended_at:
        try:
            end = datetime.fromisoformat(record.ended_at)
        except ValueError:
            pass
    return max(0.0, (end - start).total_seconds())


def _estimate_eta(
    elapsed: Optional[float],
    epochs_completed: int,
    total_epochs: int,
    state: str,
) -> Optional[float]:
    if state in JobState.TERMINAL:
        return 0.0
    if not elapsed or epochs_completed <= 0 or total_epochs <= 0:
        return None
    per_epoch = elapsed / epochs_completed
    remaining = max(0, total_epochs - epochs_completed)
    return per_epoch * remaining


def _as_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        result = float(value)
        return result if result == result else None  # drop NaN
    except (TypeError, ValueError):
        return None
