"""Training callbacks: early stopping and checkpoint management."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn


def _mode_for_monitor(monitor: str) -> str:
    """Return ``"min"`` for loss-like metrics, ``"max"`` for accuracy-like."""
    return "max" if "acc" in monitor.lower() else "min"


def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric key to watch (e.g. ``"val_loss"``).
        patience: Epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        enabled: If ``False``, :meth:`step` never signals a stop.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled
        self.mode = _mode_for_monitor(monitor)
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metrics: Mapping[str, float]) -> bool:
        """Update state with the latest metrics; return whether to stop."""
        if not self.enabled:
            return False
        current = metrics[self.monitor]
        if _is_improvement(current, self.best, self.mode, self.min_delta):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CheckpointManager:
    """Save best/last/periodic checkpoints and prune old periodic ones.

    Args:
        directory: Directory to write checkpoints into.
        monitor: Metric used to determine the best checkpoint.
        save_best: Save ``best_model.pth`` on improvement.
        save_last: Overwrite ``last_model.pth`` each epoch.
        save_every_n_epochs: If set, also save ``epoch_NNN.pth`` periodically.
        max_keep: Cap on periodic checkpoints retained (oldest pruned first).
    """

    def __init__(
        self,
        directory: str | Path,
        monitor: str = "val_loss",
        save_best: bool = True,
        save_last: bool = True,
        save_every_n_epochs: int | None = None,
        max_keep: int | None = None,
        extra_state: Mapping | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = _mode_for_monitor(monitor)
        self.save_best = save_best
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.max_keep = max_keep
        self.extra_state = dict(extra_state or {})
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self._periodic: list[Path] = []

    def _save(self, model: nn.Module, path: Path, epoch: int, metrics: Mapping[str, float], extra: Mapping) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": dict(metrics),
        }
        payload.update(self.extra_state)
        payload.update(extra)
        torch.save(payload, path)

    def update(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Mapping[str, float],
        extra: Mapping | None = None,
    ) -> bool:
        """Persist checkpoints for this epoch.

        Returns:
            ``True`` if this epoch produced a new best checkpoint.
        """
        extra = extra or {}
        is_best = False

        if self.save_last:
            self._save(model, self.directory / "last_model.pth", epoch, metrics, extra)

        if self.save_best and _is_improvement(metrics[self.monitor], self.best, self.mode, 0.0):
            self.best = metrics[self.monitor]
            self._save(model, self.directory / "best_model.pth", epoch, metrics, extra)
            is_best = True

        if self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0:
            path = self.directory / f"epoch_{epoch:03d}.pth"
            self._save(model, path, epoch, metrics, extra)
            self._periodic.append(path)
            self._prune_periodic()

        return is_best

    def _prune_periodic(self) -> None:
        if self.max_keep is None:
            return
        while len(self._periodic) > self.max_keep:
            oldest = self._periodic.pop(0)
            oldest.unlink(missing_ok=True)
