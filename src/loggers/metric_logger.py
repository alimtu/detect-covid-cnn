"""Composite logger fanning metrics out to console, CSV and TensorBoard."""

from __future__ import annotations

import logging
from pathlib import Path

from src.loggers.csv_logger import CSVLogger
from src.loggers.tensorboard_logger import TensorBoardLogger


class MetricLogger:
    """Aggregate per-epoch logging across the configured sinks.

    Args:
        config: The full experiment configuration (``logging`` section is read).
        csv_path: Destination for the CSV history.
        tensorboard_dir: Directory for TensorBoard events.
        logger: Console logger to use.
    """

    def __init__(
        self,
        config,
        csv_path: str | Path,
        tensorboard_dir: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        log_cfg = config["logging"]
        self.console = logger or logging.getLogger("covid.train")
        self.console_enabled = bool(log_cfg.get("console", True))

        self.csv = CSVLogger(csv_path) if bool(log_cfg.get("csv", True)) else None
        self.tensorboard = TensorBoardLogger(
            tensorboard_dir, enabled=bool(log_cfg.get("tensorboard", False))
        )

    def log_epoch(self, epoch: int, epochs: int, lr: float, metrics: dict[str, float]) -> None:
        """Log one epoch's metrics to all enabled sinks."""
        if self.console_enabled:
            self.console.info(
                "Epoch %02d/%02d | lr %.2e | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
                epoch,
                epochs,
                lr,
                metrics.get("train_loss", float("nan")),
                metrics.get("train_acc", float("nan")),
                metrics.get("val_loss", float("nan")),
                metrics.get("val_acc", float("nan")),
            )
        if self.csv is not None:
            self.csv.log({"epoch": epoch, "lr": lr, **metrics})
        self.tensorboard.log_scalars({"lr": lr, **metrics}, step=epoch)

    def info(self, message: str, *args) -> None:
        """Proxy an informational message to the console logger."""
        if self.console_enabled:
            self.console.info(message, *args)

    def close(self) -> None:
        self.tensorboard.close()
