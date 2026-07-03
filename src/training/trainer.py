"""Reusable training loop wiring together the modular training components.

Supports mixed precision (AMP on CUDA), gradient clipping, gradient
accumulation, backbone freeze/unfreeze, pluggable optimizers/schedulers/losses,
early stopping, checkpointing and multi-sink logging.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import ModelBundle, freeze_backbone, unfreeze_all
from src.training.callbacks import CheckpointManager, EarlyStopping
from src.training.optimizers import build_optimizer
from src.training.schedulers import SchedulerBundle, build_scheduler
from src.loggers import MetricLogger


class Trainer:
    """Train a model according to the experiment configuration.

    Args:
        bundle: The model and its classifier head.
        dataloaders: Mapping with ``"train"`` and ``"val"`` loaders.
        device: Compute device.
        config: The full experiment configuration.
        criterion: The loss function.
        metric_logger: Composite logger for per-epoch metrics.
        checkpoint_manager: Manages best/last/periodic checkpoints.
        early_stopping: Early stopping callback.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        dataloaders: dict[str, DataLoader],
        device: torch.device,
        config,
        criterion: nn.Module,
        metric_logger: MetricLogger,
        checkpoint_manager: CheckpointManager,
        early_stopping: EarlyStopping,
    ) -> None:
        self.bundle = bundle
        self.model = bundle.model.to(device)
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.device = device
        self.config = config
        self.criterion = criterion
        self.metric_logger = metric_logger
        self.checkpoints = checkpoint_manager
        self.early_stopping = early_stopping
        self.logger = logging.getLogger("covid.trainer")

        train_cfg = config["training"]
        self.epochs = int(train_cfg["epochs"])
        self.grad_clip = train_cfg.get("gradient_clip")
        self.accumulation_steps = max(int(train_cfg.get("gradient_accumulation_steps", 1)), 1)

        model_cfg = config["model"]
        self.unfreeze_after_epoch = model_cfg.get("unfreeze_after_epoch")
        if bool(model_cfg.get("freeze_backbone", False)):
            freeze_backbone(bundle)
            self.logger.info("Backbone frozen; training classifier head only.")

        # Mixed precision is only enabled on CUDA.
        requested_amp = bool(train_cfg.get("mixed_precision", False))
        self.use_amp = requested_amp and device.type == "cuda"
        if requested_amp and not self.use_amp:
            self.logger.warning("mixed_precision requested but unsupported on %s; disabling.", device.type)
        self.scaler = self._make_grad_scaler(self.use_amp)

        self._build_optimizer_and_scheduler()

        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }

    @staticmethod
    def _make_grad_scaler(enabled: bool):
        """Create an AMP GradScaler using the newer API when available."""
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):  # pragma: no cover - older torch
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def _build_optimizer_and_scheduler(self) -> None:
        """(Re)build the optimizer and scheduler over currently trainable params."""
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = build_optimizer(trainable, self.config)
        self.scheduler_bundle: SchedulerBundle = build_scheduler(
            self.optimizer, self.config, steps_per_epoch=len(self.train_loader), epochs=self.epochs
        )

    def _maybe_unfreeze(self, epoch: int) -> None:
        if self.unfreeze_after_epoch is not None and epoch == int(self.unfreeze_after_epoch):
            unfreeze_all(self.bundle)
            self._build_optimizer_and_scheduler()
            self.logger.info("Unfroze full network at epoch %d; optimizer/scheduler rebuilt.", epoch)

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        self.optimizer.zero_grad()
        num_batches = len(self.train_loader)

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) / self.accumulation_steps

            self.scaler.scale(loss).backward()

            is_step = ((i + 1) % self.accumulation_steps == 0) or (i + 1 == num_batches)
            if is_step:
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler_bundle.step_mode == "batch":
                    self.scheduler_bundle.scheduler.step()

            running_loss += loss.item() * self.accumulation_steps * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        return running_loss / max(total, 1), correct / max(total, 1)

    def _step_epoch_scheduler(self, val_loss: float) -> None:
        mode = self.scheduler_bundle.step_mode
        if mode == "plateau":
            self.scheduler_bundle.scheduler.step(val_loss)
        elif mode == "epoch":
            self.scheduler_bundle.scheduler.step()

    def fit(self) -> dict[str, list[float]]:
        """Run the training loop and return the per-epoch history."""
        self.logger.info("Training for up to %d epochs on %s", self.epochs, self.device)
        model_name = str(self.config["model"]["name"])

        for epoch in range(1, self.epochs + 1):
            self._maybe_unfreeze(epoch)

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate()
            self._step_epoch_scheduler(val_loss)

            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            for key, value in metrics.items():
                self.history[key].append(value)

            lr = self.optimizer.param_groups[0]["lr"]
            self.metric_logger.log_epoch(epoch, self.epochs, lr, metrics)

            is_best = self.checkpoints.update(
                self.model, epoch, metrics, extra={"model_name": model_name}
            )
            if is_best:
                self.metric_logger.info("  -> new best (%s=%.4f)", self.checkpoints.monitor, metrics[self.checkpoints.monitor])

            if self.early_stopping.step(metrics):
                self.metric_logger.info("Early stopping at epoch %d.", epoch)
                break

        return self.history
