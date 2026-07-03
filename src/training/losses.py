"""Loss function construction, including a configurable Focal Loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    Down-weights easy examples to focus training on hard/minority cases.

    Args:
        gamma: Focusing parameter; ``0`` reduces to cross-entropy.
        alpha: Optional per-class weights (tensor of shape ``(num_classes,)``).
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - target_probs) ** self.gamma
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, target)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    config,
    class_weights: torch.Tensor | None,
    device: torch.device,
) -> nn.Module:
    """Build a loss function from the ``loss`` config section.

    Args:
        config: The full experiment configuration.
        class_weights: Resolved per-class weights (or ``None`` for uniform).
        device: Device the weights should live on.

    Returns:
        The configured loss module.

    Raises:
        ValueError: If the loss name is unsupported.
    """
    loss_cfg = config["loss"]
    name = str(loss_cfg["name"]).lower()
    weight = class_weights.to(device) if class_weights is not None else None

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    if name == "focal":
        focal_cfg = loss_cfg.get("focal", {})
        gamma = float(focal_cfg.get("gamma", 2.0)) if focal_cfg else 2.0
        return FocalLoss(gamma=gamma, alpha=weight)
    raise ValueError(f"Unsupported loss '{name}'.")
