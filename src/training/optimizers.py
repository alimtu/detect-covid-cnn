"""Optimizer construction from configuration."""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import SGD, Adam, AdamW, Optimizer


def build_optimizer(params: Iterable[torch.nn.Parameter], config) -> Optimizer:
    """Build an optimizer from the ``optimizer`` config section.

    Only parameters with ``requires_grad=True`` should be passed in (this keeps
    frozen backbones out of the optimizer).

    Args:
        params: Iterable of parameters to optimize.
        config: The full experiment configuration.

    Returns:
        The configured optimizer.

    Raises:
        ValueError: If the optimizer name is unsupported.
    """
    opt_cfg = config["optimizer"]
    name = str(opt_cfg["name"]).lower()
    lr = float(opt_cfg["learning_rate"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay, betas=tuple(opt_cfg.get("betas", (0.9, 0.999))))
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay, betas=tuple(opt_cfg.get("betas", (0.9, 0.999))))
    if name == "sgd":
        return SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(opt_cfg.get("momentum", 0.0)),
            nesterov=bool(opt_cfg.get("nesterov", False)),
        )
    raise ValueError(f"Unsupported optimizer '{name}'.")
