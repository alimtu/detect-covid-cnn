"""Learning-rate scheduler construction from configuration.

Returns both the scheduler and a ``step_mode`` telling the trainer when to call
``scheduler.step()``:

* ``"plateau"`` - once per epoch, with the monitored validation metric.
* ``"epoch"``   - once per epoch, no argument.
* ``"batch"``   - once per optimizer step (e.g. OneCycleLR).
* ``"none"``    - no scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


@dataclass
class SchedulerBundle:
    """A scheduler paired with its stepping mode."""

    scheduler: object | None
    step_mode: str  # "plateau" | "epoch" | "batch" | "none"


def build_scheduler(
    optimizer: Optimizer,
    config,
    steps_per_epoch: int,
    epochs: int,
) -> SchedulerBundle:
    """Build a scheduler from the ``scheduler`` config section.

    Args:
        optimizer: The optimizer to schedule.
        config: The full experiment configuration.
        steps_per_epoch: Number of optimizer steps per epoch (for OneCycleLR).
        epochs: Total number of epochs (for OneCycleLR).

    Returns:
        A :class:`SchedulerBundle`.

    Raises:
        ValueError: If the scheduler name is unsupported.
    """
    sched_cfg = config["scheduler"]
    name = str(sched_cfg["name"]).lower()

    if name == "none":
        return SchedulerBundle(None, "none")
    if name == "reduce_on_plateau":
        cfg = sched_cfg.get("reduce_on_plateau", {})
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=str(cfg.get("mode", "min")),
            factor=float(cfg.get("factor", 0.1)),
            patience=int(cfg.get("patience", 2)),
        )
        return SchedulerBundle(scheduler, "plateau")
    if name == "cosine":
        cfg = sched_cfg.get("cosine", {})
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.get("t_max", epochs)),
            eta_min=float(cfg.get("eta_min", 0.0)),
        )
        return SchedulerBundle(scheduler, "epoch")
    if name == "step":
        cfg = sched_cfg.get("step", {})
        scheduler = StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 10)),
            gamma=float(cfg.get("gamma", 0.1)),
        )
        return SchedulerBundle(scheduler, "epoch")
    if name == "onecycle":
        cfg = sched_cfg.get("onecycle", {})
        scheduler = OneCycleLR(
            optimizer,
            max_lr=float(cfg.get("max_lr", 0.01)),
            epochs=epochs,
            steps_per_epoch=max(steps_per_epoch, 1),
            pct_start=float(cfg.get("pct_start", 0.3)),
        )
        return SchedulerBundle(scheduler, "batch")
    raise ValueError(f"Unsupported scheduler '{name}'.")
