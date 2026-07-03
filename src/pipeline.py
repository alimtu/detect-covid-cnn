"""End-to-end experiment pipeline: train + evaluate into a fresh folder.

This module extracts the orchestration that ``run.py`` historically performed so
that it can be reused from *any* frontend (the CLI, the Streamlit dashboard, a
background worker) without duplicating logic. The command-line entry point and
the UI now both call :func:`run_experiment`.

The function is a faithful sequencing of the existing backend components; it adds
no new training behaviour. It only:

* accepts an already-loaded :class:`~src.config.Config` (so callers may build a
  config in memory instead of on disk),
* exposes an ``on_experiment_created`` hook so a caller can learn the experiment
  folder name as soon as it exists (used by the live monitor), and
* records a small ``run_meta.json`` (wall-clock training time + device) next to
  the other artifacts, which the dashboard surfaces as columns.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import torch

from src.config import Config, ExperimentManager, validate_config
from src.datasets import build_dataloaders, compute_class_weights
from src.evaluation import evaluate_model
from src.loggers import MetricLogger
from src.models import build_model_from_config
from src.training import CheckpointManager, EarlyStopping, Trainer, build_loss
from src.utils import get_device, get_logger, set_seed
from src.utils.visualization import plot_history

# Callback invoked with the experiment manager immediately after its folder is
# created, before training begins.
ExperimentCreatedHook = Callable[[ExperimentManager], None]


@dataclass
class ExperimentResult:
    """Everything a caller might want after a completed run."""

    experiment: ExperimentManager
    history: dict[str, list[float]]
    metrics: dict[str, Any]
    training_time_seconds: float
    device: str


def run_experiment(
    config: Config,
    *,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
    on_experiment_created: Optional[ExperimentCreatedHook] = None,
    validate: bool = True,
) -> ExperimentResult:
    """Train and evaluate a single experiment described by ``config``.

    Args:
        config: The fully resolved experiment configuration.
        device: Optional device override; auto-selected (MPS/CUDA/CPU) if ``None``.
        logger: Optional logger; a module logger is used if ``None``.
        on_experiment_created: Optional hook called with the
            :class:`ExperimentManager` right after its folder is created. Used by
            the dashboard to locate the live ``training_history.csv``.
        validate: Whether to run :func:`validate_config` first.

    Returns:
        An :class:`ExperimentResult` with the experiment, per-epoch history,
        final test metrics, training wall-time and the device string.
    """
    logger = logger or get_logger("covid.run")

    if validate:
        validate_config(config)

    repro = config["reproducibility"]
    set_seed(int(repro["seed"]), deterministic=bool(repro.get("deterministic", True)))
    device = device or get_device()

    experiment = ExperimentManager.create(config)
    logger.info("Experiment: %s (%s)", experiment.name, experiment.dir)
    if on_experiment_created is not None:
        on_experiment_created(experiment)

    # Data.
    bundle, dataloaders = build_dataloaders(config)
    num_classes = config.get("model.num_classes") or len(bundle.class_names)
    logger.info(
        "Classes %s | train=%d val=%d test=%d",
        bundle.class_names, len(bundle.train), len(bundle.val), len(bundle.test),
    )

    # Model.
    model_bundle = build_model_from_config(config["model"], num_classes=int(num_classes))

    # Loss (resolve class weights if requested).
    class_weights = compute_class_weights(
        bundle.train_targets, int(num_classes), config.get("loss.class_weights")
    )
    criterion = build_loss(config, class_weights, device)

    # Logging + callbacks.
    metric_logger = MetricLogger(
        config,
        csv_path=experiment.history_path,
        tensorboard_dir=experiment.tensorboard_dir,
        logger=logger,
    )
    ckpt_cfg = config["training"]["checkpoint"]
    checkpoint_manager = CheckpointManager(
        directory=experiment.checkpoints_dir,
        monitor=str(ckpt_cfg.get("monitor", "val_loss")),
        save_best=bool(ckpt_cfg.get("save_best", True)),
        save_last=bool(ckpt_cfg.get("save_last", True)),
        save_every_n_epochs=ckpt_cfg.get("save_every_n_epochs"),
        max_keep=ckpt_cfg.get("max_keep"),
        extra_state={"class_names": bundle.class_names},
    )
    es_cfg = config["training"]["early_stopping"]
    early_stopping = EarlyStopping(
        monitor=str(es_cfg.get("monitor", "val_loss")),
        patience=int(es_cfg.get("patience", 5)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
        enabled=bool(es_cfg.get("enabled", True)),
    )

    # Train.
    trainer = Trainer(
        bundle=model_bundle,
        dataloaders=dataloaders,
        device=device,
        config=config,
        criterion=criterion,
        metric_logger=metric_logger,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
    )
    started_at = datetime.now()
    start = time.perf_counter()
    history = trainer.fit()
    training_time_seconds = time.perf_counter() - start
    plot_history(history, experiment.plots_dir)

    # Evaluate the best checkpoint on the test set.
    best_path = experiment.checkpoints_dir / "best_model.pth"
    if best_path.is_file():
        checkpoint = torch.load(best_path, map_location=device)
        model_bundle.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded best checkpoint (epoch %s) for evaluation.", checkpoint.get("epoch"))

    metrics = evaluate_model(
        model=model_bundle.model,
        loader=dataloaders["test"],
        device=device,
        class_names=bundle.class_names,
        config=config,
        plots_dir=experiment.plots_dir,
    )
    experiment.save_metrics(metrics)
    experiment.save_summary(config, metrics)

    # Additive run metadata (device + wall time) for the dashboard. Written as a
    # separate file so no existing backend artifact format changes.
    _write_run_meta(
        experiment,
        device=str(device),
        training_time_seconds=training_time_seconds,
        started_at=started_at,
        epochs_completed=len(history.get("train_loss", [])),
    )
    metric_logger.close()

    logger.info(
        "Test accuracy %.4f | precision %.4f | recall %.4f | f1 %.4f",
        metrics.get("accuracy", float("nan")),
        metrics.get("precision_macro", float("nan")),
        metrics.get("recall_macro", float("nan")),
        metrics.get("f1_macro", float("nan")),
    )
    logger.info("All artifacts saved under %s", experiment.dir)

    return ExperimentResult(
        experiment=experiment,
        history=history,
        metrics=metrics,
        training_time_seconds=training_time_seconds,
        device=str(device),
    )


def _write_run_meta(
    experiment: ExperimentManager,
    *,
    device: str,
    training_time_seconds: float,
    started_at: datetime,
    epochs_completed: int,
) -> None:
    """Persist wall-clock + device metadata beside the experiment artifacts."""
    meta = {
        "device": device,
        "training_time_seconds": round(float(training_time_seconds), 3),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "epochs_completed": int(epochs_completed),
    }
    path = experiment.dir / "run_meta.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
