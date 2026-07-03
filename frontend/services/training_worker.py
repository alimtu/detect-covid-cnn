"""Entry point executed inside the spawned training child process.

The parent (:mod:`frontend.services.job_manager`) writes a :class:`JobRecord`
JSON file, then launches :func:`run_training_job` in a fresh ``spawn`` process
with that file's path as the sole (string) argument. The worker:

1. marks the job ``running`` and records its PID,
2. calls :func:`src.pipeline.run_experiment` (the *same* pipeline the CLI uses),
   recording the experiment folder name as soon as it is created,
3. marks the job ``completed`` (with final metrics) or ``failed`` (with the
   traceback).

Heavy imports (torch, backend) happen inside the function so that when the
``spawn`` bootstrap re-imports this module the cost is minimal.
"""

from __future__ import annotations

from pathlib import Path


def run_training_job(job_file: str) -> None:
    """Run one training job described by the JSON file at ``job_file``.

    This function is the multiprocessing target. It must remain importable at
    module scope and accept only picklable (string) arguments.
    """
    import os
    import traceback
    from datetime import datetime

    from frontend.services.job_models import JobRecord, JobState

    path = Path(job_file)
    record = JobRecord.load(path)
    if record is None:  # pragma: no cover - defensive
        return

    def persist() -> None:
        record.save(path)

    record.state = JobState.RUNNING
    record.pid = os.getpid()
    record.started_at = datetime.now().isoformat(timespec="seconds")
    persist()

    try:
        # Imported lazily: keeps the spawn re-import lightweight and defers the
        # torch import to the child that actually needs it.
        from src.config import Config
        from src.pipeline import run_experiment
        from src.utils import get_logger

        from frontend.services.device_service import resolve_device

        logger = get_logger("covid.job")
        config = Config(record.config)
        device = resolve_device(record.device or "auto")
        record.device = str(device)
        record.total_epochs = int(config.get("training.epochs", record.total_epochs))
        persist()

        def _on_created(experiment) -> None:
            record.experiment_name = experiment.name
            record.experiment_dir = str(experiment.dir)
            persist()

        result = run_experiment(
            config,
            device=device,
            logger=logger,
            on_experiment_created=_on_created,
        )

        record.state = JobState.COMPLETED
        record.ended_at = datetime.now().isoformat(timespec="seconds")
        record.device = result.device
        # Store only scalar metrics for the notification/summary panels.
        record.final_metrics = {
            k: v for k, v in result.metrics.items() if not isinstance(v, (list, dict))
        }
        persist()
    except Exception:  # noqa: BLE001 - surface any failure to the UI
        record.state = JobState.FAILED
        record.ended_at = datetime.now().isoformat(timespec="seconds")
        record.error = traceback.format_exc()
        persist()
