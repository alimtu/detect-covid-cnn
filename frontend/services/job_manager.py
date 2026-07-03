"""Background training job manager (process-based).

Runs each experiment in a dedicated ``spawn`` child process that calls the
backend :func:`src.pipeline.run_experiment` directly — no shell, no CLI. This
keeps the Streamlit UI responsive, survives script reruns, and lays the ground
for a future multi-job queue.

State is file-based: each job is a JSON :class:`JobRecord` under
``outputs/.jobs/``. The on-disk record is the source of truth, so job status
survives even a full app reload; the in-memory process handles are only used to
stop/terminate running jobs within the current server lifetime.

Live per-epoch metrics are **not** pushed from the worker — the UI reads the
``training_history.csv`` that the backend's ``CSVLogger`` already appends each
epoch (see :func:`frontend.services.monitor_service`).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from frontend.services import paths
from frontend.services.job_models import JobRecord, JobState
from frontend.services.training_worker import run_training_job

# Spawn context is mandatory: fork is unsafe with torch/MPS on macOS.
_CTX = mp.get_context("spawn")

# In-memory handles to live child processes, keyed by job_id. Module-level so it
# survives Streamlit reruns (imported modules persist in the server process).
_PROCESSES: dict[str, "mp.process.BaseProcess"] = {}


def _job_path(job_id: str) -> Path:
    return paths.JOBS_DIR / f"{job_id}.json"


def submit(config: dict[str, Any], *, label: str = "", device: str = "auto") -> str:
    """Start a background training job for ``config`` and return its ``job_id``.

    Args:
        config: A fully resolved experiment config dict.
        label: Optional human-friendly label shown in the UI.
        device: Preference passed to the worker (``auto``/``mps``/``cuda``/``cpu``).
    """
    paths.ensure_runtime_dirs()
    job_id = f"job_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"

    record = JobRecord(
        job_id=job_id,
        state=JobState.PENDING,
        created_at=datetime.now().isoformat(timespec="seconds"),
        label=label or str(config.get("model", {}).get("name", "experiment")),
        model=str(config.get("model", {}).get("name", "")),
        total_epochs=int(config.get("training", {}).get("epochs", 0) or 0),
        device=device,
        config=config,
    )
    path = _job_path(job_id)
    record.save(path)

    process = _CTX.Process(
        target=run_training_job,
        args=(str(path),),
        name=f"train-{job_id}",
        daemon=False,
    )
    process.start()
    _PROCESSES[job_id] = process
    return job_id


def get_job(job_id: str) -> Optional[JobRecord]:
    """Load a single job record from disk (reconciling liveness)."""
    record = JobRecord.load(_job_path(job_id))
    if record is not None:
        _reconcile(record)
    return record


def list_jobs() -> list[JobRecord]:
    """Return all known jobs, newest first (reconciled against process liveness)."""
    if not paths.JOBS_DIR.is_dir():
        return []
    records: list[JobRecord] = []
    for path in paths.JOBS_DIR.glob("job_*.json"):
        record = JobRecord.load(path)
        if record is not None:
            _reconcile(record)
            records.append(record)
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records


def active_jobs() -> list[JobRecord]:
    """Return jobs that are pending or running."""
    return [r for r in list_jobs() if r.state in JobState.ACTIVE]


def latest_job() -> Optional[JobRecord]:
    """Return the most recently created job, if any."""
    jobs = list_jobs()
    return jobs[0] if jobs else None


def running_experiment_dirs() -> set[str]:
    """Return experiment dirs currently owned by active jobs (for status merge)."""
    return {
        r.experiment_dir
        for r in active_jobs()
        if r.experiment_dir is not None
    }


def stop(job_id: str) -> bool:
    """Terminate a running job. Returns whether a live process was signalled."""
    process = _PROCESSES.get(job_id)
    signalled = False
    if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():  # pragma: no cover - stubborn child
            os.kill(process.pid, signal.SIGKILL)
        signalled = True

    record = JobRecord.load(_job_path(job_id))
    if record is not None and record.state in JobState.ACTIVE:
        record.state = JobState.STOPPED
        record.ended_at = datetime.now().isoformat(timespec="seconds")
        record.error = record.error or "Stopped by user."
        record.save(_job_path(job_id))
    _PROCESSES.pop(job_id, None)
    return signalled


def clear_finished() -> int:
    """Delete job records that have reached a terminal state. Returns the count."""
    removed = 0
    for record in list_jobs():
        if record.state in JobState.TERMINAL:
            path = _job_path(record.job_id)
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
            _PROCESSES.pop(record.job_id, None)
    return removed


def _reconcile(record: JobRecord) -> None:
    """Mark an active job failed if its process is known-dead this session.

    We only trust a *known* dead handle from this server process. If the handle
    is absent (e.g. after an app reload) we leave the state as-is; the worker
    itself writes the terminal state, so a genuinely finished job is already
    reflected on disk.
    """
    if record.state not in JobState.ACTIVE:
        return
    process = _PROCESSES.get(record.job_id)
    if process is not None and not process.is_alive() and process.exitcode not in (0, None):
        record.state = JobState.FAILED
        record.ended_at = datetime.now().isoformat(timespec="seconds")
        record.error = record.error or (
            f"Worker process exited unexpectedly (code {process.exitcode})."
        )
        record.save(_job_path(record.job_id))
        _PROCESSES.pop(record.job_id, None)
