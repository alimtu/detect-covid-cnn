"""Shared data structures for background training jobs.

Kept in a tiny dependency-free module so both the parent process (job manager,
running inside Streamlit) and the spawned child (training worker) can import it
without pulling in heavy libraries or streamlit.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


class JobState:
    """Lifecycle states for a training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

    TERMINAL = {COMPLETED, FAILED, STOPPED}
    ACTIVE = {PENDING, RUNNING}


@dataclass
class JobRecord:
    """Serializable state of a single training job (the on-disk source of truth)."""

    job_id: str
    state: str
    created_at: str
    label: str = ""
    model: str = ""
    total_epochs: int = 0
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_dir: Optional[str] = None
    device: Optional[str] = None
    pid: Optional[int] = None
    error: Optional[str] = None
    final_metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    def save(self, path: Path) -> None:
        """Atomically write this record to ``path`` (temp file + os.replace)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @classmethod
    def load(cls, path: Path) -> Optional["JobRecord"]:
        """Load a record, tolerating a concurrent write (returns ``None``)."""
        try:
            with path.open("r", encoding="utf-8") as handle:
                return cls.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError):
            return None
