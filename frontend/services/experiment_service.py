"""Read, summarise and manage experiment folders under ``outputs/``.

Everything here operates on the artifacts the backend already writes
(``summary.json``, ``metrics.json``, ``training_history.csv``, ``plots/``,
``checkpoints/``) plus the additive ``run_meta.json`` from
:mod:`src.pipeline`. It performs no training or evaluation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from frontend.services import paths

_METRIC_KEYS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


@dataclass
class ExperimentRecord:
    """Flattened, display-ready view of one experiment folder."""

    experiment_id: str
    path: Path
    created_at: Optional[str]
    model: str
    epochs: Any
    batch_size: Any
    learning_rate: Any
    optimizer: str
    scheduler: str
    loss: str
    accuracy: Optional[float]
    precision_macro: Optional[float]
    recall_macro: Optional[float]
    f1_macro: Optional[float]
    roc_auc_macro: Optional[float]
    ap_macro: Optional[float]
    training_time_seconds: Optional[float]
    device: Optional[str]
    status: str
    summary: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        """Return a dict suited to a pandas DataFrame row."""
        return {
            "Experiment": self.experiment_id,
            "Date": _short_date(self.created_at),
            "Model": self.model,
            "Epochs": self.epochs,
            "Batch": self.batch_size,
            "LR": self.learning_rate,
            "Optimizer": self.optimizer,
            "Scheduler": self.scheduler,
            "Accuracy": self.accuracy,
            "Macro F1": self.f1_macro,
            "Train time": _fmt_duration(self.training_time_seconds),
            "Device": (self.device or "—").upper() if self.device else "—",
            "Status": self.status,
        }


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def is_experiment_dir(path: Path) -> bool:
    """A directory counts as an experiment if it has a config or summary."""
    return path.is_dir() and (
        (path / "config.yaml").is_file() or (path / "summary.json").is_file()
    )


def load_record(path: Path) -> ExperimentRecord:
    """Build an :class:`ExperimentRecord` from an experiment directory."""
    summary = _read_json(path / "summary.json")
    metrics = summary.get("final_metrics") or _read_json(path / "metrics.json")
    run_meta = _read_json(path / "run_meta.json")

    model_cfg = summary.get("model", {})
    optim_cfg = summary.get("optimizer", {})
    sched_cfg = summary.get("scheduler", {})
    loss_cfg = summary.get("loss", {})
    train_cfg = summary.get("training", {})

    roc = metrics.get("roc_auc") or {}
    ap = metrics.get("average_precision") or {}

    return ExperimentRecord(
        experiment_id=path.name,
        path=path,
        created_at=summary.get("created_at") or run_meta.get("finished_at"),
        model=str(model_cfg.get("name", "—")),
        epochs=train_cfg.get("epochs", "—"),
        batch_size=train_cfg.get("batch_size", "—"),
        learning_rate=optim_cfg.get("learning_rate", "—"),
        optimizer=str(optim_cfg.get("name", "—")),
        scheduler=str(sched_cfg.get("name", "—")),
        loss=str(loss_cfg.get("name", "—")),
        accuracy=_as_float(metrics.get("accuracy")),
        precision_macro=_as_float(metrics.get("precision_macro")),
        recall_macro=_as_float(metrics.get("recall_macro")),
        f1_macro=_as_float(metrics.get("f1_macro")),
        roc_auc_macro=_as_float(roc.get("macro")),
        ap_macro=_as_float(ap.get("macro")),
        training_time_seconds=_as_float(run_meta.get("training_time_seconds")),
        device=run_meta.get("device"),
        status="completed" if (path / "metrics.json").is_file() else "incomplete",
        summary=summary,
    )


def list_experiments(output_root: Path | None = None) -> list[ExperimentRecord]:
    """Return all experiment records under ``output_root`` (newest first)."""
    root = output_root or paths.OUTPUTS_ROOT
    if not root.is_dir():
        return []
    records = [
        load_record(child)
        for child in root.iterdir()
        if not child.name.startswith(".") and is_experiment_dir(child)
    ]
    records.sort(key=lambda r: r.experiment_id, reverse=True)
    return records


def load_history(path: Path) -> pd.DataFrame:
    """Load an experiment's per-epoch ``training_history.csv`` (may be empty)."""
    csv_path = path / "training_history.csv"
    if not csv_path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def load_metrics(path: Path) -> dict[str, Any]:
    """Load an experiment's full ``metrics.json``."""
    return _read_json(path / "metrics.json")


def load_config_dict(path: Path) -> dict[str, Any]:
    """Load an experiment's resolved ``config.yaml`` as a plain dict."""
    from frontend.services import config_service

    config_path = path / "config.yaml"
    if not config_path.is_file():
        return {}
    return config_service.load_config_file(config_path)


def list_checkpoints(path: Path) -> list[str]:
    """Return checkpoint filenames available for an experiment."""
    ckpt_dir = path / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    preferred = ["best_model.pth", "last_model.pth"]
    found = {p.name for p in ckpt_dir.glob("*.pth")}
    ordered = [name for name in preferred if name in found]
    ordered += sorted(found - set(preferred))
    return ordered


def plot_path(path: Path, filename: str) -> Optional[Path]:
    """Return a plot path if it exists (e.g. ``confusion_matrix.png``)."""
    candidate = path / "plots" / filename
    return candidate if candidate.is_file() else None


def classification_report(path: Path) -> Optional[str]:
    """Return the saved classification report text if present."""
    candidate = path / "plots" / "classification_report.txt"
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return None


# --------------------------------------------------------------------------- #
# Mutating actions
# --------------------------------------------------------------------------- #
def delete_experiment(path: Path) -> None:
    """Permanently delete an experiment folder."""
    if path.is_dir() and path.parent == paths.OUTPUTS_ROOT:
        shutil.rmtree(path)


def rename_experiment(path: Path, new_name: str) -> Path:
    """Rename an experiment folder, returning the new path.

    Raises:
        FileExistsError: If the target name already exists.
        ValueError: If ``new_name`` is empty or contains path separators.
    """
    clean = new_name.strip()
    if not clean or "/" in clean or "\\" in clean:
        raise ValueError("Invalid experiment name.")
    target = path.parent / clean
    if target.exists():
        raise FileExistsError(f"'{clean}' already exists.")
    path.rename(target)
    return target


def export_archive(path: Path, dest_dir: Path) -> Path:
    """Zip an experiment folder into ``dest_dir`` and return the archive path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_base = dest_dir / path.name
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=path)
    return Path(archive)


def open_in_file_manager(path: Path) -> bool:
    """Reveal a folder in the OS file manager. Returns whether it was launched.

    This is a pure UI convenience (Finder/Explorer) — not part of any training
    pipeline. It is best-effort and never raises.
    """
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("win"):
            subprocess.run(["explorer", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        return True
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Aggregate helpers (Dashboard)
# --------------------------------------------------------------------------- #
def best_record(
    records: list[ExperimentRecord], metric: str = "f1_macro"
) -> Optional[ExperimentRecord]:
    """Return the record with the highest value of ``metric`` (or ``None``)."""
    scored = [r for r in records if getattr(r, metric) is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: getattr(r, metric))


def to_dataframe(records: list[ExperimentRecord]) -> pd.DataFrame:
    """Return a display DataFrame of experiment rows."""
    return pd.DataFrame([r.as_row() for r in records])


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _short_date(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"
