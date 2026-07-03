"""Canonical project paths used across the frontend.

Centralising these keeps the services free of scattered string literals and makes
the on-disk layout easy to reason about.
"""

from __future__ import annotations

from pathlib import Path

# Project root = two levels up from this file (frontend/services/paths.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
BASE_CONFIG_PATH: Path = CONFIGS_DIR / "base.yaml"
EXPERIMENT_CONFIGS_DIR: Path = CONFIGS_DIR / "experiments"
SAVED_CONFIGS_DIR: Path = CONFIGS_DIR / "saved"

OUTPUTS_ROOT: Path = PROJECT_ROOT / "outputs"
JOBS_DIR: Path = OUTPUTS_ROOT / ".jobs"

APP_SETTINGS_PATH: Path = CONFIGS_DIR / "app_settings.json"

FONTS_DIR: Path = PROJECT_ROOT / "fonts" / "IRANYekanX" / "FaNum"


def ensure_runtime_dirs() -> None:
    """Create directories the app writes into (idempotent)."""
    for directory in (OUTPUTS_ROOT, JOBS_DIR, SAVED_CONFIGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
