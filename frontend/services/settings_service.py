"""Persistent application settings (project defaults, theme, device).

Settings are stored as JSON at :data:`~frontend.services.paths.APP_SETTINGS_PATH`
and provide the defaults the Train page uses when seeding a fresh configuration,
plus UI preferences (theme, auto-save).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from frontend.services import paths


@dataclass
class AppSettings:
    """User-editable project defaults and preferences."""

    default_dataset_path: str = "DATA"
    default_output_root: str = "outputs"
    default_device: str = "auto"  # auto | mps | cuda | cpu
    default_image_size: int = 224
    default_optimizer: str = "adamw"
    default_scheduler: str = "reduce_on_plateau"
    default_model: str = "densenet121"
    default_epochs: int = 5
    default_batch_size: int = 32
    theme: str = "dark"  # dark | light
    auto_save_config: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


def load_settings() -> AppSettings:
    """Load settings from disk, falling back to defaults for missing keys."""
    path = paths.APP_SETTINGS_PATH
    if not path.is_file():
        return AppSettings()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return AppSettings.from_dict(data)
    except (json.JSONDecodeError, OSError):
        return AppSettings()


def save_settings(settings: AppSettings) -> None:
    """Persist settings to disk (creating the configs directory if needed)."""
    paths.APP_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with paths.APP_SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(settings.to_dict(), handle, indent=2)
