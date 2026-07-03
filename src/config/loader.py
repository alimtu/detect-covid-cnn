"""Layered YAML configuration loading.

Configuration is composed from a base file plus small per-experiment override
files. An override file may declare its parent with a top-level ``base:`` key
(a path relative to the override file). Bases are resolved recursively and
deep-merged, so an experiment file only needs to specify what changes.

Example ``configs/experiments/resnet50.yaml``::

    base: ../base.yaml
    model:
      name: resnet50
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

# Key used by an override file to reference its parent config.
_BASE_KEY = "base"


class Config(Mapping):
    """Immutable, nested view over a configuration dictionary.

    Supports attribute access (``cfg.training.epochs``), item access
    (``cfg["training"]["epochs"]``) and dotted lookups
    (``cfg.get("training.epochs", default)``).
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data: dict[str, Any] = dict(data)

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, Mapping) else value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, Mapping) else value

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Look up ``key`` supporting dotted paths like ``"training.epochs"``."""
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, Mapping) else node

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copied plain dictionary representation."""

        def _plain(value: Any) -> Any:
            if isinstance(value, Config):
                return value.to_dict()
            if isinstance(value, Mapping):
                return {k: _plain(v) for k, v in value.items()}
            return value

        return {k: _plain(v) for k, v in self._data.items()}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({self._data!r})"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], dict(value))
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(path: Path, _seen: set[Path]) -> dict[str, Any]:
    """Load ``path`` and merge it onto its (recursively resolved) base."""
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular config inheritance detected at: {path}")
    _seen.add(path)

    data = _read_yaml(path)
    base_ref = data.pop(_BASE_KEY, None)
    if base_ref is None:
        return data

    base_path = (path.parent / str(base_ref)).resolve()
    base_data = _resolve(base_path, _seen)
    return _deep_merge(base_data, data)


def load_config(path: str | Path) -> Config:
    """Load a (possibly layered) configuration file.

    Args:
        path: Path to a base or experiment-override YAML file.

    Returns:
        The fully merged :class:`Config`.

    Raises:
        FileNotFoundError: If any referenced file is missing.
        ValueError: If a circular ``base`` reference is detected.
    """
    merged = _resolve(Path(path), set())
    return Config(merged)
