"""Compute-device detection helpers for the UI.

Wraps :func:`src.utils.get_device` and adds a human-readable label plus a list
of the backends actually available on this machine (for the Settings page).
"""

from __future__ import annotations

import torch

from src.utils import get_device


def resolve_device(prefer: str | None = None) -> torch.device:
    """Return the device to use, honouring an optional preference.

    Args:
        prefer: One of ``"auto"``, ``"mps"``, ``"cuda"``, ``"cpu"`` (or ``None``).
            ``"auto"``/``None`` defer to the backend's MPS -> CUDA -> CPU order.
    """
    if prefer in (None, "auto"):
        return get_device()
    return get_device(prefer=prefer)


def available_devices() -> list[str]:
    """Return the compute backends available on this machine (plus ``auto``)."""
    devices = ["auto", "cpu"]
    if torch.backends.mps.is_available():
        devices.insert(1, "mps")
    if torch.cuda.is_available():
        devices.insert(1, "cuda")
    return devices


def device_label(device: torch.device | str) -> str:
    """Return a short uppercase label such as ``MPS`` / ``CUDA`` / ``CPU``."""
    name = str(device).split(":", 1)[0]
    return name.upper()
