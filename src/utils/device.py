"""Hardware device selection with priority MPS -> CUDA -> CPU."""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available compute device.

    Selection priority is Apple Silicon (MPS) -> CUDA -> CPU. An explicit
    preference can be passed to force a specific backend when available.

    Args:
        prefer: Optional device string ("mps", "cuda" or "cpu"). Ignored if
            the requested backend is unavailable.

    Returns:
        The selected :class:`torch.device`.
    """
    if prefer:
        prefer = prefer.lower()
        if prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "cpu":
            return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
