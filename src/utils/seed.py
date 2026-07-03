"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all relevant RNGs for reproducible runs.

    Args:
        seed: The seed value applied to Python, NumPy and PyTorch RNGs.
        deterministic: If ``True``, request deterministic cuDNN behaviour.
            Note that full determinism is not guaranteed on all backends
            (e.g. MPS), but results become as reproducible as possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
