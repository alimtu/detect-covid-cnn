"""Optional TensorBoard logger.

TensorBoard is an optional dependency. If it is unavailable the logger degrades
gracefully to a no-op so training never fails because of logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

_logger = logging.getLogger("covid.tensorboard")


class TensorBoardLogger:
    """Thin wrapper around ``SummaryWriter`` that no-ops when unavailable.

    Args:
        log_dir: Directory for TensorBoard event files.
        enabled: Whether TensorBoard logging is requested.
    """

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self._writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))
        except Exception as exc:  # noqa: BLE001 - optional dependency
            _logger.warning("TensorBoard unavailable (%s); disabling TB logging.", exc)
            self.enabled = False

    def log_scalars(self, metrics: Mapping[str, float], step: int) -> None:
        """Log a mapping of scalar metrics at ``step``."""
        if self._writer is None:
            return
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
