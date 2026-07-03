"""CSV logger for per-epoch training history."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


class CSVLogger:
    """Append per-epoch metric rows to a CSV file.

    The header is written from the keys of the first logged row.

    Args:
        path: Destination CSV path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: Mapping[str, object]) -> None:
        """Append one row, writing the header on first use."""
        write_header = self._fieldnames is None
        if write_header:
            self._fieldnames = list(row.keys())
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(dict(row))
