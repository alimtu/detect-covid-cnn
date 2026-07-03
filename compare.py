"""Compare metrics across experiments.

Usage:
    python compare.py                       # compare all experiments in outputs/
    python compare.py --experiments experiment_001 experiment_002
    python compare.py --output-root outputs

Reads each experiment's ``summary.json`` and prints a side-by-side table of the
key test metrics, then writes ``comparison.txt`` / ``comparison.csv`` /
``comparison.json`` under the output root.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils import get_logger

_METRIC_KEYS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare experiments.")
    parser.add_argument("--output-root", default="outputs", help="Root folder of experiments.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Specific experiment folder names to compare (default: all).",
    )
    return parser.parse_args()


def _load_summaries(root: Path, names: list[str] | None) -> dict[str, dict]:
    """Load ``summary.json`` for the requested (or all) experiments."""
    if names:
        dirs = [root / n for n in names]
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "summary.json").is_file())

    summaries: dict[str, dict] = {}
    for exp_dir in dirs:
        summary_path = exp_dir / "summary.json"
        if not summary_path.is_file():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summaries[exp_dir.name] = json.load(handle)
    return summaries


def _row_values(summary: dict) -> dict[str, float]:
    final = summary.get("final_metrics", {})
    return {key: float(final.get(key, float("nan"))) for key in _METRIC_KEYS}


def _format_table(summaries: dict[str, dict]) -> str:
    header = f"{'experiment':<20}{'model':<18}" + "".join(f"{k:>16}" for k in _METRIC_KEYS)
    lines = [header, "-" * len(header)]
    for name, summary in summaries.items():
        model = str(summary.get("model", {}).get("name", "?"))
        values = _row_values(summary)
        row = f"{name:<20}{model:<18}" + "".join(f"{values[k]:>16.4f}" for k in _METRIC_KEYS)
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logger = get_logger("covid.compare")
    root = Path(args.output_root)
    if not root.is_dir():
        logger.error("Output root not found: %s", root)
        return

    summaries = _load_summaries(root, args.experiments)
    if not summaries:
        logger.error("No experiment summaries found in %s. Run an experiment first.", root)
        return

    table = _format_table(summaries)
    print(table)

    (root / "comparison.txt").write_text(table + "\n", encoding="utf-8")

    comparison = {
        name: {"model": s.get("model", {}).get("name"), **_row_values(s)}
        for name, s in summaries.items()
    }
    with (root / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "model", *_METRIC_KEYS])
        for name, values in comparison.items():
            writer.writerow([name, values["model"], *[values[k] for k in _METRIC_KEYS]])

    best = max(summaries, key=lambda n: _row_values(summaries[n]).get("f1_macro", float("-inf")))
    logger.info("Best experiment by macro F1: %s", best)
    logger.info("Comparison written to %s", root / "comparison.txt")


if __name__ == "__main__":
    main()
