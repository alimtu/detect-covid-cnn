"""Run a full experiment: train + evaluate into a fresh experiment folder.

Usage:
    python run.py --config configs/experiments/densenet121.yaml

Everything about the run is driven by the config. Results are written to an
auto-incremented ``outputs/experiment_NNN_<model>/`` folder containing the
resolved config, per-epoch history, metrics, plots and checkpoints.

The orchestration itself lives in :func:`src.pipeline.run_experiment` so the
exact same pipeline powers the Streamlit dashboard's background training.
"""

from __future__ import annotations

import argparse

from src.config import load_config, validate_config
from src.pipeline import run_experiment
from src.utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate an experiment.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to a base or experiment-override YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("covid.run")

    config = load_config(args.config)
    validate_config(config)
    run_experiment(config, logger=logger, validate=False)


if __name__ == "__main__":
    main()
