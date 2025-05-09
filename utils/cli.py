"""Command-line interface utilities for the house price prediction model."""

import argparse
from pathlib import Path

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a house-price model.")
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="CSV file containing the raw house-price data.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="If set, save the best model pipeline to <data-file>.joblib",
    )
    return parser.parse_args(argv) 