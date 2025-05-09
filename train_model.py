#!/usr/bin/env python
"""House Price Prediction Model
--------------------------------
A robust, production-ready script that trains and evaluates
both Linear Regression and Random Forest Regressor models
on a cleaned London house-price dataset. The best model is
selected based on cross-validated RMSE and can optionally
be saved to disk.

Usage
-----
python train_model.py --data-file london_house_prices.csv --save-model

Author: Adil Hafiz <adilhafiz@hotmail.co.uk>
Date: 2025-05-09
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from data.processor import load_data, clean_data, build_preprocessor
from models.trainer import (
    build_models,
    evaluate_models,
    train_final_model,
    save_model,
    predict_example,
)
from utils.cli import parse_args

###############################################################################
# Logging configuration
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    raw_df = load_data(args.data_file)
    X, y = clean_data(raw_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    best_name, best_model, _ = evaluate_models(models, X_train, y_train)
    best_model = train_final_model(best_model, X_train, y_train)

    # Evaluate on hold-out validation set
    y_pred = best_model.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_pred, squared=False)
    logger.info("Validation RMSE of best model (%s): %.2f", best_name, rmse_val)

    if args.save_model:
        save_model(best_model, args.data_file)

    # Show a prediction example
    predict_example(best_model, X_train)

if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:]) 