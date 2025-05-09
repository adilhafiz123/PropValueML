"""Model training and evaluation module for house price prediction."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def build_models(preprocessor) -> dict[str, Pipeline]:
    """Return candidate model pipelines keyed by name."""
    models: dict[str, Pipeline] = {}

    models["linear_regression"] = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])

    models["random_forest"] = Pipeline([
        ("preprocess", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])
    return models

def evaluate_models(
    models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series
) -> tuple[str, Pipeline, float]:
    """Cross-validate each model and return the best."""
    best_name: str | None = None
    best_model: Pipeline | None = None
    best_rmse = float("inf")

    for name, pipe in models.items():
        logger.info("Evaluating %s …", name)
        neg_mse_scores = cross_val_score(
            pipe,
            X,
            y,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        rmse_scores = np.sqrt(-neg_mse_scores)
        rmse = rmse_scores.mean()
        logger.info("%s RMSE: %.2f ± %.2f", name, rmse, rmse_scores.std())

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipe
            best_name = name

    assert best_model is not None and best_name is not None  # mypy safety
    logger.info("Best model: %s (RMSE %.2f)", best_name, best_rmse)
    return best_name, best_model, best_rmse

def train_final_model(best_model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit the best pipeline on the full training data."""
    logger.info("Training final %s model on full data …", best_model.named_steps["model"].__class__.__name__)
    best_model.fit(X, y)
    return best_model

def save_model(pipe: Pipeline, ref_path: Path) -> Path:
    """Serialize the fitted pipeline to <ref_path>.joblib"""
    out_path = ref_path.with_suffix(".joblib")
    joblib.dump(pipe, out_path)
    logger.info("Model saved to %s", out_path)
    return out_path

def predict_example(pipe: Pipeline, X_template: pd.DataFrame) -> None:
    """Demonstrate predicting a single property given a dict of features."""
    example_features = {col: X_template[col].iloc[0] for col in X_template.columns}

    logger.info("\nExample input features: %s", example_features)
    pred_price = pipe.predict(pd.DataFrame([example_features]))[0]
    logger.info("Predicted price: £%.0f", pred_price) 