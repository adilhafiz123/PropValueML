"""Data processing module for house price prediction model."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the raw data set and perform basic cleaning."""
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop obvious duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Removed %d duplicate rows", before - len(df))

    # Handle missing target values
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    return df

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature engineering & target separation."""
    # Basic date features
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df.drop(columns=["date"], inplace=True)

    # Remove extreme price outliers (1st-99th percentile)
    lower, upper = np.percentile(df["price"], [1, 99])
    df = df[(df["price"] >= lower) & (df["price"] <= upper)]

    # Separate target
    y = df.pop("price")
    X = df
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer that scales numeric and encodes categorical."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info("Numeric features: %s", numeric_features)
    logger.info("Categorical features: %s", categorical_features)

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor 