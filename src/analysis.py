"""
Machine learning analysis module.

Provides functions to run:
  - Exploratory statistics
  - Linear regression (baseline)
  - MLP regression and classification via scikit-learn
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MLP_DEFAULTS
from src.data_loader import _coerce_numeric_columns, prepare_features


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Return basic descriptive statistics for a DataFrame.

    Returns
    -------
    dict
        Keys: ``shape``, ``dtypes``, ``missing_per_column``,
        ``describe_numeric``.
    """
    numeric = df.select_dtypes(include=np.number)
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_column": df.isna().sum().to_dict(),
        "describe_numeric": numeric.describe().to_dict() if not numeric.empty else {},
    }


def infer_task(y: pd.Series) -> str:
    """Determine whether the target suggests regression or classification."""
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        return "regression"
    if pd.api.types.is_numeric_dtype(y) and y.nunique() <= 20:
        return "classification"
    return "classification"


def run_analysis(
    df: pd.DataFrame,
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
    task: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    mlp_hidden: Tuple[int, ...] = MLP_DEFAULTS.hidden_layer_sizes,
    mlp_max_iter: int = MLP_DEFAULTS.max_iter,
) -> Dict[str, Any]:
    """Run a complete analysis pipeline on a DataFrame.

    Steps
    -----
    1. Compute descriptive statistics.
    2. Prepare features and target.
    3. Train/test split.
    4. Fit a linear regression baseline (regression tasks only).
    5. Fit an MLP model (regression or classification).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str | None
        Name of the target column.  If *None*, only statistics are returned.
    features : list[str] | None
        Explicit feature list.  *None* = auto-detect numeric columns.
    task : str
        ``"regression"``, ``"classification"``, or ``"auto"``.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.
    mlp_hidden : tuple[int, ...]
        Hidden layer sizes for the MLP.
    mlp_max_iter : int
        Maximum iterations for the MLP solver.

    Returns
    -------
    dict
        Nested dictionary with statistics, model metrics, and predictions.
    """
    np.random.seed(random_state)
    df = _coerce_numeric_columns(df)

    results: Dict[str, Any] = {"basic_stats": compute_statistics(df)}

    if target is None:
        results["note"] = "No target column specified; returning statistics only."
        return results

    X, y = prepare_features(df, target, features)

    if task == "auto":
        task = infer_task(y)

    results["task"] = task
    results["target"] = target
    results["features"] = list(X.columns)

    # Train / test split
    stratify = y if task == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )

    # Linear regression baseline
    if task == "regression":
        results["linear_regression"] = _fit_linear_regression(X_train, X_test, y_train, y_test, X.columns)
    else:
        results["linear_regression"] = None

    # MLP model
    results["mlp"] = _fit_mlp(
        task, X_train, X_test, y_train, y_test,
        mlp_hidden, mlp_max_iter, random_state,
    )

    return results


# -------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------

def _fit_linear_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    col_names: pd.Index,
) -> Dict[str, Any]:
    """Fit an OLS linear regression and return metrics."""
    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return {
        "coefficients": dict(zip(col_names, np.ravel(lr.coef_))),
        "intercept": float(lr.intercept_),
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "residuals": (y_test.values - y_pred).tolist(),
    }


def _fit_mlp(
    task: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    hidden: Tuple[int, ...],
    max_iter: int,
    random_state: int,
) -> Dict[str, Any]:
    """Fit an MLP (regressor or classifier) and return metrics + predictions."""
    if task == "regression":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=hidden,
                max_iter=max_iter,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=random_state,
            )),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        return {
            "type": "MLPRegressor",
            "hidden_layers": hidden,
            "r2": float(r2_score(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "residuals": (y_test.values - y_pred).tolist(),
            "test_index": y_test.index.tolist(),
        }

    # Classification
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            early_stopping=False,
            random_state=random_state,
        )),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "type": "MLPClassifier",
        "hidden_layers": hidden,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
        "classes": sorted(y_test.unique().tolist()),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "test_index": y_test.index.tolist(),
    }
