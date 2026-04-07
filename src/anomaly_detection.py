"""
Anomaly detection module.

Combines two complementary methods:
  - **Isolation Forest** -- tree-based unsupervised anomaly detector.
  - **Z-score filtering** -- flags values more than *k* standard deviations
    away from the column mean.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

from src.config import ANOMALY_DEFAULTS
from src.data_loader import _coerce_numeric_columns


def detect_anomalies(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    contamination: float = ANOMALY_DEFAULTS.contamination,
    zscore_threshold: float = ANOMALY_DEFAULTS.zscore_threshold,
    random_state: int = ANOMALY_DEFAULTS.random_state,
) -> Dict[str, Any]:
    """Detect anomalies using Isolation Forest and Z-score.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    features : list[str] | None
        Columns to analyse.  *None* = all numeric columns.
    contamination : float
        Expected proportion of anomalies (Isolation Forest parameter).
    zscore_threshold : float
        Absolute Z-score above which a value is flagged.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``features``, ``index``, ``isoforest_labels`` (-1 = anomaly,
        1 = normal), ``isoforest_scores``, ``zscore_outliers``,
        ``anomaly_indices`` (union of both methods).
    """
    df = _coerce_numeric_columns(df)

    if features is None:
        features = list(df.select_dtypes(include=np.number).columns)

    X = df[features].dropna()
    idx = X.index

    if X.empty:
        return _empty_result(features)

    # Isolation Forest
    iso = IsolationForest(
        contamination=contamination, random_state=random_state,
    )
    iso.fit(X)
    iso_labels = iso.predict(X)          # 1 = normal, -1 = anomaly
    iso_scores = iso.decision_function(X)

    # Z-score
    z = X.apply(zscore)
    z_outlier_mask = (z.abs() > zscore_threshold).any(axis=1)

    # Combined anomaly mask
    combined_mask = (iso_labels == -1) | z_outlier_mask.values

    return {
        "features": features,
        "index": idx.tolist(),
        "isoforest_labels": iso_labels.tolist(),
        "isoforest_scores": iso_scores.tolist(),
        "zscore_outliers": z_outlier_mask.astype(int).tolist(),
        "anomaly_indices": idx[combined_mask].tolist(),
    }


def _empty_result(features: List[str]) -> Dict[str, Any]:
    """Return a well-formed but empty result dictionary."""
    return {
        "features": features,
        "index": [],
        "isoforest_labels": [],
        "isoforest_scores": [],
        "zscore_outliers": [],
        "anomaly_indices": [],
    }
