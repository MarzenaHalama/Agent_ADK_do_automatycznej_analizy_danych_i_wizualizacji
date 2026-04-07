"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def regression_df() -> pd.DataFrame:
    """Small synthetic regression dataset."""
    rng = np.random.RandomState(0)
    n = 100
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    y = 3 * x1 + 2 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture()
def classification_df() -> pd.DataFrame:
    """Small synthetic classification dataset (3 classes)."""
    rng = np.random.RandomState(42)
    n = 150
    features = rng.randn(n, 4)
    labels = np.array(["setosa"] * 50 + ["versicolor"] * 50 + ["virginica"] * 50)
    # Shift means to make classes separable
    features[:50] += 2
    features[100:] -= 2
    df = pd.DataFrame(features, columns=["f1", "f2", "f3", "f4"])
    df["species"] = labels
    return df


@pytest.fixture()
def timeseries_df() -> pd.DataFrame:
    """Small time series with one clear anomaly."""
    dates = pd.date_range("2025-09-01", periods=30, freq="D")
    rng = np.random.RandomState(7)
    liters = rng.normal(230, 15, 30)
    liters[5] = 1600  # anomaly
    return pd.DataFrame({"date": dates, "liters": liters})


@pytest.fixture()
def dirty_csv_df() -> pd.DataFrame:
    """DataFrame simulating BOM and comma-decimal CSV quirks."""
    return pd.DataFrame({
        "\ufeffcol_a": ["1,5", "2,3", "3,7"],
        " col_b ": ["10", "20", "30"],
        "target": [1, 2, 3],
    })
