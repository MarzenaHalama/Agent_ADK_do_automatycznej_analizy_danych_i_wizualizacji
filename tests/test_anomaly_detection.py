"""Tests for the anomaly_detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import detect_anomalies


class TestDetectAnomalies:
    """Core anomaly detection tests."""

    def test_returns_required_keys(self, timeseries_df):
        result = detect_anomalies(timeseries_df, features=["liters"])
        expected_keys = {
            "features", "index", "isoforest_labels",
            "isoforest_scores", "zscore_outliers", "anomaly_indices",
        }
        assert expected_keys == set(result.keys())

    def test_detects_obvious_anomaly(self, timeseries_df):
        """The spike at index 5 (1600 litres) must be flagged."""
        result = detect_anomalies(
            timeseries_df, features=["liters"], contamination=0.1,
        )
        assert 5 in result["anomaly_indices"]

    def test_labels_length_matches_data(self, timeseries_df):
        result = detect_anomalies(timeseries_df, features=["liters"])
        n_valid = timeseries_df["liters"].dropna().shape[0]
        assert len(result["isoforest_labels"]) == n_valid

    def test_scores_are_floats(self, timeseries_df):
        result = detect_anomalies(timeseries_df, features=["liters"])
        assert all(isinstance(s, float) for s in result["isoforest_scores"])

    def test_normal_data_few_anomalies(self):
        """A purely normal distribution should yield very few anomalies."""
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"val": rng.normal(100, 5, 500)})
        result = detect_anomalies(df, features=["val"], contamination=0.01)
        anomaly_ratio = len(result["anomaly_indices"]) / 500
        assert anomaly_ratio < 0.05

    def test_auto_feature_selection(self, timeseries_df):
        """When features is None, all numeric columns are used."""
        result = detect_anomalies(timeseries_df, features=None)
        assert "liters" in result["features"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        result = detect_anomalies(df, features=["x"])
        assert result["isoforest_labels"] == []
        assert result["anomaly_indices"] == []


class TestZscoreDetection:
    """Verify the Z-score component of anomaly detection."""

    def test_extreme_value_flagged_by_zscore(self):
        """A value 10 std away should be flagged."""
        vals = [100.0] * 99 + [10000.0]
        df = pd.DataFrame({"val": vals})
        result = detect_anomalies(df, features=["val"], zscore_threshold=3.0)
        assert 99 in result["anomaly_indices"]

    def test_moderate_values_not_flagged(self):
        """Values within 2 std should not trip the Z > 3 filter."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"val": rng.normal(50, 1, 200)})
        result = detect_anomalies(
            df, features=["val"], contamination=0.01, zscore_threshold=3.0,
        )
        zscore_flags = sum(result["zscore_outliers"])
        assert zscore_flags <= 3  # at most a few false positives


class TestContaminationParameter:
    """Verify that contamination controls sensitivity."""

    def test_higher_contamination_more_anomalies(self, timeseries_df):
        low = detect_anomalies(timeseries_df, features=["liters"], contamination=0.01)
        high = detect_anomalies(timeseries_df, features=["liters"], contamination=0.20)
        assert len(high["anomaly_indices"]) >= len(low["anomaly_indices"])
